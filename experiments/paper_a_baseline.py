"""Baselines for the do-no-harm-gate paper, evaluated on the SAME benches/protocol as run_recipe.

Modes (env GCM_BASELINE):
  sft   -- SFT-LoRA: train a read-LoRA on the FULL (truncated) context, NO compression, NO gate. The natural
           "why not just LoRA-tune the reader on full context?" baseline. Cost = full L-token KV at inference.
  gist  -- Gist-lite (Mu et al. 2023): K gist tokens inserted after the ctx; a gist mask blocks query/answer from
           attending the ctx (they may attend only the gist tokens). Train read-LoRA + gist embeddings. At inference
           the gist KV replaces the ctx. This is "learned compression WITHOUT our proj-MLP/manifold/distill/recon/gate"
           -- the closest prior-art compression baseline.

Single GPU per cell (the autonomous runner already pins one). Reuses GCMModel.mc_loglik / _gen_batch for eval so the
numbers are directly comparable to run_recipe. Writes RECIPE_EVAL <tag> {...} like run_recipe (same downstream parser).

Launch:  CUDA_VISIBLE_DEVICES=g GCM_BASELINE=sft GCM_TRAIN=bfcl_live_multiple ... python run_baseline.py <model>
"""
import os
import json
import math
import sys
import string
from pathlib import Path

import torch

from gcm import GCMModel, enable_torch_linear_attention
enable_torch_linear_attention()
from gcm.lora import checkpoint_lora_context, set_lora_enabled  # noqa: E402
from mem_embedding.gcm.runtime import Runtime                  # noqa: E402
from mem_embedding.gcm.data import load_items, score_gen, MC_BENCHES, options_for  # noqa: E402


def env(k, d):
    return os.environ.get(k, d)


def atomic_json_dump(payload, path):
    temp_path = f"{path}.tmp.{os.getpid()}"
    with open(temp_path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _install_llmlingua_cache_compat(compressor):
    """Adapt LLMLingua 0.2.x's legacy cache lists to Transformers 5 caches."""
    from transformers import DynamicCache

    def get_ppl(
        text,
        granularity="sentence",
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        return_kv=False,
        end=None,
        condition_mode="none",
        condition_pos_id=0,
    ):
        if input_ids is None:
            tokenized = compressor.tokenizer(text, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(compressor.device)
            attention_mask = tokenized["attention_mask"].to(compressor.device)
        past_length = (
            past_key_values[0][0].shape[2]
            if past_key_values is not None
            else 0
        )
        if end is None:
            end = input_ids.shape[1]
        end = min(end, past_length + compressor.max_position_embeddings)
        cache = (
            DynamicCache(past_key_values, config=compressor.model.config)
            if past_key_values is not None
            else None
        )
        with torch.no_grad():
            response = compressor.model(
                input_ids[:, past_length:end],
                attention_mask=attention_mask[:, :end],
                past_key_values=cache,
                use_cache=True,
            )
        legacy_cache = [
            [key, value]
            for key, value, *_ in response.past_key_values
        ]
        shift_logits = response.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., past_length + 1:end].contiguous()
        active = (attention_mask[:, past_length:end] == 1)[..., :-1].reshape(-1)
        active_logits = shift_logits.reshape(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.reshape(-1)[active]
        loss = torch.nn.functional.cross_entropy(
            active_logits,
            active_labels,
            reduction="none",
        )
        if condition_mode == "before":
            loss = loss[:condition_pos_id]
        elif condition_mode == "after":
            loss = loss[condition_pos_id:]
        result = loss.mean() if granularity == "sentence" else loss
        return (result, legacy_cache) if return_kv else result

    compressor.get_ppl = get_ppl


def main():
    dev = "cuda:0"
    torch.cuda.set_device(0)
    SEED = int(env("GCM_SEED", "1234"))
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    MODEL = sys.argv[1]; mname = MODEL.rstrip("/").split("/")[-1]
    MODE = env("GCM_BASELINE", "sft").lower()
    K = int(env("GCM_K", "64")); LORA = int(env("GCM_LORA", "64"))
    LR = float(env("GCM_LR", "2e-4")); ACCUM = int(env("GCM_ACCUM", "8"))
    NTR = int(env("GCM_NTRAIN", "3000")); NVAL = int(env("GCM_NVAL", "128"))
    MAXCTX = int(env("GCM_MAXCTX", "4096")); GOLD_MAX = int(env("GCM_GOLD_MAX", "64"))
    SOURCE_MAXCTX = int(env("GCM_SOURCE_MAXCTX", str(MAXCTX)))
    NCHUNKS = int(env("GCM_NCHUNKS", "8"))  # length knob for the NIAH/length-sweep family (was hardcoded 8)
    RAG_BUDGET = int(env("GCM_RAG_BUDGET", "2048"))  # rag: token budget of retrieved passages (the retrieval "compression" knob)
    RAG_CHUNK = int(env("GCM_RAG_CHUNK", "128"))     # rag: passage size in tokens
    TOME_RATIO = float(env("GCM_TOME_RATIO", "0.5")) # tome: fraction of ctx tokens KEPT after merging
    TOME_SIM = env("GCM_TOME_SIM", "embed")          # tome: similarity space (embed | random control)
    GEN_MAX = int(env("GCM_GEN_MAX", "64")); GEN_BS = int(env("GCM_GEN_BS", "16"))
    STRICT_EVAL = env("GCM_STRICT_EVAL", "1") == "1"
    MAXSTEPS = int(env("GCM_MAX_STEPS", "1500"))
    TAG = env("GCM_TAG", f"base_{MODE}")
    DEFB = "bfcl_live_multiple,toolace,squad_v2,narrativeqa"
    TRAIN_BENCHES = env("GCM_TRAIN", env("GCM_BENCHES", DEFB)).split(",")
    EVAL_BENCHES = env("GCM_EVAL", env("GCM_BENCHES", DEFB)).split(",")

    rt = Runtime(MODEL, device=dev, n_memory=K, max_ctx_tokens=SOURCE_MAXCTX, max_new_tokens=GOLD_MAX)
    # GCMModel gives us the read-LoRA (and, for gist, the K memory tokens used AS gist embeddings) + the eval methods.
    model = GCMModel(rt.model, rt.tok, n_memory=K, lora_rank=LORA, normalize=False, norm_mode="off").to(dev)
    base, embed = model.base, model._embed
    outdir = str(
        Path(env("GCM_OUT", Path(__file__).resolve().parents[1] / "out"))
        / mname
    )
    os.makedirs(outdir, exist_ok=True)
    adapter_path = f"{outdir}/{TAG}_adapters.pt"
    load_adapter = env("GCM_LOAD_ADAPTER", "")
    if MODE == "sft" and env("GCM_GRAD_CKPT", "1") != "0":
        try:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False,
                    "context_fn": checkpoint_lora_context(base),
                }
            )
        except TypeError:
            base.gradient_checkpointing_enable()
    if load_adapter:
        model.load_adapters(load_adapter, map_location=dev)

    def prep(it):
        g = rt.tok(" " + str(it.gold).strip() + "\n", add_special_tokens=False).input_ids[:GOLD_MAX]
        return (rt.ctx_ids(it), rt.query_ids(it), torch.tensor([g], device=dev)) if g else None

    def source_token_count(it):
        text = "\n".join(str(value) for value in (getattr(it, "chunks", []) or []))
        return len(rt.tok(text, add_special_tokens=False).input_ids)

    def document_id(it):
        import hashlib
        text = "\n".join(str(value) for value in (getattr(it, "chunks", []) or []))
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    pool = []
    for b in TRAIN_BENCHES:
        pool += [x for it in load_items(b, NTR, NCHUNKS, 0, "train") if (x := prep(it))]
    val_pool = [(x, b, it) for b in EVAL_BENCHES for it in load_items(b, NVAL, NCHUNKS, 1, os.environ.get("GCM_EVAL_SPLIT","validation")) if (x := prep(it))]
    def _item_sig(x):
        import hashlib
        h = hashlib.sha256()
        for tensor in x:
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.digest()
    if env("GCM_LEAKAGE_GUARD", "1").lower() not in ("0", "off", "false"):
        val_signatures = {_item_sig(x) for x, _, _ in val_pool}
        pool = [x for x in pool if _item_sig(x) not in val_signatures]

    # ---- gist mask: over [ctx(lc) ; gist(K) ; query(lq) ; gold(lg)] block the query+gold from seeing the ctx;
    # they may attend only the K gist tokens (which DID see the ctx). Causal elsewhere. (Mu et al. 2023.) ----
    def gist_mask(lc, K_, lq, lg, dtype, device):
        L = lc + K_ + lq + lg
        mn = torch.finfo(dtype).min
        m = torch.triu(torch.full((L, L), mn, device=device, dtype=dtype), diagonal=1)
        qs = lc + K_
        m[qs:, :lc] = mn                                     # query+gold cannot attend the ctx (only the gist tokens)
        return m.view(1, 1, L, L)

    # ---- training ----
    if MODE == "sft":
        params = list(model.lora_params)                    # SFT: only the read-LoRA trains (frozen base, no M)
    else:                                                    # gist / cart: read-LoRA + the K memory/gist/cartridge tokens
        params = list(model.lora_params) + [model.compressor.mem]
    opt = torch.optim.AdamW(params, lr=LR)
    warm = max(1, int(0.05 * (MAXSTEPS // ACCUM)))
    def lr_at(o):
        return (o + 1) / warm if o < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (o - warm) / max(1, MAXSTEPS // ACCUM - warm))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    n = len(pool); import random as _r; perm = list(range(n)); _r.Random(SEED).shuffle(perm)
    TXL_W = int(env("GCM_TXL_WINDOW", "1024"))          # txl/streaming bounded-KV window (no compression)
    LL_RATE = float(env("GCM_LL_RATE", "0.33"))         # LLMLingua-2 target keep-rate (hard prompt compression)
    LL_TARGET = int(env("GCM_LL_TARGET", "0"))          # paper runs set an explicit reader-token budget
    MATCH_READER_BUDGET = env("GCM_MATCH_READER_BUDGET", "0") == "1"
    MATCH_CHUNK = int(env("GCM_MATCH_CHUNK", "4096"))
    MATCH_K = int(env("GCM_MATCH_K", str(K)))
    _PC = None; _SC = None
    if MODE == "llmlingua":                             # LLMLingua-2 (token classifier) — official model
        from llmlingua import PromptCompressor
        _PC = PromptCompressor(model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                               use_llmlingua2=True, device_map=dev)
    elif MODE in ("longllmlingua", "llmlingua_orig"):   # EXACT LLMLingua / LongLLMLingua (perplexity-based, authors' compressor LM)
        from llmlingua import PromptCompressor
        _PC = PromptCompressor(model_name=env("GCM_LL_LM", "NousResearch/Llama-2-7b-hf"),
                               use_llmlingua2=False, device_map=dev)
        if not hasattr(_PC.model.generation_config, "return_legacy_cache"):
            _install_llmlingua_cache_compat(_PC)
    elif MODE == "selective_context":                   # EXACT Selective-Context (Li 2023) — authors' pkg + gpt2 default
        from selective_context import SelectiveContext
        _SC = SelectiveContext(model_type=env("GCM_SC_LM", "gpt2"), lang="en")
    opt.zero_grad()
    _EVAL_ONLY = ("txl", "llmlingua", "rag", "longllmlingua", "llmlingua_orig", "selective_context", "tome", "imp")
    train_steps = 0 if MODE in _EVAL_ONLY or load_adapter else MAXSTEPS
    if train_steps:
        # Runtime freezes the base in eval mode. Gradient checkpointing requires training mode even
        # when only LoRA parameters receive gradients.
        model.train()
    for step in range(train_steps):  # eval-only modes and loaded adapters do not train
        cid, qid, gold = pool[perm[step % n]]
        c = cid[:, :MAXCTX]
        set_lora_enabled(base, True)
        if MODE == "sft":
            seq = torch.cat([embed(c), embed(qid), embed(gold)], 1)
            keep = int(gold.shape[1]) + 1
            import contextlib
            offload = env("GCM_SFT_OFFLOAD", "0") == "1"
            cm = torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else contextlib.nullcontext()
            with cm:
                lg = base(
                    inputs_embeds=seq,
                    use_cache=False,
                    logits_to_keep=keep,
                ).logits[0, :gold.shape[1]].float()
        elif MODE == "cart":                                 # Cartridge-lite: FIXED amortized prefix (no ctx); [mem ; query ; gold]
            cart = model.compressor.mem.unsqueeze(0).to(embed(qid).dtype)
            seq = torch.cat([cart, embed(qid), embed(gold)], 1)
            P = K + qid.shape[1]
            lg = base(inputs_embeds=seq, use_cache=False).logits[0, P - 1:P - 1 + gold.shape[1]].float()
        else:                                                # gist: [ctx ; K gist ; query ; gold] with the gist mask
            gist = model.compressor.mem.unsqueeze(0).to(embed(c).dtype)
            seq = torch.cat([embed(c), gist, embed(qid), embed(gold)], 1)
            lc, lq, lg_ = c.shape[1], qid.shape[1], gold.shape[1]
            mask = gist_mask(lc, K, lq, lg_, seq.dtype, seq.device)
            pos = torch.arange(seq.shape[1], device=seq.device).unsqueeze(0)
            P = lc + K + lq
            lg = base(inputs_embeds=seq, attention_mask=mask, position_ids=pos, use_cache=False).logits[0, P - 1:P - 1 + lg_].float()
        loss = torch.nn.functional.cross_entropy(lg, gold[0])
        (loss / ACCUM).backward()
        if (step + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad()
        if step % 50 == 0:
            print(f"[{MODE}] step {step}/{MAXSTEPS} loss {float(loss):.4f}", flush=True)

    model.eval()
    if train_steps:
        # Save before full-split evaluation so a later scorer failure never discards training.
        model.save_adapters(adapter_path)
    torch.cuda.empty_cache()

    # ---- gist memory builder (encode ctx -> K gist token hiddens, reused as the cheap prefix at inference) ----
    @torch.no_grad()
    def gist_mem(c):
        set_lora_enabled(base, True)
        gist = model.compressor.mem.unsqueeze(0).to(embed(c).dtype)
        seq = torch.cat([embed(c), gist], 1)
        lc = c.shape[1]
        mn = torch.finfo(seq.dtype).min
        L = lc + K; mask = torch.triu(torch.full((L, L), mn, device=seq.device, dtype=seq.dtype), diagonal=1).view(1, 1, L, L)
        pos = torch.arange(L, device=seq.device).unsqueeze(0)
        h = base.model(inputs_embeds=seq, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state
        return h[:, -K:, :]                                  # K gist vectors (embedding space)

    def _reader_budget(source_tokens: int) -> int:
        """Match the realized K-per-source-chunk memory read, capped by the reader."""
        if MATCH_READER_BUDGET:
            source_tokens = min(max(0, source_tokens), SOURCE_MAXCTX)
            chunks = max(1, math.ceil(source_tokens / max(1, MATCH_CHUNK)))
            return min(MAXCTX, chunks * MATCH_K)
        return min(MAXCTX, LL_TARGET if LL_TARGET > 0 else MAXCTX)

    def _ll_target_for_reader(text: str, reader_budget: int) -> int:
        if reader_budget <= 0:
            return -1
        reader_n = max(1, len(rt.tok(text, add_special_tokens=False).input_ids))
        oai_n = max(1, _PC.get_token_length(text, use_oai_tokenizer=True))
        return max(8, round(reader_budget * oai_n / reader_n))

    def _reader_budget_embed(text: str, reader_budget: int):
        ids = rt.tok(text, add_special_tokens=False).input_ids
        budget = min(reader_budget, MAXCTX)
        if len(ids) > budget:
            left = budget // 2
            ids = ids[:left] + ids[-(budget - left):]
        return embed(torch.tensor([ids], device=dev))

    # ---- eval (reuse GCMModel.mc_loglik / _gen_batch) ----
    torch.set_grad_enabled(False)
    from collections import defaultdict
    by_bench = defaultdict(list)
    for (x, b, it) in val_pool:
        by_bench[b].append((x, it))
    per_bench = {}
    records = []
    for b, rows in by_bench.items():
        d = per_bench.setdefault(b, {"no_ctx": [], "full": [], "method": []})
        is_mc = b in MC_BENCHES
        for s0 in range(0, len(rows), GEN_BS):
            chunk = rows[s0:s0 + GEN_BS]
            cids = [xx[0] for (xx, it) in chunk]; qids = [xx[1] for (xx, it) in chunk]; its = [it for (xx, it) in chunk]
            method_qids = qids
            source_counts = [source_token_count(it) for it in its]
            method_meta = [
                {
                    "compressor_failed": False,
                    "method_source_tokens": int(cids[k].shape[1]),
                    "reader_budget_tokens": _reader_budget(int(cids[k].shape[1])),
                }
                for k in range(len(cids))
            ]
            # method prefix: sft = LoRA-on full ctx; gist = K gist vectors; cart = FIXED prefix; txl = bounded raw window
            method_um = (MODE not in _EVAL_ONLY)   # eval-only modes read raw/compressed/retrieved text with the FROZEN base (no adapter)
            if MODE == "sft":
                mpref = [embed(c[:, :MAXCTX]) for c in cids]
            elif MODE == "cart":
                mpref = [model.compressor.mem.unsqueeze(0) for _ in cids]
            elif MODE == "txl":                              # GENERIC sliding-window READ (last W-4 tokens + 4 sinks) fed to the frozen base. NOT StreamingLLM (that = the `streaming` kvpress press); report as generic truncation only.
                def _win(c, k):
                    budget = method_meta[k]["reader_budget_tokens"] if MATCH_READER_BUDGET else TXL_W
                    if c.shape[1] <= budget:
                        return embed(c)
                    sinks = min(4, budget)
                    tail = budget - sinks
                    ids = c[:, :sinks] if tail == 0 else torch.cat([c[:, :sinks], c[:, -tail:]], 1)
                    return embed(ids)
                mpref = [_win(c, k) for k, c in enumerate(cids)]
            elif MODE == "llmlingua":                        # LLMLingua-2 hard compression: prune ctx text -> frozen base reads the shortened text
                def _ll(c, it, k):
                    contexts = [rt.tok.decode(c[0], skip_special_tokens=True)]
                    method_meta[k]["method_source_tokens"] = int(c.shape[1])
                    txt = "\n\n".join(contexts)
                    try:
                        reader_budget = method_meta[k]["reader_budget_tokens"]
                        target = _ll_target_for_reader(txt, reader_budget)
                        kwargs = {
                            "force_tokens": ["\n", "?", "!", "."],
                            "drop_consecutive": True,
                            "force_reserve_digit": b in {
                                "ruler_niah", "numerical_niah", "coding_niah",
                                "longbench_v2", "infbench_choice",
                            },
                        }
                        if target > 0:
                            kwargs["target_token"] = target
                        else:
                            kwargs["rate"] = LL_RATE
                        comp = _PC.compress_prompt_llmlingua2(contexts, **kwargs)["compressed_prompt"]
                    except Exception as e:
                        method_meta[k]["compressor_failed"] = True
                        method_meta[k]["compressor_error"] = repr(e)
                        print(f"[llmlingua] compress failed: {repr(e)[:80]}", flush=True)
                        if env("GCM_COMPRESS_FALLBACK", "0") != "1":
                            raise
                        comp = txt
                    return _reader_budget_embed(comp, reader_budget)
                mpref = [_ll(cids[k], its[k], k) for k in range(len(cids))]
            elif MODE == "rag":                              # BM25 retrieve-then-read: rank passages by query relevance, frozen base reads top passages within a token budget
                def _rag(qid, c, k):
                    ids = c[0, :SOURCE_MAXCTX].tolist()
                    if not ids:
                        return embed(qid[:, :0])
                    cw = RAG_CHUNK
                    passages = [ids[i:i + cw] for i in range(0, len(ids), cw)]
                    N = len(passages)
                    df = {}
                    for p in passages:
                        for t in set(p):
                            df[t] = df.get(t, 0) + 1
                    avgdl = max(1.0, sum(len(p) for p in passages) / max(1, N)); k1, bb = 1.5, 0.75
                    qterms = set(qid[0].tolist())
                    from collections import Counter
                    def sc(p):
                        tf = Counter(p); s = 0.0
                        for t in qterms:
                            if t in tf:
                                idf = math.log(1 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
                                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - bb + bb * len(p) / avgdl))
                        return s
                    ranked = sorted(range(N), key=lambda i: (-sc(passages[i]), i))
                    sel, tot = [], 0
                    budget = (
                        method_meta[k]["reader_budget_tokens"]
                        if MATCH_READER_BUDGET
                        else RAG_BUDGET
                    )
                    for i in ranked:
                        if tot + len(passages[i]) > budget and sel:
                            continue
                        sel.append(i); tot += len(passages[i])
                        if tot >= budget:
                            break
                    sel.sort()  # restore reading order
                    out = [t for i in sel for t in passages[i]][:min(MAXCTX, budget)]
                    return embed(torch.tensor([out], device=dev))
                mpref = [_rag(qids[k], cids[k], k) for k in range(len(cids))]
            elif MODE in ("longllmlingua", "llmlingua_orig"):   # EXACT (Long)LLMLingua: perplexity-based prune -> frozen base reads shortened text
                def _llp(c, it, k):
                    contexts = [rt.tok.decode(c[0], skip_special_tokens=True)]
                    method_meta[k]["method_source_tokens"] = int(c.shape[1])
                    full_text = "\n\n".join(contexts)
                    try:
                        reader_budget = method_meta[k]["reader_budget_tokens"]
                        target = _ll_target_for_reader(full_text, reader_budget)
                        if MODE == "longllmlingua":
                            kwargs = {
                                "question": str(getattr(it, "query", "")),
                                "condition_in_question": "after_condition",
                                "reorder_context": "sort",
                                "dynamic_context_compression_ratio": 0.4,
                                "condition_compare": True,
                                "context_budget": "+100",
                                "token_budget_ratio": 1.4,
                                "rank_method": "longllmlingua",
                                "concate_question": True,
                            }
                            if target > 0:
                                kwargs["target_token"] = target
                            else:
                                kwargs["rate"] = LL_RATE
                            comp = _PC.compress_prompt(contexts, **kwargs)["compressed_prompt"]
                        else:
                            kwargs = {
                                "question": "",
                                "rank_method": "llmlingua",
                                "condition_in_question": "none",
                                "iterative_size": 200,
                                "context_budget": "+100",
                                "token_budget_ratio": 1.4,
                            }
                            if target > 0:
                                kwargs["target_token"] = target
                            else:
                                kwargs["rate"] = LL_RATE
                            comp = _PC.compress_prompt(contexts, **kwargs)["compressed_prompt"]
                    except Exception as e:
                        method_meta[k]["compressor_failed"] = True
                        method_meta[k]["compressor_error"] = repr(e)
                        print(f"[{MODE}] compress failed: {repr(e)[:90]}", flush=True)
                        if env("GCM_COMPRESS_FALLBACK", "0") != "1":
                            raise
                        comp = full_text
                    return _reader_budget_embed(comp, reader_budget)
                mpref = [_llp(cids[k], its[k], k) for k in range(len(cids))]
                if MODE == "longllmlingua":
                    method_qids = [q[:, :0] for q in qids]  # official output already concatenates the question
            elif MODE == "selective_context":                    # EXACT Selective-Context: drop low-self-information lexical units
                def _sc(c, it):
                    txt = "".join(map(str, getattr(it, "chunks", []) or [])) or rt.tok.decode(c[0], skip_special_tokens=True)
                    try:
                        comp = _SC(txt, reduce_ratio=max(0.05, 1.0 - LL_RATE))[0]
                    except Exception as e:
                        print(f"[sc] compress failed: {repr(e)[:90]}", flush=True); comp = txt
                    cc = rt.tok(comp, return_tensors="pt", truncation=True, max_length=MAXCTX).input_ids.to(dev)
                    return embed(cc)
                mpref = [_sc(cids[k], its[k]) for k in range(len(cids))]
            elif MODE == "tome":                                 # ToMe (Bolya 2022) input-side token merging: bipartite soft-match merge similar ctx tokens, order-preserving
                def _tome(it):
                    ids = rt.tok("".join(map(str, getattr(it, "chunks", []) or [])), add_special_tokens=False,
                                 truncation=True, max_length=MAXCTX, return_tensors="pt").input_ids.to(dev)
                    if ids.shape[1] == 0:
                        return embed(rt.query_ids(it))[:, :1]
                    x = embed(ids)[0].float()                    # L x d
                    L0 = x.shape[0]
                    size = torch.ones(L0, device=dev); pos = torch.arange(L0, device=dev).float()
                    target = max(8, int(TOME_RATIO * L0))
                    guard = 0
                    while x.shape[0] > target and guard < 64:
                        guard += 1; n = x.shape[0]
                        ai = torch.arange(0, n, 2, device=dev); bi = torch.arange(1, n, 2, device=dev)
                        a, b = x[ai], x[bi]
                        if TOME_SIM == "random":
                            vals = torch.rand(a.shape[0], device=dev); mj = torch.randint(0, b.shape[0], (a.shape[0],), device=dev)
                        else:
                            an = a / (a.norm(dim=-1, keepdim=True) + 1e-6); bn = b / (b.norm(dim=-1, keepdim=True) + 1e-6)
                            sim = an @ bn.T; best = sim.max(dim=1); vals, mj = best.values, best.indices
                        r = min(n - target, a.shape[0])
                        sel = vals.argsort(descending=True)[:r]     # a-tokens to merge into their matched b
                        j = mj[sel]; wa = size[ai][sel]
                        num = (size[bi].unsqueeze(-1) * b).clone(); den = size[bi].clone()
                        num.index_add_(0, j, wa.unsqueeze(-1) * a[sel]); den.index_add_(0, j, wa)
                        nb = num / den.unsqueeze(-1); nbp = pos[bi].clone()
                        nbp.scatter_reduce_(0, j, pos[ai][sel], reduce="amin", include_self=True)
                        keep = torch.ones(a.shape[0], dtype=torch.bool, device=dev); keep[sel] = False
                        x = torch.cat([a[keep], nb], 0); size = torch.cat([size[ai][keep], den], 0); pos = torch.cat([pos[ai][keep], nbp], 0)
                        o = pos.argsort(); x, size, pos = x[o], size[o], pos[o]   # restore token order
                    return x.unsqueeze(0).to(embed(ids).dtype)
                mpref = [_tome(it) for it in its]
            elif MODE == "imp":                                  # Mode A (Paper B): plug-and-play importance-routing prefilter.
                # Score each ctx token by cheap O(L) signals (query-relevance + surprisal, F20); KEEP top-p VERBATIM
                # (protect the un-mergeable needle, F14/F21); drop the redundant rest. No training; frozen base.
                def _fulldoc_bm25(it, submode):   # FAIR-vs-RAG fix: chunk/BM25 family needs no forward -> retrieve over the FULL (untruncated) doc, like RAG. Fixes the ctx>MAXCTX truncation asymmetry (F44).
                    import math as _m
                    from collections import Counter as _C
                    toks = rt.tok("".join(map(str, getattr(it, "chunks", []) or [])), add_special_tokens=False).input_ids
                    L = len(toks)
                    if L == 0:
                        return embed(rt.query_ids(it))
                    W = 256 if submode in ("chunk", "hier", "auto") else 32
                    budget = max(8, int(LL_RATE * min(L, MAXCTX)))
                    qset = set(rt.query_ids(it)[0].tolist())
                    units = [(i, min(i + W, L)) for i in range(0, L, W)]
                    segs = [toks[a:b] for a, b in units]; N = len(segs); df = {}
                    for sg in segs:
                        for t in set(sg): df[t] = df.get(t, 0) + 1
                    avgdl = max(1.0, L / max(1, N)); k1, bb = 1.5, 0.75
                    def _sc(sg):
                        tf = _C(sg); v = 0.0
                        for t in qset:
                            if t in tf:
                                idf = _m.log(1 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
                                v += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - bb + bb * len(sg) / avgdl))
                        return v
                    ranked = sorted(range(N), key=lambda i: _sc(segs[i]), reverse=True)
                    sel, tot = [], 0
                    for i in ranked:
                        if tot + len(segs[i]) > budget and sel: break
                        sel.append(i); tot += len(segs[i])
                    _ord = env("GCM_IMP_CHUNK_ORDER", "reading")   # GDN anti-forgetting module: chunk placement
                    if _ord == "rel_last":                          # best-BM25 chunk placed LAST (nearest query) to beat GDN recency-forgetting
                        selset = set(sel); order = [i for i in reversed(ranked) if i in selset]
                    elif _ord == "replay":                          # reading order + re-inject the top-1 chunk right before the query
                        sel.sort(); order = sel + [ranked[0]]
                    else:                                           # reading order (default)
                        sel.sort(); order = sel
                    keep_ids = [t for i in order for t in segs[i]][:MAXCTX]
                    return embed(torch.tensor([keep_ids], device=dev))
                def _imp(it):
                    ids = rt.tok("".join(map(str, getattr(it, "chunks", []) or [])), add_special_tokens=False,
                                 truncation=True, max_length=MAXCTX, return_tensors="pt").input_ids.to(dev)
                    if ids.shape[1] == 0:
                        return embed(rt.query_ids(it))
                    _FD = env("GCM_IMP_FULLDOC", "1") == "1"
                    _M0 = env("GCM_IMP_MODE", "span")
                    _doc_too_long = ids.shape[1] >= MAXCTX   # doc does not fit -> span (needs a forward) cannot see it all -> must retrieve over full doc
                    _go_fd = _M0 in ("chunk", "bm25span", "hier") or (_M0 == "auto" and (_doc_too_long or not bool(options_for(it))))
                    if _FD and _go_fd:
                        return _fulldoc_bm25(it, _M0)
                    L = ids.shape[1]
                    with torch.no_grad():
                        E = embed(ids)[0].float()
                        qe = embed(rt.query_ids(it))[0].float().mean(0, keepdim=True)
                        En = E / (E.norm(dim=-1, keepdim=True) + 1e-6); qn = qe / (qe.norm() + 1e-6)
                        qdot = (En @ qn.T).squeeze(-1)                       # query relevance (F20: 0.95 word-needle)
                        lg = base(inputs_embeds=embed(ids), use_cache=False).logits[0].float()
                        lp = lg.log_softmax(-1); surp = torch.zeros(L, device=dev)
                        surp[1:] = -lp[:-1].gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)   # surprisal (F20: 0.84 numeric)
                        z = lambda t: (t - t.mean()) / (t.std() + 1e-6)
                        _sig = env("GCM_IMP_SIGNAL", "both")                 # query/surprisal/both/lex/qlex/all
                        import math as _m
                        _qset = set(rt.query_ids(it)[0].tolist())
                        _tf = {}
                        for _t in ids[0].tolist(): _tf[_t] = _tf.get(_t, 0) + 1
                        lex = torch.tensor([(_m.log((L + 1) / (_tf[_t] + 0.5)) if _t in _qset else 0.0) for _t in ids[0].tolist()], device=dev, dtype=torch.float)  # IDF-weighted query-term match (BM25-style; downweights boilerplate)
                        if _sig == "query": score = z(qdot)
                        elif _sig == "surprisal": score = z(surp)
                        elif _sig == "lex": score = z(lex)
                        elif _sig == "qlex": score = z(qdot) + z(lex)
                        elif _sig == "all": score = z(qdot) + z(surp) + z(lex)
                        else: score = z(qdot) + z(surp)
                    keep = max(8, int(LL_RATE * L))
                    MODE = env("GCM_IMP_MODE", "span")
                    W = max(1, int(env("GCM_IMP_SPAN", "32")))
                    CW = int(env("GCM_IMP_CHUNK", "256"))
                    toks = ids[0].tolist()
                    def _spanmax(sc, w):
                        nb = (L + w - 1) // w
                        pad = torch.full((nb * w - L,), float("-inf"), device=dev)
                        return torch.cat([sc, pad]).view(nb, w).max(1).values
                    def _keep_spans(sc, w, budget):
                        bs = _spanmax(sc, w)
                        blocks = bs.topk(min(max(1, budget // w), (L + w - 1) // w)).indices.sort().values
                        return torch.cat([torch.arange(int(b) * w, min(int(b) * w + w, L), device=dev) for b in blocks])
                    def _units(w):
                        return [(i, min(i + w, L)) for i in range(0, L, w)]
                    def _bm25(units):
                        k1, bb = 1.5, 0.75; udf = {}
                        for a, b_ in units:
                            for t in set(toks[a:b_]): udf[t] = udf.get(t, 0) + 1
                        Nu = len(units); avgdl = max(1.0, L / Nu); out = []
                        for a, b_ in units:
                            seg = toks[a:b_]; tfc = {}
                            for t in seg: tfc[t] = tfc.get(t, 0) + 1
                            sm = 0.0
                            for t in _qset:
                                if t in tfc:
                                    idf = _m.log(1 + (Nu - udf.get(t, 0) + 0.5) / (udf.get(t, 0) + 0.5))
                                    sm += idf * tfc[t] * (k1 + 1) / (tfc[t] + k1 * (1 - bb + bb * len(seg) / avgdl))
                            out.append(sm)
                        return torch.tensor(out, device=dev, dtype=torch.float)
                    def _sel_units(us, units, budget):
                        order = us.argsort(descending=True); sel = []; tot = 0
                        for j in order.tolist():
                            a, b_ = units[j]
                            if tot + (b_ - a) > budget and sel: continue
                            sel.append(j); tot += (b_ - a)
                            if tot >= budget: break
                        sel.sort(); return [i for j in sel for i in range(units[j][0], units[j][1])]
                    if MODE == "chunk":
                        units = _units(CW); bm = _bm25(units)
                        ms = torch.stack([surp[a:b_].mean() for a, b_ in units])
                        mq = torch.stack([qdot[a:b_].mean() for a, b_ in units])
                        idx = torch.tensor(_sel_units(z(bm) + z(ms) + z(mq), units, keep), device=dev, dtype=torch.long)
                    elif MODE == "bm25span":
                        units = _units(W); bm = _bm25(units)
                        ms = torch.stack([surp[a:b_].mean() for a, b_ in units])
                        idx = torch.tensor(_sel_units(z(bm) + z(ms), units, keep), device=dev, dtype=torch.long)
                    elif MODE == "hier":
                        units = _units(CW); bm = _bm25(units)
                        toks1 = _sel_units(bm, units, min(2 * keep, L))
                        sub = torch.tensor(sorted(toks1), device=dev, dtype=torch.long)
                        subsc = score[sub]; ns = len(sub); nb = (ns + W - 1) // W
                        pad = torch.full((nb * W - ns,), float("-inf"), device=dev)
                        bs = torch.cat([subsc, pad]).view(nb, W).max(1).values
                        blocks = bs.topk(min(max(1, keep // W), nb)).indices.sort().values
                        idx = torch.cat([sub[int(b) * W:min(int(b) * W + W, ns)] for b in blocks])
                    elif MODE == "qfree":
                        idx = _keep_spans(z(surp), W, keep) if W > 1 else z(surp).topk(min(keep, L)).indices
                    elif MODE == "auto":  # input-driven route chunk<->span; routers {peak,mc,qover}; optional adaptive budget (F39)
                        _router = env("GCM_IMP_AUTO_ROUTER", "mc")
                        units = _units(CW); bm = _bm25(units)
                        peak = float(bm.max() / (bm.mean() + 1e-6)) if bm.numel() > 1 else 0.0
                        if _router == "mc":
                            go_chunk = not bool(options_for(it))            # options present (MC/reasoning) -> span; extractive -> chunk
                        elif _router == "qover":
                            ov = len(_qset & set(toks)) / max(1, len(_qset))
                            go_chunk = ov >= float(env("GCM_IMP_AUTO_TAU", "0.5"))
                        else:  # peak (BM25 max/mean)
                            go_chunk = peak >= float(env("GCM_IMP_AUTO_TAU", "3.0"))
                        kk = keep
                        if env("GCM_IMP_ADAPT_BUDGET", "0") == "1" and go_chunk:
                            kk = max(8, int(0.25 * L))                      # tighten budget when a lexical anchor localizes the answer (targets F39)
                        if go_chunk:
                            ms = torch.stack([surp[a:b_].mean() for a, b_ in units])
                            mq = torch.stack([qdot[a:b_].mean() for a, b_ in units])
                            idx = torch.tensor(_sel_units(z(bm) + z(ms) + z(mq), units, kk), device=dev, dtype=torch.long)
                        else:
                            idx = _keep_spans(score, W, kk) if W > 1 else score.topk(min(kk, L)).indices
                    else:  # span (default): token score -> top-p spans
                        idx = _keep_spans(score, W, keep) if W > 1 else score.topk(min(keep, L)).indices
                    if idx.numel() == 0:
                        idx = torch.arange(min(keep, L), device=dev)
                    idx = idx.sort().values
                    return embed(ids[:, idx])
                mpref = [_imp(it) for it in its]
            else:
                mpref = [gist_mem(c[:, :MAXCTX]) for c in cids]
            if is_mc:
                for j, it in enumerate(its):
                    opts = options_for(it)
                    if not opts:
                        continue
                    letters = list(string.ascii_uppercase)[:len(opts)]
                    def _mc(prefix, um, query_ids=None):
                        sc = model.mc_loglik(prefix, query_ids if query_ids is not None else qids[j], letters, um)
                        pred = letters[max(range(len(sc)), key=lambda k: sc[k])]
                        return {
                            "score": float(pred == str(it.gold).strip()),
                            "prediction": pred,
                            "option_loglik": [float(value) for value in sc],
                        }
                    no_result = _mc(None, False)
                    full_result = _mc(embed(cids[j][:, :MAXCTX]), False)
                    method_result = _mc(mpref[j], method_um, method_qids[j])
                    no = no_result["score"]
                    full = full_result["score"]
                    method = method_result["score"]
                    d["no_ctx"].append(no)
                    d["full"].append(full)
                    d["method"].append(method)
                    records.append({
                        "bench": b, "item_id": str(getattr(it, "item_id", s0 + j)),
                        "document_id": document_id(it), "gold": str(getattr(it, "gold", "")),
                        "seed": SEED, "is_mc": True,
                        "source_tokens": source_counts[j],
                        "ctx_tokens": int(cids[j].shape[1]), "query_tokens": int(qids[j].shape[1]),
                        "method_tokens": int(mpref[j].shape[1]),
                        "truncation": {
                            "source_load": source_counts[j] > int(cids[j].shape[1]),
                            "feasible_raw": source_counts[j] > min(int(cids[j].shape[1]), MAXCTX),
                            "method_source": source_counts[j] > int(method_meta[j]["method_source_tokens"]),
                            "raw_side": "right",
                            "window_policy": "sink+tail" if MODE == "txl" else None,
                        },
                        "method_meta": method_meta[j],
                        "scores": {"no_ctx": no, "feasible_raw": full, "method": method},
                        "predictions": {
                            "no_ctx": no_result["prediction"],
                            "feasible_raw": full_result["prediction"],
                            "method": method_result["prediction"],
                        },
                        "option_loglik": {
                            "labels": letters,
                            "no_ctx": no_result["option_loglik"],
                            "feasible_raw": full_result["option_loglik"],
                            "method": method_result["option_loglik"],
                        },
                    })
            else:
                t_no = model._gen_batch([None] * len(chunk), qids, False, GEN_MAX, GEN_BS)
                t_full = model._gen_batch([embed(c[:, :MAXCTX]) for c in cids], qids, False, GEN_MAX, GEN_BS)
                t_m = model._gen_batch(mpref, method_qids, method_um, GEN_MAX, GEN_BS)
                def _sc(txt, it):
                    sc = 0.0
                    try:
                        sc = float(score_gen(b, txt, it)[0])
                    except Exception:
                        if STRICT_EVAL:
                            raise
                    g = str(getattr(it, "gold", "")).strip()
                    if sc == 0.0 and g and g in (txt or "") and env("GCM_GEN_NOFALLBACK", "1") != "1":
                        sc = 1.0
                    return sc
                for j, it in enumerate(its):
                    no = _sc(t_no[j], it)
                    full = _sc(t_full[j], it)
                    method = _sc(t_m[j], it)
                    d["no_ctx"].append(no)
                    d["full"].append(full)
                    d["method"].append(method)
                    records.append({
                        "bench": b, "item_id": str(getattr(it, "item_id", s0 + j)),
                        "document_id": document_id(it), "gold": str(getattr(it, "gold", "")),
                        "seed": SEED, "is_mc": False,
                        "source_tokens": source_counts[j],
                        "ctx_tokens": int(cids[j].shape[1]), "query_tokens": int(qids[j].shape[1]),
                        "method_tokens": int(mpref[j].shape[1]),
                        "truncation": {
                            "source_load": source_counts[j] > int(cids[j].shape[1]),
                            "feasible_raw": source_counts[j] > min(int(cids[j].shape[1]), MAXCTX),
                            "method_source": source_counts[j] > int(method_meta[j]["method_source_tokens"]),
                            "raw_side": "right",
                            "window_policy": "sink+tail" if MODE == "txl" else None,
                        },
                        "method_meta": method_meta[j],
                        "scores": {"no_ctx": no, "feasible_raw": full, "method": method},
                        "predictions": {
                            "no_ctx": t_no[j],
                            "feasible_raw": t_full[j],
                            "method": t_m[j],
                        },
                    })
            torch.cuda.empty_cache()
    if not records:
        raise RuntimeError(f"{TAG} evaluated zero items")
    agg = {m: float(sum(sum(d[m]) for d in per_bench.values()) / max(1, sum(len(d[m]) for d in per_bench.values())))
           for m in ("no_ctx", "full", "method")}
    agg["mode"] = MODE
    per_bench_acc = {b: {k: float(sum(d[k]) / max(1, len(d[k]))) for k in d} for b, d in per_bench.items()}
    payload = {"status": "ok", "model": mname, "cell": TAG, "baseline": MODE, "seed": SEED,
               "config": {"K": K, "lora": LORA, "lr": LR, "maxctx": MAXCTX,
                          "source_maxctx": SOURCE_MAXCTX,
                          "match_reader_budget": MATCH_READER_BUDGET,
                          "match_chunk": MATCH_CHUNK if MATCH_READER_BUDGET else None,
                          "match_k": MATCH_K if MATCH_READER_BUDGET else None,
                          "n_train": len(pool), "n_val": len(val_pool), "max_steps": MAXSTEPS,
                          "ll_target_reader_tokens": LL_TARGET,
                          "compress_fallback_enabled": env("GCM_COMPRESS_FALLBACK", "0") == "1",
                          "strict_eval": STRICT_EVAL,
                          "trainable_params": sum(p.numel() for p in params),
                          "gold_substring_fallback": env("GCM_GEN_NOFALLBACK", "1") != "1"},
               "agg": agg, "per_bench": per_bench_acc, "records": records}
    atomic_json_dump(payload, f"{outdir}/{TAG}.json")
    print("RECIPE_EVAL", TAG, agg, flush=True)


if __name__ == "__main__":
    main()
