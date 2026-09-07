"""Frozen-base runtime for v1.7 benchmark testing.

The base is loaded once and frozen. Every method differs ONLY in the prefix embeds it puts before the
query (and, for Gist, the masked context length); generation, MC log-likelihood, and the Gist attention
mask are shared here. All forwards are cache-free (re-forward the growing sequence) to avoid the legacy
KV-cache API differences we hit across transformers versions.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from llm_infra.models import load_base
from llm_infra.prompting import format_query_block

try:
    from transformers import DynamicCache
except Exception:  # noqa: BLE001
    DynamicCache = None


class Runtime:
    def __init__(
        self,
        base: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        n_memory: int = 64,
        max_ctx_tokens: int = 1024,
        max_input_tokens: int = 2048,
        max_new_tokens: int = 16,
        eager: bool = False,
    ):
        # eager attention is required when a method passes a custom 4D mask (Gist).
        loaded = load_base(
            base, device=device, torch_dtype=dtype,
            attn_implementation="eager" if eager else None,
        )
        self.model = loaded.model
        self.tok = loaded.tokenizer
        self.d = loaded.hidden_size
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if getattr(self.tok, "pad_token", None) is None:
            self.tok.pad_token = self.tok.eos_token
        self.dev = torch.device(device)
        self.bdt = self.model.get_input_embeddings().weight.dtype
        self.embed = self.model.get_input_embeddings()
        self.K = n_memory
        self.max_ctx = max_ctx_tokens
        self.max_in = max_input_tokens
        self.max_new = max_new_tokens

    # -- tokenisation -------------------------------------------------------
    def ctx_ids(self, item: Any) -> torch.Tensor:
        text = "\n".join(str(c) for c in item.chunks)
        return self.tok(
            text, return_tensors="pt", truncation=True, max_length=self.max_ctx
        ).input_ids.to(self.dev)

    def query_ids(self, item: Any) -> torch.Tensor:
        q = format_query_block(item.query).lstrip("\n")
        return self.tok(
            q, return_tensors="pt", truncation=True, max_length=self.max_in
        ).input_ids.to(self.dev)

    # -- gist attention mask ------------------------------------------------
    def gist_mask(self, seq_len: int, pre_len: int, ctx_len: int) -> torch.Tensor | None:
        """4D additive mask: causal everywhere, EXCEPT rows >= pre_len (query + output) cannot attend to
        cols < ctx_len (the raw context), forcing them through the gist tokens. ctx_len<=0 -> no mask."""
        if ctx_len <= 0:
            return None
        mn = torch.finfo(self.bdt).min
        m = torch.triu(torch.full((seq_len, seq_len), mn, device=self.dev, dtype=self.bdt), diagonal=1)
        m[pre_len:, :ctx_len] = mn
        return m.view(1, 1, seq_len, seq_len)

    # -- prediction primitives ---------------------------------------------
    @torch.no_grad()
    def generate(self, pre: torch.Tensor | None, ctx_len: int, qids: torch.Tensor) -> str:
        """Greedy decode from [pre ; query]. ctx_len>0 applies the Gist mask (rebuilt as the seq grows)."""
        seq = torch.cat([pre, self.embed(qids)], dim=1) if pre is not None else self.embed(qids)
        pre_len = pre.shape[1] if pre is not None else 0
        ids: list[int] = []
        for _ in range(self.max_new):
            am = self.gist_mask(seq.shape[1], pre_len, ctx_len)
            lg = self.model(
                inputs_embeds=seq,
                attention_mask=am,
                use_cache=False,
                logits_to_keep=1,
            ).logits[:, -1]
            nxt = int(lg.argmax(-1))
            if nxt == getattr(self.tok, "eos_token_id", -1):
                break
            ids.append(nxt)
            seq = torch.cat([seq, self.embed(torch.tensor([[nxt]], device=self.dev))], dim=1)
        return self.tok.decode(ids, skip_special_tokens=True)

    @torch.no_grad()
    def mc_loglik(
        self, pre: torch.Tensor | None, ctx_len: int, qids: torch.Tensor, opts: list
    ) -> list[float]:
        """Mean log-likelihood of each option appended after [pre ; query]."""
        qe = self.embed(qids)
        base = torch.cat([pre, qe], dim=1) if pre is not None else qe
        P = base.shape[1]
        pre_len = pre.shape[1] if pre is not None else 0
        option_ids = [
            self.tok.encode(" " + str(o).strip(), add_special_tokens=False)
            for o in opts
        ]
        if option_ids and all(len(ids) == 1 for ids in option_ids):
            am = self.gist_mask(P, pre_len, ctx_len)
            lp = self.model(
                inputs_embeds=base,
                attention_mask=am,
                use_cache=False,
                logits_to_keep=1,
            ).logits[0, -1].float().log_softmax(-1)
            return [float(lp[ids[0]]) for ids in option_ids]
        out: list[float] = []
        for oid in option_ids:
            if not oid:
                out.append(-1e9)
                continue
            full = torch.cat([base, self.embed(torch.tensor([oid], device=self.dev))], dim=1)
            am = self.gist_mask(full.shape[1], pre_len, ctx_len)
            keep = len(oid) + 1
            lg = self.model(
                inputs_embeds=full,
                attention_mask=am,
                use_cache=False,
                logits_to_keep=keep,
            ).logits[0]
            lp = F.log_softmax(lg[:len(oid)].float(), dim=-1)
            out.append(float(lp[torch.arange(len(oid)), torch.tensor(oid, device=self.dev)].mean()))
        return out

    # -- faithful Cartridge: per-layer trainable KV-cache (prefix-tuning) -------------------
    def cart_cache(self, zk: torch.Tensor, zv: torch.Tensor):
        """Build a past_key_values from per-layer trainable K/V. zk,zv: (L, n_kv, p, head_dim). Gradients flow
        back into zk/zv (they are the trainable Cartridge Z)."""
        assert DynamicCache is not None, "DynamicCache unavailable in this transformers"
        cache = DynamicCache()
        for layer in range(zk.shape[0]):
            cache.update(zk[layer].unsqueeze(0), zv[layer].unsqueeze(0), layer)
        return cache

    def kv_logits(
        self,
        zk: torch.Tensor,
        zv: torch.Tensor,
        seq_embeds: torch.Tensor,
        logits_to_keep: int | torch.Tensor = 0,
    ) -> torch.Tensor:
        """Logits for seq_embeds (1,L,d) attending the Cartridge KV (p virtual positions before it). Re-injects a
        fresh cache each call (cache-free style) so it is safe for both training (grad to Z) and decode."""
        p, sl = zk.shape[2], seq_embeds.shape[1]
        pos = torch.arange(p, p + sl, device=self.dev).unsqueeze(0)
        am = torch.ones(1, p + sl, device=self.dev, dtype=torch.long)
        out = self.model(inputs_embeds=seq_embeds, past_key_values=self.cart_cache(zk, zv),
                         attention_mask=am, position_ids=pos, use_cache=False,
                         logits_to_keep=logits_to_keep)
        return out.logits

    @torch.no_grad()
    def generate_kv(self, zk: torch.Tensor, zv: torch.Tensor, qids: torch.Tensor) -> str:
        ids: list[int] = []
        for _ in range(self.max_new):
            sid = torch.cat([qids, torch.tensor([ids], device=self.dev, dtype=qids.dtype)], 1) if ids else qids
            nxt = int(self.kv_logits(zk, zv, self.embed(sid))[:, -1].argmax(-1))
            if nxt == getattr(self.tok, "eos_token_id", -1):
                break
            ids.append(nxt)
        return self.tok.decode(ids, skip_special_tokens=True)

    @torch.no_grad()
    def mc_loglik_kv(self, zk: torch.Tensor, zv: torch.Tensor, qids: torch.Tensor, opts: list) -> list[float]:
        qe = self.embed(qids); out: list[float] = []
        option_ids = [
            self.tok.encode(" " + str(o).strip(), add_special_tokens=False)
            for o in opts
        ]
        if option_ids and all(len(ids) == 1 for ids in option_ids):
            lp = self.kv_logits(zk, zv, qe, logits_to_keep=1)[0, -1].float().log_softmax(-1)
            return [float(lp[ids[0]]) for ids in option_ids]
        for oid in option_ids:
            if not oid:
                out.append(-1e9); continue
            seq = torch.cat([qe, self.embed(torch.tensor([oid], device=self.dev))], dim=1)
            keep = len(oid) + 1
            lg = self.kv_logits(zk, zv, seq, logits_to_keep=keep)[0]
            lp = F.log_softmax(lg[:len(oid)].float(), dim=-1)
            out.append(float(lp[torch.arange(len(oid)), torch.tensor(oid, device=self.dev)].mean()))
        return out
