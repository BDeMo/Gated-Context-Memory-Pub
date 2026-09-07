"""Truncated-BPTT trainer for memory wrappers.

The trainer is wrapper-agnostic: it takes any `Wrapper` (`mem_embedding`,
`mem_feature`, or `mem_weight`) and trains its parameters with the joint loss
defined in `llm_infra.losses`. The base model is frozen and not touched.

For Phase 0 the trainer uses BPTT window = 1 (single-chunk recurrence
backward). Phase 1 promotes to truncated BPTT windows of 2-4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch
from loguru import logger
from torch import nn

from llm_infra.datasets import LongContextItem
from llm_infra.losses import JointLossOutputs, joint_loss
from llm_infra.prompting import format_query_block, tokenize_to_ids
from llm_infra.wrappers import BaseCall, ChunkEncoding, MemoryState, Wrapper


@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    lambda_kl: float = 1.0
    kl_temperature: float = 1.0
    bptt_window: int = 1
    max_steps: int = 100
    log_every: int = 10
    max_input_tokens: int = 1024
    # Encourage consecutive Δm to point in different directions so the wrapper
    # cannot collapse to "walk memory along one constant direction, ignore
    # chunks". Loss term is λ_div · mean(cos(Δm_t, Δm_{t-1})), only added when
    # the wrapper exposes ``MemoryState.extra["last_delta"]`` and there are
    # ≥2 chunks per item. 0.0 disables.
    lambda_div: float = 0.0
    # Reconstruction-loss weight (B4): when > 0 and the wrapper exposes a
    # ``reconstruction_head: nn.Linear``, add an L2 loss between the head's
    # output on m_T (mean over K) and the per-chunk hidden mean (averaged
    # across chunks). Forces m_T to actually carry summarized chunk content.
    lambda_recon: float = 0.0
    # If > 0, append (step, update_alpha, apply_alpha, readout_alpha) rows to
    # ``alpha_log_path`` (CSV) every N steps. Lightweight diagnostic for the
    # "gate stays stuck" hypothesis. Set both to 0/None to disable.
    alpha_log_every: int = 0
    alpha_log_path: str | None = None
    # Answer-head direct supervision (Phase Q ships 2026-06-02). When > 0
    # AND wrapper.answer_head is not None AND answer_label_to_idx maps
    # item.gold to a valid class index, add CE on `answer_head(pool(m_T))`
    # vs the class label. This bypasses the frozen base and gives the
    # wrapper a clean training signal — should kill the per-item
    # memorization mode that surfaced in Phase K.
    lambda_answer_head: float = 0.0
    answer_label_to_idx: dict[str, int] = field(default_factory=dict)


@dataclass
class TrainStepOutputs:
    step: int
    loss_total: float
    loss_ce: float
    loss_kl: float
    loss_div: float = 0.0
    loss_recon: float = 0.0
    loss_answer_head: float = 0.0
    answer_head_acc: float = 0.0  # 1.0 / 0.0 per-step single-item accuracy
    cos_consec_mean: float = 0.0


@dataclass
class BPTTTrainer:
    model: Any
    tokenizer: Any
    wrapper: Wrapper
    encoder_fn: Callable[[list[str]], ChunkEncoding]
    embed_fn: Callable[[torch.Tensor], torch.Tensor]
    config: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        params = list(self._trainable_params())
        if not params:
            raise ValueError(
                "wrapper has no trainable parameters; ensure it inherits from "
                "nn.Module and registers parameters before constructing the trainer"
            )
        self.optim = torch.optim.AdamW(
            params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self._device = next(self.model.parameters()).device
        self._alpha_log_fh: Any = None
        if self.config.alpha_log_every and self.config.alpha_log_path:
            import os
            os.makedirs(os.path.dirname(self.config.alpha_log_path) or ".", exist_ok=True)
            self._alpha_log_fh = open(self.config.alpha_log_path, "w", encoding="utf-8")
            self._alpha_log_fh.write(
                "step,update_alpha,apply_alpha,readout_alpha_mean,readout_alpha_std,cos_consec\n"
            )
            self._alpha_log_fh.flush()

    def _trainable_params(self):
        if isinstance(self.wrapper, nn.Module):
            yield from (p for p in self.wrapper.parameters() if p.requires_grad)

    def train(self, items: Iterable[LongContextItem]) -> list[TrainStepOutputs]:
        items_list = list(items)
        history: list[TrainStepOutputs] = []
        step = 0
        while step < self.config.max_steps:
            for item in items_list:
                if step >= self.config.max_steps:
                    break
                out = self.train_step(item, step)
                history.append(out)
                if step % self.config.log_every == 0:
                    extras = ""
                    if out.loss_recon != 0.0:
                        extras += f" recon={out.loss_recon:.4f}"
                    if out.loss_answer_head != 0.0 or self.config.lambda_answer_head > 0:
                        extras += (
                            f" ans={out.loss_answer_head:.4f}"
                            f" ans_acc={out.answer_head_acc:.0f}"
                        )
                    logger.info(
                        f"step={step} total={out.loss_total:.4f} ce={out.loss_ce:.4f} "
                        f"kl={out.loss_kl:.4f} div={out.loss_div:.4f}"
                        f"{extras} cos_consec={out.cos_consec_mean:+.3f}"
                    )
                step += 1
        return history

    def train_step(self, item: LongContextItem, step: int) -> TrainStepOutputs:
        if isinstance(self.wrapper, nn.Module):
            self.wrapper.train()

        memory: MemoryState = self.wrapper.init_memory(batch=1, device=self._device)
        deltas: list[torch.Tensor] = []
        chunk_means: list[torch.Tensor] = []  # for reconstruction loss
        for chunk in item.chunks:
            enc = self.encoder_fn([chunk])
            memory = self.wrapper.update(memory, enc)
            d = memory.extra.get("last_delta") if memory.extra else None
            if isinstance(d, torch.Tensor):
                deltas.append(d)
            if self.config.lambda_recon > 0 and isinstance(enc.hidden, torch.Tensor):
                mask = enc.mask.unsqueeze(-1).to(enc.hidden.dtype)
                denom = mask.sum(dim=1).clamp(min=1)
                chunk_means.append((enc.hidden * mask).sum(dim=1) / denom)

        teacher_logits, student_logits, labels, loss_mask = self._teacher_student_forward(item, memory)
        loss_out: JointLossOutputs = joint_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            loss_mask=loss_mask,
            lambda_kl=self.config.lambda_kl,
            temperature=self.config.kl_temperature,
        )

        # Δm diversity regularization: penalize correlation between consecutive
        # chunk-deltas so the wrapper cannot collapse to a single constant
        # direction. Cosine averaged across batch and K tokens.
        div_val = 0.0
        cos_val = 0.0
        if self.config.lambda_div > 0.0 and len(deltas) >= 2:
            cos_terms: list[torch.Tensor] = []
            for prev, curr in zip(deltas[:-1], deltas[1:]):
                cos = torch.nn.functional.cosine_similarity(prev, curr, dim=-1)
                cos_terms.append(cos.mean())
            cos_mean = torch.stack(cos_terms).mean()
            div_loss = self.config.lambda_div * cos_mean
            total = loss_out.total + div_loss
            div_val = float(div_loss.item())
            cos_val = float(cos_mean.item())
        else:
            total = loss_out.total
            if len(deltas) >= 2:
                with torch.no_grad():
                    cos_terms = []
                    for prev, curr in zip(deltas[:-1], deltas[1:]):
                        cos_terms.append(
                            torch.nn.functional.cosine_similarity(prev, curr, dim=-1).mean()
                        )
                    cos_val = float(torch.stack(cos_terms).mean().item())

        # Reconstruction loss: ||recon_head(mean_k m_T) - mean_chunks(mean_t h_chunk)||^2
        recon_val = 0.0
        recon_head = getattr(self.wrapper, "reconstruction_head", None)
        if (
            self.config.lambda_recon > 0
            and recon_head is not None
            and chunk_means
            and isinstance(memory.payload, torch.Tensor)
        ):
            mem_pool = memory.payload.mean(dim=1)
            target = torch.stack(chunk_means, dim=0).mean(dim=0).to(mem_pool.dtype)
            recon_pred = recon_head(mem_pool)
            recon_term = self.config.lambda_recon * torch.nn.functional.mse_loss(
                recon_pred, target.detach()
            )
            total = total + recon_term
            recon_val = float(recon_term.item())

        # Answer-head direct supervision (Phase Q). Adds CE on
        # `wrapper.answer_head(pool(m_T))` vs the gold class index. Only
        # fires when (a) head is built, (b) λ > 0, (c) item.gold maps to
        # a known class. Mis-mapped golds are silently skipped so eval-
        # set OOV doesn't crash training.
        answer_val = 0.0
        answer_acc = 0.0
        ans_head = getattr(self.wrapper, "answer_head", None)
        if (
            self.config.lambda_answer_head > 0
            and ans_head is not None
            and isinstance(memory.payload, torch.Tensor)
            and self.config.answer_label_to_idx
        ):
            cls_idx = self.config.answer_label_to_idx.get(item.gold, -1)
            if cls_idx >= 0:
                pool = getattr(self.wrapper, "answer_head_pool", "mean")
                m_T = memory.payload  # [B, K, D]
                if pool == "first":
                    feats = m_T[:, 0, :]
                elif pool == "max":
                    feats = m_T.max(dim=1).values
                else:
                    feats = m_T.mean(dim=1)
                logits = ans_head(feats.to(ans_head.weight.dtype))
                target_idx = torch.tensor(
                    [cls_idx], dtype=torch.long, device=logits.device
                )
                ans_term = self.config.lambda_answer_head * torch.nn.functional.cross_entropy(
                    logits, target_idx
                )
                total = total + ans_term
                answer_val = float(ans_term.item())
                with torch.no_grad():
                    pred_idx = int(logits.argmax(dim=-1).item())
                    answer_acc = 1.0 if pred_idx == cls_idx else 0.0

        self.optim.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(
            [p for p in self._trainable_params()], self.config.grad_clip
        )
        self.optim.step()

        if (
            self._alpha_log_fh is not None
            and self.config.alpha_log_every
            and step % self.config.alpha_log_every == 0
        ):
            self._dump_alpha_row(step, cos_val)

        return TrainStepOutputs(
            step=step,
            loss_total=float(total.item()),
            loss_ce=float(loss_out.ce.item()),
            loss_kl=float(loss_out.kl.item()),
            loss_div=div_val,
            loss_recon=recon_val,
            loss_answer_head=answer_val,
            answer_head_acc=answer_acc,
            cos_consec_mean=cos_val,
        )

    def _dump_alpha_row(self, step: int, cos_val: float) -> None:
        if self._alpha_log_fh is None:
            return
        upd = getattr(self.wrapper, "update_alpha", None)
        upd_val = float(upd.detach().mean().item()) if isinstance(upd, torch.Tensor) else float("nan")
        apply_block = getattr(self.wrapper, "apply_block", None)
        if apply_block is not None and hasattr(apply_block, "alpha"):
            app_val = float(apply_block.alpha.detach().mean().item())
        else:
            app_val = float("nan")
        readout_block = getattr(self.wrapper, "readout_block", None)
        if readout_block is not None and hasattr(readout_block, "alpha"):
            ro = readout_block.alpha.detach().float()
            ro_mean = float(ro.mean().item())
            ro_std = float(ro.std().item()) if ro.numel() > 1 else 0.0
        else:
            ro_mean = float("nan")
            ro_std = float("nan")
        self._alpha_log_fh.write(
            f"{step},{upd_val:.6f},{app_val:.6f},{ro_mean:.6f},{ro_std:.6f},{cos_val:.6f}\n"
        )
        self._alpha_log_fh.flush()

    def _teacher_student_forward(
        self, item: LongContextItem, memory: MemoryState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run teacher (full-context) and student (memory + query+gold) forwards.

        We force both to share the same answer-span tokens so the per-token KL
        and CE align positionally.
        """

        # Build the answer span we will score on.
        gold_text = item.gold + (getattr(self.tokenizer, "eos_token", "") or "")
        query_text = format_query_block(item.query)

        # Teacher: full chunks + query + gold.
        teacher_text = "\n\n".join(item.chunks) + query_text + gold_text
        teacher_input = tokenize_to_ids(
            self.tokenizer, teacher_text, max_length=self.config.max_input_tokens
        ).to(self._device)
        teacher_attn = (teacher_input != self.tokenizer.pad_token_id).long()

        # We need to know where the answer span starts to align with student.
        gold_ids = self.tokenizer(gold_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            self._device
        )
        n_gold = gold_ids.shape[1]
        teacher_t = teacher_input.shape[1]
        teacher_mask = torch.zeros_like(teacher_input)
        teacher_mask[:, max(0, teacher_t - n_gold):] = 1

        with torch.no_grad():
            teacher_out = self.model(
                input_ids=teacher_input,
                attention_mask=teacher_attn,
                use_cache=False,
            )

        # Student: query+gold prefixed by memory via inputs_embeds.
        qg_ids = tokenize_to_ids(
            self.tokenizer, query_text + gold_text, max_length=self.config.max_input_tokens
        ).to(self._device)
        qg_embeds = self.embed_fn(qg_ids)
        qg_attn = torch.ones(qg_embeds.shape[:2], dtype=torch.long, device=self._device)

        student_call = BaseCall(
            inputs_embeds=qg_embeds,
            attention_mask=qg_attn,
            extra={"memory": memory},
        )
        student_call = self.wrapper.apply(student_call, memory)
        cleanup = student_call.extra.pop("_cleanup", None)

        try:
            kwargs = {}
            if student_call.inputs_embeds is not None:
                kwargs["inputs_embeds"] = student_call.inputs_embeds
            if student_call.input_ids is not None:
                kwargs["input_ids"] = student_call.input_ids
            if student_call.attention_mask is not None:
                kwargs["attention_mask"] = student_call.attention_mask
            student_out = self.model(use_cache=False, **kwargs)
        finally:
            if cleanup is not None:
                cleanup()

        # Align: take the last n_gold positions of each output for both
        # teacher logits and student logits; CE labels = gold ids; mask = ones.
        student_t = (
            student_call.inputs_embeds.shape[1]
            if student_call.inputs_embeds is not None
            else student_call.input_ids.shape[1]  # type: ignore[union-attr]
        )

        if student_t < n_gold + 1 or teacher_t < n_gold + 1:
            raise RuntimeError(
                "answer span longer than sequence; raise max_input_tokens or shorten the gold"
            )

        teacher_slice = teacher_out.logits[:, teacher_t - n_gold - 1: teacher_t, :]
        student_slice = student_out.logits[:, student_t - n_gold - 1: student_t, :]

        labels = torch.cat(
            [torch.full((1, 1), -100, dtype=torch.long, device=self._device), gold_ids],
            dim=1,
        )
        # labels and slice are length n_gold+1; first position is a no-op via mask
        mask = torch.ones_like(labels)
        mask[:, 0] = 0  # do not score the prefix-shift position
        # Replace -100 in labels with 0 so cross-entropy index lookup is safe;
        # mask zeros it out anyway.
        labels = labels.clamp_min(0)

        return teacher_slice, student_slice, labels, mask
