"""On-policy distillation + REINFORCE trainers for memory wrappers.

These complement the teacher-forced ``BPTTTrainer`` in
``llm_infra.train``. The SFT trainer scores the wrapper at gold-token
positions only, which makes the wrapper match the teacher distribution
*at the answer position assuming the prefix is the gold prefix*. In
practice the wrapper sometimes never gets to that position under free
generation (collapses to a default class prefix instead). The two
trainers here fix that train/test mismatch by scoring the wrapper on
its own samples:

1. ``OnPolicyDistillTrainer`` (forward-KL on student samples):

   * sample a short trajectory τ from the wrapper-augmented student
   * teacher (full-context base, no wrapper) computes next-token logits
     at every position of τ
   * loss = mean KL(student || teacher) at the τ positions, with
     gradients flowing back through the wrapper's recurrence

   This is the standard "on-policy distillation" recipe used by
   GKD / DistilLLM, simplified to a single rollout per step.

2. ``RLTrainer`` (REINFORCE with leave-one-out baseline):

   * sample N rollouts from the student
   * reward = contains_match(rollout_text, item.gold) ∈ {0, 1}
     (other reward families are easy to plug in)
   * advantage_i = r_i - mean_{j ≠ i} r_j   (GRPO-style)
   * loss = -E_i[ advantage_i · Σ_t log p_student(τ_i,t) ]
   * optional KL-to-teacher penalty (λ_kl) keeps the wrapper from
     drifting away from the teacher policy

Both trainers re-use the wrapper's ``init_memory`` / ``update`` /
``apply`` contract; nothing is base-model-specific.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

from llm_infra.datasets import LongContextItem
from llm_infra.generation import generate_greedy
from llm_infra.losses import masked_kl
from llm_infra.metrics import contains_match, exact_match, token_f1
from llm_infra.prompting import format_query_block, tokenize_to_ids
from llm_infra.wrappers import BaseCall, ChunkEncoding, MemoryState, Wrapper


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #


def _trainable_params(wrapper: Wrapper):
    if isinstance(wrapper, nn.Module):
        yield from (p for p in wrapper.parameters() if p.requires_grad)


def _build_memory(
    wrapper: Wrapper,
    item: LongContextItem,
    encoder_fn: Callable[[list[str]], ChunkEncoding],
    device: torch.device,
) -> MemoryState:
    """Run the wrapper's recurrence over all chunks of ``item``.

    Caller controls whether grads flow by being inside / outside a
    ``torch.no_grad()`` block. The function itself is grad-transparent.
    """

    memory = wrapper.init_memory(batch=1, device=device)
    for chunk in item.chunks:
        enc = encoder_fn([chunk])
        memory = wrapper.update(memory, enc)
    return memory


def _student_call_for_query(
    wrapper: Wrapper,
    memory: MemoryState,
    item: LongContextItem,
    tokenizer: Any,
    embed_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    device: torch.device,
    max_input_tokens: int,
    extra_token_ids: torch.Tensor | None = None,
) -> tuple[BaseCall, int, int]:
    """Build the wrapper-augmented base call for ``query [+ extra]``.

    ``extra_token_ids`` (shape ``[1, T_extra]``) — if provided, the
    given token ids are appended after the formatted query, embedded
    via ``embed_fn``, and included in the call's ``inputs_embeds``.
    Used by trainers that need to re-score a sampled trajectory under
    the wrapper.

    Returns ``(base_call, query_len, extra_len)`` so callers know the
    split between prefix / query / extra positions in the final
    ``inputs_embeds``.
    """

    query_text = format_query_block(item.query).lstrip("\n")
    query_ids = tokenize_to_ids(tokenizer, query_text, max_length=max_input_tokens).to(device)
    if extra_token_ids is not None:
        if extra_token_ids.dim() == 1:
            extra_token_ids = extra_token_ids.unsqueeze(0)
        if extra_token_ids.shape[0] != query_ids.shape[0]:
            raise ValueError("extra_token_ids batch must match query batch")
        full_ids = torch.cat([query_ids, extra_token_ids.to(device)], dim=1)
    else:
        full_ids = query_ids
    full_embeds = embed_fn(full_ids)
    attn = torch.ones(full_embeds.shape[:2], dtype=torch.long, device=device)
    call = BaseCall(
        inputs_embeds=full_embeds,
        attention_mask=attn,
        extra={"memory": memory},
    )
    call = wrapper.apply(call, memory)
    return call, int(query_ids.shape[1]), int(extra_token_ids.shape[1]) if extra_token_ids is not None else 0


# --------------------------------------------------------------------------- #
# On-policy distillation                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class OnPolicyDistillConfig:
    """Hyperparameters for ``OnPolicyDistillTrainer``."""

    lr: float = 5e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_steps: int = 500
    log_every: int = 10

    max_input_tokens: int = 2048
    max_new_tokens: int = 8
    """How many tokens to sample per rollout. The KL is averaged across
    these positions, so a longer rollout gives smoother gradients at
    higher per-step cost."""

    sample_temperature: float = 1.0
    sample_top_k: int = 0
    kl_temperature: float = 1.0

    lambda_div: float = 0.0
    """Δm consecutive-cosine penalty (kept for parity with SFT)."""


@dataclass
class OPDStepOutputs:
    step: int
    loss_total: float
    loss_kl: float
    loss_div: float = 0.0
    cos_consec_mean: float = 0.0
    reward_mean: float = 0.0
    n_tokens_sampled: int = 0


@dataclass
class OnPolicyDistillTrainer:
    """Forward-KL distillation on the student's own samples.

    The teacher is the *frozen base* with full chunks in context (no
    wrapper). The student is the *frozen base + wrapper*. At each step:

    1. Build memory once with no_grad and use it to autoregressively
       sample a trajectory τ of length ``max_new_tokens`` from the
       student (sampling temperature configurable).
    2. Rebuild memory *with* grad and run a single forward pass on
       ``[wrapper-prefix, query, τ]``; the student logits at each τ
       position give the on-policy probabilities.
    3. Run a frozen teacher forward on
       ``[chunks, query, τ]`` to get teacher logits at the same τ
       positions.
    4. Loss = mean KL(student || teacher) over τ positions, weighted
       by ``kl_temperature``.

    Compared to teacher-forced KL (the ``BPTTTrainer`` recipe), this
    closes the train/test mismatch: the wrapper is graded on what it
    *actually* generates, not on a hypothetical "if you saw the gold
    prefix you would say…" distribution.
    """

    model: Any
    tokenizer: Any
    wrapper: Wrapper
    encoder_fn: Callable[[list[str]], ChunkEncoding]
    embed_fn: Callable[[torch.Tensor], torch.Tensor]
    config: OnPolicyDistillConfig = field(default_factory=OnPolicyDistillConfig)

    def __post_init__(self):
        params = list(_trainable_params(self.wrapper))
        if not params:
            raise ValueError("wrapper has no trainable params")
        self.optim = torch.optim.AdamW(
            params, lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self._device = next(self.model.parameters()).device

    def train(self, items: Iterable[LongContextItem]) -> list[OPDStepOutputs]:
        items_list = list(items)
        history: list[OPDStepOutputs] = []
        step = 0
        while step < self.config.max_steps:
            for item in items_list:
                if step >= self.config.max_steps:
                    break
                out = self.train_step(item, step)
                history.append(out)
                if step % self.config.log_every == 0:
                    logger.info(
                        f"[opd] step={step} total={out.loss_total:.4f} "
                        f"kl={out.loss_kl:.4f} div={out.loss_div:.4f} "
                        f"cos_consec={out.cos_consec_mean:+.3f} "
                        f"reward={out.reward_mean:.3f} n_tok={out.n_tokens_sampled}"
                    )
                step += 1
        return history

    def train_step(self, item: LongContextItem, step: int) -> OPDStepOutputs:
        if isinstance(self.wrapper, nn.Module):
            self.wrapper.train()
        cfg = self.config

        # ---- 1. Rollout from student (no grad) ---------------------------
        with torch.no_grad():
            mem_rollout = _build_memory(self.wrapper, item, self.encoder_fn, self._device)
            base_call, query_len, _ = _student_call_for_query(
                self.wrapper,
                mem_rollout,
                item,
                self.tokenizer,
                self.embed_fn,
                device=self._device,
                max_input_tokens=cfg.max_input_tokens,
            )
            gen = generate_greedy(
                self.model,
                self.tokenizer,
                inputs_embeds=base_call.inputs_embeds,
                attention_mask=base_call.attention_mask,
                max_new_tokens=cfg.max_new_tokens,
                stop_strings=(),
                do_sample=cfg.sample_temperature > 0,
                temperature=cfg.sample_temperature,
                top_k=cfg.sample_top_k,
            )
        sampled_ids = torch.tensor(gen.token_ids, dtype=torch.long, device=self._device).unsqueeze(0)
        if sampled_ids.shape[1] == 0:
            # Defensive: nothing to distill on; report a no-op step.
            return OPDStepOutputs(step=step, loss_total=0.0, loss_kl=0.0,
                                  n_tokens_sampled=0)

        reward = float(contains_match(gen.text, item.gold))

        # ---- 2. Student re-forward (with grad) ---------------------------
        mem_grad = _build_memory(self.wrapper, item, self.encoder_fn, self._device)
        deltas: list[torch.Tensor] = []
        if mem_grad.extra is not None:
            # Refetch the per-step deltas for diversity regularization.
            # This requires re-running update with grad — done below by
            # constructing memory from scratch (which builds the graph).
            pass  # deltas not tracked; we keep it simple and skip div for OPD
        student_call, _, sample_len = _student_call_for_query(
            self.wrapper,
            mem_grad,
            item,
            self.tokenizer,
            self.embed_fn,
            device=self._device,
            max_input_tokens=cfg.max_input_tokens,
            extra_token_ids=sampled_ids,
        )
        kwargs: dict[str, Any] = {"use_cache": False}
        if student_call.inputs_embeds is not None:
            kwargs["inputs_embeds"] = student_call.inputs_embeds
        if student_call.attention_mask is not None:
            kwargs["attention_mask"] = student_call.attention_mask
        student_out = self.model(**kwargs)
        student_T = int(student_call.inputs_embeds.shape[1])
        # Logits that *predict* sampled token i live at position
        # (student_T - sample_len + i - 1). Slice the [last_sample_len] window
        # ending one before the final position.
        student_slice = student_out.logits[:, student_T - sample_len - 1: student_T - 1, :]

        # ---- 3. Teacher forward on (chunks + query + sample) -----------
        gold_text = item.gold
        chunks_text = "\n\n".join(item.chunks)
        query_text = format_query_block(item.query)
        teacher_text = chunks_text + query_text
        teacher_ids = tokenize_to_ids(
            self.tokenizer, teacher_text, max_length=cfg.max_input_tokens
        ).to(self._device)
        # Append the sampled ids (the student's actual rollout, not gold).
        teacher_full = torch.cat([teacher_ids, sampled_ids], dim=1)
        teacher_attn = (teacher_full != self.tokenizer.pad_token_id).long()
        del gold_text
        with torch.no_grad():
            teacher_out = self.model(
                input_ids=teacher_full, attention_mask=teacher_attn, use_cache=False
            )
        teacher_T = int(teacher_full.shape[1])
        teacher_slice = teacher_out.logits[:, teacher_T - sample_len - 1: teacher_T - 1, :]

        # ---- 4. KL(student || teacher) on sample positions -------------
        # masked_kl expects [B, T, V] with mask[:, 1:] applied; we
        # pad a dummy first position so the shift is harmless.
        pad_s = torch.zeros(1, 1, student_slice.shape[-1], device=self._device, dtype=student_slice.dtype)
        pad_t = torch.zeros(1, 1, teacher_slice.shape[-1], device=self._device, dtype=teacher_slice.dtype)
        s_padded = torch.cat([pad_s, student_slice], dim=1)
        t_padded = torch.cat([pad_t, teacher_slice], dim=1)
        mask = torch.ones(1, s_padded.shape[1], dtype=torch.long, device=self._device)
        mask[:, 0] = 0
        kl = masked_kl(s_padded, t_padded, mask, temperature=cfg.kl_temperature)
        total = kl

        self.optim.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(list(_trainable_params(self.wrapper)), cfg.grad_clip)
        self.optim.step()

        return OPDStepOutputs(
            step=step,
            loss_total=float(total.item()),
            loss_kl=float(kl.item()),
            reward_mean=reward,
            n_tokens_sampled=int(sampled_ids.shape[1]),
        )


# --------------------------------------------------------------------------- #
# REINFORCE                                                                   #
# --------------------------------------------------------------------------- #


REWARD_FUNCTIONS = {
    "contains": contains_match,
    "exact": exact_match,
    "f1": token_f1,
}


@dataclass
class RLConfig:
    """Hyperparameters for ``RLTrainer``."""

    lr: float = 5e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_steps: int = 500
    log_every: int = 10

    max_input_tokens: int = 2048
    max_new_tokens: int = 8

    n_rollouts: int = 4
    """Rollouts per step. With leave-one-out baseline, ``N >= 2``
    is required for variance reduction; ``N = 4`` is a good default."""

    sample_temperature: float = 1.0
    sample_top_k: int = 0
    sample_top_p: float = 0.95

    reward_fn: str = "contains"
    """One of ``contains | exact | f1``."""

    lambda_kl: float = 0.0
    """Optional KL-to-teacher penalty (anchors the policy)."""

    kl_temperature: float = 1.0

    reward_shift: float = 0.0
    """Constant subtracted from rewards before computing advantages.
    Useful when reward is sparse (mostly 0); a small negative shift
    encourages exploration. 0 = pure advantage."""


@dataclass
class RLStepOutputs:
    step: int
    loss_total: float
    loss_pg: float
    loss_kl: float = 0.0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    advantage_abs_mean: float = 0.0
    n_rollouts: int = 0


@dataclass
class RLTrainer:
    """REINFORCE on the wrapper-augmented student with binary reward.

    Per step:

    1. Build memory once with no_grad and sample ``N`` rollouts from
       the wrapper-augmented student.
    2. Score each rollout's *decoded text* against ``item.gold`` with
       ``REWARD_FUNCTIONS[cfg.reward_fn]`` (contains / em / f1).
    3. Compute leave-one-out advantages
       ``A_i = r_i - mean_{j != i} r_j``  (no learned baseline).
    4. Rebuild memory with grad, recompute the student log-prob of
       each rollout's tokens, and form
       ``loss_pg = -mean_i ( A_i · sum_t log p(τ_i,t) )``.
    5. Optional KL-to-teacher penalty (``λ_kl > 0``) computes the
       teacher full-context distribution at each rollout position and
       adds ``λ_kl · KL(student || teacher)``.

    All N rollouts share a single re-forward of the wrapper recurrence
    (the chunk attention is the heaviest part); per-rollout cost is
    one student forward + one teacher forward, batched if memory
    permits but here run sequentially for simplicity.
    """

    model: Any
    tokenizer: Any
    wrapper: Wrapper
    encoder_fn: Callable[[list[str]], ChunkEncoding]
    embed_fn: Callable[[torch.Tensor], torch.Tensor]
    config: RLConfig = field(default_factory=RLConfig)

    def __post_init__(self):
        params = list(_trainable_params(self.wrapper))
        if not params:
            raise ValueError("wrapper has no trainable params")
        if self.config.reward_fn not in REWARD_FUNCTIONS:
            raise ValueError(
                f"unknown reward_fn={self.config.reward_fn!r}; "
                f"choices={sorted(REWARD_FUNCTIONS)}"
            )
        self.optim = torch.optim.AdamW(
            params, lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self._device = next(self.model.parameters()).device
        self._reward_fn = REWARD_FUNCTIONS[self.config.reward_fn]

    def train(self, items: Iterable[LongContextItem]) -> list[RLStepOutputs]:
        items_list = list(items)
        history: list[RLStepOutputs] = []
        step = 0
        while step < self.config.max_steps:
            for item in items_list:
                if step >= self.config.max_steps:
                    break
                out = self.train_step(item, step)
                history.append(out)
                if step % self.config.log_every == 0:
                    logger.info(
                        f"[rl] step={step} total={out.loss_total:.4f} "
                        f"pg={out.loss_pg:.4f} kl={out.loss_kl:.4f} "
                        f"r_mean={out.reward_mean:.3f} r_std={out.reward_std:.3f} "
                        f"|A|_mean={out.advantage_abs_mean:.3f} N={out.n_rollouts}"
                    )
                step += 1
        return history

    def _sample_rollouts(
        self, item: LongContextItem, memory: MemoryState, n: int
    ) -> list[tuple[torch.Tensor, str, float]]:
        """Return list of (token_ids[1,T], decoded_text, reward)."""

        cfg = self.config
        rollouts: list[tuple[torch.Tensor, str, float]] = []
        base_call, _, _ = _student_call_for_query(
            self.wrapper,
            memory,
            item,
            self.tokenizer,
            self.embed_fn,
            device=self._device,
            max_input_tokens=cfg.max_input_tokens,
        )
        for _ in range(n):
            gen = generate_greedy(
                self.model,
                self.tokenizer,
                inputs_embeds=base_call.inputs_embeds,
                attention_mask=base_call.attention_mask,
                max_new_tokens=cfg.max_new_tokens,
                stop_strings=(),
                do_sample=True,
                temperature=cfg.sample_temperature,
                top_k=cfg.sample_top_k,
                top_p=cfg.sample_top_p,
            )
            ids = torch.tensor(gen.token_ids, dtype=torch.long, device=self._device).unsqueeze(0)
            reward = float(self._reward_fn(gen.text, item.gold))
            rollouts.append((ids, gen.text, reward))
        return rollouts

    def train_step(self, item: LongContextItem, step: int) -> RLStepOutputs:
        if isinstance(self.wrapper, nn.Module):
            self.wrapper.train()
        cfg = self.config

        # ---- 1. Rollouts (no grad) -------------------------------------
        with torch.no_grad():
            mem_rollout = _build_memory(self.wrapper, item, self.encoder_fn, self._device)
            rollouts = self._sample_rollouts(item, mem_rollout, cfg.n_rollouts)
        if not rollouts or all(r[0].shape[1] == 0 for r in rollouts):
            return RLStepOutputs(step=step, loss_total=0.0, loss_pg=0.0, n_rollouts=0)

        rewards = [r[2] for r in rollouts]
        n_eff = len(rewards)
        r_mean = sum(rewards) / n_eff
        r_var = sum((r - r_mean) ** 2 for r in rewards) / n_eff
        r_std = float(r_var ** 0.5)
        # Leave-one-out baseline:
        #   A_i = r_i - (sum_j r_j - r_i) / (N - 1)   if N > 1
        #   A_i = r_i - reward_shift                  if N = 1
        if n_eff > 1:
            total_r = sum(rewards)
            advantages = [
                ri - (total_r - ri) / (n_eff - 1) - cfg.reward_shift
                for ri in rewards
            ]
        else:
            advantages = [rewards[0] - cfg.reward_shift]
        adv_abs_mean = float(sum(abs(a) for a in advantages) / n_eff)

        # ---- 2. Re-forward with grad on each rollout -------------------
        mem_grad = _build_memory(self.wrapper, item, self.encoder_fn, self._device)
        pg_terms: list[torch.Tensor] = []
        kl_terms: list[torch.Tensor] = []
        chunks_text = "\n\n".join(item.chunks)
        query_text = format_query_block(item.query)
        for (ids, _text, _r), adv in zip(rollouts, advantages):
            if ids.shape[1] == 0:
                continue
            student_call, _, sample_len = _student_call_for_query(
                self.wrapper,
                mem_grad,
                item,
                self.tokenizer,
                self.embed_fn,
                device=self._device,
                max_input_tokens=cfg.max_input_tokens,
                extra_token_ids=ids,
            )
            kwargs: dict[str, Any] = {"use_cache": False}
            if student_call.inputs_embeds is not None:
                kwargs["inputs_embeds"] = student_call.inputs_embeds
            if student_call.attention_mask is not None:
                kwargs["attention_mask"] = student_call.attention_mask
            student_out = self.model(**kwargs)
            student_T = int(student_call.inputs_embeds.shape[1])
            student_slice = student_out.logits[:, student_T - sample_len - 1: student_T - 1, :]
            logp = F.log_softmax(student_slice, dim=-1)
            # Gather log p at the actual sampled tokens
            sampled_logp = logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)  # [1, T]
            seq_logp = sampled_logp.sum(dim=1).squeeze(0)  # scalar
            pg_terms.append(-adv * seq_logp)

            if cfg.lambda_kl > 0.0:
                # Teacher full-context forward on the same rollout
                teacher_full = torch.cat(
                    [tokenize_to_ids(self.tokenizer, chunks_text + query_text,
                                     max_length=cfg.max_input_tokens).to(self._device),
                     ids],
                    dim=1,
                )
                teacher_attn = (teacher_full != self.tokenizer.pad_token_id).long()
                with torch.no_grad():
                    teacher_out = self.model(
                        input_ids=teacher_full, attention_mask=teacher_attn, use_cache=False
                    )
                teacher_T = int(teacher_full.shape[1])
                teacher_slice = teacher_out.logits[:, teacher_T - sample_len - 1: teacher_T - 1, :]
                pad_s = torch.zeros(1, 1, student_slice.shape[-1], device=self._device, dtype=student_slice.dtype)
                pad_t = torch.zeros(1, 1, teacher_slice.shape[-1], device=self._device, dtype=teacher_slice.dtype)
                s_padded = torch.cat([pad_s, student_slice], dim=1)
                t_padded = torch.cat([pad_t, teacher_slice], dim=1)
                mask = torch.ones(1, s_padded.shape[1], dtype=torch.long, device=self._device)
                mask[:, 0] = 0
                kl_terms.append(masked_kl(s_padded, t_padded, mask, temperature=cfg.kl_temperature))

        if not pg_terms:
            return RLStepOutputs(step=step, loss_total=0.0, loss_pg=0.0, n_rollouts=0)

        pg_loss = torch.stack(pg_terms).mean()
        if kl_terms:
            kl_loss = torch.stack(kl_terms).mean()
            total = pg_loss + cfg.lambda_kl * kl_loss
        else:
            kl_loss = torch.zeros((), device=self._device)
            total = pg_loss

        self.optim.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(list(_trainable_params(self.wrapper)), cfg.grad_clip)
        self.optim.step()

        return RLStepOutputs(
            step=step,
            loss_total=float(total.item()),
            loss_pg=float(pg_loss.item()),
            loss_kl=float(kl_loss.item()),
            reward_mean=r_mean,
            reward_std=r_std,
            advantage_abs_mean=adv_abs_mean,
            n_rollouts=n_eff,
        )
