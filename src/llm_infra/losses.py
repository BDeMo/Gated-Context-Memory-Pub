"""Losses for the BPTT trainer.

Joint loss per v0 doc:

    L = CE(student, gold) + lambda * KL(student || teacher)

CE is masked to the answer span (loss_mask). KL is over the teacher's predicted
distribution at every answer-span position. Teacher logits come from the same
frozen base run with the full context; student logits come from the same base
run with the wrapper-supplied memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class JointLossOutputs:
    total: torch.Tensor
    ce: torch.Tensor
    kl: torch.Tensor
    lambda_kl: float


def masked_ce(logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on tokens where `loss_mask == 1`.

    Args:
        logits:    `[B, T, V]` student logits over the full sequence.
        labels:    `[B, T]` gold token ids; values where mask is 0 are ignored.
        loss_mask: `[B, T]` bool/long where 1 marks the answer span.
    """

    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    shifted_mask = loss_mask[..., 1:].contiguous().float()

    flat_logits = shifted_logits.view(-1, shifted_logits.size(-1))
    flat_labels = shifted_labels.view(-1)
    per_token = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    per_token = per_token.view_as(shifted_labels) * shifted_mask

    denom = shifted_mask.sum().clamp_min(1.0)
    return per_token.sum() / denom


def masked_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL(student || teacher) over answer-span positions only.

    Both logits must be aligned at the answer span (caller's responsibility).
    `loss_mask[i, t] == 1` selects positions to score.
    """

    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student/teacher logits shape mismatch: {student_logits.shape} vs {teacher_logits.shape}"
        )

    shifted_student = student_logits[..., :-1, :] / temperature
    shifted_teacher = teacher_logits[..., :-1, :] / temperature
    shifted_mask = loss_mask[..., 1:].contiguous().float()

    log_p_student = F.log_softmax(shifted_student, dim=-1)
    p_teacher = F.softmax(shifted_teacher, dim=-1)
    log_p_teacher = torch.log(p_teacher.clamp_min(1e-12))
    per_token_kl = (p_teacher * (log_p_teacher - log_p_student)).sum(dim=-1)

    per_token_kl = per_token_kl * shifted_mask
    denom = shifted_mask.sum().clamp_min(1.0)
    return (per_token_kl.sum() / denom) * (temperature**2)


def joint_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    lambda_kl: float = 1.0,
    temperature: float = 1.0,
) -> JointLossOutputs:
    ce = masked_ce(student_logits, labels, loss_mask)
    kl = masked_kl(student_logits, teacher_logits, loss_mask, temperature=temperature)
    total = ce + lambda_kl * kl
    return JointLossOutputs(total=total, ce=ce, kl=kl, lambda_kl=lambda_kl)
