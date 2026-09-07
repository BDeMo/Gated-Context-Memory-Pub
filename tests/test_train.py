from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gcm.model import GCMModel
from gcm.train import train_compressor


def _tiny_model():
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        eos_token_id=2,
        pad_token_id=0,
    )
    base = LlamaForCausalLM(config)
    tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    return GCMModel(
        base,
        tokenizer,
        n_memory=4,
        proj_mult=1,
        lora_rank=2,
    )


def test_short_training_updates_adapters_without_nonfinite_values():
    torch.manual_seed(7)
    model = _tiny_model()
    items = [
        (
            torch.tensor([[3, 4, 5, 6]]),
            torch.tensor([[7, 8]]),
            torch.tensor([[9]]),
        ),
        (
            torch.tensor([[10, 11, 12, 13]]),
            torch.tensor([[14, 15]]),
            torch.tensor([[16]]),
        ),
    ]
    before = [parameter.detach().clone() for parameter in model.trainable_parameters()]

    train_compressor(
        model,
        items,
        max_steps=4,
        batch_size=2,
        lr=1e-3,
        lam_distill=0.0,
        log_every=100,
    )

    after = model.trainable_parameters()
    assert any(not torch.equal(old, new) for old, new in zip(before, after))
    assert all(torch.isfinite(parameter).all() for parameter in after)


def test_adversarial_update_modes_remain_finite():
    items = [
        (
            torch.tensor([[3, 4, 5, 6]]),
            torch.tensor([[7, 8]]),
            torch.tensor([[9]]),
        ),
        (
            torch.tensor([[10, 11, 12, 13]]),
            torch.tensor([[14, 15]]),
            torch.tensor([[16]]),
        ),
    ]
    for mode in ("alternating", "simultaneous"):
        torch.manual_seed(11)
        model = _tiny_model()
        before = [parameter.detach().clone() for parameter in model.trainable_parameters()]
        train_compressor(
            model,
            items,
            max_steps=2,
            batch_size=1,
            lr=1e-3,
            lam_distill=0.0,
            lam_adv=0.1,
            adv_layer=1,
            adv_mode=mode,
            log_every=100,
        )
        after = model.trainable_parameters()
        assert any(not torch.equal(old, new) for old, new in zip(before, after))
        assert all(torch.isfinite(parameter).all() for parameter in after)


def test_unknown_adversarial_update_mode_is_rejected():
    model = _tiny_model()
    try:
        train_compressor(model, [], adv_mode="joint")
    except ValueError as exc:
        assert "adversarial update mode" in str(exc)
    else:
        raise AssertionError("unknown adversarial mode should fail")


def test_adapter_checkpoint_restores_all_inference_state(tmp_path):
    source = _tiny_model()
    with torch.no_grad():
        source.compressor.mem.fill_(0.25)
        source.compressor.embed_scale.fill_(3.5)
        source.rec_slot.fill_(0.75)
        for module in source.base.modules():
            if hasattr(module, "lora_B"):
                module.lora_B.fill_(0.5)
    path = tmp_path / "adapters.pt"
    source.save_adapters(str(path))

    restored = _tiny_model()
    restored.load_adapters(str(path), map_location="cpu")

    assert torch.equal(restored.compressor.mem, source.compressor.mem)
    assert torch.equal(restored.compressor.embed_scale, source.compressor.embed_scale)
    assert torch.equal(restored.rec_slot, source.rec_slot)
    source_lora = [
        module.lora_B
        for module in source.base.modules()
        if hasattr(module, "lora_B")
    ]
    restored_lora = [
        module.lora_B
        for module in restored.base.modules()
        if hasattr(module, "lora_B")
    ]
    assert all(torch.equal(left, right) for left, right in zip(source_lora, restored_lora))
