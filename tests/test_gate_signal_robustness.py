from types import SimpleNamespace

import torch

from gcm.model import GCMModel


class GateBase(torch.nn.Module):
    def __init__(self, peak=2.0):
        super().__init__()
        self.peak = peak

    def forward(self, inputs_embeds, **_kwargs):
        batch, length, _ = inputs_embeds.shape
        logits = torch.zeros(batch, length, 8)
        logits[..., 3] = self.peak
        logits[..., 4] = 1.0
        return SimpleNamespace(logits=logits)


def _model(peak=2.0):
    model = object.__new__(GCMModel)
    torch.nn.Module.__init__(model)
    model.base = GateBase(peak)
    model._embed = torch.nn.Embedding(8, 4)
    return model


def test_gate_core_signals_survive_optional_signal_failure(monkeypatch):
    monkeypatch.setenv("GCM_GATE_SIGNALS", "conf,margin,neg_entropy,neg_recon,dlogit")
    model = _model()
    model.encode = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("optional failure"))
    memory = torch.randn(1, 2, 4)
    context = torch.tensor([[1, 2]])
    query = torch.tensor([[5, 6]])

    signals = model.gate_signal(memory, context, query)

    assert signals["conf"] > 0.0
    assert signals["margin"] > 0.0
    assert signals["neg_entropy"] < 0.0
    assert signals["neg_recon"] == 0.0
    assert signals["dlogit"] == 0.0
    assert all(torch.isfinite(torch.tensor(value)) for value in signals.values())


def test_negative_entropy_is_higher_for_a_more_confident_distribution(monkeypatch):
    monkeypatch.setenv("GCM_GATE_SIGNALS", "conf,margin,neg_entropy")
    memory = torch.randn(1, 2, 4)
    context = torch.tensor([[1, 2]])
    query = torch.tensor([[5, 6]])

    uncertain = _model(peak=1.0).gate_signal(memory, context, query)
    confident = _model(peak=8.0).gate_signal(memory, context, query)

    assert confident["neg_entropy"] > uncertain["neg_entropy"]
