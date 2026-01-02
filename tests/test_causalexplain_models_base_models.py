import numpy as np
import pytest
import torch

import causalexplain.models._base_models as base_models
from causalexplain.models._base_models import DFF, MDN, MLP
from causalexplain.models._loss import MMDLoss


@pytest.mark.parametrize(
    "activation,expected_cls",
    [
        ("gelu", torch.nn.GELU),
        ("selu", torch.nn.SELU),
        ("tanh", torch.nn.Tanh),
        ("linear", torch.nn.Identity),
        ("sigmoid", torch.nn.Sigmoid),
    ],
)
@pytest.mark.parametrize(
    "loss_name,expected_loss",
    [
        ("mae", torch.nn.L1Loss),
        ("mmd", MMDLoss),
        ("binary_crossentropy", torch.nn.BCEWithLogitsLoss),
        ("crossentropy", torch.nn.CrossEntropyLoss),
    ],
)
def test_mlp_instantiation_covers_additional_branches(activation, expected_cls, loss_name, expected_loss):
    mlp = MLP(
        input_size=2,
        layers_dimensions=[3],
        activation=activation,
        batch_size=2,
        lr=0.01,
        loss=loss_name,
        dropout=0.0,
    )
    assert isinstance(mlp.activation, expected_cls)
    assert isinstance(mlp.loss_fn, expected_loss) or callable(mlp.loss_fn)


def test_mlp_training_and_validation_steps_return_losses(monkeypatch):
    mlp = MLP(
        input_size=2,
        layers_dimensions=[1],
        activation="tanh",
        batch_size=2,
        lr=0.01,
        loss="mae",
        dropout=0.0,
    )
    mlp.log = lambda *_, **__: None
    batch = (torch.zeros((2, 1)), torch.ones((2, 1)))
    train_loss = mlp.training_step(batch, 0)
    val_loss = mlp.validation_step(batch, 0)
    assert train_loss > 0
    assert val_loss > 0


def test_mlp_invalid_activation():
    with pytest.raises(ValueError):
        MLP(
            input_size=2,
            layers_dimensions=[2],
            activation="unsupported",
            batch_size=1,
            lr=0.01,
            loss="mse",
            dropout=0.1,
        )


def test_mlp_invalid_loss():
    with pytest.raises(ValueError):
        MLP(
            input_size=2,
            layers_dimensions=[2],
            activation="relu",
            batch_size=1,
            lr=0.01,
            loss="not-a-loss",
            dropout=0.1,
        )


def test_mlp_forward_and_predict_shape():
    torch.manual_seed(0)
    mlp = MLP(
        input_size=3,
        layers_dimensions=[2],
        activation="relu",
        batch_size=2,
        lr=0.01,
        loss="mse",
        dropout=0.0,
    )
    x = torch.ones((2, 2))
    out = mlp.forward(x)
    assert out.shape == (2, 1)

    np_out = mlp.predict(np.ones((2, 2), dtype=np.float32))
    assert np_out.shape == (2, 1)


def test_mlp_noise_is_used(monkeypatch):
    mlp = MLP(
        input_size=2,
        layers_dimensions=[],
        activation="relu",
        batch_size=2,
        lr=0.01,
        loss="mse",
        dropout=0.0,
    )
    with torch.no_grad():
        mlp.head.weight.copy_(torch.tensor([[0.0, 1.0]]))
        mlp.head.bias.zero_()

    x = torch.zeros((2, 1))

    monkeypatch.setattr(
        base_models.torch, "randn",
        lambda *args, **kwargs: torch.zeros(*args, **kwargs))
    out_zero = mlp.forward(x)

    monkeypatch.setattr(
        base_models.torch, "randn",
        lambda *args, **kwargs: torch.ones(*args, **kwargs))
    out_one = mlp.forward(x)

    assert not torch.allclose(out_zero, out_one)


def _assert_noise_device(monkeypatch, device):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    captured = {}

    def fake_randn(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        if "dtype" not in kwargs:
            kwargs["dtype"] = torch.float32
        return torch.zeros(*args, **kwargs)

    try:
        monkeypatch.setattr(base_models.torch, "randn", fake_randn)
        mlp = MLP(
            input_size=2,
            layers_dimensions=[],
            activation="relu",
            batch_size=2,
            lr=0.01,
            loss="mse",
            dropout=0.0,
        ).to(device)
        x = torch.zeros((2, 1), device=device, dtype=torch.float32)
        mlp.forward(x)

        assert captured["args"] == (x.shape[0], 1)
        assert captured["kwargs"]["device"] == x.device
    finally:
        torch.set_default_dtype(old_dtype)


def test_mlp_noise_device_cpu(monkeypatch):
    _assert_noise_device(monkeypatch, torch.device("cpu"))


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.backends.cuda.is_built()),
    reason="CUDA not available",
)
def test_mlp_noise_device_cuda(monkeypatch):
    _assert_noise_device(monkeypatch, torch.device("cuda"))


@pytest.mark.skipif(
    not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
    reason="MPS not available",
)
def test_mlp_noise_device_mps(monkeypatch):
    _assert_noise_device(monkeypatch, torch.device("mps"))


def test_mlp_forward_cpu_golden():
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(123)
    try:
        mlp = MLP(
            input_size=3,
            layers_dimensions=[2],
            activation="relu",
            batch_size=2,
            lr=0.01,
            loss="mse",
            dropout=0.0,
        )
        x = torch.ones((2, 2), dtype=torch.float32)
        out = mlp.forward(x)
        expected = torch.tensor([[-0.2612704], [-0.2612704]])
        assert torch.allclose(out, expected, atol=1e-6)
    finally:
        torch.set_default_dtype(old_dtype)


def test_dff_invalid_loss():
    with pytest.raises(ValueError):
        DFF(input_size=1, hidden_size=1, batch_size=1, lr=0.01, loss="oops")


def test_dff_training_and_validation_steps(monkeypatch):
    dff = DFF(input_size=1, hidden_size=1, batch_size=1, lr=0.01, loss="mae")
    monkeypatch.setattr(dff, "forward", lambda x: torch.zeros((x.shape[0], 1)))
    dff.log = lambda *_, **__: None
    x = torch.ones((1, 1))
    y = torch.zeros((1, 1))
    train_loss = dff.training_step((x, y), 0)
    dff.validation_step((x, y), 0)
    assert train_loss >= 0


@pytest.mark.parametrize("kernel", ["multiscale", "rbf"])
def test_mdn_static_mmd_loss_kernels(kernel):
    x = torch.tensor([[0.0], [1.0]])
    y = torch.tensor([[0.5], [1.5]])

    loss_val = MDN.mmd_loss(x, y, kernel)
    assert loss_val >= 0


def test_mdn_common_step_branches(monkeypatch):
    mdn = MDN(
        input_size=1,
        hidden_size=1,
        num_gaussians=1,
        lr=0.01,
        batch_size=1,
        loss_function="loglikelihood",
    )

    # Bypass the real forward call to test the loglikelihood path deterministically.
    def fake_forward(x):
        return (
            torch.tensor([[1.0]]),
            torch.tensor([[1.0]]),
            torch.tensor([[0.0]]),
        )

    monkeypatch.setattr(mdn, "forward", fake_forward)
    x = torch.zeros((1, 1))
    y = torch.zeros((1, 1))
    loss = mdn.common_step((x, y))
    assert loss >= 0

    # Now exercise the mmd branch with controlled outputs.
    mdn.loss_fn = "mmd"
    monkeypatch.setattr(mdn, "forward", lambda _x: fake_forward(_x))
    monkeypatch.setattr(mdn, "g_sample", lambda *_, **__: torch.zeros((1, 1)))
    monkeypatch.setattr(mdn, "mmd_loss", lambda *_, **__: torch.tensor(0.5))
    mmd_loss = mdn.common_step((x, y))
    assert mmd_loss == torch.tensor(0.5)


def test_mdn_forward_and_validation_step():
    mdn = MDN(
        input_size=1,
        hidden_size=2,
        num_gaussians=2,
        lr=0.01,
        batch_size=2,
        loss_function="loglikelihood",
    )

    class Wrapper:
        def __init__(self, tensor):
            self.tensor = tensor
            self.shape = tensor.shape

        def to_device(self, _device):
            return self.tensor

    wrapped = Wrapper(torch.zeros((2, 0)))
    pi, sigma, mu = mdn.forward(wrapped)
    assert pi.shape[0] == 2 and sigma.shape[0] == 2 and mu.shape[0] == 2
    mdn.log = lambda *_, **__: None
    mdn.validation_step((torch.zeros((1, 1)), torch.zeros((1, 1))), 0)
    optim = mdn.configure_optimizers()
    assert "optimizer" in optim and "lr_scheduler" in optim


def test_mdn_gaussian_probability_and_sampling():
    pi = torch.tensor([[0.6], [0.4]])
    sigma = torch.tensor([[1.0], [1.0]])
    mu = torch.tensor([[0.0], [0.5]])

    probs = MDN.gaussian_probability(torch.ones((2, 1)), mu, sigma)
    assert probs.shape == mu.shape

    samples = MDN.g_sample(pi, sigma, mu)
    assert isinstance(samples, torch.Tensor)
    assert samples.shape[0] == pi.shape[0]

    added_noise = MDN.add_noise(torch.ones((2, 1)))
    assert added_noise.shape[1] == 2

    pi_full = torch.ones((2, 2)) * 0.5
    sigma_full = torch.ones((2, 2, 1))
    mu_full = torch.zeros((2, 2, 1))
    sampled = MDN.sample(pi_full, sigma_full, mu_full)
    assert sampled.shape[0] == pi_full.shape[0]
