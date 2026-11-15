"""Checks of the reference simulator against closed forms and against torch.

If these fail, no fixture output can be trusted."""

import inspect

import nir
import numpy as np
import pytest

from nirconf.reference import Convention, simulate

F = np.float32
HAS_V_RESET = "v_reset" in {f.name for f in __import__("dataclasses").fields(nir.LIF)}
HAS_TYPE_CHECK = "type_check" in inspect.signature(nir.NIRGraph.__init__).parameters


def chain(nodes, n_in, n_out, type_check=True):
    keys = ["input", *nodes, "output"]
    nodes = {"input": nir.Input(np.array(n_in)), **nodes, "output": nir.Output(np.array(n_out))}
    kw = {"type_check": type_check} if HAS_TYPE_CHECK else {}
    return nir.NIRGraph(nodes=nodes, edges=list(zip(keys[:-1], keys[1:])), **kw)


def lif(n=1, tau=20.0, r=1.0, v_leak=0.0, thr=1.0, v_reset=0.0):
    f = lambda v: np.full(n, v, F)  # noqa: E731
    kw = {"v_reset": f(v_reset)} if HAS_V_RESET else {}
    if v_reset and not HAS_V_RESET:
        pytest.skip("nir < 1.0.6 has no v_reset")
    return nir.LIF(tau=f(tau), r=f(r), v_leak=f(v_leak), v_threshold=f(thr), **kw)


def const_input(T, value, n=1):
    return np.full((T, 1, n), value, F)


def test_affine_matches_numpy():
    rng = np.random.default_rng(0)
    W, b = rng.normal(size=(5, 3)).astype(F), rng.normal(size=5).astype(F)
    g = chain({"fc": nir.Affine(W, b)}, [3], [5])
    x = rng.normal(size=(4, 2, 3)).astype(F)
    np.testing.assert_allclose(simulate(g, x, 1.0), x @ W.T.astype(np.float64) + b, rtol=1e-12)


def test_lif_euler_closed_form_first_spike():
    # v_t = c * (1 - (1-a)^t) with a = dt/tau, gain a*r ; constant input c
    dt, tau, c, thr = 1.0, 20.0, 3.0, 1.0
    a = dt / tau
    g = chain({"lif": lif(tau=tau, thr=thr)}, [1], [1])
    y = simulate(g, const_input(50, c), dt)[:, 0, 0]
    t_star = next(t for t in range(1, 200) if c * (1 - (1 - a) ** t) > thr)
    assert np.argmax(y > 0) == t_star - 1  # step index t_star-1 is the t_star-th update


def test_lif_reset_value_vs_subtract():
    g = chain({"lif": lif(tau=1e9, thr=1.0, v_reset=-0.25)}, [1], [1])  # ~no leak
    x = const_input(3, 0.6 / (1.0 / 1e9))  # gain dt/tau => jump of 0.6 per step
    _, tr = simulate(g, x, 1.0, record=True)
    v_seq = []
    # value reset: 0.6, 1.2->spike->-0.25, 0.35
    y = tr["lif"][:, 0, 0]
    assert list(y) == [0.0, 1.0, 0.0]
    ys = simulate(g, x, 1.0, Convention(reset="subtract"))[:, 0, 0]
    assert list(ys) == [0.0, 1.0, 0.0]


def test_if_euler_gain_uses_dt():
    g = chain({"if": nir.IF(r=np.ones(1, F), v_threshold=np.full(1, 1e-4 * 2.5, F))}, [1], [1])
    y = simulate(g, const_input(6, 1.0), 1e-4)[:, 0, 0]
    # v after k steps = k*1e-4 ; spikes when > 2.5e-4: step 3 (k=3), reset to 0, then k=6
    assert list(y) == [0, 0, 1, 0, 0, 1]


def test_if_unit_gain_variant_ignores_dt():
    g = chain({"if": nir.IF(r=np.ones(1, F), v_threshold=np.full(1, 2.5, F))}, [1], [1])
    y = simulate(g, const_input(6, 1.0), 1e-4, Convention(input_gain="unit"))[:, 0, 0]
    assert list(y) == [0, 0, 1, 0, 0, 1]


def test_strict_vs_ge_threshold():
    g = chain({"if": nir.IF(r=np.ones(1, F), v_threshold=np.full(1, 2.0, F))}, [1], [1])
    x = const_input(2, 1.0)
    assert list(simulate(g, x, 1.0)[:, 0, 0]) == [0, 0]
    assert list(simulate(g, x, 1.0, Convention(compare="ge"))[:, 0, 0]) == [0, 1]


def test_exact_decay_variant():
    g = chain({"li": nir.LI(tau=np.full(1, 10.0, F), r=np.ones(1, F), v_leak=np.zeros(1, F))}, [1], [1])
    y = simulate(g, const_input(5, 1.0), 1.0, Convention(decay="exact"))[:, 0, 0]
    a = np.exp(-0.1)
    expect = [(1 - a) * sum(a**j for j in range(t + 1)) for t in range(5)]
    np.testing.assert_allclose(y, expect, rtol=1e-12)


def test_cubalif_uses_updated_current():
    n = nir.CubaLIF(tau_syn=np.full(1, 2.0, F), tau_mem=np.full(1, 4.0, F), r=np.ones(1, F),
                    v_leak=np.zeros(1, F), v_threshold=np.full(1, 100.0, F), w_in=np.ones(1, F))
    g = chain({"lif": n}, [1], [1])
    _, tr = simulate(g, const_input(2, 1.0), 1.0, record=True)
    # I1 = 0.5 ; v1 = 0.25*I1 = 0.125 (membrane sees the updated current)
    # checked through CubaLI equivalent below
    if not hasattr(nir, "CubaLI"):
        pytest.skip("nir < 1.0.5 has no CubaLI")
    li = nir.CubaLI(tau_syn=n.tau_syn, tau_mem=n.tau_mem, r=n.r, v_leak=n.v_leak, w_in=n.w_in)
    y = simulate(chain({"li": li}, [1], [1]), const_input(2, 1.0), 1.0)[:, 0, 0]
    np.testing.assert_allclose(y[0], 0.125)
    # I2 = 0.5 + 0.5*(1-0.5) = 0.75 ; v2 = 0.125*0.75 + 0.25*0.75 = 0.28125
    np.testing.assert_allclose(y[1], 0.28125)


def test_recurrent_edge_is_one_step_delayed():
    # LIF with huge threshold except neuron feeds itself; use Linear rec with weight 1 on LI
    li = nir.LI(tau=np.full(1, 1.0, F), r=np.ones(1, F), v_leak=np.zeros(1, F))  # dt/tau = 1 => v = x
    nodes = {"input": nir.Input(np.array([1])), "li": li, "rec": nir.Linear(np.ones((1, 1), F)), "output": nir.Output(np.array([1]))}
    g = nir.NIRGraph(nodes=nodes, edges=[("input", "li"), ("li", "rec"), ("rec", "li"), ("li", "output")])
    x = np.array([1, 0, 0, 0], F).reshape(4, 1, 1)
    y = simulate(g, x, 1.0)[:, 0, 0]
    # v_t = x_t + v_{t-1}  (recurrent input from previous step)
    assert list(y) == [1, 1, 1, 1]


def test_delay_shifts_by_steps():
    g = chain({"d": nir.Delay(np.array([2.0], F))}, [1], [1])
    x = np.array([1, 0, 0, 0, 0], F).reshape(5, 1, 1)
    assert list(simulate(g, x, 1.0)[:, 0, 0]) == [0, 0, 1, 0, 0]


torch = pytest.importorskip("torch")


@pytest.mark.parametrize("stride,padding,groups", [(1, 1, 1), (2, 0, 1), (1, 1, 2)])
def test_conv2d_matches_torch(stride, padding, groups):
    rng = np.random.default_rng(1)
    w = rng.normal(size=(4, 4 // groups, 3, 3)).astype(F)
    b = rng.normal(size=4).astype(F)
    node = nir.Conv2d(input_shape=(7, 7), weight=w, stride=stride, padding=padding, dilation=1, groups=groups, bias=b)
    x = rng.normal(size=(1, 2, 4, 7, 7)).astype(F)
    # type_check=False: NIR derives a grouped conv's input channels as weight.shape[1]
    # (= C_in/groups), so a valid grouped graph fails type inference (see test below).
    y = simulate(chain({"c": node}, [4, 7, 7], list(node.output_type["output"]), type_check=groups == 1), x, 1.0)
    t = torch.nn.functional.conv2d(torch.tensor(x[0], dtype=torch.float64), torch.tensor(w, dtype=torch.float64),
                                   torch.tensor(b, dtype=torch.float64), stride=stride, padding=padding, groups=groups)
    np.testing.assert_allclose(y[0], t.numpy(), rtol=1e-10, atol=1e-10)


def test_conv1d_matches_torch():
    rng = np.random.default_rng(2)
    w, b = rng.normal(size=(3, 2, 3)).astype(F), rng.normal(size=3).astype(F)
    node = nir.Conv1d(input_shape=10, weight=w, stride=1, padding=1, dilation=1, groups=1, bias=b)
    x = rng.normal(size=(1, 2, 2, 10)).astype(F)
    y = simulate(chain({"c": node}, [2, 10], [3, 10]), x, 1.0)
    t = torch.nn.functional.conv1d(torch.tensor(x[0], dtype=torch.float64), torch.tensor(w, dtype=torch.float64), torch.tensor(b, dtype=torch.float64), padding=1)
    np.testing.assert_allclose(y[0], t.numpy(), rtol=1e-10, atol=1e-10)


def test_pools_match_torch():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(1, 2, 2, 6, 6)).astype(F)
    sp = nir.SumPool2d(kernel_size=np.array([2, 2]), stride=np.array([2, 2]), padding=np.array([0, 0]))
    ap = nir.AvgPool2d(kernel_size=np.array([2, 2]), stride=np.array([2, 2]), padding=np.array([0, 0]))
    xs = torch.tensor(x[0], dtype=torch.float64)
    ys = simulate(chain({"p": sp}, [2, 6, 6], [2, 3, 3]), x, 1.0)[0]
    ya = simulate(chain({"p": ap}, [2, 6, 6], [2, 3, 3]), x, 1.0)[0]
    np.testing.assert_allclose(ya, torch.nn.functional.avg_pool2d(xs, 2).numpy(), rtol=1e-12)
    np.testing.assert_allclose(ys, 4 * torch.nn.functional.avg_pool2d(xs, 2).numpy(), rtol=1e-12)


def test_flatten_excludes_batch():
    f = nir.Flatten(input_type={"input": np.array([2, 3, 3])}, start_dim=0, end_dim=-1)
    x = np.arange(2 * 4 * 18, dtype=F).reshape(2, 4, 2, 3, 3)
    y = simulate(chain({"f": f}, [2, 3, 3], [18]), x, 1.0)
    assert y.shape == (2, 4, 18)
    np.testing.assert_array_equal(y, x.reshape(2, 4, 18))


def test_nir_grouped_conv_input_type_ignores_groups():
    """Documents a NIR-core issue: Conv2d.input_type uses weight.shape[1] (C_in/groups)."""
    if not HAS_TYPE_CHECK:
        pytest.skip("nir 1.0.4 has no NIRGraph type_check argument")
    w = np.zeros((4, 2, 3, 3), F)  # groups=2 => C_in = 4
    node = nir.Conv2d(input_shape=(7, 7), weight=w, stride=1, padding=1, dilation=1, groups=2, bias=np.zeros(4, F))
    assert int(node.input_type["input"][0]) == 2  # should be 4
    with pytest.raises(ValueError, match="type mismatch"):
        chain({"c": node}, [4, 7, 7], [4, 7, 7])
