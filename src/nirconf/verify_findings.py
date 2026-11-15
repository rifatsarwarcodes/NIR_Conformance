"""Direct checks for every finding claimed in docs/FINDINGS.md.

The conformance matrix says *that* a framework disagrees with the reference;
the fingerprint suggests *why*. Each finding is then confirmed here by a
targeted test that does not depend on the fingerprint search:

* parameter inspection of the imported module,
* a minimal hand-built graph with an obvious expected answer, or
* an invariance test: change only parameter X in the NIR graph; if the
  framework's output is bit-identical, the importer ignores X.

    python -m nirconf.verify_findings <results_dir>

Runs every check whose framework is importable in the current environment and
writes ``<results_dir>/verified/<env>.json``.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
import sys
import traceback
import warnings
from pathlib import Path
from typing import Callable, Dict, List

import nir
import numpy as np

from . import adapters
from .reference import Convention, simulate

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

FIX = Path(__file__).resolve().parents[2] / "fixtures"
F = np.float32
CHECKS: List[dict] = []


# which environment a framework's checks run in: "latest" = nir 1.0.8, "pinned" = nir 1.0.4
DEFAULT_ENVS = {"nir": ("latest",), "nirtorch": ("latest",), "snntorch": ("latest",), "norse": ("latest", "pinned"),
                "spyx": ("latest",), "sinabs": ("pinned",), "rockpool": ("pinned",)}


def check(cid: str, framework: str, claim: str, envs=None):
    def deco(fn: Callable[[], str]):
        CHECKS.append({"id": cid, "framework": framework, "claim": claim, "fn": fn, "envs": envs or DEFAULT_ENVS[framework]})
        return fn
    return deco


def load(suite: str, name: str, legacy: bool = False):
    d = FIX / suite / name
    f = d / ("graph_legacy.nir" if legacy and (d / "graph_legacy.nir").exists() else "graph.nir")
    meta = json.loads((d / "meta.json").read_text())
    return nir.read(f), np.load(d / "input.npy"), np.load(d / "expected.npy"), meta["dt"]


def run(fw, g, x, dt):
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return adapters.run(fw, g, x, dt)


def has_v_reset() -> bool:
    import dataclasses

    return "v_reset" in {f.name for f in dataclasses.fields(nir.LIF)}


def lif(n, tau, thr=1.0, r=1.0, v_leak=0.0, v_reset=0.0):
    kw = dict(tau=np.full(n, tau, F), r=np.full(n, r, F), v_leak=np.full(n, v_leak, F), v_threshold=np.full(n, thr, F))
    if has_v_reset():
        kw["v_reset"] = np.full(n, v_reset, F)
    return nir.LIF(**kw)


def chain(nodes: Dict[str, nir.NIRNode], n_in, n_out) -> nir.NIRGraph:
    keys = ["input", *nodes, "output"]
    nodes = {"input": nir.Input(np.array(n_in)), **nodes, "output": nir.Output(np.array(n_out))}
    return nir.NIRGraph(nodes=nodes, edges=list(zip(keys[:-1], keys[1:])))


def invariant(fw, g, x, dt, mutate) -> bool:
    g2 = copy.deepcopy(g)
    mutate(g2)
    y1, y2 = run(fw, g, x, dt), run(fw, g2, x, dt)
    return bool(np.array_equal(y1, y2))


def expect_raises(fn) -> str:
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    raise AssertionError("expected an exception, none raised")


def assert_(cond, msg=""):
    if not cond:
        raise AssertionError(msg)


# =========================================================================== NIR core


@check("NIR-1", "nir", "Conv1d/Conv2d input_type ignores `groups`: input channels are taken as weight.shape[1] (= C_in/groups), so a valid grouped conv fails graph type inference")
def _():
    w = np.zeros((4, 2, 3, 3), F)  # groups=2 => C_in should be 4
    node = nir.Conv2d(input_shape=(7, 7), weight=w, stride=1, padding=1, dilation=1, groups=2, bias=np.zeros(4, F))
    got = int(node.input_type["input"][0])
    assert_(got == 2, f"input channels {got}")
    err = expect_raises(lambda: chain({"c": node}, [4, 7, 7], [4, 7, 7]))
    return f"Conv2d(weight (4,2,3,3), groups=2).input_type = {node.input_type['input'].tolist()} (should start with 4); building Input([4,7,7])->conv raises {err[:90]}"


@check("NIR-2", "nir", "nir >= 1.0.6 always serialises `v_reset` for LIF/IF/CubaLIF, even when it was not given; nir <= 1.0.4 rejects the key, so older readers cannot load any spiking-neuron file from current NIR")
def _():
    import tempfile

    import h5py

    assert_(has_v_reset(), "run in the nir 1.0.8 environment")
    g = chain({"lif": nir.LIF(tau=np.ones(2, F), r=np.ones(2, F), v_leak=np.zeros(2, F), v_threshold=np.ones(2, F))}, [2], [2])
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "g.nir"
        nir.write(p, g)
        with h5py.File(p) as f:
            keys = sorted(f["node/nodes/lif"].keys())
    assert_("v_reset" in keys)
    return f"LIF built without v_reset is written with keys {keys}"


@check("NIR-3", "nir", "A .nir file stores no time step; every importer picks its own dt (see capability table)")
def _():
    import h5py

    with h5py.File(FIX / "dt1e-4" / "lif" / "graph.nir") as f:
        keys = []
        f.visit(keys.append)
    assert_(not any(k.split("/")[-1] in ("dt", "timestep", "time_step") for k in keys))
    return f"datasets in a LIF graph file: {[k for k in keys if '/' in k and 'nodes/lif' in k]}"


# =========================================================================== nirtorch legacy loader


@check("NT-1", "nirtorch", "nirtorch.load (legacy GraphExecutor, used by Norse, Sinabs, Rockpool): execution order is not topological for fan-in graphs, and the executor returns the last-executed node, so a fan-in graph returns one branch's Linear output instead of the neuron's spikes")
def _():
    from norse.torch import from_nir

    g, x, ref, dt = load("dt1", "fan_in")
    net = from_nir(g, dt=dt)
    order = [n.name for n in net.execution_order]
    assert_(order.index("fc_b") > order.index("lif"), str(order))
    y = run("norse", g, x, dt)
    fcb = x.astype(np.float64) @ g.nodes["fc_b"].weight.T
    assert_(np.allclose(y, fcb, atol=1e-5))
    return f"execution order {order}; model output == fc_b(x) (a Linear output, not spikes)"


@check("NT-2", "nirtorch", "nirtorch.load silently skips nodes the importer cannot map (returns None): no error, the node is absent from the execution order and the output is wrong")
def _():
    from norse.torch import from_nir

    g, x, ref, dt = load("dt1", "li")
    net = from_nir(g, dt=dt)
    order = [n.name for n in net.execution_order]
    assert_("li" not in order, str(order))
    return f"Linear->LI graph imports without error; execution order {order}"


@check("NT-3", "nirtorch", "With an unsupported node in the graph, nirtorch.load's result is not deterministic: repeated imports of the same file sometimes raise and sometimes run (with a wrong output), even with PYTHONHASHSEED fixed; graphs of supported nodes are stable", envs=("latest", "pinned"))
def _():
    import collections
    import hashlib

    cases = [("norse", "dt1", "scale"), ("norse", "dt1", "delay"), ("norse", "dt1", "two_layer")]
    if importlib.util.find_spec("sinabs") is not None:
        cases.append(("sinabs", "dt1e-4", "flatten"))
    out, counts = [], {}
    for fw, suite, name in cases:
        c = collections.Counter()
        for _ in range(30):
            g, x, ref, dt = load(suite, name, legacy=not has_v_reset() or fw == "sinabs")
            try:
                y = run(fw, g, x, dt)
                c["ok:" + hashlib.sha1(np.ascontiguousarray(y).tobytes()).hexdigest()[:6]] += 1
            except Exception as e:  # noqa: BLE001
                c["error:" + type(e).__name__] += 1
        counts[name] = len(c)
        out.append(f"{fw}/{suite}/{name}: {dict(c)}")
    assert_(counts["two_layer"] == 1, "control graph must be stable")
    assert_(any(v > 1 for k, v in counts.items() if k != "two_layer"), out)
    return "30 imports each, one process: " + "; ".join(out)


# =========================================================================== snnTorch


def _snn_modules(net, cls):
    return [m for m in net.modules() if m.__class__.__name__ == cls]


@check("ST-1", "snntorch", "LIF/CubaLIF import hardcodes dt = 1e-4 s, so time constants in any other unit give wrong dynamics (no dt argument)")
def _():
    import inspect

    from snntorch.import_nir import import_from_nir

    assert_("dt" not in inspect.signature(import_from_nir).parameters)
    g, x, ref, dt = load("dt1", "lif")  # tau = 20 (steps), dt = 1
    m = _snn_modules(import_from_nir(copy.deepcopy(g)), "Leaky")[0]
    beta = float(np.asarray(m.beta.detach() if hasattr(m.beta, "detach") else m.beta).ravel()[0])
    assert_(abs(beta - (1 - 1e-4 / 20)) < 1e-6, beta)
    return f"tau=20 with dt=1 should give beta=0.95; imported Leaky.beta={beta:.6f} (= 1 - 1e-4/20)"


@check("ST-2", "snntorch", "nir.IF is imported as a *leaky* neuron: snn.Leaky(beta=0.9, reset_mechanism='subtract', init_hidden=False); its output is a (spk, mem) tuple, which breaks any layer after it")
def _():
    from snntorch.import_nir import import_from_nir

    g, x, ref, dt = load("dt1", "if")
    m = _snn_modules(import_from_nir(copy.deepcopy(g)), "Leaky")[0]
    beta, reset, ih = float(m.beta), m.reset_mechanism, m.init_hidden
    assert_(abs(beta - 0.9) < 1e-6 and reset == "subtract" and ih is False)
    g2, x2, _, dt2 = load("dt1", "sumpool")
    err = expect_raises(lambda: run("snntorch", g2, x2, dt2))
    return f"IF -> Leaky(beta={beta:.2f}, reset={reset}, init_hidden={ih}); Conv->IF->SumPool2d raises {err[:70]}"


@check("ST-3", "snntorch", "v_reset is ignored: LIF is always imported with reset_mechanism='zero'")
def _():
    from snntorch.import_nir import import_from_nir

    g, x, ref, dt = load("dt1e-4", "lif_v_reset")
    m = _snn_modules(import_from_nir(copy.deepcopy(g)), "Leaky")[0]
    assert_(m.reset_mechanism == "zero")
    same = invariant("snntorch", g, x, dt, lambda gg: setattr(gg.nodes["lif"], "v_reset", np.zeros_like(gg.nodes["lif"].v_reset)))
    assert_(same)
    return "v_reset=-0.5*thr imported as reset_mechanism='zero'; output bit-identical with v_reset=0"


@check("ST-4", "snntorch", "nir.Flatten start/end dims are passed to torch.nn.Flatten unchanged, but NIR dims exclude the batch dimension: start_dim=0 flattens the batch")
def _():
    from snntorch.import_nir import import_from_nir

    g, x, ref, dt = load("dt1", "flatten")
    flat = [m for m in import_from_nir(copy.deepcopy(g)).modules() if m.__class__.__name__ == "Flatten"][0]
    assert_(flat.start_dim == 0)
    err = expect_raises(lambda: run("snntorch", g, x, dt))
    return f"torch Flatten(start_dim={flat.start_dim}); running raises {err[:60]}"


@check("ST-5", "snntorch", "Recurrent LIF graphs (LIF <-> Linear) fail to import", envs=("latest", "pinned"))
def _():
    from snntorch.import_nir import import_from_nir

    g, x, ref, dt = load("dt1e-4", "recurrent_lif", legacy=not has_v_reset())
    return expect_raises(lambda: import_from_nir(g))


@check("ST-6", "snntorch", "Import rejects per-neuron thresholds, v_leak != 0, and CubaLIF with r != tau_mem/dt")
def _():
    from snntorch.import_nir import import_from_nir

    out = []
    for name in ["lif_hetero", "lif_v_leak", "cubalif"]:
        g, *_ = load("dt1e-4", name, legacy=not has_v_reset())
        out.append(f"{name}: {expect_raises(lambda: import_from_nir(g))[:60]}")
    return " | ".join(out)


# =========================================================================== Norse


def _ref_dropped(g, key, x, dt):
    g2 = copy.deepcopy(g)
    n = g2.nodes[key]
    g2.nodes[key] = nir.Scale(np.ones(tuple(int(v) for v in n.input_type["input"]), F))
    return simulate(g2, x, dt)


@check("NO-1", "norse", "LI, I, CubaLI and AvgPool2d have no importer and are silently dropped: no error, output equals the graph without that node")
def _():
    out = []
    for name, key in [("li", "li"), ("i", "i"), ("cubali", "li")]:
        if name == "cubali" and not hasattr(nir, "CubaLI"):
            continue
        g, x, ref, dt = load("dt1", name)
        y = run("norse", g, x, dt)
        assert_(np.allclose(y, _ref_dropped(g, key, x, dt), atol=1e-5), name)
        out.append(name)
    g, x, ref, dt = load("dt1", "avgpool", legacy=not has_v_reset())
    y = run("norse", g, x, dt)
    assert_(list(y.shape[2:]) == [2, 6, 6])
    out.append(f"avgpool (output shape {list(y.shape[2:])}, pool missing)")
    return "dropped without error: " + ", ".join(out)


@check("NO-2", "norse", "LIF `r` is ignored (only a logged warning)")
def _():
    g, x, ref, dt = load("dt1", "lif_r2", legacy=not has_v_reset())
    assert_(invariant("norse", g, x, dt, lambda gg: setattr(gg.nodes["lif"], "r", np.ones_like(gg.nodes["lif"].r))))
    return "lif_r2 output bit-identical with r=2 and r=1"


@check("NO-3", "norse", "`v_reset` is ignored: neurons reset to 0", envs=("latest",))
def _():
    assert_(has_v_reset(), "needs nir >= 1.0.6")
    g, x, ref, dt = load("dt1", "lif_v_reset")
    assert_(invariant("norse", g, x, dt, lambda gg: setattr(gg.nodes["lif"], "v_reset", np.zeros_like(gg.nodes["lif"].v_reset))))
    return "lif_v_reset output bit-identical with v_reset=-0.5*thr and v_reset=0"


@check("NO-4", "norse", "IF integration ignores dt (v += x per step) while LIF uses dt: the same IF graph gives identical output at dt=1 and dt=1e-4")
def _():
    from norse.torch import from_nir

    g, x, ref, dt = load("dt1", "if", legacy=not has_v_reset())
    y1, y2 = run("norse", g, x, 1.0), run("norse", g, x, 1e-4)
    gl, xl, _, _ = load("dt1", "lif", legacy=not has_v_reset())
    l1, l2 = run("norse", gl, xl, 1.0), run("norse", gl, xl, 1e-4)
    assert_(np.array_equal(y1, y2) and not np.array_equal(l1, l2))
    return f"IF spikes at dt=1 vs dt=1e-4 identical ({int(y1.sum())}); LIF spikes differ ({int(l1.sum())} vs {int(l2.sum())})"


@check("NO-5", "norse", "Membranes start at v_leak, not 0 (NIR does not specify the initial state)")
def _():
    g, x, ref, dt = load("dt1", "lif_v_leak", legacy=not has_v_reset())
    y = run("norse", g, x, dt)
    assert_(np.array_equal(y, simulate(g, x, dt, Convention(init_v="v_leak"))))
    assert_(not np.array_equal(y, ref))
    return "lif_v_leak output equals the reference started at v=v_leak, exactly"


@check("NO-6", "norse", "CubaLIF: the synaptic current is given the membrane's v_leak, so with v_leak != 0 a neuron with no input spikes")
def _():
    n = 4
    kw = dict(tau_syn=np.full(n, 10.0, F), tau_mem=np.full(n, 20.0, F), r=np.ones(n, F), v_leak=np.full(n, 0.6, F), v_threshold=np.ones(n, F), w_in=np.ones(n, F))
    g = chain({"lif": nir.CubaLIF(**kw)}, [n], [n])
    x = np.zeros((200, 1, n), F)
    y = run("norse", g, x, 1.0)
    ref = simulate(g, x, 1.0)
    assert_(ref.sum() == 0 and y.sum() > 0, (ref.sum(), y.sum()))
    return f"zero input, v_leak=0.6, threshold=1: reference 0 spikes (v settles at 0.6); Norse {int(y.sum())} spikes"


@check("NO-7", "norse", "nir.Flatten dims are used as torch dims (batch included), so Flatten(start_dim=0) -> Linear crashes")
def _():
    g, x, ref, dt = load("dt1", "flatten", legacy=not has_v_reset())
    return expect_raises(lambda: run("norse", g, x, dt))


@check("NO-8", "norse", "nir.Conv1d is not supported (crashes at run time)")
def _():
    g, x, ref, dt = load("dt1", "conv1d", legacy=not has_v_reset())
    return expect_raises(lambda: run("norse", g, x, dt))


# =========================================================================== Sinabs


@check("SB-1", "sinabs", "sinabs pins nir<=1.0.4, so it cannot read LIF/IF/CubaLIF files written by current NIR")
def _():
    import importlib.metadata as md

    req = [r for r in md.requires("sinabs") or [] if r.startswith("nir")]
    err = expect_raises(lambda: nir.read(FIX / "dt1" / "lif" / "graph.nir"))
    return f"requires {req}; nir {nir.__version__} reading a nir-1.0.8 LIF file: {err[:80]}"


@check("SB-2", "sinabs", "nir.Linear (bias-free weights, which snnTorch/Norse/Spyx export) has no importer")
def _():
    g, x, ref, dt = load("dt1", "linear")
    return expect_raises(lambda: run("sinabs", g, x, dt))


@check("SB-3", "sinabs", "nir.Conv1d import crashes: it calls .detach() on the numpy weight array")
def _():
    g, x, ref, dt = load("dt1", "conv1d", legacy=True)
    err = expect_raises(lambda: run("sinabs", g, x, dt))
    assert_("detach" in err)
    return err


@check("SB-4", "sinabs", "IF is imported with multi-spike output and reset-by-subtraction (NIR: at most one spike per step, reset to v_reset)")
def _():
    g, x, ref, dt = load("dt1", "conv2d", legacy=True)
    y = run("sinabs", g, x, dt)
    assert_(np.array_equal(y, simulate(g, x, dt, Convention(reset="subtract", multispike=True))))
    return f"conv2d->IF output equals reference with reset=subtract, multispike=True; max spikes/step = {int(y.max())}"


@check("SB-5", "sinabs", "Recurrent edges are silently ignored (output unchanged when the recurrent weights are zeroed)")
def _():
    g, x, ref, dt = load("dt1", "recurrent_lif", legacy=True)
    assert_(invariant("sinabs", g, x, dt, lambda gg: setattr(gg.nodes["rec"], "weight", gg.nodes["rec"].weight * 0)))
    return "recurrent_lif output bit-identical with recurrent weights zeroed"


@check("SB-6", "sinabs", "LIF `r` is ignored without warning (LI checks r == 1, LIF does not)")
def _():
    rng = np.random.default_rng(0)
    g = chain({"fc": nir.Affine(rng.normal(0, 3, (4, 3)).astype(F), np.zeros(4, F)), "lif": lif(4, 20.0, r=2.0)}, [3], [4])
    x = (rng.random((50, 2, 3)) < 0.3).astype(F)
    assert_(invariant("sinabs", g, x, 1.0, lambda gg: setattr(gg.nodes["lif"], "r", np.ones_like(gg.nodes["lif"].r))))
    return "Affine->LIF output bit-identical with r=2 and r=1"


# =========================================================================== Rockpool


@check("RP-1", "rockpool", "nir.LIF import always crashes: the importer reads node.tau_mem, but nir.LIF's field is `tau`")
def _():
    g, x, ref, dt = load("dt1", "lif", legacy=True)
    from rockpool.nn.modules.torch.nir import from_nir

    err = expect_raises(lambda: from_nir(g))
    assert_("tau_mem" in err)
    return err


@check("RP-2", "rockpool", "nir.LI time constants are truncated to integers (tau.int()): tau = 2e-3 s becomes 0")
def _():
    from rockpool.nn.modules.torch.nir import from_nir

    g, x, ref, dt = load("dt1e-4", "li")
    net = from_nir(g)
    taus = [float(np.asarray(m.tau.detach()).ravel()[0]) for m in net.modules() if m.__class__.__name__.startswith("ExpSyn")]
    assert_(taus and taus[0] == 0.0, taus)
    return f"file tau = 0.002; imported ExpSynTorch.tau = {taus[0]}"


@check("RP-3", "rockpool", "The simulation dt is inferred as min(tau / (1 + r)) instead of being given; with r = 1 it is tau/2")
def _():
    from rockpool.nn.modules.torch.nir import from_nir

    g, x, ref, dt = load("dt1", "cubalif", legacy=True)
    net = from_nir(g)
    dts = [float(m.dt) for m in net.modules() if hasattr(m, "dt") and m.__class__.__name__.startswith("LIF")]
    assert_(dts and abs(dts[0] - 10.0) < 1e-6, dts)
    return f"file intends dt = 1 (tau_mem = 20 steps, r = 1); imported LIFTorch.dt = {dts[0]}"


@check("RP-4", "rockpool", "No importer for IF, I, Conv1d/Conv2d, Flatten, Delay or Threshold")
def _():
    from rockpool.nn.modules.torch.nir import from_nir

    out = []
    for name in ["if", "i", "conv2d", "conv1d", "flatten", "delay", "threshold"]:
        g, *_ = load("dt1", name, legacy=True)
        out.append(expect_raises(lambda: from_nir(g)).split("<class '")[-1].split("'>")[0])
    return "NotImplementedError for: " + ", ".join(out)


# =========================================================================== Spyx


@check("SX-1", "spyx", "Spikes are reported one step late: a neuron's step-t output is computed from its membrane *before* the step-t update")
def _():
    g = chain({"lif": lif(1, 20.0, thr=1.0)}, [1], [1])
    x = np.zeros((5, 1, 1), F)
    x[0] = 1000.0  # one huge input at t=0
    y = run("spyx", g, x, 1.0)[:, 0, 0]
    r = simulate(g, x, 1.0)[:, 0, 0]
    assert_(r[0] == 1 and y[0] == 0 and y[1] > 0, (r, y))
    return f"input at t=0 only: reference spikes {r.astype(int).tolist()}, Spyx {y.astype(int).tolist()}"


@check("SX-2", "spyx", "LIF `r`, `v_leak` and `v_reset` are all ignored")
def _():
    out = []
    for name, field in [("lif_r2", "r"), ("lif_v_leak", "v_leak"), ("lif_v_reset", "v_reset")]:
        g, x, ref, dt = load("dt1", name)
        val = 1.0 if field == "r" else 0.0
        assert_(invariant("spyx", g, x, dt, lambda gg: setattr(gg.nodes["lif"], field, np.full_like(getattr(gg.nodes["lif"], field), val))), field)
        out.append(field)
    return "outputs bit-identical when each is reset to its default: " + ", ".join(out)


@check("SX-3", "spyx", "Only the first branch of a non-chain graph is imported: other branches are dropped without error")
def _():
    g, x, ref, dt = load("dt1", "fan_in")
    assert_(invariant("spyx", g, x, dt, lambda gg: setattr(gg.nodes["fc_b"], "weight", gg.nodes["fc_b"].weight * 0)))
    return "fan_in output bit-identical with the second branch's weights zeroed"


@check("SX-4", "spyx", "Scale, Delay, I, CubaLI and AvgPool2d are silently dropped")
def _():
    g, x, ref, dt = load("dt1", "scale")
    assert_(invariant("spyx", g, x, dt, lambda gg: setattr(gg.nodes["sc"], "scale", gg.nodes["sc"].scale * 3)))
    g, x, ref, dt = load("dt1", "delay")
    assert_(invariant("spyx", g, x, dt, lambda gg: setattr(gg.nodes["delay"], "delay", gg.nodes["delay"].delay * 0)))
    g, x, ref, dt = load("dt1", "i")
    assert_(np.allclose(run("spyx", g, x, dt), _ref_dropped(g, "i", x, dt), atol=1e-5))
    g, x, ref, dt = load("dt1", "avgpool")
    shape = list(run("spyx", g, x, dt).shape[2:])
    assert_(shape == [2, 6, 6])
    return f"Scale x3 and Delay -> 0 leave output bit-identical; I output == its input; AvgPool2d output shape {shape} (pool missing)"


@check("SX-5", "spyx", "Flatten after a conv flattens in Spyx's channels-last (H, W, C) order, but NIR's following weights expect channels-first (C, H, W) order")
def _():
    g, x, ref, dt = load("dt1", "flatten")
    y = run("spyx", g, x, dt)
    # same Linear->LIF, fed the input flattened in (H, W, C) order
    g2 = chain({"fc": g.nodes["fc"], "lif": g.nodes["lif"]}, [18], [8])
    x_hwc = np.transpose(x, (0, 1, 3, 4, 2)).reshape(x.shape[0], x.shape[1], 18)
    x_chw = x.reshape(x.shape[0], x.shape[1], 18)
    y_hwc, y_chw = run("spyx", g2, x_hwc, dt), run("spyx", g2, x_chw, dt)
    assert_(np.array_equal(y, y_hwc) and not np.array_equal(y, y_chw))
    return "output equals Spyx(Linear->LIF) on the input flattened as (H,W,C), not as NIR's (C,H,W)"


@check("SX-6", "spyx", "nir.Conv1d import fails (shape error at run time)")
def _():
    g, x, ref, dt = load("dt1", "conv1d")
    return expect_raises(lambda: run("spyx", g, x, dt))


# =========================================================================== runner


def main(results: Path) -> None:
    env = f"nir{nir.__version__}"
    kind = "latest" if has_v_reset() else "pinned"
    out = []
    for c in CHECKS:
        if kind not in c["envs"]:
            continue
        fw = c["framework"]
        mod = {"nir": "nir", "nirtorch": "norse"}.get(fw, fw)
        if importlib.util.find_spec(mod) is None:
            continue
        rec = {k: c[k] for k in ("id", "framework", "claim")}
        rec["env"] = env
        rec["framework_version"] = adapters.framework_version(fw) if fw not in ("nir",) else nir.__version__
        try:
            rec["evidence"] = c["fn"]()
            rec["confirmed"] = True
        except Exception as e:  # noqa: BLE001
            rec["confirmed"] = False
            rec["evidence"] = f"{type(e).__name__}: {str(e)[:200]}"
            rec["trace"] = traceback.format_exc(limit=-2)
        out.append(rec)
        print(f"{rec['id']:6s} {'CONFIRMED' if rec['confirmed'] else 'NOT CONFIRMED':13s} {rec['evidence'][:150]}")
    d = results / "verified"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{env}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results"))
