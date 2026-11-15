"""Fixture definitions and generator.

Each fixture is a small NIR graph plus a fixed input spike train, a time step
``dt``, and the reference output produced by ``reference.simulate`` under the
reference convention. Every fixture exists in two *suites* with identical
dimensionless dynamics (dt/tau = 0.05):

* ``dt1``    : dt = 1.0,  tau = 20      (time measured in steps)
* ``dt1e-4`` : dt = 1e-4, tau = 2e-3    (time measured in seconds)

Usage::

    python -m nirconf.fixtures fixtures/
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import nir
import numpy as np

from .reference import membrane_margin, simulate

SUITES = {"dt1": 1.0, "dt1e-4": 1e-4}
T, B = 100, 2
MIN_MARGIN = 1e-4
F32 = np.float32


@dataclass
class Spec:
    name: str
    description: str
    build: Callable[[float, float, np.random.Generator], Tuple[nir.NIRGraph, Tuple[int, ...]]]
    targets: List[str] = field(default_factory=list)  # issue / question it probes
    spiking_output: bool = True
    p_in: float = 0.2


def _f(x) -> np.ndarray:
    return np.asarray(x, dtype=F32)


def _full(n, v) -> np.ndarray:
    return np.full(n, v, dtype=F32)


def _g(nodes: Dict[str, nir.NIRNode], edges, in_shape, out_shape) -> nir.NIRGraph:
    nodes = {"input": nir.Input(np.array(in_shape)), **nodes, "output": nir.Output(np.array(out_shape))}
    return nir.NIRGraph(nodes=nodes, edges=edges)


def _chain(nodes: Dict[str, nir.NIRNode], in_shape, out_shape) -> nir.NIRGraph:
    keys = ["input", *nodes.keys(), "output"]
    return _g(nodes, list(zip(keys[:-1], keys[1:])), in_shape, out_shape)


NI, N = 6, 8


def _w(rng, n_out, n_in, s, positive=False):
    w = rng.normal(0.0, 1.0, (n_out, n_in))
    if positive:
        w = np.abs(w)
    return _f(s * w)


# --- builders: (dt, scale, rng) -> (graph, input_shape) ---------------------------


def lif_node(dt, n=N, thr=1.0, r=1.0, v_leak=0.0, v_reset=0.0, hetero=False, rng=None):
    tau = 20 * dt
    if hetero:
        taus = _f(tau * rng.uniform(0.5, 2.0, n))
        thrs = _f(thr * rng.uniform(0.5, 1.5, n))
    else:
        taus, thrs = _full(n, tau), _full(n, thr)
    return nir.LIF(tau=taus, r=_full(n, r), v_leak=_full(n, v_leak * thr), v_threshold=thrs, v_reset=_full(n, v_reset * thr))


def cubalif_node(dt, n=N, thr=1.0, r=1.0, v_leak=0.0, w_in=1.0):
    return nir.CubaLIF(
        tau_syn=_full(n, 10 * dt), tau_mem=_full(n, 20 * dt), r=_full(n, r), v_leak=_full(n, v_leak * thr),
        v_threshold=_full(n, thr), v_reset=_full(n, 0.0), w_in=_full(n, w_in),
    )


def b_affine(dt, s, rng):
    return _chain({"fc": nir.Affine(_w(rng, 5, 3, s), _f(rng.normal(0, 0.1, 5)))}, [3], [5]), (3,)


def b_linear(dt, s, rng):
    return _chain({"fc": nir.Linear(_w(rng, 5, 3, s))}, [3], [5]), (3,)


def b_scale(dt, s, rng):
    return _chain({"fc": nir.Linear(_w(rng, N, NI, s, True)), "sc": nir.Scale(_f(rng.uniform(0.5, 2.0, N))), "lif": lif_node(dt)}, [NI], [N]), (NI,)


def b_li(dt, s, rng):
    return _chain({"fc": nir.Linear(_w(rng, N, NI, s)), "li": nir.LI(tau=_full(N, 20 * dt), r=_full(N, 1.0), v_leak=_full(N, 0.0))}, [NI], [N]), (NI,)


def b_i(dt, s, rng):
    return _chain({"fc": nir.Linear(_w(rng, N, NI, s)), "i": nir.I(r=_full(N, 1.0))}, [NI], [N]), (NI,)




def b_lif_mixed(dt, s, rng):
    return _chain({"fc": nir.Linear(_w(rng, N, NI, s)), "lif": lif_node(dt)}, [NI], [N]), (NI,)


def b_if(dt, s, rng, mixed=False, v_reset=0.0):
    thr = 1.0 * dt  # Euler step of dv/dt = R I is dt*R*x: threshold scales with dt
    node = nir.IF(r=_full(N, 1.0), v_threshold=_full(N, thr), v_reset=_full(N, v_reset * thr))
    return _chain({"fc": nir.Linear(_w(rng, N, NI, s, positive=not mixed)), "if": node}, [NI], [N]), (NI,)


def _cuba_fixture(**kw):
    def b(dt, s, rng):
        r = kw.get("r", 1.0)
        kw2 = dict(kw)
        if r == "tau_over_dt":
            kw2["r"] = 20 * dt / dt
        return _chain({"fc": nir.Linear(_w(rng, N, NI, s, True)), "lif": cubalif_node(dt, **kw2)}, [NI], [N]), (NI,)
    return b


def b_cubali(dt, s, rng):
    node = nir.CubaLI(tau_syn=_full(N, 10 * dt), tau_mem=_full(N, 20 * dt), r=_full(N, 1.0), v_leak=_full(N, 0.0), w_in=_full(N, 1.0))
    return _chain({"fc": nir.Linear(_w(rng, N, NI, s)), "li": node}, [NI], [N]), (NI,)


def b_two_layer(dt, s, rng):
    return _chain({
        "fc1": nir.Linear(_w(rng, N, NI, s, True)), "lif1": lif_node(dt),
        "fc2": nir.Linear(_w(rng, 4, N, s * 1.5, True)), "lif2": lif_node(dt, n=4),
    }, [NI], [4]), (NI,)


def b_two_layer_affine(dt, s, rng):
    return _chain({
        "fc1": nir.Affine(_w(rng, N, NI, s, True), _f(rng.uniform(0, 0.02, N))), "lif1": lif_node(dt),
        "fc2": nir.Affine(_w(rng, 4, N, s * 1.5, True), _f(rng.uniform(0, 0.02, 4))), "lif2": lif_node(dt, n=4),
    }, [NI], [4]), (NI,)


def b_recurrent(dt, s, rng, cuba=False):
    neuron = cubalif_node(dt) if cuba else lif_node(dt)
    wrec = _f(rng.normal(0, 0.5 * s, (N, N)))
    nodes = {"fc": nir.Affine(_w(rng, N, NI, s, True), _full(N, 0.0)), "lif": neuron, "rec": nir.Linear(wrec)}
    edges = [("input", "fc"), ("fc", "lif"), ("lif", "rec"), ("rec", "lif"), ("lif", "output")]
    return _g(nodes, edges, [NI], [N]), (NI,)


def b_fan_in(dt, s, rng):
    nodes = {"fc_a": nir.Linear(_w(rng, N, NI, s, True)), "fc_b": nir.Linear(_w(rng, N, NI, s, True)), "lif": lif_node(dt)}
    edges = [("input", "fc_a"), ("input", "fc_b"), ("fc_a", "lif"), ("fc_b", "lif"), ("lif", "output")]
    return _g(nodes, edges, [NI], [N]), (NI,)




def b_conv2d(dt, s, rng):
    w = _f(s * np.abs(rng.normal(0, 1, (2, 1, 3, 3))))
    conv = nir.Conv2d(input_shape=(6, 6), weight=w, stride=1, padding=1, dilation=1, groups=1, bias=_full(2, 0.0))
    ifn = nir.IF(r=np.ones((2, 6, 6), F32), v_threshold=np.full((2, 6, 6), dt, F32), v_reset=np.zeros((2, 6, 6), F32))
    return _chain({"conv": conv, "if": ifn}, [1, 6, 6], [2, 6, 6]), (1, 6, 6)


def b_conv2d_stride(dt, s, rng):
    w = _f(s * np.abs(rng.normal(0, 1, (3, 2, 3, 3))))
    conv = nir.Conv2d(input_shape=(7, 7), weight=w, stride=2, padding=0, dilation=1, groups=1, bias=_full(3, 0.0))
    ifn = nir.IF(r=np.ones((3, 3, 3), F32), v_threshold=np.full((3, 3, 3), dt, F32), v_reset=np.zeros((3, 3, 3), F32))
    return _chain({"conv": conv, "if": ifn}, [2, 7, 7], [3, 3, 3]), (2, 7, 7)


def b_conv1d(dt, s, rng):
    w = _f(s * np.abs(rng.normal(0, 1, (3, 2, 3))))
    conv = nir.Conv1d(input_shape=10, weight=w, stride=1, padding=1, dilation=1, groups=1, bias=_full(3, 0.0))
    ifn = nir.IF(r=np.ones((3, 10), F32), v_threshold=np.full((3, 10), dt, F32), v_reset=np.zeros((3, 10), F32))
    return _chain({"conv": conv, "if": ifn}, [2, 10], [3, 10]), (2, 10)


def _conv_if(dt, s, rng, cout=2, hw=6):
    w = _f(s * np.abs(rng.normal(0, 1, (cout, 1, 3, 3))))
    conv = nir.Conv2d(input_shape=(hw, hw), weight=w, stride=1, padding=1, dilation=1, groups=1, bias=_full(cout, 0.0))
    ifn = nir.IF(r=np.ones((cout, hw, hw), F32), v_threshold=np.full((cout, hw, hw), dt, F32), v_reset=np.zeros((cout, hw, hw), F32))
    return conv, ifn


def b_sumpool(dt, s, rng):
    conv, ifn = _conv_if(dt, s, rng)
    pool = nir.SumPool2d(kernel_size=np.array([2, 2]), stride=np.array([2, 2]), padding=np.array([0, 0]))
    return _chain({"conv": conv, "if": ifn, "pool": pool}, [1, 6, 6], [2, 3, 3]), (1, 6, 6)


def b_avgpool(dt, s, rng):
    conv, ifn = _conv_if(dt, s, rng)
    pool = nir.AvgPool2d(kernel_size=np.array([2, 2]), stride=np.array([2, 2]), padding=np.array([0, 0]))
    return _chain({"conv": conv, "if": ifn, "pool": pool}, [1, 6, 6], [2, 3, 3]), (1, 6, 6)


def b_flatten(dt, s, rng):
    flat = nir.Flatten(input_type={"input": np.array([2, 3, 3])}, start_dim=0, end_dim=-1)
    return _chain({"flat": flat, "fc": nir.Linear(_w(rng, N, 18, s, True)), "lif": lif_node(dt)}, [2, 3, 3], [N]), (2, 3, 3)


def b_conv_net(dt, s, rng):
    conv, ifn = _conv_if(dt, s, rng)
    pool = nir.SumPool2d(kernel_size=np.array([2, 2]), stride=np.array([2, 2]), padding=np.array([0, 0]))
    flat = nir.Flatten(input_type={"input": np.array([2, 3, 3])}, start_dim=0, end_dim=-1)
    fc = nir.Affine(_w(rng, 4, 18, s * 4, True), _full(4, 0.0))
    return _chain({"conv": conv, "if": ifn, "pool": pool, "flat": flat, "fc": fc, "lif": lif_node(dt, n=4)}, [1, 6, 6], [4]), (1, 6, 6)


def b_threshold(dt, s, rng):
    li = nir.LI(tau=_full(N, 20 * dt), r=_full(N, 1.0), v_leak=_full(N, 0.0))
    return _chain({"fc": nir.Linear(_w(rng, N, NI, s, True)), "li": li, "thr": nir.Threshold(_full(N, 1.0))}, [NI], [N]), (NI,)


def b_delay(dt, s, rng):
    d = _f(dt * rng.integers(1, 5, N))
    return _chain({"fc": nir.Linear(_w(rng, N, NI, s, True)), "delay": nir.Delay(d), "lif": lif_node(dt)}, [NI], [N]), (NI,)


def _pos(builder_kw):
    def b(dt, s, rng):
        node = lif_node(dt, rng=rng, **builder_kw)
        return _chain({"fc": nir.Linear(_w(rng, N, NI, s, True)), "lif": node}, [NI], [N]), (NI,)
    return b


SPECS: List[Spec] = [
    # stateless
    Spec("affine", "Affine 3->5 (non-square, with bias). Catches transposed weights.", b_affine, ["weight layout"], spiking_output=False),
    Spec("linear", "Linear 3->5 (non-square). Catches transposed weights / swapped in/out features.", b_linear, ["weight layout"], spiking_output=False),
    # single neuron types
    Spec("li", "Linear -> LI (r=1, v_leak=0). Analog membrane output.", b_li, ["LI support"], spiking_output=False),
    Spec("i", "Linear -> I (integrator). Analog output.", b_i, ["I support"], spiking_output=False),
    Spec("lif", "Linear -> LIF, canonical (r=1, v_leak=0, v_reset=0, uniform threshold).", _pos({}), ["LIF baseline"]),
    Spec("lif_mixed_sign", "LIF driven by mixed-sign weights (membrane goes negative).", b_lif_mixed, ["negative membrane"]),
    Spec("lif_hetero", "LIF with per-neuron tau and threshold.", _pos({"hetero": True}), ["per-neuron parameters (#176)"]),
    Spec("lif_r2", "LIF with r=2. A framework that ignores r fires at the wrong times.", _pos({"r": 2.0}), ["r handling"]),
    Spec("lif_v_leak", "LIF with v_leak = 0.4*threshold.", _pos({"v_leak": 0.4}), ["v_leak handling"]),
    Spec("lif_v_reset", "LIF with v_reset = -0.5*threshold.", _pos({"v_reset": -0.5}), ["v_reset handling"]),
    Spec("lif_overshoot", "LIF with strong input so v overshoots 2-3x threshold. Separates reset-to-value / subtract / multi-spike.", _pos({}), ["reset semantics", "single vs multi spike"], p_in=0.5),
    Spec("if", "Linear -> IF, positive weights.", lambda dt, s, rng: b_if(dt, s, rng), ["IF baseline", "IF input gain (dt*R)"]),
    Spec("if_mixed_sign", "IF with mixed-sign weights (membrane goes negative; no lower bound in NIR).", lambda dt, s, rng: b_if(dt, s, rng, mixed=True), ["IF lower bound"]),
    Spec("if_v_reset", "IF with v_reset = -0.5*threshold.", lambda dt, s, rng: b_if(dt, s, rng, v_reset=-0.5), ["v_reset handling"]),
    Spec("cubalif", "Linear -> CubaLIF (r=1, w_in=1, v_leak=0).", _cuba_fixture(), ["CubaLIF baseline"]),
    Spec("cubalif_r_tau_dt", "CubaLIF with r = tau_mem/dt (the only r snnTorch accepts).", _cuba_fixture(r="tau_over_dt"), ["r handling"]),
    Spec("cubalif_w_in", "CubaLIF with w_in = 2 (elementwise input weight, #190).", _cuba_fixture(w_in=2.0), ["w_in (#190)"]),
    Spec("cubalif_v_leak", "CubaLIF with v_leak = 0.4*threshold. The synaptic current must still decay to 0.", _cuba_fixture(v_leak=0.4), ["v_leak handling", "synapse leak"]),
    Spec("cubali", "Linear -> CubaLI. Analog output.", b_cubali, ["CubaLI support"], spiking_output=False),
    Spec("scale", "Linear -> Scale -> LIF.", b_scale, ["Scale support"]),
    Spec("threshold", "Linear -> LI -> Threshold (stateless spike).", b_threshold, ["Threshold support"]),
    Spec("delay", "Linear -> Delay (1-4 steps per neuron) -> LIF.", b_delay, ["Delay support"]),
    # structure
    Spec("two_layer", "Linear -> LIF -> Linear -> LIF.", b_two_layer, ["composition"]),
    Spec("two_layer_affine", "Affine -> LIF -> Affine -> LIF (with biases).", b_two_layer_affine, ["composition", "bias"]),
    Spec("recurrent_lif", "Affine -> LIF with a recurrent Linear (LIF -> rec -> LIF).", lambda dt, s, rng: b_recurrent(dt, s, rng), ["recurrence timing"]),
    Spec("recurrent_cubalif", "Affine -> CubaLIF with a recurrent Linear.", lambda dt, s, rng: b_recurrent(dt, s, rng, cuba=True), ["recurrence timing"]),
    Spec("fan_in", "Two Linear branches from the input summed into one LIF.", b_fan_in, ["fan-in / non-chain graphs"]),
    Spec("conv2d", "Conv2d (1->2, 3x3, pad 1) -> IF, 6x6 input.", b_conv2d, ["Conv2d", "channel layout"]),
    Spec("conv2d_stride", "Conv2d (2->3, 3x3, stride 2, no pad) -> IF, 7x7 input.", b_conv2d_stride, ["Conv2d stride/padding"]),
    Spec("conv1d", "Conv1d (2->3, k=3, pad 1) -> IF.", b_conv1d, ["Conv1d"]),
    Spec("sumpool", "Conv2d -> IF -> SumPool2d(2).", b_sumpool, ["SumPool2d"]),
    Spec("avgpool", "Conv2d -> IF -> AvgPool2d(2).", b_avgpool, ["AvgPool2d"], spiking_output=False),
    Spec("flatten", "Flatten(start_dim=0) over (2,3,3) -> Linear -> LIF. NIR dims exclude batch.", b_flatten, ["Flatten dims"]),
    Spec("conv_net", "Conv2d -> IF -> SumPool2d -> Flatten -> Affine -> LIF.", b_conv_net, ["composition"]),
]


def _scale_weights(graph: nir.NIRGraph, k: float) -> None:
    for n in graph.nodes.values():
        if isinstance(n, (nir.Affine, nir.Linear, nir.Conv1d, nir.Conv2d)):
            n.weight = _f(n.weight * k)


def _make(spec: Spec, suite: str, seed: int):
    """Build a fixture and tune its weight scale so the reference output is
    informative: spiking outputs fire on 10-35% of (step, neuron) entries, and
    every membrane stays >= MIN_MARGIN (relative) away from the threshold:
    float32 drift over 100 steps is ~1e-6, so 1e-4 leaves a 100x safety factor."""
    dt = SUITES[suite]
    best = None
    for attempt in range(40):
        rng = np.random.default_rng(seed * 1000 + attempt)
        graph, in_shape = spec.build(dt, 1.0, rng)
        x = (rng.random((T, B) + in_shape) < spec.p_in).astype(F32)
        for k in np.geomspace(0.02, 50, 60):
            g = copy.deepcopy(graph)
            _scale_weights(g, k)
            y = simulate(g, x, dt)
            if spec.spiking_output:
                rate = float(np.mean(y > 0))
                lo, hi = (0.10, 0.35) if spec.name != "lif_overshoot" else (0.3, 0.9)
                if not (lo <= rate <= hi):
                    continue
            else:
                if not np.all(np.isfinite(y)) or np.max(np.abs(y)) == 0:
                    continue
            margin = membrane_margin(g, x, dt)
            if margin >= MIN_MARGIN:
                if spec.name == "lif_overshoot":
                    # require real overshoot: a single step can push v past 2*threshold
                    _, tr = simulate(g, x, dt, record=True)
                    if np.max(tr["fc"]) * 0.05 < 2.0:  # dt/tau = 0.05, threshold = 1
                        continue
                best = (g, x, y, margin, float(k), seed * 1000 + attempt)
                break
        if best:
            break
    if best is None:
        raise RuntimeError(f"could not tune fixture {spec.name}/{suite}")
    return best


def write_legacy(src: Path, dst: Path) -> str:
    """Write a copy readable by nir <= 1.0.4 (which predates ``v_reset`` and
    ``CubaLI``). Only done when it does not change the graph's meaning: an
    all-zero ``v_reset`` is dropped (1.0.4 resets to 0). Returns a status
    string: 'identical', 'written', or why it is not expressible."""
    import shutil

    import h5py

    shutil.copy(src, dst)
    changed = False
    with h5py.File(dst, "r+") as f:
        nodes = f["node/nodes"]
        for key in list(nodes.keys()):
            grp = nodes[key]
            typ = grp["type"][()]
            typ = typ.decode() if isinstance(typ, bytes) else str(typ)
            if typ == "CubaLI":
                f.close()
                dst.unlink()
                return "not expressible: CubaLI does not exist in nir 1.0.4"
            if "v_reset" in grp:
                if np.any(grp["v_reset"][()] != 0):
                    f.close()
                    dst.unlink()
                    return "not expressible: non-zero v_reset does not exist in nir 1.0.4"
                del grp["v_reset"]
                changed = True
    if not changed:
        dst.unlink()
        return "identical"
    return "written"


def generate(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    index = []
    for suite, dt in SUITES.items():
        for i, spec in enumerate(SPECS):
            g, x, y, margin, k, seed = _make(spec, suite, seed=i + 1)
            d = out_dir / suite / spec.name
            d.mkdir(parents=True, exist_ok=True)
            nir.write(d / "graph.nir", g)
            legacy = write_legacy(d / "graph.nir", d / "graph_legacy.nir")
            np.save(d / "input.npy", x)
            np.save(d / "expected.npy", y.astype(np.float64))
            meta = {
                "name": spec.name, "suite": suite, "dt": dt, "T": T, "B": B,
                "description": spec.description, "targets": spec.targets,
                "spiking_output": spec.spiking_output, "input_shape": list(x.shape[2:]),
                "output_shape": list(y.shape[2:]), "seed": seed, "weight_scale": k,
                "min_threshold_margin": None if not np.isfinite(margin) else margin,
                "output_spike_rate": float(np.mean(y > 0)) if spec.spiking_output else None,
                "nir_version_written": nir.__version__ if hasattr(nir, "__version__") else "unknown",
                "convention": "reference (see docs/conventions.md)",
                "legacy_file": legacy,
            }
            (d / "meta.json").write_text(json.dumps(meta, indent=2))
            index.append({"suite": suite, "name": spec.name})
            print(f"{suite:7s} {spec.name:20s} k={k:8.3f} margin={margin:.3f} rate={meta['output_spike_rate']}")
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    generate(Path(sys.argv[1] if len(sys.argv) > 1 else "fixtures"))
