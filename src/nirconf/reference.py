"""A slow, readable float64 reference simulator for flat NIR graphs.

NIR defines neurons as continuous-time ODEs and does not say how to discretise
them. This simulator fixes one documented convention (the *reference
convention*, see ``Convention()`` defaults and docs/conventions.md) and lets
each choice be switched, so that a framework's output can be explained as
"the reference, except for X" (see ``fingerprint.py``).

Reference convention (all defaults):

* forward Euler with step ``dt``: ``v <- v + dt/tau * (v_leak - v + r * x)``
* IF / I integrate ``v <- v + dt * r * x`` (the literal Euler step of ``dv/dt = R I``)
* CubaLIF/CubaLI: the synaptic current is updated first and the membrane uses
  the *updated* current
* spike when ``v > v_threshold`` (strict), at most one spike per step
* reset to ``v_reset`` in the same step the spike is emitted
* all states start at 0
* edges in a directed acyclic graph propagate within the same time step; an edge
  that closes a cycle (a recurrent edge) carries the value from the previous step
* several edges into one node are summed

Shapes: NIR shapes exclude the batch dimension. Arrays here are ``(B, ...)``
per step and ``(T, B, ...)`` per sequence.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import nir
import numpy as np


@dataclass(frozen=True)
class Convention:
    """Discretisation choices. Defaults = the reference convention."""

    decay: str = "euler"  # "euler": 1 - dt/tau ; "exact": exp(-dt/tau)
    input_gain: str = "euler"  # "euler": dt*r/tau (LIF) and dt*r (IF) ; "unit": r ; "unit_no_r": 1
    reset: str = "value"  # "value": v=v_reset ; "zero": v=0 ; "subtract": v-=v_threshold ; "none"
    compare: str = "gt"  # "gt": v > thr ; "ge": v >= thr
    multispike: bool = False  # emit floor(v/thr) spikes per step (sinabs-style)
    ignore_v_leak: bool = False
    ignore_r: bool = False
    dt_scale: float = 1.0  # framework uses dt * dt_scale instead of dt
    if_min_v: Optional[str] = None  # None ; "neg_threshold": clamp IF membrane at -thr (sinabs)
    synapse_v_leak: bool = False  # CubaLIF: synaptic current leaks towards v_leak (norse-style)
    init_v: str = "zero"  # "zero" ; "v_leak": membranes start at v_leak (norse)
    if_beta: Optional[float] = None  # IF simulated as a leaky neuron v <- beta*v + gain*x (snnTorch maps IF to Leaky(beta=0.9))

    def describe(self) -> str:
        ref = Convention()
        diffs = []
        for f in dataclasses.fields(self):
            a, b = getattr(self, f.name), getattr(ref, f.name)
            if a != b:
                diffs.append(f"{f.name}={a}")
        return "reference" if not diffs else ", ".join(diffs)


REFERENCE = Convention()

# nir < 1.0.5 has no CubaLI; keep the simulator importable there.
_CUBALI = getattr(nir, "CubaLI", type("_MissingCubaLI", (), {}))


# ----------------------------------------------------------------------------- helpers


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _pair(x) -> Tuple[int, ...]:
    a = np.atleast_1d(np.asarray(x)).astype(int).tolist()
    return tuple(a)


def _conv_nd(x: np.ndarray, node, nd: int) -> np.ndarray:
    """Direct (slow, obvious) N-d convolution. x: (B, C_in, *spatial)."""
    w = _arr(node.weight)  # (C_out, C_in/groups, *k)
    stride = _pair(node.stride)
    dilation = _pair(node.dilation)
    if len(stride) == 1:
        stride = stride * nd
    if len(dilation) == 1:
        dilation = dilation * nd
    k = w.shape[2:]
    pad = node.padding
    if isinstance(pad, str):
        if pad == "valid":
            pads = [(0, 0)] * nd
        else:  # same (stride 1)
            pads = []
            for d in range(nd):
                total = dilation[d] * (k[d] - 1)
                pads.append((total // 2, total - total // 2))
    else:
        p = _pair(pad)
        if len(p) == 1:
            p = p * nd
        pads = [(int(pi), int(pi)) for pi in p]
    x = np.pad(x, [(0, 0), (0, 0)] + pads)
    B, Cin = x.shape[:2]
    Cout = w.shape[0]
    groups = int(node.groups)
    cin_g, cout_g = Cin // groups, Cout // groups
    spatial = x.shape[2:]
    out_sp = [
        (spatial[d] - dilation[d] * (k[d] - 1) - 1) // stride[d] + 1 for d in range(nd)
    ]
    y = np.zeros((B, Cout, *out_sp))
    for g in range(groups):
        xg = x[:, g * cin_g : (g + 1) * cin_g]
        wg = w[g * cout_g : (g + 1) * cout_g]
        for kidx in np.ndindex(*k):
            sl = [slice(None), slice(None)]
            for d in range(nd):
                start = kidx[d] * dilation[d]
                sl.append(slice(start, start + stride[d] * (out_sp[d] - 1) + 1, stride[d]))
            patch = xg[tuple(sl)]  # (B, cin_g, *out_sp)
            wk = wg[(slice(None), slice(None)) + tuple(kidx)]  # (cout_g, cin_g)
            y[:, g * cout_g : (g + 1) * cout_g] += np.einsum("bc...,oc->bo...", patch, wk)
    bias = getattr(node, "bias", None)
    if bias is not None:
        y += _arr(bias).reshape((1, Cout) + (1,) * nd)
    return y


def _pool2d(x: np.ndarray, node, avg: bool) -> np.ndarray:
    k = _pair(node.kernel_size)
    s = _pair(node.stride)
    p = _pair(node.padding)
    if len(k) == 1:
        k = k * 2
    if len(s) == 1:
        s = s * 2
    if len(p) == 1:
        p = p * 2
    x = np.pad(x, [(0, 0), (0, 0), (p[0], p[0]), (p[1], p[1])])
    H, W = x.shape[2:]
    oh, ow = (H - k[0]) // s[0] + 1, (W - k[1]) // s[1] + 1
    y = np.zeros(x.shape[:2] + (oh, ow))
    for i in range(k[0]):
        for j in range(k[1]):
            y += x[:, :, i : i + s[0] * (oh - 1) + 1 : s[0], j : j + s[1] * (ow - 1) + 1 : s[1]]
    if avg:
        y /= k[0] * k[1]
    return y


def _flatten(x: np.ndarray, node) -> np.ndarray:
    # NIR dims exclude batch: shift by one.
    nd = x.ndim - 1
    start = node.start_dim if node.start_dim >= 0 else nd + node.start_dim
    end = node.end_dim if node.end_dim >= 0 else nd + node.end_dim
    start += 1
    end += 1
    shape = x.shape[:start] + (int(np.prod(x.shape[start : end + 1])),) + x.shape[end + 1 :]
    return x.reshape(shape)


# ----------------------------------------------------------------------------- neurons


def _v_reset(node) -> np.ndarray:
    """nir < 1.0.6 has no v_reset field: it then means reset to 0."""
    vr = getattr(node, "v_reset", None)
    return np.zeros_like(_arr(node.v_threshold)) if vr is None else _arr(vr)


class _State:
    def __init__(self):
        self.v: Optional[np.ndarray] = None
        self.i: Optional[np.ndarray] = None
        self.buf: List[np.ndarray] = []


def _decay(dt: float, tau: np.ndarray, c: Convention) -> np.ndarray:
    return np.exp(-dt / tau) if c.decay == "exact" else 1.0 - dt / tau


def _leaky_step(v, x, tau, r, v_leak, dt, c: Convention):
    """One step of tau dv/dt = (v_leak - v) + r x."""
    if c.ignore_r:
        r = np.ones_like(r)
    if c.ignore_v_leak:
        v_leak = np.zeros_like(v_leak)
    a = _decay(dt, tau, c)
    if c.input_gain == "euler":
        gain = (dt / tau) * r if c.decay == "euler" else (1 - a) * r
    elif c.input_gain == "unit":
        gain = r
    else:
        gain = np.ones_like(r)
    # leak towards v_leak with the same factor as the decay
    return a * v + (1 - a) * v_leak + gain * x


def _integrator_step(v, x, r, dt, c: Convention, leaky_ok: bool = False):
    if leaky_ok and c.if_beta is not None:
        v = c.if_beta * v
    if c.ignore_r:
        r = np.ones_like(r)
    if c.input_gain == "euler":
        gain = dt * r
    elif c.input_gain == "unit":
        gain = r
    else:
        gain = np.ones_like(r)
    return v + gain * x


def _fire(v, thr, v_reset, c: Convention, sink=None):
    if sink is not None:
        sink.append(float(np.min(np.abs(v - thr) / np.abs(thr))))
    if c.multispike:
        z = np.where(v >= thr, np.floor(v / thr), 0.0) if c.compare == "ge" else np.where(
            v > thr, np.floor(v / thr), 0.0
        )
    else:
        z = (v >= thr) if c.compare == "ge" else (v > thr)
        z = z.astype(np.float64)
    fired = z > 0
    if c.reset == "value":
        v = np.where(fired, v_reset, v)
    elif c.reset == "zero":
        v = np.where(fired, 0.0, v)
    elif c.reset == "subtract":
        v = v - z * thr
    return z, v


def _neuron_step(node, st: _State, x: np.ndarray, dt: float, c: Convention, sink=None) -> np.ndarray:
    if st.v is None:
        st.v = np.zeros_like(x)
        st.i = np.zeros_like(x)
        if c.init_v == "v_leak" and hasattr(node, "v_leak"):
            st.v = st.v + _arr(node.v_leak)
    if isinstance(node, nir.LI):
        st.v = _leaky_step(st.v, x, _arr(node.tau), _arr(node.r), _arr(node.v_leak), dt, c)
        return st.v.copy()
    if isinstance(node, nir.LIF):
        st.v = _leaky_step(st.v, x, _arr(node.tau), _arr(node.r), _arr(node.v_leak), dt, c)
        z, st.v = _fire(st.v, _arr(node.v_threshold), _v_reset(node), c, sink)
        return z
    if isinstance(node, nir.I):
        st.v = _integrator_step(st.v, x, _arr(node.r), dt, c)
        return st.v.copy()
    if isinstance(node, nir.IF):
        st.v = _integrator_step(st.v, x, _arr(node.r), dt, c, leaky_ok=True)
        z, st.v = _fire(st.v, _arr(node.v_threshold), _v_reset(node), c, sink)
        if c.if_min_v == "neg_threshold":
            st.v = np.maximum(st.v, -_arr(node.v_threshold))
        return z
    if isinstance(node, (_CUBALI, nir.CubaLIF)):
        tau_syn = _arr(node.tau_syn)
        w_in = _arr(node.w_in)
        syn_leak = _arr(node.v_leak) if c.synapse_v_leak else np.zeros_like(tau_syn)
        # synapse: tau_syn dI/dt = -I + w_in S  (unit resistance)
        st.i = _leaky_step(st.i, w_in * x, tau_syn, np.ones_like(tau_syn), syn_leak, dt, dataclasses.replace(c, ignore_r=False, ignore_v_leak=False))
        st.v = _leaky_step(st.v, st.i, _arr(node.tau_mem), _arr(node.r), _arr(node.v_leak), dt, c)
        if isinstance(node, _CUBALI):
            return st.v.copy()
        z, st.v = _fire(st.v, _arr(node.v_threshold), _v_reset(node), c, sink)
        return z
    raise TypeError(type(node))


_STATEFUL = (nir.LI, nir.LIF, nir.I, nir.IF, _CUBALI, nir.CubaLIF, nir.Delay)


def _stateless(node, x: np.ndarray, sink=None) -> np.ndarray:
    if isinstance(node, (nir.Input, nir.Output)):
        return x
    if isinstance(node, nir.Affine):
        return x @ _arr(node.weight).T + _arr(node.bias)
    if isinstance(node, nir.Linear):
        return x @ _arr(node.weight).T
    if isinstance(node, nir.Scale):
        return x * _arr(node.scale)
    if isinstance(node, nir.Threshold):
        if sink is not None:
            thr = _arr(node.threshold)
            sink.append(float(np.min(np.abs(x - thr) / np.abs(thr))))
        return (x > _arr(node.threshold)).astype(np.float64)
    if isinstance(node, nir.Flatten):
        return _flatten(x, node)
    if isinstance(node, nir.Conv1d):
        return _conv_nd(x, node, 1)
    if isinstance(node, nir.Conv2d):
        return _conv_nd(x, node, 2)
    if isinstance(node, nir.SumPool2d):
        return _pool2d(x, node, avg=False)
    if isinstance(node, nir.AvgPool2d):
        return _pool2d(x, node, avg=True)
    raise TypeError(f"reference simulator does not support {type(node).__name__}")


# ----------------------------------------------------------------------------- graph


def _order(graph: nir.NIRGraph) -> Tuple[List[str], set]:
    """Topological order from 'input' plus the set of back (recurrent) edges."""
    succ: Dict[str, List[str]] = {k: [] for k in graph.nodes}
    for a, b in graph.edges:
        succ[a].append(b)
    inputs = [k for k, n in graph.nodes.items() if isinstance(n, nir.Input)]
    color: Dict[str, int] = {}
    back = set()
    post: List[str] = []

    def dfs(u):
        color[u] = 1
        for v in succ[u]:
            if color.get(v, 0) == 0:
                dfs(v)
            elif color[v] == 1:
                back.add((u, v))
        color[u] = 2
        post.append(u)

    for s in inputs:
        if color.get(s, 0) == 0:
            dfs(s)
    for k in graph.nodes:  # unreachable nodes, if any
        if color.get(k, 0) == 0:
            dfs(k)
    return post[::-1], back


def simulate(
    graph: nir.NIRGraph,
    x: np.ndarray,
    dt: float,
    convention: Convention = REFERENCE,
    record: bool = False,
    _sink: Optional[list] = None,
):
    """Run ``graph`` on ``x`` of shape (T, B, ...). Returns (T, B, ...) output
    of the single Output node (and, if ``record``, a dict of per-node traces)."""
    c = convention
    dt_eff = dt * c.dt_scale
    order, back = _order(graph)
    preds: Dict[str, List[Tuple[str, bool]]] = {k: [] for k in graph.nodes}
    for a, b in graph.edges:
        preds[b].append((a, (a, b) in back))
    inputs = [k for k, n in graph.nodes.items() if isinstance(n, nir.Input)]
    outputs = [k for k, n in graph.nodes.items() if isinstance(n, nir.Output)]
    assert len(inputs) == 1 and len(outputs) == 1, "single input/output graphs only"
    states = {k: _State() for k in graph.nodes}
    prev: Dict[str, np.ndarray] = {}
    out_seq = []
    traces: Dict[str, List[np.ndarray]] = {k: [] for k in graph.nodes}
    x = _arr(x)
    for t in range(x.shape[0]):
        cur: Dict[str, np.ndarray] = {}
        for k in order:
            node = graph.nodes[k]
            if k == inputs[0]:
                inp = x[t]
            else:
                parts = []
                for p, is_back in preds[k]:
                    if is_back:
                        if p in prev:
                            parts.append(prev[p])
                    else:
                        parts.append(cur[p])
                if not parts:
                    # only recurrent inputs, first step: zero of the node's input shape
                    shape = tuple(int(s) for s in node.input_type["input"])
                    inp = np.zeros((x.shape[1],) + shape)
                else:
                    inp = parts[0]
                    for q in parts[1:]:
                        inp = inp + q
            if isinstance(node, nir.Delay):
                st = states[k]
                steps = np.round(_arr(node.delay) / dt_eff).astype(int)
                st.buf.append(inp)
                y = np.zeros_like(inp)
                for idx in np.ndindex(*steps.shape):
                    lag = steps[idx]
                    if len(st.buf) > lag:
                        y[(slice(None),) + idx] = st.buf[-1 - lag][(slice(None),) + idx]
                cur[k] = y
            elif isinstance(node, _STATEFUL):
                cur[k] = _neuron_step(node, states[k], inp, dt_eff, c, _sink)
            else:
                cur[k] = _stateless(node, inp, _sink)
            if record:
                traces[k].append(cur[k])
        prev = cur
        out_seq.append(cur[outputs[0]])
    y = np.stack(out_seq)
    if record:
        return y, {k: np.stack(v) for k, v in traces.items() if v}
    return y


def membrane_margin(graph: nir.NIRGraph, x: np.ndarray, dt: float) -> float:
    """Smallest |v - threshold| / |threshold| over every spiking neuron and step
    (pre-reset), under the reference convention. Fixtures keep this well away
    from 0 so a float32 framework cannot flip a spike through rounding alone."""
    sink: list = []
    simulate(graph, x, dt, REFERENCE, _sink=sink)
    return min(sink) if sink else float("inf")
