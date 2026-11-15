"""Framework adapters: NIR graph + input (T, B, ...) + dt  ->  output (T, B, ...).

Each adapter uses only the framework's public NIR import function and its
documented calling convention. Layout conversions an adapter performs (batch
vs time order, channels-last for Spyx) are the framework's documented
conventions, not fixes. Any parameter the importer cannot take (for example
``dt``) is recorded in ``CAPABILITIES`` rather than worked around.
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict

import numpy as np

CAPABILITIES: Dict[str, Dict[str, str]] = {
    "snntorch": {"import": "snntorch.import_nir.import_from_nir(graph)", "dt": "not configurable (hardcoded 1e-4 in the importer)", "stepping": "per step, init_hidden=True"},
    "norse": {"import": "norse.torch.from_nir(graph, dt=dt)", "dt": "configurable", "stepping": "per step, explicit state"},
    "sinabs": {"import": "sinabs.from_nir(graph, batch_size=B, num_timesteps=T)", "dt": "not configurable (time constants are read in steps)", "stepping": "whole sequence, (B*T, ...) squeezed"},
    "rockpool": {"import": "rockpool.nn.modules.torch.nir.from_nir(graph)", "dt": "not configurable (inferred from tau/(1+r))", "stepping": "whole sequence, (B, T, N)"},
    "spyx": {"import": "spyx.nir.from_nir(graph, input, dt=dt)", "dt": "configurable", "stepping": "whole sequence, (T, B, ...) channels-last"},
}


NOTES: Dict[str, set] = {}


def _note(msg: str) -> None:
    NOTES.setdefault("current", set()).add(msg)


def _unwrap(out):
    """Lenient: if a graph's output is a tuple (e.g. (spikes, membrane)), keep
    the first element and record that the output was not a plain tensor."""
    while isinstance(out, (tuple, list)):
        _note("graph output was a tuple; first element used")
        out = out[0]
    return out


def _torch_step_loop(net, x: np.ndarray, pass_state: bool = True) -> np.ndarray:
    import torch

    outs, state = [], None
    with torch.no_grad():
        for t in range(x.shape[0]):
            xt = torch.tensor(x[t])
            ret = net(xt, state) if (pass_state and state is not None) else net(xt)
            if isinstance(ret, tuple):
                out, state = ret[0], ret[1]
            else:
                out = ret
            outs.append(_unwrap(out).detach().cpu().numpy())
    return np.stack(outs)


def run_snntorch(graph, x, dt):
    from snntorch.import_nir import import_from_nir

    net = import_from_nir(graph)
    return _torch_step_loop(net, x)


def run_norse(graph, x, dt):
    from norse.torch import from_nir

    net = from_nir(graph, dt=dt)
    return _torch_step_loop(net, x)


def run_sinabs(graph, x, dt):
    import sinabs
    import torch

    T, B = x.shape[:2]
    net = sinabs.from_nir(graph, batch_size=B, num_timesteps=T)
    xb = np.moveaxis(x, 0, 1).reshape((B * T,) + x.shape[2:])  # batch-major squeeze
    with torch.no_grad():
        ret = net(torch.tensor(xb))
    out = ret[0] if isinstance(ret, tuple) else ret  # (output, state)
    out = _unwrap(out).detach().cpu().numpy()
    return np.moveaxis(out.reshape((B, T) + out.shape[1:]), 1, 0)


def run_rockpool(graph, x, dt):
    import torch
    from rockpool.nn.modules.torch.nir import from_nir

    net = from_nir(graph)
    xb = np.moveaxis(x, 0, 1)  # (B, T, ...)
    with torch.no_grad():
        ret = net(torch.tensor(xb))
    out = ret[0] if isinstance(ret, tuple) else ret  # (output, state)
    out = _unwrap(out).detach().cpu().numpy()
    return np.moveaxis(out, 1, 0)


def run_spyx(graph, x, dt):
    import jax.numpy as jnp
    import spyx.nir

    xin = x
    if x.ndim == 5:  # (T, B, C, H, W) -> channels-last (T, B, H, W, C), Spyx's documented layout
        xin = np.transpose(x, (0, 1, 3, 4, 2))
    _, out = spyx.nir.from_nir(graph, jnp.array(xin), dt=dt)
    out = np.asarray(out)
    if out.ndim == 5:
        out = np.transpose(out, (0, 1, 4, 2, 3))
    return out


ADAPTERS: Dict[str, Callable] = {
    "snntorch": run_snntorch,
    "norse": run_norse,
    "sinabs": run_sinabs,
    "rockpool": run_rockpool,
    "spyx": run_spyx,
}


def framework_version(name: str) -> str:
    import importlib.metadata as md

    try:
        return md.version(name)
    except Exception:
        return "unknown"


def run(name: str, graph, x, dt):
    NOTES.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ADAPTERS[name](graph, x.astype(np.float32), dt)
