<!-- DRAFTS for synsense/sinabs — not posted. Sinabs 3.1.3, nir 1.0.4, nirtorch 2.0.2 -->

# 1. `nir<=1.0.4` pin: cannot read LIF/IF/CubaLIF files from current NIR

nir ≥ 1.0.6 always writes `v_reset`. Under the pin, `nir.read` of any spiking-neuron file exported by current snnTorch, Norse or Spyx fails with `TypeError: LIF.__init__() got an unexpected keyword argument 'v_reset'`. The installer also downgrades `nir` from 1.0.8 to 1.0.4 for everything else in the environment. See the NIR issue on forward compatibility.

# 2. `from_nir`: `nir.Linear` is not supported

Only `nir.Affine` is handled. snnTorch, Norse and Spyx export bias-free layers as `nir.Linear`, so most of their graphs fail with:
- `ValueError: Cannot find the input of an empty graph`, or
- `KeyError: 'fc'` (from the nirtorch graph).

Suggest mapping `nir.Linear` → `nn.Linear(bias=False)`.

# 3. `from_nir`: `nir.Conv1d` crashes

`conv.weight.data = node.weight.detach().clone().to(float)` (`sinabs/nir.py:49`). NIR stores numpy arrays, so this raises `'numpy.ndarray' object has no attribute 'detach'`. Conv2d already uses `torch.tensor(node.weight).float()`; the same works here. (`.to(float)` would also make the weight float64.)

# 4. `from_nir`: recurrent edges are silently ignored

For `Affine → LIF` with a recurrent `Linear` (LIF → rec → LIF), the output is bit-identical when the recurrent weights are zeroed. The recurrence is not imported, and there is no error.

# 5. `from_nir`: LIF `r` is ignored

The LI branch checks `r == 1` and raises otherwise. The LIF branch does not check `r`, and its output is bit-identical for r = 2 and r = 1. Suggest the same check, or applying `r` as an input scale.

(For documentation: IF/LIF import uses Sinabs's defaults of multi-spike output and subtract-threshold reset. NIR's equations specify at most one spike per step and reset to `v_reset`. Time constants are read in steps, and there is no `dt` argument. See NIR #208.)
