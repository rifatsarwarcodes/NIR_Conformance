<!-- DRAFTS for Rockpool — not posted. Rockpool 3.1.0, nir 1.0.4, nirtorch 2.0.2 -->

# 1. `rockpool.nn.modules.torch.nir.from_nir`: every `nir.LIF` import fails

The `nir.ir.LIF` branch passes `tau_mem=_to_tensor(node.tau_mem)` and `dt=…node.tau_mem…`, but `nir.LIF`'s field is `tau`. `tau_mem` belongs to `CubaLIF`. As a result, any graph containing a LIF fails:

```
# upstream/repros/rockpool_lif.py
AttributeError: 'LIF' object has no attribute 'tau_mem'      (rockpool/nn/modules/torch/nir.py:118)
```

**Fix:** use `node.tau` in that branch.

# 2. `from_nir`: LI time constants are truncated to integers

`ExpSynTorch(shape=…, tau=_to_tensor(node.tau).int(), …)`. A time constant in seconds such as 2 ms becomes `tau = 0`: imported `ExpSynTorch.tau = 0.0` for a file with τ = 0.002. Suggest dropping `.int()`.

# 3. `from_nir`: dt is inferred as `min(tau / (1 + r))`

The simulation dt is reconstructed from `r`. For a graph with r = 1 and τ_mem = 20 (meant for dt = 1), the imported `LIFTorch.dt` is 10. The export side writes `r = τ·exp(−dt/τ)/dt`, which is not the inverse of `dt = τ/(1+r)`, so a Rockpool → NIR → Rockpool round trip probably changes dt as well. That round trip was not tested here. Suggest letting the caller pass `dt` explicitly.

(Also: the `nir` extra pins `nir<=1.0.4`, which cannot read LIF/IF/CubaLIF files from nir ≥ 1.0.6 (see the NIR issue). There is no import for IF, I, Conv1d/Conv2d, Flatten, Delay or Threshold, and these raise clearly.)
