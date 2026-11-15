<!-- DRAFTS for norse/norse — not posted. Norse 1.1.0 with nirtorch 2.6 / nir 1.0.8 (also nirtorch 2.0.2 / nir 1.0.4) -->

# 1. `from_nir`: CubaLIF synaptic current decays towards `v_leak`

`_import_norse_module` builds the CubaLIF synapse as `LIBoxCell(LIBoxParameters(tau_mem_inv=1/tau_syn, v_leak=node.v_leak))`. NIR's synapse is `τ_syn dI/dt = −I + w_in·S`, with no leak term, so the current should decay to 0. With `v_leak != 0` the current settles at `v_leak` and is added to the membrane on top of the membrane's own leak. A neuron with zero input then fires.

```
# upstream/repros/norse_cubalif_vleak.py — v_leak = 0.6, threshold = 1, zero input, 200 steps
spikes with zero input: 6   (expected 0: v settles at v_leak = 0.6)
```

**Fix:** `v_leak=torch.zeros_like(...)` for the synapse.

# 2. `from_nir`: unsupported nodes are dropped silently

There are no branches for `nir.LI`, `nir.I`, `nir.CubaLI`, `nir.Scale`, `nir.AvgPool2d`, `nir.Delay`, `nir.Threshold` or `nir.Conv1d`. `_import_norse_module` returns `None`, and nirtorch's legacy loader then skips the node without error. The results:
- `Linear → LI` returns the Linear output.
- `Conv → IF → AvgPool2d` returns the un-pooled IF output with shape `[2, 6, 6]`.
- Graphs with Delay or Scale sometimes raise and sometimes run when the same file is imported repeatedly (see the nirtorch issue).

Suggest raising `NotImplementedError` for unknown nodes. Separately, LI maps directly onto `LIBoxCell`, and Scale onto a multiply, so both could be supported.

# 3. `from_nir`: LIF `r` and `v_reset` are ignored

- `r != 1` only logs a warning: output is bit-identical for r = 2 and r = 1.
- `v_reset` is never read: output is bit-identical for `v_reset = −0.5·thr` and 0.

`r` could be applied as an input scale, as is already done for CubaLIF, and `reset_method` could use `v_reset`.

# 4. `from_nir`: `nir.Flatten` dims include the batch dimension

`torch.nn.Flatten(node.start_dim, node.end_dim)`: NIR dims exclude batch, so `Flatten(start_dim=0)` → Linear fails with `mat1 and mat2 shapes cannot be multiplied (1x36 and 18x8)`. Suggest adding 1 to non-negative dims, as Sinabs does.

(Possibly worth documenting rather than a bug: IF ignores `dt`, so `v += x` per step, while LIF uses it. The same IF graph gives identical spikes at dt = 1 and dt = 1e-4. Membranes also start at `v_leak` rather than 0. NIR doesn't specify either. See NIR #208.)
