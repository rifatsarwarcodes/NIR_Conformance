<!-- DRAFTS for kmheckel/spyx — not posted. Spyx 1.0.0, nir 1.0.8 -->

# 1. `spyx.nir.from_nir`: only the first branch of a non-chain graph is imported

`_build_model` orders edges by following the first outgoing edge from `input` (`_order_edge_list`), so other branches are dropped without error. For `input → fc_a → lif`, `input → fc_b → lif`, the output is bit-identical whether `fc_b`'s weights are strong or zero:

```
# upstream/repros/spyx_fan_in.py
spikes with strong branch b: 76 | with branch b zeroed: 76
```

Suggest raising for graphs that are not a chain until fan-in is supported.

# 2. `from_nir`: Flatten after a conv uses channels-last element order

Spyx convs are channels-last, so `Flatten` produces `(H, W, C)` order. The NIR `Linear` weights that follow were written for NIR's channels-first `(C, H, W)` order. As a result, a conv → flatten → dense network imported from another framework computes with permuted weights. Check: the output of `Flatten → Linear → LIF` equals Spyx's `Linear → LIF` applied to the input flattened as `(H, W, C)`, not as `(C, H, W)`. Fix: transpose to channels-first before flattening, or permute the next layer's weight columns on import.

# 3. `from_nir`: unsupported nodes are dropped silently

`_nir_node_to_spyx_module` returns `None` for `Scale`, `Delay`, `I`, `CubaLI` and `AvgPool2d`, and `_build_model` skips them:
- Scale ×3 and Delay → 0 leave the output bit-identical.
- An `I` node outputs its input.
- `AvgPool2d` output keeps the un-pooled shape.

Suggest raising `NotImplementedError`.

# 4. `from_nir`: LIF `r`, `v_leak` and `v_reset` are ignored

Only `tau` (via `beta = 1 − dt/τ`) and `v_threshold` are read. Each of `r`, `v_leak` and `v_reset` can be changed without changing the output. The import uses `v ← β·v + x`, which equals NIR's LIF only if `r = τ/dt`. (Spyx's recurrent export, `_spyx_recurrent_to_nirgraph`, writes `r = β`, which may be worth checking against NIR's equation as well.)

(For documentation: step t's spike is computed from the membrane *before* the step-t update. An input only at t = 0 gives reference spikes `[1,0,0,0,0]` and Spyx `[0,1,1,1,1]`, and the delay adds up per layer. NIR doesn't pin this down either; see NIR #208.)
