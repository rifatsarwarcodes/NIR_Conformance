<!-- DRAFTS for jeshraghian/snntorch — not posted. snnTorch 1.0.0, nirtorch 2.6, nir 1.0.8 -->

# 1. `import_from_nir`: `nir.IF` becomes a leaky neuron, and its output breaks downstream layers

`_nir_to_snntorch_module` maps `nir.IF` to `snn.Leaky(beta=0.9, threshold=vthr*r, init_hidden=False, reset_delay=False)`. That causes three problems:
- **It leaks:** the membrane decays by 10% per step, but `nir.IF` (`dv/dt = R·I`) has no leak.
- **It subtracts on reset:** `reset_mechanism` is left at the default `subtract`, but NIR resets to `v_reset`.
- **It returns a tuple:** with `init_hidden=False` the module returns a `(spk, mem)` tuple, ignoring the `init_hidden=True` that the node map passes. Any layer after an IF then crashes, for example SumPool2d (`'tuple' object has no attribute 'pow'`), AvgPool2d, or a following conv.

On our IF fixtures (with the tuple's first element taken), the output matches a reference IF with leak factor 0.9 and subtract reset exactly. Suggested mapping:

```python
snn.Leaky(beta=1.0, threshold=vthr, reset_mechanism="zero", init_hidden=init_hidden, reset_delay=False)
```

Repro: `upstream/repros/snntorch_if_dt.py` prints `IF -> Leaky beta 0.90 reset subtract`.

# 2. `import_from_nir`: dt is hardcoded to 1e-4 for LIF/CubaLIF

`dt = 1e-4` is hardcoded in the LIF, CubaLIF and recurrent branches, and `import_from_nir` takes no `dt` argument. A graph whose time constants are in steps (τ = 20, dt = 1) is imported with β = 1 − 1e-4/20 = 0.999995 instead of 0.95. Our LIF fixtures pass at dt = 1e-4 and fail at dt = 1. Suggest `import_from_nir(graph, dt=1e-4)`, threaded through to every branch.

# 3. `v_reset` is ignored

LIF and CubaLIF are always created with `reset_mechanism="zero"`. With `v_reset = -0.5·threshold`, the output is bit-identical to `v_reset = 0`. Either support a reset value or raise when `v_reset != 0`.

# 4. `nir.Flatten` dims are applied with the batch dimension included

`torch.nn.Flatten(node.start_dim, node.end_dim)` is used as is. NIR's dims exclude the batch dimension (as Sinabs's importer and NIR's own `calc_flatten_output` assume), so `Flatten(start_dim=0)` flattens the batch. A following Linear then fails with `mat1 and mat2 shapes cannot be multiplied (1x36 and 18x8)`. Suggest adding 1 to non-negative dims.

# 5. Recurrent LIF subgraphs do not import

`LIF ⇄ Linear` cycles are wrapped into a nested `NIRGraph`, and then:
- nirtorch 2.6: `AttributeError: 'GraphModule' object has no attribute 'items'` (in `nirtorch/nir_interpreter.py:_construct_state_recursive`)
- nirtorch 2.0.2: `KeyError: module name can't contain ".", got: lif.input`

(Also seen, not separate bugs: no import for LI, I, CubaLI, Scale, Delay or Threshold, which raise clearly. Per-neuron thresholds and `v_leak != 0` are rejected by assertions.)
