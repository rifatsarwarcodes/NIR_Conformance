# Reference convention and open questions in the NIR spec

NIR defines each neuron as a continuous-time ODE. It does not say how a simulator should step that ODE in discrete time. To compare frameworks we need one exact rule. This file states the rule the reference simulator (`src/nirconf/reference.py`) uses, and lists every choice where frameworks were observed to differ.

The reference rule is **one reasonable reading of the spec, not a ruling**. Where the spec is silent, the conformance matrix reports a framework as "DIFF" against this reading, and `docs/FINDINGS.md` labels the item *ambiguity* rather than *bug*. Proposing which reading should be normative is the point of the NIR #208 discussion. See `upstream/nir-208-comment.md`.

## The reference rule

| Choice | Reference | Why |
|---|---|---|
| Time stepping | Forward Euler with step `dt`: `v ← v + dt/τ · (v_leak − v + r·x)` | The literal Euler step of NIR's LIF/LI equation. It is also what snnTorch (via threshold rescaling), Norse (`LIFBox`) and Spyx (`beta = 1 − dt/τ`) compute for LIF |
| IF / I input | `v ← v + dt · r · x` | The literal Euler step of `dv/dt = R·I`. **No framework does this** (see ambiguity A2) |
| CubaLIF order | Synaptic current first, and the membrane uses the *updated* current | Matches Norse and snnTorch `Synaptic` |
| Spike condition | `v > v_threshold` (strict) | The NIR docstrings say `v > v_thr` |
| Spikes per step | At most one | NIR's `z ∈ {0, 1}` |
| Reset | `v ← v_reset` in the same step as the spike | NIR's equations. Before nir 1.0.6 there was no `v_reset` field, which means reset to 0 |
| Initial state | All states 0 | The spec says nothing (A4) |
| Graph timing | Edges in a directed acyclic graph propagate within a step; an edge that closes a cycle carries the previous step's value | The spec says nothing (A6) |
| Fan-in | Several edges into one node are summed | NIR graph semantics |
| Batch | NIR shapes exclude the batch dimension | NIR convention (it matters for `Flatten`) |

The simulator runs in float64 on the same float32 parameters stored in the file. Fixtures are tuned so that every membrane stays at least 1e-4 (relative) away from its threshold at every step. Float32 drift over 100 steps is about 1e-6, so a framework cannot flip a spike through rounding alone. Spiking outputs must match **exactly**. Analog outputs must match to rtol 1e-4 and atol 1e-5·max|ref|.

## Switches the fingerprint search tries

When a framework disagrees, `compare.py` searches for the smallest set of these changes (up to 4) under which the reference reproduces the framework's output bit-for-bit:

| Switch | Values tried | Example of a framework that matches it |
|---|---|---|
| `decay` | `exact` (`exp(−dt/τ)`) | Sinabs |
| `input_gain` | `unit` (`v += r·x`), `unit_no_r` (`v += x`) | Spyx (LIF), every framework's IF |
| `reset` | `zero`, `subtract`, `none` | snnTorch (ignores `v_reset`), Sinabs (subtract) |
| `compare` | `ge` (`v ≥ thr`) | — |
| `multispike` | floor(v/thr) spikes per step | Sinabs |
| `ignore_v_leak`, `ignore_r` | true | Norse (r), Spyx (r) |
| `dt_scale` | the framework uses its own dt (1, 1e-3, 1e-4, or Rockpool's `τ/(1+r)`) | snnTorch (hardcoded 1e-4) |
| `if_min_v` | IF membrane clamped at −threshold | Sinabs (in code, not reached by these fixtures) |
| `synapse_v_leak` | CubaLIF current decays towards `v_leak` | Norse (in code) |
| `init_v` | membranes start at `v_leak` | Norse |
| `if_beta` | IF leaks with factor 0.9 per step | snnTorch |
| structural | a node silently dropped; recurrent edges dropped | Norse, Spyx, Sinabs |
| output delay | spikes reported k steps late | Spyx (one step per spiking layer) |

The search only produces hypotheses. Every finding claimed in `FINDINGS.md` is confirmed by a separate direct check in `src/nirconf/verify_findings.py`: an invariance test, a module parameter inspection, or a minimal hand-built graph.

## Open questions in the spec (observed to cause divergence)

- **A1. No time step in the file.** A `.nir` file stores time constants, not `dt` (check NIR-3). The same file means different dynamics in each framework unless the user passes the matching `dt`:

  | Framework | How it picks dt |
  |---|---|
  | snnTorch | Hardcoded `1e-4` |
  | Norse | Argument, default `1e-3` |
  | Spyx | Argument, default `1` |
  | Sinabs | None; time constants are read as steps |
  | Rockpool | Inferred as `min(τ/(1+r))` |

  Options for the spec: store an intended `dt` in graph metadata, or state that time constants are in seconds and every importer must take `dt` explicitly.
- **A2. What does an input value mean for IF/I?** Under NIR's ODE, `dv/dt = R·I` needs `v += dt·R·x`. Norse, snnTorch and Spyx all use `v += R·x` (or `v += x`) for IF. Yet Norse and snnTorch use `dt` for LIF (check NO-4). Either "inputs are charge per step (impulses)" or "inputs are currents" should be stated once, for all neuron types.
- **A3. Reset.** Reset to value or subtract the threshold? At most one spike per step, or several? The spec equations say reset to `v_reset`, one spike per step. Sinabs defaults to subtract and multi-spike; snnTorch's IF subtracts.
- **A4. Initial state.** The spec says nothing. Norse starts membranes at `v_leak`; others start at 0 (check NO-5).
- **A5. When a spike appears.** Is step t's spike computed from the membrane after or before the step-t update? Spyx reports it one step late, before the update (check SX-1). For an N-layer network this shifts the output by N steps.
- **A6. Timing of recurrent edges.** One-step delay (the reference) or within the same step? This is not exercised separately yet, because only Norse and Spyx run the recurrent fixtures at all.
- **A7. Unsupported nodes.** The spec says nothing about what an importer must do with a node it cannot map. nirtorch's legacy loader skips it silently (NT-2), which produces wrong outputs and even non-deterministic behaviour (NT-3). Recommendation: importers must raise.
