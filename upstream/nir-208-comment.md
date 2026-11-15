<!-- DRAFT for neuromorphs/NIR#208 — not posted -->

I built a prototype of this to see what shared fixtures would catch. Sharing the results in case they're useful for the design here.

**What it is**
- 34 small graphs, one per primitive or edge case: LIF/IF/LI/I/CubaLIF/CubaLI, Affine/Linear/Scale, Conv1d/Conv2d, SumPool2d/AvgPool2d, Flatten, Delay, Threshold, recurrent LIF/CubaLIF, fan-in, and small composed nets.
- Each graph comes at two time scales, with the same dimensionless dynamics:
  - `dt=1`, τ=20 steps
  - `dt=1e-4`, τ=2 ms
- Each fixture is a directory holding `graph.nir`, `input.npy` (T=100, B=2 spike trains), `expected.npy` and `meta.json` (dt, description, what it targets).
- Expected outputs come from a slow float64 NumPy reference simulator: forward Euler, the reset and threshold rules from the NIR docstrings, states starting at 0.
- Parameters are tuned so that no membrane comes within 1e-4 of threshold. A float32 framework therefore has to match the spikes **exactly**.
- A harness runs every fixture through snnTorch 1.0.0, Norse 1.1.0, Spyx 1.0.0, Sinabs 3.1.3 and Rockpool 3.1.0, and writes a pass/diff/error matrix.
- For each mismatch, it searches for the smallest set of convention changes that reproduces the framework's output bit-for-bit. Examples: Euler vs exact decay, reset-to-value vs subtract, multi-spike, ignoring `r`, a different dt, a node silently dropped, output delayed k steps.

**What it found.** Each item is confirmed by a separate direct test.
- **Silent wrong results are the main problem.**
  - nirtorch's legacy `load` skips nodes it can't map, so Norse silently drops LI, I, CubaLI and AvgPool2d.
  - The same loader returns the wrong node for fan-in graphs.
  - With a skipped node, it is not even deterministic: the same file sometimes raises and sometimes runs.
  - Spyx imports only the first branch of a non-chain graph, and drops Scale and Delay.
  - Several importers ignore `r`, `v_leak` or `v_reset` with at most a log warning.
- **Compatibility.** nir ≥ 1.0.6 always writes `v_reset`, and nir ≤ 1.0.4 rejects it. Sinabs and Rockpool pin `nir<=1.0.4`, so they cannot read any LIF/IF/CubaLIF file written by current NIR.
- **Spec questions the fixtures force.** These are where frameworks disagree with each other; I'd suggest settling them in the spec.
  1. dt is not stored. snnTorch hardcodes 1e-4, Norse defaults to 1e-3, Spyx to 1, Sinabs reads τ in steps, and Rockpool infers `τ/(1+r)`.
  2. For IF, is the input charge per step (`v += R·x`, what every framework does) or a current (`v += dt·R·x`, the literal ODE)? Norse and snnTorch use dt for LIF but not for IF.
  3. Reset to `v_reset` vs subtract threshold; at most one spike per step vs several.
  4. Initial state: 0 vs `v_leak`.
  5. Whether step t's spike comes from the membrane after or before the step-t update. Spyx reports one step late per layer.
  6. Recurrent-edge timing.
  7. What an importer must do with an unsupported node. I'd propose: it must raise.

**A design suggestion** from doing this. Fixtures only work if the expected outputs follow one stated convention. So I'd suggest the fixture `meta.json` (or NIR graph metadata) carry an explicit `dt` and convention block, and #208 decide the defaults for points 1–7. Divergent frameworks could then declare which points they deviate on, and the harness would check them against the matching variant.

The full matrix, the per-framework findings with minimal repros, and the code are here: <LINK — to be added when published>. Happy to turn this into a PR in whatever layout you prefer, for example `tests/fixtures/<primitive>/` plus a pytest runner, once the conventions are agreed.
