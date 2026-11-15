# Findings

Run on 2026-11-16 on an M4 Mac (CPU only).

| Package | Version |
|---|---|
| nir | 1.0.8 and 1.0.4 |
| nirtorch | 2.6 (with nir 1.0.8) and 2.0.2 (with nir 1.0.4) |
| snnTorch | 1.0.0 |
| Norse | 1.1.0 |
| Sinabs | 3.1.3 |
| Rockpool | 3.1.0 |
| Spyx | 1.0.0 |
| torch | 2.15.0 |
| jax | 0.10.2 |

Sinabs and Rockpool's NIR extra require `nir<=1.0.4`, so they were tested in a second environment.

Every item below has a check ID. `python -m nirconf.verify_findings results` re-runs these checks in the current environment, and the evidence it printed is in `results/verified/*.json`. **All 36 distinct checks passed: 26 under nir 1.0.8 and 19 under nir 1.0.4, with 9 run in both.** The conformance matrix is in `results/matrix.md` (and `results/matrix.html`).

Each finding has one of four labels:

| Label | Meaning |
|---|---|
| **bug** | Contradicts the NIR spec text or the framework's own code intent |
| **silent** | An unsupported feature that produces a wrong result without an error (the most harmful kind) |
| **gap** | An unsupported feature that fails loudly |
| **ambiguity** | The spec does not decide the question (see `conventions.md`) |

## Headline numbers

There are 68 fixtures: 34 graphs × 2 time-step suites (`dt = 1` with τ = 20, and `dt = 1e-4` with τ = 2 ms).

| Framework | PASS (identical to reference) | Ran but differs (no error) | Raised | Unreadable | Flaky |
|---|---|---|---|---|---|
| Norse 1.1.0, nir 1.0.8 | 31 | 25 | 8 | 0 | 4 |
| Norse 1.1.0, nir 1.0.4 | 31 | 19 | 8 | 6 | 4 |
| snnTorch 1.0.0, nir 1.0.8 | 12 | 21 | 35 | 0 | 0 |
| snnTorch 1.0.0, nir 1.0.4 | 11 | 6 | 45 | 6 | 0 |
| Spyx 1.0.0, nir 1.0.8 | 4 | 60 | 4 | 0 | 0 |
| Sinabs 3.1.3, nir 1.0.4 | 2 | 14 | 44 | 6 | 2 |
| Rockpool 3.1.0, nir 1.0.4 | 4 | 12 | 46 | 6 | 0 |

"Ran but differs" counts both wrong values and wrong output shapes. "Flaky" means repeated runs of the same file gave different outcomes: some raised, some ran (NT-3).

"Differs" is measured against the reference convention. Several differences are ambiguities, not bugs: for Spyx most of its 60 come from A2 and A5. The findings below separate the two.

- **Most faithful importer:** Norse. It passes every canonical LIF, IF, CubaLIF, recurrent, Conv2d and SumPool2d fixture at `dt = 1`.
- **Biggest risk:** silent failure. An unsupported node, branch or parameter usually yields a plausible-looking wrong output rather than an error. This holds for Norse, Spyx and Sinabs, and for nirtorch's legacy loader that Norse, Sinabs and Rockpool share.

## NIR core

| ID | Label | Finding | Evidence |
|---|---|---|---|
| NIR-1 | bug | `Conv1d`/`Conv2d.input_type` ignores `groups`. Input channels are taken as `weight.shape[1]` (= C_in/groups), so every valid grouped conv fails graph type inference | `Conv2d(weight (4,2,3,3), groups=2).input_type = [2,7,7]`; building `Input([4,7,7]) → conv` raises `Type inference error`. Source: `nir/ir/conv.py:57` and `:124` |
| NIR-2 | bug (compatibility) | From nir 1.0.6, `v_reset` is always written for LIF/IF/CubaLIF, even when not given. nir ≤ 1.0.4 rejects the key (`unexpected keyword argument 'v_reset'`). **Every LIF/IF/CubaLIF file written by current NIR is unreadable by Sinabs 3.1.3 and Rockpool 3.1.0**, which pin `nir<=1.0.4`. `CubaLI` (added in 1.0.5) is also unreadable there | A LIF built without `v_reset` is written with keys `[r, tau, type, v_leak, v_reset, v_threshold]`. Versions checked: 1.0.5 has no `v_reset`; 1.0.6 does |
| NIR-3 | ambiguity A1 | Files store no time step | The LIF dataset contains only `r, tau, type, v_leak, v_reset, v_threshold` |

## nirtorch legacy loader (`nirtorch.load`, used by Norse, Sinabs and Rockpool)

`nirtorch.load` prints a deprecation warning in favour of `nir_to_torch`. snnTorch uses `nir_to_torch` and passes the fan-in fixture.

| ID | Label | Finding | Evidence |
|---|---|---|---|
| NT-1 | bug, silent | Execution order is not topological for fan-in graphs, and the executor returns the last-executed node (`graph_executor.py:166-167`). A two-branch graph returns one branch's Linear output instead of the neuron's spikes | Execution order `['input', 'fc_a', 'lif', 'output', 'fc_b']`; the model output equals `fc_b(x)` exactly |
| NT-2 | silent | Nodes the importer cannot map (it returns `None`) are skipped without error (`graph_executor.py:153`) | A Linear→LI graph imports; the execution order is `['input', 'fc']` and the output is the Linear output |
| NT-3 | bug | With such a skipped node the result is **not deterministic**: repeated imports of the same file, in one process with `PYTHONHASHSEED` fixed, sometimes raise and sometimes run (with a wrong output). The likely cause is node sets ordered by object identity (`nirtorch/graph.py:79-84`). Graphs of supported nodes are stable | Over 30 imports each (saved run, nirtorch 2.6; counts vary run to run): Norse `scale` raised in 13 and ran in 17, Norse `delay` raised in 12 and ran in 18, and the control graph `two_layer` gave the same result all 30 times. Under nirtorch 2.0.2, Sinabs `flatten` raised in 5 of 30. The runner repeats Norse, Sinabs and Rockpool fixtures 30 times and marks such cells FLAKY |

## snnTorch 1.0.0

| ID | Label | Finding | Evidence |
|---|---|---|---|
| ST-1 | bug, silent | LIF/CubaLIF import hardcodes `dt = 1e-4` (`import_nir.py`, no `dt` argument). Time constants in any other unit give wrong dynamics | τ = 20 with dt = 1 should give β = 0.95; imported `Leaky.beta = 0.999995`. LIF fixtures pass at dt = 1e-4 and fail at dt = 1 |
| ST-2 | bug, silent | `nir.IF` is imported as a **leaky** neuron: `snn.Leaky(beta=0.9, reset_mechanism='subtract', init_hidden=False)`. Its output is a `(spk, mem)` tuple, which crashes any layer after it | Module inspection. The IF fixtures match the reference exactly with `if_beta=0.9, reset=subtract`. Conv→IF→SumPool2d raises `'tuple' object has no attribute 'pow'` |
| ST-3 | bug, silent | `v_reset` is ignored: LIF always gets `reset_mechanism='zero'` | `v_reset = −0.5·thr` gives output bit-identical to `v_reset = 0` |
| ST-4 | bug | `Flatten` dims go to `torch.nn.Flatten` unchanged. NIR dims exclude batch, so `start_dim=0` flattens the batch | `Flatten(start_dim=0)` → `mm: mat1 and mat2 shapes cannot be multiplied (1x36 and 18x8)` |
| ST-5 | bug | Recurrent LIF graphs do not import | nirtorch 2.6: `'GraphModule' object has no attribute 'items'`. nirtorch 2.0.2: `module name can't contain ".", got: lif.input` |
| ST-6 | gap | Rejects per-neuron thresholds, `v_leak ≠ 0`, and CubaLIF unless `r = τ_mem/dt`. There is no importer for LI, I, CubaLI, Scale, Delay or Threshold | Assertion messages (loud, good) |

## Norse 1.1.0

| ID | Label | Finding | Evidence |
|---|---|---|---|
| NO-1 | silent | LI, I, CubaLI and AvgPool2d have no importer and are dropped without error (through NT-2) | The outputs equal the reference with that node removed; AvgPool2d output shape is `[2,6,6]` instead of `[2,3,3]` |
| NO-2 | bug, silent | LIF `r` is ignored (only a log warning) | `lif_r2` output is bit-identical with r = 2 and r = 1 |
| NO-3 | bug, silent | `v_reset` is ignored (reset to 0) | Bit-identical with `v_reset = −0.5·thr` and 0 |
| NO-4 | ambiguity A2, inconsistent | IF ignores `dt` (`v += x`) while LIF uses it | The same IF graph gives identical spikes (171) at dt = 1 and dt = 1e-4; LIF gives 179 vs 0 |
| NO-5 | ambiguity A4 | Membranes start at `v_leak` | `lif_v_leak` equals the reference started at `v = v_leak`, exactly |
| NO-6 | bug, silent | CubaLIF passes the membrane's `v_leak` to the *synaptic* current, so with `v_leak ≠ 0` a neuron with no input fires | Zero input, `v_leak = 0.6`, threshold 1: the reference gives 0 spikes (v settles at 0.6); Norse gives 24 |
| NO-7 | bug | `Flatten` dims are used as torch dims (same as ST-4) | `mat1 and mat2 shapes cannot be multiplied (1x36 and 18x8)` |
| NO-8 | gap | Conv1d is not supported (crashes at run time) | Shape error in `iaf.py:95` |

## Sinabs 3.1.3 (needs nir ≤ 1.0.4)

| ID | Label | Finding | Evidence |
|---|---|---|---|
| SB-1 | compatibility | Cannot read current NIR LIF/IF/CubaLIF files (NIR-2) | `requires nir<=1.0.4`; reading a nir-1.0.8 LIF file raises `unexpected keyword argument 'v_reset'` |
| SB-2 | gap | No importer for `nir.Linear`, the bias-free layer that snnTorch, Norse and Spyx export. Most fixtures fail on it with `KeyError` or `empty graph` | `Cannot find the input of an empty graph` |
| SB-3 | bug | Conv1d import calls `.detach()` on the numpy weight array | `'numpy.ndarray' object has no attribute 'detach'` (`sinabs/nir.py:49`) |
| SB-4 | ambiguity A3 | IF is imported with multi-spike output and reset-by-subtraction | The conv2d→IF output equals the reference with `reset=subtract, multispike=True`; up to 2 spikes per step |
| SB-5 | silent | Recurrent edges are ignored | `recurrent_lif` output is bit-identical with the recurrent weights zeroed |
| SB-6 | bug, silent | LIF `r` is ignored. LI checks `r == 1` but LIF does not | Affine→LIF output is bit-identical with r = 2 and r = 1 |

## Rockpool 3.1.0 (needs nir ≤ 1.0.4)

| ID | Label | Finding | Evidence |
|---|---|---|---|
| RP-1 | bug | **Every `nir.LIF` import crashes.** The importer reads `node.tau_mem`, but `nir.LIF`'s field is `tau` | `'LIF' object has no attribute 'tau_mem'` (`rockpool/nn/modules/torch/nir.py:118`) |
| RP-2 | bug, silent | LI time constants are truncated to integers (`tau.int()`): τ = 2 ms becomes 0 | File τ = 0.002; imported `ExpSynTorch.tau = 0.0` |
| RP-3 | ambiguity A1, silent | The simulation dt is inferred as `min(τ/(1+r))`; with r = 1 it is τ/2 | The file intends dt = 1 (τ_mem = 20 steps); imported `LIFTorch.dt = 10.0` |
| RP-4 | gap | No importer for IF, I, Conv1d/Conv2d, Flatten, Delay or Threshold | `NotImplementedError` for each (loud, good) |

## Spyx 1.0.0

| ID | Label | Finding | Evidence |
|---|---|---|---|
| SX-1 | ambiguity A5 | Spikes are reported one step late: step t's output comes from the membrane *before* the step-t update. The delay adds up per layer | Input only at t = 0: the reference gives `[1,0,0,0,0]`, Spyx `[0,1,1,1,1]` |
| SX-2 | bug, silent | LIF `r`, `v_leak` and `v_reset` are all ignored. The input gain is always 1 (A2) | Each output is bit-identical when that field is reset to its default |
| SX-3 | bug, silent | Only the first branch of a non-chain graph is imported (the edge walker follows a single chain) | `fan_in` output is bit-identical with the second branch's weights zeroed |
| SX-4 | silent | Scale, Delay, I, CubaLI and AvgPool2d are dropped without error | Scale ×3 and Delay → 0 leave the output bit-identical; I output equals its input; AvgPool2d shape is `[2,6,6]` |
| SX-5 | bug, silent | `Flatten` flattens in Spyx's channels-last `(H, W, C)` order, but the NIR weights that follow expect channels-first `(C, H, W)` order. A conv→flatten→dense network computes with scrambled weights | The output equals Spyx's Linear→LIF on the input flattened as (H,W,C), not as (C,H,W) |
| SX-6 | gap | Conv1d fails at run time | `add got incompatible shapes for broadcasting` |

## What this suite does not cover yet

- **Export:** framework → NIR → framework round trips. Rockpool's export formula `r = τ·e^(−dt/τ)/dt` and its import formula `dt = τ/(1+r)` are not inverses, which suggests a round-trip bug. Not tested.
- **Other frameworks:** SpikingJelly 2.0, Lava-dl (archived), hxtorch/jaxsnn, Nengo, SpiNNaker2, and the Rust/FPGA backends.
- **Delay timing in combination with recurrence**, and Upsample (`#202/#203`).
- **GPU or MPS execution.** Every run here is on CPU.
- **Conventions beyond those listed:** a framework may match the reference for a reason outside the search space. Only the direct checks above are claimed as findings.
