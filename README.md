# NIR Conformance Suite

[![conformance](https://github.com/rifatsarwarcodes/NIR_Conformance/actions/workflows/conformance.yml/badge.svg)](https://github.com/rifatsarwarcodes/NIR_Conformance/actions/workflows/conformance.yml)

The Neuromorphic Intermediate Representation (NIR) is the file format SNN frameworks use to exchange models. This suite checks whether five frameworks give the same answer when they import the same `.nir` file. Each framework's output is compared with a slow, readable float64 reference simulator, and the suite explains every disagreement it finds.

**Status (2026-11-16):** working prototype. The full pipeline runs in about 2 minutes on an M4 CPU.
- 68 fixtures × 7 framework/NIR-version combinations, compared.
- 36 findings, each confirmed by a direct check.
- Upstream reports are drafted but **not posted** (see `upstream/`).

## Result in one table

| Framework | PASS | Ran but differs (no error) | Raised | Unreadable file | Flaky |
|---|---|---|---|---|---|
| Norse 1.1.0 (nir 1.0.8) | 31 | 25 | 8 | 0 | 4 |
| snnTorch 1.0.0 (nir 1.0.8) | 12 | 21 | 35 | 0 | 0 |
| Spyx 1.0.0 (nir 1.0.8) | 4 | 60 | 4 | 0 | 0 |
| Sinabs 3.1.3 (nir 1.0.4) | 2 | 14 | 44 | 6 | 2 |
| Rockpool 3.1.0 (nir 1.0.4) | 4 | 12 | 46 | 6 | 0 |

(Out of 68 fixtures per framework. The full matrix, including snnTorch and Norse under nir 1.0.4, is in [results/matrix.md](results/matrix.md) and [results/matrix.html](results/matrix.html).)

The most important findings (all in [docs/FINDINGS.md](docs/FINDINGS.md)):

1. **Silent wrong results are the main failure mode.** nirtorch's legacy loader skips nodes it can't map and returns the wrong node for fan-in graphs. With an unsupported node present, the same file sometimes raises and sometimes runs. Norse, Sinabs and Rockpool use this loader. Spyx imports only the first branch of a graph and drops Scale and Delay. Several importers ignore `r`, `v_leak` or `v_reset`.
2. **Current NIR files cannot be read by Sinabs or Rockpool.** nir ≥ 1.0.6 always writes `v_reset`, and both frameworks pin `nir<=1.0.4`, which rejects it.
3. **Concrete importer bugs**, each with a minimal repro in `upstream/repros/`:
   - Rockpool can't import any `nir.LIF` (it reads `tau_mem`, but the field is `tau`).
   - snnTorch imports `nir.IF` as a leaky neuron and hardcodes `dt = 1e-4`.
   - Norse gives CubaLIF's synaptic current the membrane's `v_leak`.
   - Spyx's Flatten uses channels-last order.
   - NIR's own grouped-conv `input_type` ignores `groups`.
4. **Seven places where the NIR spec leaves the answer open:** dt, IF input meaning, reset rule, initial state, spike timing, recurrent timing, and unsupported nodes. They are listed in [docs/conventions.md](docs/conventions.md) as input for NIR issue #208.

## How it works

```
fixtures.py        34 graphs x 2 time scales -> fixtures/<suite>/<name>/{graph.nir, graph_legacy.nir, input.npy, expected.npy, meta.json}
reference.py       float64 forward-Euler simulator with switchable conventions (the source of expected.npy)
adapters.py        one function per framework: its public NIR importer + documented calling convention, nothing else
run_framework.py   run all fixtures through one framework in the current environment -> results/raw/<fw>@nir<ver>/
compare.py         verdict per cell + "fingerprint": smallest set of convention / structural changes that reproduces the output
verify_findings.py one direct check per claimed finding (invariance test, module inspection, or minimal graph)
```

- **Fixtures are hard to pass by accident.** Weights are tuned so outputs fire on 10–35% of entries, and no membrane comes within 1e-4 (relative) of its threshold. Float32 frameworks must therefore match spikes exactly.
- **Two suites.**
  - `dt1`: dt = 1, τ = 20 steps.
  - `dt1e-4`: dt = 1e-4, τ = 2 ms.
  Both have the same dimensionless dynamics, so a dt-handling problem shows up as a pass in one suite and a fail in the other.
- **Two environments**, because Sinabs and Rockpool's NIR extra pin `nir<=1.0.4`:
  - `.venv-latest`: nir 1.0.8, nirtorch 2.6, snnTorch, Norse, Spyx.
  - `.venv-pinned`: nir 1.0.4, nirtorch 2.0.2, all five frameworks.
- **Legacy copies.** `graph_legacy.nir` is written so the pinned readers can still be tested. It is only written when it doesn't change the graph's meaning: an all-zero `v_reset` is dropped. Otherwise the cell is marked READ.
- **The fingerprint only suggests why a framework differs.** Only checks in `verify_findings.py` are claimed as findings.
- **Repeats for unstable frameworks.** Because of finding NT-3, Norse, Sinabs and Rockpool run each fixture 30 times, and cells whose outcomes differ are marked FLAKY. The verdicts are reproducible: `matrix.md` was byte-identical across full reruns.

## Run it

```bash
# one-time setup (uv, Python 3.11): two environments with the exact versions tested
uv venv --python 3.11 .venv-latest && uv pip install --python .venv-latest/bin/python -r requirements-latest.txt -e .
uv venv --python 3.11 .venv-pinned && uv pip install --python .venv-pinned/bin/python -r requirements-pinned.txt -e .

./run_all.sh        # tests -> fixtures -> 7 framework runs -> compare -> verify (about 2 minutes on an M4)
```

`requirements-latest.txt` holds nir 1.0.8, nirtorch 2.6, snnTorch, Norse and Spyx. `requirements-pinned.txt` holds nir 1.0.4 and nirtorch 2.0.2, plus Sinabs and Rockpool, which need that NIR version, and snnTorch and Norse again for comparison. A fresh clone set up this way reproduced the committed `results/matrix.md` byte for byte (macOS arm64, 2026-11-16). `results.json` and `results/verified/` also store repeat counts for flaky cells, and those vary from run to run by design.

CI ([.github/workflows/conformance.yml](.github/workflows/conformance.yml)) runs the full pipeline on Ubuntu on every push and every Monday, in two jobs:

| Job | Versions | What makes it fail |
|---|---|---|
| `tested` | The pinned requirements | The pipeline breaks, or any finding in `docs/FINDINGS.md` stops reproducing |
| `newest` | The same packages unpinned | Never fails the build; it is an early warning when a framework release changes behaviour |

Both jobs upload the matrix and evidence as artifacts.

On Linux, add `--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match` to get CPU-only torch wheels.

The disk cost is mostly the two virtual environments, about 1.1 GB each. Fixtures and results together are about 14 MB.

## Layout

```
README.md            this file
run_all.sh           full pipeline
pyproject.toml       package `nirconf`
src/nirconf/         reference, fixtures, adapters, runner, compare, verify_findings, probe
tests/               17 checks of the reference simulator against closed forms and torch conv/pool
fixtures/            generated (deterministic; checked: byte-identical across regenerations)
results/
  matrix.md, matrix.html, results.json   conformance matrix + fingerprints
  verified/nir1.0.8.json, nir1.0.4.json  evidence for every finding
  raw/, logs/                            per-framework outputs and logs
docs/
  FINDINGS.md        all findings, labelled bug / silent / gap / ambiguity
  conventions.md     the reference convention and the open spec questions (A1-A7)
upstream/            DRAFT issue texts + minimal repros (not posted)
```

## Limitations

- **Import only.** Export and round trips are not tested yet. The Rockpool export and import formulas for dt don't look like inverses.
- **Five frameworks.** SpikingJelly 2.0, Nengo, hxtorch/jaxsnn and SpiNNaker2 are not covered yet. Lava-dl was archived in May 2026.
- **The reference convention is a choice.** Where the spec is silent, a framework can be marked DIFF without being wrong. `FINDINGS.md` labels those *ambiguity*.
- **CPU only.**

## License

MIT. See [LICENSE](LICENSE).

## Next steps

1. Review the drafts in `upstream/`, then post the #208 comment and agree on the fixture layout and conventions with the maintainers.
2. File the clear bugs: NIR-2, NT-1/NT-2, RP-1, ST-2, NO-6 and SX-3/SX-5.
3. Add export round-trip tests and SpikingJelly 2.0. Then add the Rust runtime from project 02 as another backend.
