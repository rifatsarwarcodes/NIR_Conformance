"""Run every fixture through one framework and store raw outputs.

    python -m nirconf.run_framework <framework> <fixtures_dir> <results_dir>

Writes ``<results_dir>/raw/<framework>@nir<version>/<suite>/<fixture>.npy``
and a ``status.json`` with one record per fixture: stage reached
(read / import+run), error type and message, wall time.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Frameworks built on nirtorch's legacy loader give run-to-run different
# results for some graphs (finding NT-3), so each fixture is repeated and
# a cell whose outcomes differ is marked flaky. With 30 repeats, a cell that
# fails 15% of the time is missed with probability 0.85**30 = 0.8%.
REPEATS = {"norse": 30, "sinabs": 30, "rockpool": 30}


def _short_error(e: BaseException) -> str:
    tb = traceback.extract_tb(e.__traceback__)
    where = ""
    for fr in reversed(tb):
        if "site-packages" in fr.filename and "nirconf" not in fr.filename:
            pkg = fr.filename.split("site-packages/")[-1]
            where = f" [{pkg}:{fr.lineno}]"
            break
    msg = str(e).strip().splitlines()[0] if str(e).strip() else ""
    return f"{type(e).__name__}: {msg[:300]}{where}"


def main(fw: str, fixtures: Path, results: Path) -> None:
    import nir

    from . import adapters

    nir_version = getattr(nir, "__version__", "unknown")
    tag = f"{fw}@nir{nir_version}"
    out_root = results / "raw" / tag
    out_root.mkdir(parents=True, exist_ok=True)
    index = json.loads((fixtures / "index.json").read_text())
    records = []
    for item in index:
        suite, name = item["suite"], item["name"]
        d = fixtures / suite / name
        meta = json.loads((d / "meta.json").read_text())
        rec = {"suite": suite, "name": name, "framework": fw, "framework_version": adapters.framework_version(fw), "nir_version": nir_version}
        t0 = time.time()
        sink = io.StringIO()
        try:
            graph = nir.read(d / "graph.nir")
            rec["file"] = "graph.nir"
        except Exception as e:  # noqa: BLE001
            rec["read_error_current"] = _short_error(e)
            legacy = d / "graph_legacy.nir"
            if not legacy.exists():
                rec.update(stage="read", error=rec["read_error_current"], legacy=meta.get("legacy_file"))
                records.append(rec)
                print(f"{suite:7s} {name:20s} READ-ERROR {rec['error'][:110]}")
                continue
            try:
                graph = nir.read(legacy)
                rec["file"] = "graph_legacy.nir"
            except Exception as e2:  # noqa: BLE001
                rec.update(stage="read", error=_short_error(e2))
                records.append(rec)
                print(f"{suite:7s} {name:20s} READ-ERROR(legacy) {rec['error'][:100]}")
                continue
        x = np.load(d / "input.npy")
        outcomes: dict = {}
        first_ok, first_err, notes = None, None, []
        for _ in range(REPEATS.get(fw, 1)):
            g = copy.deepcopy(graph)
            try:
                with contextlib.redirect_stdout(sink):
                    y = np.asarray(adapters.run(fw, g, x, meta["dt"]), dtype=np.float64)
                key = "ok:" + hashlib.sha1(np.ascontiguousarray(y).tobytes()).hexdigest()[:10]
                if first_ok is None:
                    first_ok, notes = y, sorted(adapters.NOTES.get("current", []))
            except Exception as e:  # noqa: BLE001
                key = "error:" + type(e).__name__
                first_err = first_err or _short_error(e)
            outcomes[key] = outcomes.get(key, 0) + 1
        rec["outcomes"] = outcomes
        rec["flaky"] = len(outcomes) > 1
        if first_ok is not None:
            (out_root / suite).mkdir(exist_ok=True)
            np.save(out_root / suite / f"{name}.npy", first_ok)
            rec.update(stage="ok", error=first_err, output_shape=list(first_ok.shape), notes=notes)
            print(f"{suite:7s} {name:20s} ok {list(first_ok.shape)}{'  FLAKY ' + str(outcomes) if rec['flaky'] else ''}")
        else:
            rec.update(stage="run", error=first_err)
            print(f"{suite:7s} {name:20s} ERROR {rec['error'][:110]}")
        rec["seconds"] = round(time.time() - t0, 3)
        rec["stdout_lines"] = len(sink.getvalue().splitlines())
        records.append(rec)
    (out_root / "status.json").write_text(json.dumps({"framework": fw, "tag": tag, "capabilities": adapters.CAPABILITIES.get(fw, {}), "records": records}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
