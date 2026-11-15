"""Compare framework outputs with the reference and explain each mismatch.

    python -m nirconf.compare <fixtures_dir> <results_dir>

Writes results/results.json, results/matrix.md and results/report.html.

Verdicts per (framework tag, suite, fixture):

* ``pass``     output equals the reference (spikes exactly; analog values to
               rtol 1e-4 / atol 1e-5 * max|ref|, i.e. float32 precision)
* ``mismatch`` ran, but the output differs. The *fingerprint* is the smallest
               set (up to 3) of convention changes under which the reference
               reproduces the framework output exactly.
* ``shape``    ran, but the output shape is wrong
* ``error``    the framework's importer or the imported model raised
* ``read``     ``nir.read`` (the framework's pinned NIR version) could not read
               the file, nor its nir-1.0.4 legacy copy
* ``flaky``    repeated runs of the same file gave different outcomes (some
               raised, some ran); the verdict of the first successful run is
               kept in ``ok_verdict``
"""

from __future__ import annotations

import dataclasses
import html
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import nir
import numpy as np

from .reference import REFERENCE, Convention, simulate

MAX_DEVIATIONS = 4


def _variants(graph: nir.NIRGraph, dt: float) -> Dict[str, list]:
    dts = {1.0 / dt, 1e-3 / dt, 1e-4 / dt}
    # Rockpool infers dt = min(tau / (1 + r)) from the graph
    for n in graph.nodes.values():
        tau = getattr(n, "tau", getattr(n, "tau_mem", None))
        if tau is not None and hasattr(n, "r"):
            dts.add(float(np.min(np.asarray(tau) / (1 + np.asarray(n.r)))) / dt)
    return {
        "decay": ["exact"],
        "input_gain": ["unit", "unit_no_r"],
        "reset": ["zero", "subtract", "none"],
        "compare": ["ge"],
        "multispike": [True],
        "ignore_v_leak": [True],
        "ignore_r": [True],
        "dt_scale": sorted(s for s in dts if abs(s - 1.0) > 1e-9),
        "if_min_v": ["neg_threshold"],
        "synapse_v_leak": [True],
        "init_v": ["v_leak"],
        "if_beta": [0.9],
    }


def _drop_node(graph: nir.NIRGraph, key: str) -> Optional[nir.NIRGraph]:
    """The graph with node ``key`` replaced by an identity (what a framework
    produces when its importer returns None and the loader skips the node)."""
    import copy

    node = graph.nodes[key]
    it, ot = node.input_type.get("input"), node.output_type.get("output")
    if it is None or ot is None or list(np.asarray(it)) != list(np.asarray(ot)):
        return None  # shape-changing node: dropping it changes the output shape
    g = copy.deepcopy(graph)
    g.nodes[key] = nir.Scale(np.ones(tuple(int(v) for v in np.asarray(it)), np.float32))
    return g


def _drop_recurrent(graph: nir.NIRGraph) -> Optional[nir.NIRGraph]:
    import copy

    from .reference import _order

    _, back = _order(graph)
    if not back:
        return None
    g = copy.deepcopy(graph)
    g.edges = [e for e in g.edges if tuple(e) not in back]
    return g


def matches(y: np.ndarray, ref: np.ndarray, spiking: bool) -> bool:
    if y.shape != ref.shape:
        return False
    if spiking:
        return bool(np.allclose(y, ref, rtol=0, atol=1e-6))
    scale = max(1e-12, float(np.max(np.abs(ref))))
    return bool(np.allclose(y, ref, rtol=1e-4, atol=1e-5 * scale))


def _shifted_match(y, r, k, spiking):
    return r.shape == y.shape and y.shape[0] > k and not np.any(y[:k]) and matches(y[k:], r[:-k], spiking)


def fingerprint(graph: nir.NIRGraph, x: np.ndarray, dt: float, y: np.ndarray, spiking: bool) -> Optional[dict]:
    """Smallest set of changes under which the reference reproduces ``y``.

    Changes are: a convention switch (``Convention`` fields), a structural
    change (a node silently dropped / recurrent edges dropped) and an output
    delay of k steps (spikes reported k steps late). Switches with no effect on
    this fixture are pruned. Returns None if nothing within MAX_DEVIATIONS
    changes reproduces ``y``."""
    base = simulate(graph, x, dt)
    structs: Dict[str, nir.NIRGraph] = {}
    for k, n in graph.nodes.items():
        if isinstance(n, (nir.Input, nir.Output)):
            continue
        g2 = _drop_node(graph, k)
        if g2 is not None:
            structs[f"node '{k}' ({type(n).__name__}) silently dropped"] = g2
    g2 = _drop_recurrent(graph)
    if g2 is not None:
        structs["recurrent edge dropped"] = g2
    space = _variants(graph, dt)
    # prune switches that do not change the reference output on this fixture
    dims: Dict[str, list] = {}
    for d, vals in space.items():
        keep = []
        for v in vals:
            try:
                r = simulate(graph, x, dt, dataclasses.replace(REFERENCE, **{d: v}))
            except Exception:  # noqa: BLE001
                continue
            if r.shape != base.shape or not np.array_equal(r, base):
                keep.append(v)
        if keep:
            dims[d] = keep
    dims["_struct"] = list(structs)
    dims["_delay"] = [1, 2, 3]
    if not dims["_struct"]:
        del dims["_struct"]
    cache: Dict[tuple, Optional[np.ndarray]] = {}

    def sim(struct: Optional[str], c: Convention):
        key = (struct, c)
        if key not in cache:
            try:
                cache[key] = simulate(structs[struct] if struct else graph, x, dt, c)
            except Exception:  # noqa: BLE001
                cache[key] = None
        return cache[key]

    names = list(dims)
    for level in range(1, MAX_DEVIATIONS + 1):
        found: List[str] = []
        for combo in itertools.combinations(names, level):
            for values in itertools.product(*(dims[d] for d in combo)):
                assign = dict(zip(combo, values))
                struct = assign.pop("_struct", None)
                delay = assign.pop("_delay", 0)
                c = dataclasses.replace(REFERENCE, **assign)
                r = sim(struct, c)
                if r is None:
                    continue
                ok = _shifted_match(y, r, delay, spiking) if delay else matches(y, r, spiking)
                if ok:
                    parts = ([struct] if struct else []) + ([f"output delayed {delay} step(s)"] if delay else []) + ([c.describe()] if c != REFERENCE else [])
                    found.append(", ".join(parts))
        if found:
            uniq = sorted(set(found), key=lambda s: (0 if "dropped" in s else 1, s.count(","), s))
            return {"deviations": level, "explanations": uniq[:6]}
    return None


def evaluate(fixtures: Path, results: Path) -> dict:
    index = json.loads((fixtures / "index.json").read_text())
    tags = sorted(p.name for p in (results / "raw").iterdir() if (p / "status.json").exists())
    out = {"fixtures": [], "tags": [], "cells": []}
    metas = {}
    for item in index:
        d = fixtures / item["suite"] / item["name"]
        metas[(item["suite"], item["name"])] = json.loads((d / "meta.json").read_text())
    out["fixtures"] = list(metas.values())
    for tag in tags:
        st = json.loads((results / "raw" / tag / "status.json").read_text())
        out["tags"].append({"tag": tag, "capabilities": st.get("capabilities", {}), "framework_version": st["records"][0].get("framework_version") if st["records"] else None})
        for rec in st["records"]:
            key = (rec["suite"], rec["name"])
            meta = metas[key]
            d = fixtures / rec["suite"] / rec["name"]
            cell = {"tag": tag, "suite": rec["suite"], "name": rec["name"], "file": rec.get("file"), "read_error_current": rec.get("read_error_current"), "notes": rec.get("notes", [])}
            if rec["stage"] == "read":
                cell.update(verdict="read", detail=rec["error"])
            elif rec["stage"] == "run":
                cell.update(verdict="error", detail=rec["error"])
            else:
                y = np.load(results / "raw" / tag / rec["suite"] / f"{rec['name']}.npy")
                ref = np.load(d / "expected.npy")
                if y.shape != ref.shape:
                    cell.update(verdict="shape", detail=f"output shape {list(y.shape)}, expected {list(ref.shape)}")
                elif matches(y, ref, meta["spiking_output"]):
                    cell.update(verdict="pass", detail="")
                else:
                    graph = nir.read(d / "graph.nir")
                    x = np.load(d / "input.npy")
                    if meta["spiking_output"]:
                        diff = np.abs(y - ref) > 1e-6
                        frac = float(np.mean(diff))
                        first = int(np.argmax(diff.reshape(diff.shape[0], -1).any(axis=1)))
                        metric = {"mismatch_fraction": frac, "first_divergent_step": first, "spikes_framework": float(np.sum(y)), "spikes_reference": float(np.sum(ref))}
                        detail = f"{100 * frac:.1f}% of entries differ (first at step {first}); spikes {int(np.sum(y))} vs {int(np.sum(ref))} reference"
                    else:
                        rel = float(np.max(np.abs(y - ref)) / max(1e-12, np.max(np.abs(ref))))
                        metric = {"max_rel_error": rel}
                        detail = f"max error {rel:.3g} x max|ref|"
                    fp = fingerprint(graph, x, meta["dt"], y, meta["spiking_output"])
                    cell.update(verdict="mismatch", detail=detail, metrics=metric, fingerprint=fp)
            if rec.get("flaky"):
                cell["ok_verdict"] = cell["verdict"]
                cell["verdict"] = "flaky"
                cell["outcomes"] = rec["outcomes"]  # counts vary run to run; kept out of the markdown
                cell["detail"] = f"repeated runs of the same file: some raised, some ran; first successful run: {cell['ok_verdict']} {cell.get('detail', '')}"
            out["cells"].append(cell)
            print(f"{tag:22s} {rec['suite']:7s} {rec['name']:20s} {cell['verdict']:8s} {cell.get('fingerprint', {}) and cell['fingerprint'].get('explanations', [''])[0] if cell.get('fingerprint') else cell.get('detail', '')[:90]}")
    return out


# ----------------------------------------------------------------------------- rendering

SYMBOL = {"pass": "PASS", "mismatch": "DIFF", "shape": "SHAPE", "error": "ERR", "read": "READ", "flaky": "FLAKY"}


def _cell_text(c: dict) -> str:
    s = SYMBOL[c["verdict"]]
    if c["verdict"] == "mismatch" and c.get("metrics", {}).get("mismatch_fraction") is not None:
        s += f" {100 * c['metrics']['mismatch_fraction']:.0f}%"
    if c.get("file") == "graph_legacy.nir":
        s += "ᴸ"
    return s


def render_markdown(res: dict) -> str:
    tags = [t["tag"] for t in res["tags"]]
    lines = ["# NIR conformance matrix", ""]
    lines.append("Generated by `python -m nirconf.compare`. PASS = identical to the reference; DIFF n% = ran, n% of output entries differ; "
                 "SHAPE = wrong output shape; ERR = import or run raised; READ = this NIR version cannot read the file; "
                 "FLAKY = repeated runs of the same file gave different outcomes (finding NT-3). "
                 "ᴸ = read only the nir-1.0.4 legacy copy (the current file was unreadable).")
    lines.append("")
    summary = {t: {} for t in tags}
    for c in res["cells"]:
        summary[c["tag"]][c["verdict"]] = summary[c["tag"]].get(c["verdict"], 0) + 1
    lines += ["## Summary (68 fixtures per framework)", "", "| framework | PASS | DIFF | SHAPE | ERR | READ | FLAKY |", "|---|---|---|---|---|---|---|"]
    for t in tags:
        s = summary[t]
        lines.append(f"| {t} | {s.get('pass', 0)} | {s.get('mismatch', 0)} | {s.get('shape', 0)} | {s.get('error', 0)} | {s.get('read', 0)} | {s.get('flaky', 0)} |")
    lines.append("")
    by = {(c["tag"], c["suite"], c["name"]): c for c in res["cells"]}
    names = []
    for f in res["fixtures"]:
        if f["name"] not in names:
            names.append(f["name"])
    for suite in ["dt1", "dt1e-4"]:
        lines += [f"## Suite `{suite}`", "", "| fixture | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
        for n in names:
            row = [n] + [_cell_text(by[(t, suite, n)]) if (t, suite, n) in by else "" for t in tags]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines += ["## Mismatch explanations", "", "Smallest set of convention changes (see docs/conventions.md) under which the reference reproduces the framework output exactly.", ""]
    lines += ["| framework | suite | fixture | what differs | explanation |", "|---|---|---|---|---|"]
    for c in res["cells"]:
        if c["verdict"] == "mismatch":
            fp = c.get("fingerprint")
            expl = "; ".join(fp["explanations"][:3]) if fp else f"not explained within {MAX_DEVIATIONS} changes"
            lines.append(f"| {c['tag']} | {c['suite']} | {c['name']} | {c['detail']} | {expl} |")
    lines += ["", "## Errors", "", "| framework | suite | fixture | verdict | message |", "|---|---|---|---|---|"]
    for c in res["cells"]:
        if c["verdict"] in ("error", "read", "shape", "flaky"):
            lines.append(f"| {c['tag']} | {c['suite']} | {c['name']} | {c['verdict']} | {c['detail'].replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def render_html(res: dict) -> str:
    tags = [t["tag"] for t in res["tags"]]
    by = {(c["tag"], c["suite"], c["name"]): c for c in res["cells"]}
    names = []
    for f in res["fixtures"]:
        if f["name"] not in names:
            names.append(f["name"])
    desc = {f["name"]: f["description"] for f in res["fixtures"]}
    css = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--muted:#6b6b70;--line:#e4e4e2;--pass:#1f7a45;--passbg:#e3f3e8;--diff:#9a5b00;--diffbg:#fbefd9;--err:#b3261e;--errbg:#fbe4e2;--read:#5b4bb7;--readbg:#ebe7fb}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#151517;--fg:#ececee;--muted:#9b9ba1;--line:#2c2c30;--pass:#6fd39a;--passbg:#173524;--diff:#f2b35b;--diffbg:#3a2a10;--err:#ff8a80;--errbg:#3b1715;--read:#b5a8ff;--readbg:#262046}}
:root[data-theme="dark"]{--bg:#151517;--fg:#ececee;--muted:#9b9ba1;--line:#2c2c30;--pass:#6fd39a;--passbg:#173524;--diff:#f2b35b;--diffbg:#3a2a10;--err:#ff8a80;--errbg:#3b1715;--read:#b5a8ff;--readbg:#262046}
body{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;padding:24px 16px}
main{max-width:1200px;margin:0 auto}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px}p{color:var(--muted);margin:4px 0 12px}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{border-bottom:1px solid var(--line);padding:5px 6px;text-align:left;vertical-align:top}
th{font-weight:600;position:sticky;top:0;background:var(--bg)}td.c{white-space:nowrap;text-align:center;font-variant-numeric:tabular-nums}
.pass{color:var(--pass);background:var(--passbg)}.mismatch{color:var(--diff);background:var(--diffbg)}.error,.shape,.flaky{color:var(--err);background:var(--errbg)}.read{color:var(--read);background:var(--readbg)}
code{font-size:12px}.name{white-space:nowrap}
"""
    parts = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>NIR conformance matrix</title><style>{css}</style></head><body><main>",
             "<h1>NIR conformance matrix</h1><p>Each cell: one NIR fixture run through one framework's NIR importer, compared with a float64 reference simulator. Hover a cell for details. ᴸ = only the nir-1.0.4 legacy copy could be read.</p>"]
    for suite in ["dt1", "dt1e-4"]:
        parts.append(f"<h2>Suite <code>{suite}</code></h2><div class=wrap><table><tr><th>fixture</th>" + "".join(f"<th>{html.escape(t)}</th>" for t in tags) + "</tr>")
        for n in names:
            parts.append(f"<tr><td class=name title='{html.escape(desc[n])}'>{html.escape(n)}</td>")
            for t in tags:
                c = by.get((t, suite, n))
                if not c:
                    parts.append("<td></td>")
                    continue
                tip = c.get("detail", "")
                if c.get("fingerprint"):
                    tip += " | explained by: " + "; ".join(c["fingerprint"]["explanations"][:3])
                elif c["verdict"] == "mismatch":
                    tip += f" | not explained within {MAX_DEVIATIONS} changes"
                parts.append(f"<td class='c {c['verdict']}' title='{html.escape(tip)}'>{html.escape(_cell_text(c))}</td>")
            parts.append("</tr>")
        parts.append("</table></div>")
    parts.append("</main></body></html>")
    return "".join(parts)


def main(fixtures: Path, results: Path) -> None:
    res = evaluate(fixtures, results)
    (results / "results.json").write_text(json.dumps(res, indent=1, default=str))
    (results / "matrix.md").write_text(render_markdown(res))
    (results / "matrix.html").write_text(render_html(res))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
