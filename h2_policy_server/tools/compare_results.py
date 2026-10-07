#!/usr/bin/env python3
"""Print one comparison table from the JSON results of several evaluation runs.

Every backend writes the same scalar keys (``h2_policy_if.evaluation.METRIC_KEYS``),
so a cross-simulator comparison is a table join rather than a re-measurement.

Results from different profiles are shown as separate rows and never averaged:
a ``backend-recommended`` number is a different experiment from a
``strict-native`` one, not a better estimate of the same thing.

    python tools/compare_results.py results/*.json
    python tools/compare_results.py results/*.json --csv summary.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys

COLUMNS = [
    ("backend", "backend", 16),
    ("profile", "profile", 20),
    ("motion", "motion", 28),
    ("survival_s", "surv [s]", 9),
    ("termination_reason", "end", 10),
    ("pelvis_height_min", "pelvis min", 11),
    ("pelvis_height_final", "pelvis fin", 11),
    ("root_xy_displacement", "xy [m]", 8),
    ("joint_vel_abs_max", "|qd|max", 9),
    ("torque_abs_max", "|tau|max", 9),
    ("torque_saturation_rate", "sat", 7),
    ("foot_contact_rate", "contact", 8),
    ("nan", "NaN", 5),
    ("policy_rtt_ms_median", "rtt med", 8),
]


def load(paths: list[str]) -> list[dict]:
    """Read result files, flattening the fields the table needs.

    Args:
        paths: Result JSON paths; globs are expanded.

    Returns:
        One row dict per readable result, with provenance fields merged in.
    """
    rows = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            try:
                with open(path) as fh:
                    doc = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[skip] {path}: {exc}", file=sys.stderr)
                continue
            prov, met = doc.get("provenance", {}), dict(doc.get("metrics", {}))
            met.setdefault("backend", prov.get("backend", {}).get("name", "?"))
            met.setdefault("profile", prov.get("profile", {}).get("name", "?"))
            met["_path"] = path
            met["_solver_overrides"] = {
                k: v for k, v in prov.get("solver_overrides", {}).items() if not k.startswith("_")
            }
            met["_policy"] = prov.get("policy", {}).get("decoder_sha256", "")[:12]
            met["_wbc"] = prov.get("wbc_h12", {}).get("git_head", "")[:12]
            met["_grade"] = prov.get("provenance_grade", "?")
            met["_asym"] = prov.get("simulator_asymmetries", {})
            met["_perturb"] = prov.get("perturbations", {})
            rows.append(met)
    return rows


def fmt(value) -> str:
    """Render a metric for the table."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", help="result JSON paths or globs")
    parser.add_argument("--csv", default=None, help="also write a CSV summary here")
    args = parser.parse_args()

    rows = load(args.results)
    if not rows:
        raise SystemExit("no readable results")

    policies = {r["_policy"] for r in rows}
    wbcs = {r["_wbc"] for r in rows}
    print(f"policy decoder sha256[:12] : {', '.join(sorted(policies))}"
          + ("   *** rows do not share one policy ***" if len(policies) > 1 else ""))
    print(f"wbc_h12 HEAD[:12]          : {', '.join(sorted(wbcs))}")
    print(f"provenance grade           : {', '.join(sorted({r['_grade'] for r in rows}))}")
    print()

    header = "  ".join(f"{title:<{w}}" for _, title, w in COLUMNS)
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: (r.get("backend", ""), r.get("profile", ""))):
        print("  ".join(f"{fmt(row.get(key)):<{w}}" for key, _, w in COLUMNS))

    perturbed = [r for r in rows if r["_perturb"]]
    if perturbed:
        print()
        print("*** These rows carry a perturbation and are NOT strict baselines: ***")
        for row in perturbed:
            spec = ", ".join(f"{a}={d['value']} {d['grade']}" for a, d in row["_perturb"].items())
            print(f"  {row['backend']:<16} {row['motion']:<28} {spec}")

    asym = {}
    for row in rows:
        for name, rec in row["_asym"].items():
            if rec.get("status") == "open" and (
                not rec["affects"] or row.get("backend") in rec["affects"]
            ):
                asym.setdefault(name, {"rec": rec, "backends": set()})["backends"].add(row["backend"])
    if asym:
        print()
        print("Open simulator asymmetries touching these rows (recorded, not patched):")
        for name, entry in sorted(asym.items()):
            print(f"  {name}  [{', '.join(sorted(entry['backends']))}]")
            print(f"      {entry['rec']['resolution']}")

    tuned = [r for r in rows if r["_solver_overrides"]]
    if tuned:
        print()
        print("Non-default numerical solver settings (NUMERICAL grade, no training provenance):")
        for row in tuned:
            print(f"  {row['backend']} / {row['profile']}")
            for key, value in row["_solver_overrides"].items():
                print(f"      {key} = {value}")

    if args.csv:
        keys = [key for key, _, _ in COLUMNS]
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
