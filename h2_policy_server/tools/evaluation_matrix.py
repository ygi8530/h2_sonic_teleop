"""Config-driven batch evaluation shared by both simulators' entrypoints.

One entrypoint per simulator expands the same kind of matrix -- motions x
profiles x engines x terrain x perturbation conditions -- so the expansion,
progress reporting and summary writing live here and each entrypoint only
supplies its own defaults and command builder.

Runs are sequential by default. A physics engine that shares a GPU with another
instance of itself does not reliably give the same answer twice, and the whole
point of this harness is that a number can be reproduced, so throughput is not
traded for that.

Grades are never mixed: every row carries the grade of its condition, and the
summary reports strict baselines, TRAIN_DIST and OOD_SIM2REAL separately.
"""

from __future__ import annotations

import csv
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

_HERE = Path(__file__).resolve().parent
_EVAL = _HERE.parent / "h2_policy_if" / "evaluation.py"
_spec = importlib.util.spec_from_file_location("h2_eval_standalone", _EVAL)
ev = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ev
_spec.loader.exec_module(ev)


@dataclasses.dataclass(frozen=True)
class Condition:
    """One parameter condition in the matrix.

    Args:
        name: Short identifier that appears in filenames and the summary.
        perturb: ``AXIS=VALUE`` strings applied to the run.
        push: Whether to replay training's push event.
        grade: ``STRICT_BASELINE``, ``TRAIN_DIST`` or ``OOD_SIM2REAL``.
    """

    name: str
    perturb: tuple = ()
    push: bool = False
    grade: str = "STRICT_BASELINE"


NOMINAL = Condition(name="nominal")


def load_config(path: str) -> dict:
    """Load a YAML or JSON evaluation config.

    Args:
        path: Config file. ``.json`` is parsed as JSON, anything else as YAML.

    Returns:
        The parsed mapping.
    """
    text = Path(path).read_text()
    if path.endswith(".json"):
        return json.loads(text)
    import yaml  # noqa: PLC0415

    return yaml.safe_load(text)


def expand_conditions(cfg: dict) -> list[Condition]:
    """Turn the config's ``conditions`` block into a flat list.

    Three kinds are recognised, each switchable with ``enabled``:

    - ``nominal``: the unperturbed strict baseline.
    - ``sweeps``: one axis walked across N points. TRAIN_DIST axes take their
      bounds from ``evaluation.TRAIN_RANDOMIZATION`` unless the config overrides
      them; an OOD axis has no training range, so the config must give one.
    - ``combined``: explicit multi-axis conditions, spelled out by the operator.

    Args:
        cfg: The parsed config.

    Returns:
        The conditions to run, nominal first.

    Raises:
        SystemExit: When an OOD sweep is declared without an explicit range,
            which would mean inventing one.
    """
    block = cfg.get("conditions", {}) or {}
    out: list[Condition] = []
    if block.get("nominal", {}).get("enabled", True):
        out.append(NOMINAL)

    for sweep in block.get("sweeps", []) or []:
        if not sweep.get("enabled", True):
            continue
        axis = sweep["axis"]
        grade = ev.axis_grade(axis)
        lo, hi = _sweep_range(axis, sweep, grade)
        for value in _grid(lo, hi, int(sweep.get("points", 3))):
            out.append(Condition(
                name=f"{axis}{value:g}",
                perturb=(f"{axis}={value}",),
                grade=grade,
            ))

    for combo in block.get("combined", []) or []:
        if not combo.get("enabled", True):
            continue
        perturb = tuple(combo.get("perturb", []))
        grades = {ev.axis_grade(p.split("=", 1)[0]) for p in perturb}
        out.append(Condition(
            name=combo["name"],
            perturb=perturb,
            push=bool(combo.get("push", False)),
            grade="OOD_SIM2REAL" if "OOD_SIM2REAL" in grades else
                  ("TRAIN_DIST" if grades or combo.get("push") else "STRICT_BASELINE"),
        ))

    if block.get("push", {}).get("enabled", False):
        out.append(Condition(name="push", push=True, grade="TRAIN_DIST"))
    return out


def _sweep_range(axis: str, sweep: dict, grade: str) -> tuple[float, float]:
    """Resolve a sweep's endpoints, preferring the training range."""
    if "range" in sweep:
        lo, hi = sweep["range"]
        return float(lo), float(hi)
    if grade == "OOD_SIM2REAL":
        raise SystemExit(
            f"sweep of {axis} needs an explicit 'range': it is OOD_SIM2REAL, so training "
            "provides no bounds and this harness will not invent any."
        )
    key = ev.AXIS_SUPPORT[ev._support_key(axis)]["train_key"]
    value = ev.TRAIN_RANDOMIZATION[key].value
    lo, hi = value[axis[-1]] if isinstance(value, dict) else value
    return float(lo), float(hi)


def _grid(lo: float, hi: float, points: int) -> list[float]:
    """Evenly spaced values including both endpoints."""
    if points < 2:
        return [round((lo + hi) / 2.0, 6)]
    step = (hi - lo) / (points - 1)
    return [round(lo + i * step, 6) for i in range(points)]


def slug(text: str) -> str:
    """Make a string safe for a filename without losing what it identifies."""
    return "".join(c if c.isalnum() or c in "-._" else "-" for c in str(text))


class Progress:
    """Console progress with a remaining-time estimate.

    The estimate is the mean duration of the runs finished so far, which is the
    honest predictor here: every run is the same length and the same work, so
    the mean stabilises after a couple of rows.

    Args:
        total: Number of runs in the matrix.
    """

    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.failed = 0
        self.start = time.time()

    def tick(self, label: str, outcome: str, seconds: float) -> None:
        """Record one finished run and print the progress line."""
        self.done += 1
        # A fall, a divergence or a NaN is a result, not a failure. Only a
        # rollout that produced no JSON at all counts against the batch.
        self.failed += outcome == "NO RESULT"
        elapsed = time.time() - self.start
        eta = (elapsed / self.done) * (self.total - self.done)
        bar_len = 24
        filled = int(bar_len * self.done / self.total)
        bar = "#" * filled + "." * (bar_len - filled)
        print(f"  [{bar}] {self.done:>3}/{self.total}  {_hms(elapsed)} elapsed, "
              f"~{_hms(eta)} left   {seconds:5.1f}s  {outcome:<10} {label}",
              flush=True)

    def finish(self) -> float:
        """Print the closing line and return the total wall time [s]."""
        elapsed = time.time() - self.start
        print(f"\n  {self.done} runs in {_hms(elapsed)}"
              + (f", {self.failed} produced no result" if self.failed else ""))
        return elapsed


def _hms(seconds: float) -> str:
    """Format a duration as ``H:MM:SS`` or ``M:SS``."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def run_one(argv: list[str], cwd: str, result_path: str, timeout: int) -> tuple[dict | None, str]:
    """Execute one rollout and read back its result.

    A non-zero exit is not treated as an error: both runners exit 1 when the
    robot falls, and a fall is a result, not a failure. What matters is whether
    the JSON was written.

    Args:
        argv: The runner command.
        cwd: Directory to launch it from.
        result_path: Where the runner was told to write its JSON, as seen from here.
        timeout: Seconds before the run is abandoned.

    Returns:
        ``(result, note)`` -- the parsed JSON or None, and a short explanation
        when it is None.
    """
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"timeout after {timeout}s"
    if not os.path.exists(result_path):
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:]
        return None, (tail[0][:160] if tail else f"no result, exit {proc.returncode}")
    with open(result_path) as fh:
        return json.load(fh), ""


#: Columns of summary.csv, in order. Every runner fills the same keys, which is
#: what lets one table hold every backend.
SUMMARY_COLUMNS = (
    "motion", "backend", "profile", "terrain", "condition", "grade", "seed",
    "survival_s", "terminated", "termination_reason", "fell_at_s",
    "pelvis_height_min", "pelvis_height_final", "root_xy_displacement",
    "joint_vel_abs_max", "torque_abs_max", "torque_saturation_rate",
    "foot_contact_rate", "nan", "diverged", "no_motion",
    "policy_rtt_ms_median", "physics_substeps_mean", "video", "result",
)


def write_summary(out_dir: Path, rows: list[dict], failures: list[dict]) -> None:
    """Write ``summary.csv`` and ``summary.md`` for a finished batch.

    Args:
        out_dir: The run directory.
        rows: One dict per successful rollout, keyed by :data:`SUMMARY_COLUMNS`.
        failures: Rollouts that produced no result, with a ``note``.
    """
    with open(out_dir / "summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"# Evaluation summary — {out_dir.name}", ""]
    lines += _criteria_table(rows)
    lines += _grade_sections(rows)
    if failures:
        lines += ["", "## Runs that produced no result", "",
                  "| condition | note |", "|---|---|"]
        lines += [f"| {f['label']} | {f['note']} |" for f in failures]
    lines += ["", "---", "",
              "Grades are never merged. A STRICT_BASELINE row is the policy under "
              "training-sourced parameters only; TRAIN_DIST rows sit inside the ranges "
              "SONIC training randomised over; OOD_SIM2REAL rows use axes training never "
              "randomised, so their values are the operator's choice and carry no "
              "training provenance.", ""]
    (out_dir / "summary.md").write_text("\n".join(lines))


def _criteria_table(rows: list[dict]) -> list[str]:
    """Build the pass/fail table: one row per condition, one column per backend."""
    if not rows:
        return ["No runs produced a result.", ""]
    backends = sorted({r["backend"] for r in rows})
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    out = ["## Evaluation criteria — survived / total", "",
           "| condition | grade | " + " | ".join(backends) + " |",
           "|---|---|" + "---|" * len(backends)]
    for cond in conditions:
        cells, grade = [], ""
        for backend in backends:
            sel = [r for r in rows if r["condition"] == cond and r["backend"] == backend]
            grade = sel[0]["grade"] if sel else grade
            if not sel:
                cells.append("—")
                continue
            ok = sum(r["termination_reason"] == "completed" for r in sel)
            mark = "**PASS**" if ok == len(sel) else ("FAIL" if ok == 0 else "partial")
            cells.append(f"{ok}/{len(sel)} {mark}")
        out.append(f"| {cond} | {grade} | " + " | ".join(cells) + " |")
    return out + [""]


def _grade_sections(rows: list[dict]) -> list[str]:
    """Build one detail table per grade, so grades cannot be read as one set."""
    out: list[str] = []
    for grade in ("STRICT_BASELINE", "TRAIN_DIST", "OOD_SIM2REAL"):
        sel = [r for r in rows if r["grade"] == grade]
        if not sel:
            continue
        out += [f"## {grade}", "",
                "| motion | backend | profile | terrain | condition | seed | surv [s] | end |"
                " pelvis min [m] | \\|qd\\|max [rad/s] | \\|tau\\|max [N·m] | sat | NaN |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(sel, key=lambda x: (x["motion"], x["backend"], x["condition"])):
            out.append(
                f"| {r['motion']} | {r['backend']} | {r['profile']} | {r['terrain']} | "
                f"{r['condition']} | {r['seed']} | {r['survival_s']} | "
                f"{r['termination_reason']} | {r['pelvis_height_min']} | "
                f"{r['joint_vel_abs_max']} | {r['torque_abs_max']} | "
                f"{r['torque_saturation_rate']} | {'YES' if r['nan'] else 'no'} |"
            )
        out.append("")
    return out


def video_name(motion: str, backend: str, profile: str, condition: str, seed) -> str:
    """Filename that identifies a rollout from its parameters alone."""
    return (f"{slug(motion)}__{slug(backend)}__{slug(profile)}__"
            f"{slug(condition)}__seed{seed}.mp4")


def run_matrix(*, cfg: dict, out_dir: Path, build_argv, cwd: str,
               record: str, dry_run: bool, timeout: int) -> None:
    """Expand and execute the matrix, then write every artefact.

    ``record="failures"`` runs the matrix without video first and then re-runs
    only the rollouts that did not complete, with the same seed, so a recording
    reproduces the run it depicts. That costs a second pass over the failures
    but nothing over the passes, which is the cheap direction when most
    conditions hold.

    Args:
        cfg: The parsed config.
        out_dir: Destination ``results/<run_name>/``.
        build_argv: ``(motion, backend, profile, terrain, condition, seed,
            result_rel, video_rel) -> list[str]`` building one runner command.
            ``video_rel`` is None when that rollout is not being recorded.
        cwd: Directory the runner is launched from.
        record: ``all``, ``failures`` or ``none``.
        dry_run: Print the matrix and stop.
        timeout: Per-rollout timeout [s].
    """
    motions = cfg["motions"]
    backends = cfg["backends"]
    profiles = cfg.get("profiles", ["strict-native"])
    terrains = cfg.get("terrains", ["flat"])
    seeds = cfg.get("seeds", [0])
    seconds = float(cfg.get("seconds", 20))
    conditions = expand_conditions(cfg)

    jobs = [(m, b, p, t, c, s)
            for m in motions for b in backends for p in profiles
            for t in terrains for c in conditions for s in seeds]

    print(f"  motions    {len(motions)}   {', '.join(motions)}")
    print(f"  backends   {len(backends)}   {', '.join(backends)}")
    print(f"  profiles   {len(profiles)}   {', '.join(profiles)}")
    print(f"  terrains   {len(terrains)}   {', '.join(terrains)}")
    print(f"  conditions {len(conditions)}  "
          + ", ".join(f"{c.name}[{c.grade[:5]}]" for c in conditions))
    print(f"  seeds      {len(seeds)}   {seeds}")
    mute = sorted({b for b in backends if not ev.can_record(b)})
    print(f"  seconds    {seconds}   record {record}"
          + (f"   (no video on {', '.join(mute)}: no Kit renderer)" if mute else ""))
    print(f"  => {len(jobs)} runs, sequential, into {out_dir}\n")
    if dry_run:
        return

    # The runners execute as the container's own user, which is not the user
    # running this driver, so every directory they write into is made writable
    # by both rather than only by whoever created it first.
    for path in (out_dir.parent, out_dir, out_dir / "raw", out_dir / "videos"):
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(path, 0o777)
        except PermissionError:
            pass  # already owned by someone else and already permissive enough
    _dump_config(out_dir, cfg, record, seconds)

    rows, failures, provenance = [], [], None
    progress = Progress(len(jobs))
    for motion, backend, profile, terrain, cond, seed in jobs:
        label = f"{motion} / {backend} / {profile} / {terrain} / {cond.name} / seed{seed}"
        stem = (f"{slug(motion)}__{slug(backend)}__{slug(profile)}__"
                f"{slug(terrain)}__{slug(cond.name)}__seed{seed}")
        # The runner executes inside its own container, where the host path does
        # not exist. Paths are therefore given relative to the launch directory,
        # which resolves the same on both sides of the container boundary.
        result_rel = os.path.relpath(out_dir / "raw" / f"{stem}.json", cwd)
        # A backend that cannot record still has to produce its result: asking
        # for video it cannot make would abort the run and lose the row.
        video_rel = (os.path.relpath(
            out_dir / "videos" / video_name(motion, backend, profile, cond.name, seed), cwd)
            if record == "all" and ev.can_record(backend) else None)
        result_host = os.path.join(cwd, result_rel)
        argv = build_argv(motion, backend, profile, terrain, cond, seed,
                          result_rel, video_rel, seconds)
        # Campaign configs can ask for flags the matrix itself has no opinion on,
        # e.g. --series so tracking error can be computed post-hoc.
        argv += [str(a) for a in cfg.get("extra_args", [])]
        t0 = time.time()
        result, note = run_one(argv, cwd, result_host, timeout)
        took = time.time() - t0
        if result is None:
            progress.tick(label, "NO RESULT", took)
            failures.append({"label": label, "note": note})
            continue
        provenance = provenance or result["provenance"]
        row = _row(result, motion, backend, profile, terrain, cond, seed,
                   result_host, os.path.join(cwd, video_rel) if video_rel else None,
                   out_dir)
        rows.append(row)
        progress.tick(label, row["termination_reason"], took)

        if (record == "failures" and ev.can_record(backend)
                and row["termination_reason"] != "completed"):
            video = os.path.relpath(
                out_dir / "videos" / video_name(motion, backend, profile, cond.name, seed), cwd)
            argv = build_argv(motion, backend, profile, terrain, cond, seed,
                              result_rel, video, seconds)
            run_one(argv, cwd, result_host, timeout)
            row["video"] = os.path.relpath(os.path.join(cwd, video), out_dir)

    progress.finish()
    write_summary(out_dir, rows, failures)
    if provenance is not None:
        (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"  summary.csv / summary.md / provenance.json -> {out_dir}")


def _dump_config(out_dir: Path, cfg: dict, record: str, seconds: float) -> None:
    """Store the config the batch actually ran, so the run is reproducible."""
    payload = dict(cfg)
    payload["_resolved"] = {"record": record, "seconds": seconds,
                            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        import yaml  # noqa: PLC0415

        (out_dir / "config.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    except ImportError:
        (out_dir / "config.yaml").write_text(json.dumps(payload, indent=2))


def _row(result: dict, motion, backend, profile, terrain, cond, seed,
         result_rel, video_rel, out_dir) -> dict:
    """Flatten one result into a summary row."""
    m = result["metrics"]
    row = {k: m.get(k) for k in SUMMARY_COLUMNS}
    row.update(motion=motion, backend=backend, profile=profile, terrain=terrain,
               condition=cond.name, grade=cond.grade, seed=seed,
               result=os.path.relpath(result_rel, out_dir),
               video=os.path.relpath(video_rel, out_dir) if video_rel else "")
    return row
