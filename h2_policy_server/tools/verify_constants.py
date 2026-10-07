#!/usr/bin/env python3
"""Re-derive h2_policy_if/h2_joints.py from the wbc_h12 sources and report drift.

Run this after replacing ``wbc_h12`` with a new upstream version. It parses
``gear_sonic/envs/manager_env/robots/h2.py`` and ``h2_edu.py`` with ``ast`` --
those modules import ``isaaclab`` at module scope, so they cannot be imported in
this environment -- resolves each actuator group's ``stiffness``, ``damping`` and
``effort_limit_sim``, and compares the result against the copies this repository
serves with.

Exit status 0 means the copies are still correct.

    python tools/verify_constants.py --wbc-root ../wbc_h12
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h2_policy_if import h2_joints  # noqa: E402

ACTION_SCALE_FACTOR = 0.25  # robots/h2.py:390


def module_constants(tree: ast.Module) -> dict[str, float]:
    """Evaluate the top-level ``name = <arithmetic on earlier names>`` assignments.

    Covers ``ARMATURE_*``, ``NATURAL_FREQ``, ``DAMPING_RATIO``, ``STIFFNESS_*`` and
    ``DAMPING_*``, which is everything the actuator configs refer to.
    """
    ns: dict[str, float] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            ns[target.id] = float(eval(compile(ast.Expression(node.value), "<c>", "eval"), {}, ns))
        except Exception:
            continue  # not a numeric constant (asset paths, cfg objects, lists)
    return ns


def literal_list(tree: ast.Module, name: str) -> list:
    """Return the literal list assigned to a top-level ``name``."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found as a top-level literal assignment")


def _call_kwargs(call: ast.Call) -> dict[str, ast.AST]:
    return {kw.arg: kw.value for kw in call.keywords if kw.arg}


def find_cfg_call(tree: ast.Module, name: str) -> ast.Call:
    """Return the ``ast.Call`` assigned to a top-level ``name``."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    if not isinstance(node.value, ast.Call):
                        raise TypeError(f"{name} is not a call expression")
                    return node.value
    raise KeyError(f"{name} not found")


def actuator_groups(cfg_call: ast.Call) -> dict[str, dict[str, ast.AST]]:
    """Map each actuator group name to its ``ImplicitActuatorCfg`` keywords."""
    actuators = _call_kwargs(cfg_call).get("actuators")
    if not isinstance(actuators, ast.Dict):
        raise TypeError("actuators= is not a dict literal")
    groups = {}
    for key, value in zip(actuators.keys, actuators.values):
        if not isinstance(value, ast.Call):
            raise TypeError(f"actuator group {ast.literal_eval(key)!r} is not a call")
        groups[ast.literal_eval(key)] = _call_kwargs(value)
    return groups


def resolve_per_joint(
    node: ast.AST, ns: dict[str, float], joint_patterns: list[str]
) -> dict[str, float]:
    """Resolve one actuator keyword into ``{joint_pattern: value}``.

    Handles both forms the configs use: a scalar that applies to every joint in
    the group, and a dict keyed by joint-name regex.
    """
    value = eval(compile(ast.Expression(node), "<c>", "eval"), {}, ns) if not isinstance(node, ast.Dict) else None
    if value is not None:
        return {pattern: float(value) for pattern in joint_patterns}
    out = {}
    for key, val in zip(node.keys, node.values):
        out[ast.literal_eval(key)] = float(
            eval(compile(ast.Expression(val), "<c>", "eval"), {}, ns)
        )
    return out


def group_joint_patterns(node: ast.AST) -> list[str]:
    """Read a group's ``joint_names_expr`` list of regexes."""
    return [str(p) for p in ast.literal_eval(node)]


def build_tables(h2_src: str) -> dict[str, np.ndarray]:
    """Derive kp, kd, action_scale and default_angles in MuJoCo joint order."""
    tree = ast.parse(h2_src)
    ns = module_constants(tree)
    groups = actuator_groups(find_cfg_call(tree, "H2_CFG"))

    stiffness: dict[str, float] = {}
    damping: dict[str, float] = {}
    effort: dict[str, float] = {}
    for kwargs in groups.values():
        patterns = group_joint_patterns(kwargs["joint_names_expr"])
        stiffness.update(resolve_per_joint(kwargs["stiffness"], ns, patterns))
        damping.update(resolve_per_joint(kwargs["damping"], ns, patterns))
        effort.update(resolve_per_joint(kwargs["effort_limit_sim"], ns, patterns))

    def lookup(table: dict[str, float], joint: str, what: str) -> float:
        # Most specific pattern wins, matching how Isaac Lab resolves these.
        hits = [v for pattern, v in table.items() if re.fullmatch(pattern, joint)]
        if not hits:
            raise KeyError(f"no {what} for {joint}")
        return hits[-1]

    init = _call_kwargs(find_cfg_call(tree, "H2_CFG"))["init_state"]
    init_kwargs = _call_kwargs(init)
    defaults = {
        ast.literal_eval(k): float(ast.literal_eval(v))
        for k, v in zip(init_kwargs["joint_pos"].keys, init_kwargs["joint_pos"].values)
    }

    kp, kd, scale, default = [], [], [], []
    for joint in h2_joints.JOINT_NAMES_MUJOCO:
        k = lookup(stiffness, joint, "stiffness")
        kp.append(k)
        kd.append(lookup(damping, joint, "damping"))
        scale.append(ACTION_SCALE_FACTOR * lookup(effort, joint, "effort_limit_sim") / k)
        matches = [v for pattern, v in defaults.items() if re.fullmatch(pattern, joint)]
        default.append(matches[-1] if matches else 0.0)

    return {
        "kp": np.array(kp),
        "kd": np.array(kd),
        "action_scale": np.array(scale),
        "default_angles": np.array(default),
        "base_height": float(
            ast.literal_eval(init_kwargs["pos"])[2]
        ),
    }


def compare(name: str, source: np.ndarray, served: np.ndarray, tol: float) -> bool:
    """Print a one-line verdict and return whether the arrays agree."""
    source, served = np.asarray(source, float), np.asarray(served, float)
    if source.shape != served.shape:
        print(f"  {name:22s} SHAPE source={source.shape} served={served.shape}")
        return False
    delta = np.abs(source - served)
    ok = bool(delta.max() <= tol)
    print(f"  {name:22s} max|delta| {delta.max():.3e}  tol {tol:.0e}  {'OK' if ok else 'DRIFT'}")
    if not ok:
        for i in np.flatnonzero(delta > tol):
            print(
                f"      [{i:2d}] {h2_joints.JOINT_NAMES_MUJOCO[i]:28s} "
                f"served={served[i]:.6f} source={source[i]:.6f}"
            )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--wbc-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "wbc_h12",
        help="wbc_h12 checkout to verify against",
    )
    args = parser.parse_args()

    robots = args.wbc_root / "gear_sonic/envs/manager_env/robots"
    h2_src = (robots / "h2.py").read_text()
    edu_src = (robots / "h2_edu.py").read_text()
    mjcf = args.wbc_root / "gear_sonic/data/assets/robot_description/mjcf/h2_edu.xml"
    ok = True

    print("-- joint order (mjcf/h2_edu.xml) --")
    try:
        h2_joints.check_against_mjcf(str(mjcf))
        print(f"  {len(h2_joints.JOINT_NAMES_MUJOCO)} joints, order matches")
    except RuntimeError as exc:
        print(f"  DRIFT\n{exc}")
        ok = False

    print("-- permutations (robots/h2.py) --")
    tree = ast.parse(h2_src)
    for const, served in (
        ("H2_MUJOCO_TO_ISAACLAB_DOF", h2_joints.MUJOCO_TO_ISAACLAB),
        ("H2_ISAACLAB_TO_MUJOCO_DOF", h2_joints.ISAACLAB_TO_MUJOCO),
    ):
        ok &= compare(const, np.asarray(literal_list(tree, const)), served, 0)

    print("-- action scale provenance (robots/h2_edu.py) --")
    # h2_edu.py reuses H2_ACTION_SCALE verbatim instead of recomputing it from
    # EDU's own (tighter) effort limits. If that ever changes, every action_scale
    # below is derived from the wrong effort table.
    if "H2_EDU_ACTION_SCALE = H2_ACTION_SCALE" in edu_src:
        print("  H2_EDU_ACTION_SCALE aliases H2_ACTION_SCALE (H1031 effort limits)  OK")
    else:
        print("  DRIFT h2_edu.py no longer aliases H2_ACTION_SCALE -- recompute by hand")
        ok = False

    print("-- gains, action scale, default pose (robots/h2.py) --")
    tables = build_tables(h2_src)
    ok &= compare("kp", tables["kp"], h2_joints.KP_MUJOCO, 1e-3)
    ok &= compare("kd", tables["kd"], h2_joints.KD_MUJOCO, 1e-3)
    ok &= compare("action_scale", tables["action_scale"], h2_joints.ACTION_SCALE_MUJOCO, 1e-5)
    ok &= compare("default_angles", tables["default_angles"], h2_joints.DEFAULT_ANGLES_MUJOCO, 1e-9)
    height_ok = abs(tables["base_height"] - h2_joints.DEFAULT_BASE_HEIGHT) < 1e-9
    print(
        f"  {'base_height':22s} source={tables['base_height']:.4f} "
        f"served={h2_joints.DEFAULT_BASE_HEIGHT:.4f}  {'OK' if height_ok else 'DRIFT'}"
    )
    ok &= height_ok

    print()
    if ok:
        print("RESULT: PASS -- h2_policy_if/h2_joints.py still matches wbc_h12")
    else:
        raise SystemExit("RESULT: DRIFT -- update h2_policy_if/h2_joints.py before serving")


if __name__ == "__main__":
    main()
