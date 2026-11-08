"""Evaluate policies in "fake real" worlds, each differing from the ideal simulator in one property.

Worker mode (one world, all policies, one 20 s episode per robot):
    python eval_gap.py --axis friction --value 0.4 --policy norand=<model.pt> --policy rand=<model.pt>

Sweep mode (spawns one worker per world, appends results to a JSONL file, skips worlds already done):
    python eval_gap.py --sweep --out results.jsonl --policy norand=<model.pt> --policy rand=<model.pt>

``--task walk`` (default) follows random velocity commands. ``--task sit`` follows a fixed script:
stand 0-5 s, sit 5-15 s, stand 15-20 s, and scores the settled sit (10-15 s) and the stand-up (17-20 s).

Run from the Isaac Lab checkout with this repository on PYTHONPATH (see scripts/eval_gap_wsl.sh).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

TASKS = {"walk": "Go2-Flat-NoRand", "sit": "Go2-Sit"}
SIT_START_S, SIT_END_S = 5.0, 15.0
SIT_WINDOW_S = (10.0, 15.0)
# phases a fall is attributed to, by the time it happens: (name, start s, end s)
FALL_PHASES = [("stand", 0.0, 5.0), ("sit_down", 5.0, 7.0), ("sitting", 7.0, 15.0),
               ("stand_up", 15.0, 17.0), ("standing", 17.0, 20.0)]
STAND_WINDOW_S = (17.0, 20.0)
# a robot "held the sit" if it stayed up through the sit and its settled pose was within these
SIT_OK_JOINT_ERR = 0.15  # rad, mean absolute over joints
SIT_OK_TILT_ERR = 0.10  # norm of the projected-gravity error
MIN_SURVIVORS = 0.05  # fraction of robots needed in a scoring window before its pose errors are reported


def _step_bookkeeping(env, policy, obs, alive, fell):
    """Step all envs once, updating ``alive`` and ``fell`` in place; returns new obs and the envs still running."""
    import torch

    obs, _, dones, extras = env.step(policy(obs))
    dones = dones.bool()
    timeouts = extras.get("time_outs", torch.zeros_like(dones)).bool()
    # done envs were already reset inside step, so only count the ones still running
    running = alive & ~dones
    fell |= alive & dones & ~timeouts
    alive &= ~dones
    policy.reset(dones)
    return obs, running


def rollout_walk(env, policy, u) -> dict:
    """Run one episode per robot under random velocity commands; returns fall and tracking metrics."""
    import torch

    from isaaclab.envs import mdp

    env.reset()
    obs = env.get_observations()
    n = u.num_envs
    alive = torch.ones(n, dtype=torch.bool, device=u.device)
    fell = torch.zeros_like(alive)
    survived_steps = torch.zeros(n, device=u.device)
    err_xy = torch.zeros(n, device=u.device)
    err_yaw = torch.zeros(n, device=u.device)
    for _ in range(int(u.max_episode_length)):
        cmd = u.command_manager.get_command("base_velocity").clone()
        obs, running = _step_bookkeeping(env, policy, obs, alive, fell)
        vel = mdp.base_lin_vel(u)[:, :2]
        wz = mdp.base_ang_vel(u)[:, 2]
        err_xy += torch.where(running, torch.linalg.norm(cmd[:, :2] - vel, dim=1), 0.0)
        err_yaw += torch.where(running, (cmd[:, 2] - wz).abs(), 0.0)
        survived_steps += running.float()
    counted = survived_steps.sum().clamp(min=1)
    return {
        "fall_rate": round(fell.float().mean().item(), 4),
        "vel_err_xy": round((err_xy.sum() / counted).item(), 4),
        "vel_err_yaw": round((err_yaw.sum() / counted).item(), 4),
        "mean_survival_s": round((survived_steps.mean() * u.step_dt).item(), 2),
    }


def rollout_sit(env, policy, u, script: str = "sequence") -> dict:
    """``script``: "sequence" (stand / sit / stand), "stand" (always stand) or "sit" (always sit)."""
    import torch

    term = u.command_manager.get_term("pose_mode")
    robot = term.robot
    env.reset()
    obs = env.get_observations()
    n = u.num_envs
    alive = torch.ones(n, dtype=torch.bool, device=u.device)
    fell = torch.zeros_like(alive)
    sums = {k: torch.zeros(n, device=u.device) for k in ("sit_joint", "sit_tilt", "sit_steps", "stand_joint", "stand_steps")}
    fall_counts = dict.fromkeys((name for name, _, _ in FALL_PHASES), 0)
    for k in range(int(u.max_episode_length)):
        t = k * u.step_dt
        sit = {"sequence": SIT_START_S <= t < SIT_END_S, "stand": False, "sit": True}[script]
        term.mode[:, 0] = 1.0 if sit else 0.0
        fell_before = int(fell.sum())
        obs, running = _step_bookkeeping(env, policy, obs, alive, fell)
        if (new_falls := int(fell.sum()) - fell_before):
            fall_counts[next(name for name, lo, hi in FALL_PHASES if lo <= t < hi)] += new_falls
        joint_err = (robot.data.joint_pos.torch - term.target_joint_pos).abs().mean(dim=1)
        tilt_err = torch.linalg.norm(robot.data.projected_gravity_b.torch - term.target_gravity, dim=1)
        if SIT_WINDOW_S[0] <= t < SIT_WINDOW_S[1]:
            sums["sit_joint"] += torch.where(running, joint_err, 0.0)
            sums["sit_tilt"] += torch.where(running, tilt_err, 0.0)
            sums["sit_steps"] += running.float()
        elif STAND_WINDOW_S[0] <= t < STAND_WINDOW_S[1]:
            sums["stand_joint"] += torch.where(running, joint_err, 0.0)
            sums["stand_steps"] += running.float()
    window_steps = (SIT_WINDOW_S[1] - SIT_WINDOW_S[0]) / u.step_dt
    sit_joint = sums["sit_joint"] / sums["sit_steps"].clamp(min=1)
    sit_tilt = sums["sit_tilt"] / sums["sit_steps"].clamp(min=1)
    sat_through = sums["sit_steps"] >= window_steps - 1
    held = sat_through & (sit_joint < SIT_OK_JOINT_ERR) & (sit_tilt < SIT_OK_TILT_ERR)

    def mean_or_none(key: str, steps_key: str, enough: bool):
        """Mean of ``key`` per step counted in ``steps_key``, or None if too few robots survived."""
        # pose errors from a handful of survivors are noise, so report no value instead
        return round((sums[key].sum() / sums[steps_key].sum()).item(), 4) if enough else None

    enough_sit = sat_through.float().mean().item() >= MIN_SURVIVORS
    enough_stand = (sums["stand_steps"] > 0).float().mean().item() >= MIN_SURVIVORS
    return {
        "fall_rate": round(fell.float().mean().item(), 4),
        "sit_success": round(held.float().mean().item(), 4),
        "sit_joint_err": mean_or_none("sit_joint", "sit_steps", enough_sit),
        "sit_tilt_err": mean_or_none("sit_tilt", "sit_steps", enough_sit),
        "stand_joint_err": mean_or_none("stand_joint", "stand_steps", enough_stand),
        # share of all robots that fell during each phase
        **{f"fall_{name}": round(count / n, 4) for name, count in fall_counts.items()},
    }


def evaluate(args) -> None:
    """Worker mode: build one test world, run every policy in it, print one ``RESULT`` JSON line each."""
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.utils import to_dict

    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import resolve_task_config

    from go2_sim2real import register
    from go2_sim2real.envs import apply_eval_change

    register()
    task = TASKS[args.task]
    env_cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=["physics=newton_mjwarp"])
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    apply_eval_change(env_cfg, args.axis, args.value)
    if args.task == "sit":
        env_cfg.commands.pose_mode.resampling_time_range = (1e6, 1e6)  # the rollout scripts the command
    if args.task == "sit":
        def rollout(env, policy, u):
            """Run the scripted sit rollout and tag the metrics with the script name."""
            return {"script": args.script, **rollout_sit(env, policy, u, args.script)}
    else:
        rollout = rollout_walk

    launcher = argparse.ArgumentParser()
    add_launcher_args(launcher)
    with launch_simulation(env_cfg, launcher.parse_args([])):
        env = RslRlVecEnvWrapper(gym.make(task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        u = env.unwrapped
        for spec in args.policy:
            name, path = spec.split("=", 1)
            runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            runner.load(path)
            policy = runner.get_inference_policy(device=u.device)
            torch.manual_seed(args.seed)
            # the whole rollout, reset included, stays in inference mode: tensors created by the
            # previous policy's rollout cannot be updated in place outside it
            with torch.inference_mode():
                metrics = rollout(env, policy, u)
            result = {"axis": args.axis, "value": args.value, "policy": name, "num_envs": u.num_envs, **metrics}
            print("RESULT " + json.dumps(result), flush=True)
        env.close()


def sweep(args) -> None:
    """Sweep mode: run one worker per unfinished world and append its results to ``args.out``."""
    from go2_sim2real.envs import EVAL_AXES

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["axis"], float(r["value"])))
    axes = args.axes.split(",") if args.axes else list(EVAL_AXES)
    for axis in axes:
        for value in EVAL_AXES[axis][0]:
            if (axis, float(value)) in done:
                continue
            cmd = [sys.executable, os.path.abspath(__file__), "--task", args.task, "--script", args.script,
                   "--axis", axis, "--value", str(value),
                   "--num_envs", str(args.num_envs), "--seed", str(args.seed)]
            for spec in args.policy:
                cmd += ["--policy", spec]
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            results = [json.loads(l[7:]) for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
            if proc.returncode != 0 or len(results) != len(args.policy):
                print(f"FAILED {axis}={value} (exit {proc.returncode})\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}",
                      flush=True)
                continue
            with open(args.out, "a") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            summary = "  ".join(
                f"{r['policy']}: " + " ".join(f"{k} {v}" for k, v in r.items() if k not in ("axis", "value", "policy", "num_envs"))
                for r in results
            )
            print(f"{axis}={value} ({time.time() - t0:.0f}s)  {summary}", flush=True)


def main():
    """Parse the command line and run sweep or worker mode."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=list(TASKS), default="walk")
    p.add_argument("--script", choices=["sequence", "stand", "sit"], default="sequence",
                   help="sit task only: command script for the episode")
    p.add_argument("--axis")
    p.add_argument("--value", type=float)
    p.add_argument("--policy", action="append", required=True, help="name=path/to/model.pt (repeatable)")
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--axes", help="comma-separated subset of axes for --sweep")
    p.add_argument("--out", default="gap_results.jsonl")
    args = p.parse_args()
    if args.sweep:
        sweep(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
