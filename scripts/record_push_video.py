"""Record a stand/sit policy being pushed.

Ideal world plus fixed-speed pushes every 4 s (at 4, 8, 12, 16 s; a random direction per robot);
scripted command: stand 0-6 s, sit 6-14 s, stand 14-20 s, so pushes land twice while standing and
twice while sitting. With one robot the camera follows it; with several (``--num_envs``) it looks
down on the whole grid.

    python record_push_video.py --policy <model.pt> --out_dir <dir> [--task Go2-Sit-Step] [--push 1.5]
                                [--num_envs 16]

Writes <out_dir>/videos/<name>/clip_0000.mp4 and events.json (schedule, falls, off-pose counts)
for scripts/annotate_video.py --events. Run from the Isaac Lab checkout with this repository on
PYTHONPATH.
"""

import argparse
import json
import os

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import to_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.entrypoints.common import add_common_play_args, apply_video_recording, pre_launch_video_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import resolve_task_config

from go2_sim2real import register
from go2_sim2real.envs import apply_eval_change

SIT_S = (6.0, 14.0)
PUSH_EVERY_S = 4.0  # matches apply_eval_change("push_mps", ...)
CAMERA_OFFSET = (1.6, -1.9, 0.8)  # [m] from the robot, front-left and slightly above (one robot)
GRID_CAMERA_OFFSET = (3.4, -5.2, 5.2)  # [m] from the grid centre (several robots)
LOOKAT_HEIGHT = 0.25  # [m]
# a robot is "off pose" if it is up but its mean joint error exceeds this after settling
OFF_POSE_JOINT_ERR = 0.15  # rad, same cut-off as the sweep's "held the sit"
SETTLE_S = 2.0  # time allowed after a command change or a reset


def main():
    """Roll out the policy with the scripted command, record the video and write events.json."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--push", type=float, default=1.5)
    p.add_argument("--name", help="video subfolder under <out_dir>/videos (default push_<speed>)")
    add_common_play_args(p, agent_default="rsl_rl_cfg_entry_point", agent_help="Agent config entry point.")
    add_launcher_args(p)
    args = p.parse_args()
    args.seed = 3 if args.seed is None else args.seed
    args.task = args.task or "Go2-Sit-Step"
    args.num_envs = args.num_envs or 1
    args.video = args.video or "viz"
    args.video_length = args.video_length or 1000

    register()
    env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, overrides=["physics=newton_mjwarp"])
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    apply_eval_change(env_cfg, "push_mps", args.push)
    env_cfg.commands.pose_mode.resampling_time_range = (1e6, 1e6)  # scripted below
    pre_launch_video_config(env_cfg, args)

    with launch_simulation(env_cfg, args):
        subdir = args.name or f"push_{args.push}"
        apply_video_recording(env_cfg, args.out_dir, args, subdir=subdir, checkpoint_path=args.policy)
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        u = env.unwrapped
        runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
        runner.load(args.policy)
        policy = runner.get_inference_policy(device=u.device)
        term = u.command_manager.get_term("pose_mode")
        robot = term.robot
        dt = u.step_dt

        if args.num_envs > 1:
            centre = u.scene.env_origins.mean(dim=0).tolist()
            eye = [centre[0] + GRID_CAMERA_OFFSET[0], centre[1] + GRID_CAMERA_OFFSET[1], GRID_CAMERA_OFFSET[2]]
            for viz in u.sim.visualizers:
                viz.set_camera_view(eye, [centre[0], centre[1], 0.0])

        falls, falls_env, off_pose = [], [], []
        with torch.inference_mode():
            obs = env.get_observations()
            for k in range(args.video_length):
                t = k * dt
                term.mode[:, 0] = 1.0 if SIT_S[0] <= t < SIT_S[1] else 0.0
                if args.num_envs == 1:
                    pos = robot.data.root_pos_w.torch[0].tolist()
                    eye = [pos[0] + CAMERA_OFFSET[0], pos[1] + CAMERA_OFFSET[1], CAMERA_OFFSET[2]]
                    for viz in u.sim.visualizers:
                        viz.set_camera_view(eye, [pos[0], pos[1], LOOKAT_HEIGHT])
                obs, _, dones, extras = env.step(policy(obs))
                timeouts = extras.get("time_outs", torch.zeros_like(dones)).bool()
                for i in (dones.bool() & ~timeouts).nonzero().flatten().tolist():
                    falls.append(round(t, 2))
                    falls_env.append(i)
                    print(f"robot {i} fell at t={t:.2f}s (the env resets it)", flush=True)
                since_switch = min(t - s for s in (0.0, *SIT_S) if s <= t)
                settled = (u.episode_length_buf * dt >= SETTLE_S) & (since_switch >= SETTLE_S)
                joint_err = (robot.data.joint_pos.torch - term.target_joint_pos).abs().mean(dim=1)
                off_pose.append(int((settled & (joint_err > OFF_POSE_JOINT_ERR)).sum()))
        env.close()

    duration = args.video_length * dt
    events = {
        "segments": [[0.0, "stand"], [SIT_S[0], "sit"], [SIT_S[1], "stand"]],
        "pushes": [PUSH_EVERY_S * i for i in range(1, int(duration / PUSH_EVERY_S) + 1) if PUSH_EVERY_S * i < duration],
        "falls": falls,
        "falls_env": falls_env,
        "num_envs": args.num_envs,
        "off_pose": off_pose,
        "off_pose_joint_err": OFF_POSE_JOINT_ERR,
        "policy": args.policy,
        "push_mps": args.push,
    }
    path = os.path.join(args.out_dir, "videos", subdir, "events.json")
    with open(path, "w") as f:
        json.dump(events, f)
    print(f"wrote {path}: {len(falls)} falls at {falls}, max off-pose {max(off_pose)}", flush=True)


if __name__ == "__main__":
    main()
