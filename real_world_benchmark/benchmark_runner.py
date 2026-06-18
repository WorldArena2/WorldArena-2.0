"""Benchmark runner for real-world policy evaluation.

    python -m real_world_benchmark.benchmark_runner
    python -m real_world_benchmark.benchmark_runner /path/to/policy.py --mode dataset
    python -m real_world_benchmark.benchmark_runner /path/to/policy.py --mode live --send-action
    python -m real_world_benchmark.benchmark_runner /path/to/policy.py --mode dataset --dataset-limit 20
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_SRC = _REPO_ROOT / 'src'
if _REPO_SRC.is_dir():
    _repo_src_str = str(_REPO_SRC)
    if _repo_src_str not in sys.path:
        sys.path.insert(0, _repo_src_str)
    existing_pythonpath = os.environ.get('PYTHONPATH', '')
    pythonpath_parts = [part for part in existing_pythonpath.split(os.pathsep) if part]
    if _repo_src_str not in pythonpath_parts:
        os.environ['PYTHONPATH'] = os.pathsep.join([_repo_src_str, *pythonpath_parts])

try:
    from scipy.spatial.transform import Rotation as R
except Exception:  # pragma: no cover - optional for offline smoke tests
    R = None  # type: ignore[assignment]

try:
    from openpi.training.agilex_dataset_align import AgileXDataset
except Exception:  # pragma: no cover - optional for training-data offline mode
    AgileXDataset = None  # type: ignore[assignment]


def load_policy(path_or_module: str):
    p = Path(path_or_module)
    if p.exists() and p.suffix == '.py':
        spec = importlib.util.spec_from_file_location('user_policy', str(p))
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot import policy from file: {path_or_module}')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[arg-type]
        return mod
    return importlib.import_module(path_or_module)


def make_sample_obs() -> Dict[str, Any]:
    h, w = 240, 320
    images = {'cam_high': np.zeros((h, w, 3), dtype=np.uint8)}
    state = np.zeros((32,), dtype=np.float32)
    return {
        'images': images,
        'first_frame': images['cam_high'].copy(),
        'state': state,
        'prompt': 'benchmark example',
    }


def _make_dataset_obs(sample: Dict[str, Any], use_history: bool) -> Dict[str, Any]:
    new_obs: Dict[str, Any] = {'images': {}}

    cam_high_tensor = sample['observation.images.cam_high']
    cam_left_tensor = sample['observation.images.cam_left_wrist']
    cam_right_tensor = sample['observation.images.cam_right_wrist']

    new_obs['images']['cam_high'] = _to_uint8_hw3(_last_frame(cam_high_tensor))
    new_obs['images']['cam_left_wrist'] = _to_uint8_hw3(_last_frame(cam_left_tensor))
    new_obs['images']['cam_right_wrist'] = _to_uint8_hw3(_last_frame(cam_right_tensor))

    state_tensor = _as_numpy(sample['observation.state'])
    if state_tensor.shape[-1] == 20:
        state_tensor = np.pad(state_tensor, (0, 12), mode='constant', constant_values=0)
    new_obs['state'] = state_tensor

    if use_history and 'observation.images.cam_high_memory' in sample:
        new_obs['images']['cam_high_memory'] = _stack_frames(sample['observation.images.cam_high_memory'])

    new_obs['prompt'] = sample.get(
        'prompt',
        'Grab the towel and then wipe the table and put the towel back.',
    )
    new_obs['first_frame'] = new_obs['images']['cam_high']
    return new_obs


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def _last_frame(value: Any) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim >= 4:
        return array[-1]
    return array


def _to_uint8_hw3(frame: Any) -> np.ndarray:
    array = _as_numpy(frame)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if np.nanmax(array) <= 1.0:
            array = (array * 255).clip(0, 255).astype(np.uint8)
        else:
            array = array.clip(0, 255).astype(np.uint8)
    return array


def _stack_frames(value: Any) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim == 4 and array.shape[1] in (1, 3):
        array = np.transpose(array, (0, 2, 3, 1))
    if array.dtype != np.uint8:
        if np.nanmax(array) <= 1.0:
            array = (array * 255).clip(0, 255).astype(np.uint8)
        else:
            array = array.clip(0, 255).astype(np.uint8)
    return array


def _quat_to_rot6d(pose_7d: np.ndarray) -> np.ndarray:
    pos = pose_7d[..., :3]
    quat = pose_7d[..., 3:7]
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1e-6):
        quat = quat.copy()
        quat[norm[..., 0] < 1e-6] = [0.0, 0.0, 0.0, 1.0]
    pos = np.where(np.isclose(pos, -1.0), 0.0, pos)
    rot = R.from_quat(quat)
    rot_mat = rot.as_matrix()
    rot6d = rot_mat[..., :, :2].reshape(rot_mat.shape[:-2] + (6,))
    return np.concatenate([pos, rot6d], axis=-1)


def cont6d_to_matrix(cont6d: np.ndarray) -> np.ndarray:
    assert cont6d.shape[-1] == 6, 'The last dimension must be 6'
    cont6d = cont6d.reshape(3, 2).transpose(1, 0).reshape(6)
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = x_raw / np.linalg.norm(x_raw, axis=-1, keepdims=True)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x)
    x = x[..., None]
    y = y[..., None]
    z = z[..., None]
    return np.concatenate([x, y, z], axis=-1)


def eef_pose_base_to_camera(eef_pose: np.ndarray, is_left: bool = True) -> np.ndarray:
    t_l_base_to_cam = np.array([-0.102, -0.405, 0.655], dtype=np.float64)
    t_r_base_to_cam = np.array([-0.102, 0.405, 0.655], dtype=np.float64)
    t_cam_from_base = t_l_base_to_cam if is_left else t_r_base_to_cam

    r_cam_from_base = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
    theta = np.radians(45.0)
    c, s = np.cos(theta), np.sin(theta)
    rot_x_45 = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)

    pos_base = np.asarray(eef_pose[:3], dtype=np.float64)
    quat_base_xyzw = np.asarray(eef_pose[3:7], dtype=np.float64)
    r_eef_base = R.from_quat(quat_base_xyzw).as_matrix()

    if is_left:
        r_z180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
        r_eef_base = r_eef_base @ r_z180

    pts_rel = pos_base - t_cam_from_base
    pos_cam = rot_x_45 @ (r_cam_from_base @ pts_rel)
    r_eef_cam = rot_x_45 @ (r_cam_from_base @ r_eef_base)
    quat_cam_xyzw = R.from_matrix(r_eef_cam).as_quat()
    return np.concatenate([pos_cam.astype(np.float32), quat_cam_xyzw.astype(np.float32)], axis=0)


def eef_pose_camera_to_base(eef_pose_cam: np.ndarray, is_left: bool = True) -> np.ndarray:
    t_l_base_to_cam = np.array([-0.102, -0.405, 0.655], dtype=np.float64)
    t_r_base_to_cam = np.array([-0.102, 0.405, 0.655], dtype=np.float64)
    t_cam_from_base = t_l_base_to_cam if is_left else t_r_base_to_cam

    r_cam_from_base = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
    theta = np.radians(45.0)
    c, s = np.cos(theta), np.sin(theta)
    rot_x_45 = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)

    pos_cam = np.asarray(eef_pose_cam[:3], dtype=np.float64)
    quat_cam_xyzw = np.asarray(eef_pose_cam[3:7], dtype=np.float64)
    r_eef_cam = R.from_quat(quat_cam_xyzw).as_matrix()

    pos_base = np.linalg.inv(r_cam_from_base) @ (np.linalg.inv(rot_x_45) @ pos_cam) + t_cam_from_base
    r_base_adj = np.linalg.inv(r_cam_from_base) @ (np.linalg.inv(rot_x_45) @ r_eef_cam)

    if is_left:
        r_z180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
        r_eef_base = r_base_adj @ np.linalg.inv(r_z180)
    else:
        r_eef_base = r_base_adj

    quat_base_xyzw = R.from_matrix(r_eef_base).as_quat()
    return np.concatenate([pos_base.astype(np.float32), quat_base_xyzw.astype(np.float32)], axis=0)


def _make_live_new_obs(obs: Dict[str, Any], default_prompt: str) -> Dict[str, Any]:
    new_obs: Dict[str, Any] = {'images': {}}
    new_obs['images']['cam_high'] = _to_uint8_hw3(_last_frame(obs['img_front']))
    new_obs['images']['cam_left_wrist'] = _to_uint8_hw3(_last_frame(obs['img_left']))
    new_obs['images']['cam_right_wrist'] = _to_uint8_hw3(_last_frame(obs['img_right']))

    if 'left_end_pose' in obs and 'right_end_pose' in obs:
        left_pose = np.asarray(obs['left_end_pose'][-1])
        right_pose = np.asarray(obs['right_end_pose'][-1])
        left_gripper = np.asarray(obs['left_arm_joint_state'][-1])[-1:]
        right_gripper = np.asarray(obs['right_arm_joint_state'][-1])[-1:]
        left_pose_cam = eef_pose_base_to_camera(left_pose, is_left=True)
        right_pose_cam = eef_pose_base_to_camera(right_pose, is_left=False)
        left_pose_9d = _quat_to_rot6d(left_pose_cam)
        right_pose_9d = _quat_to_rot6d(right_pose_cam)
        new_obs['state'] = np.concatenate([left_pose_9d, left_gripper, right_pose_9d, right_gripper])
    elif 'left_arm_joint_state' in obs and 'right_arm_joint_state' in obs:
        new_obs['state'] = np.concatenate(
            [np.asarray(obs['left_arm_joint_state'][-1]), np.asarray(obs['right_arm_joint_state'][-1])]
        )
    else:
        new_obs['state'] = np.asarray(obs.get('state', np.zeros((32,), dtype=np.float32)))

    if 'img_front' in obs:
        new_obs['images']['cam_high_memory'] = _stack_frames(obs['img_front'])

    new_obs['prompt'] = obs.get('prompt', default_prompt)
    new_obs['first_frame'] = new_obs['images']['cam_high']
    return new_obs


def _get_action_array(output: Dict[str, Any]) -> np.ndarray:
    if not isinstance(output, dict):
        raise ValueError('Policy.infer must return a dict')
    if 'actions' not in output:
        raise ValueError("Return dict must contain key 'actions'")
    actions = _as_numpy(output['actions'])
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2:
        raise ValueError('actions must be a 1D or 2D array-like')
    return actions


def validate_output(out: Dict[str, Any]) -> np.ndarray:
    return _get_action_array(out)


def _build_server(args: argparse.Namespace):
    from agilex_msg.api import http_protocol

    server = http_protocol.Server(host=args.host, port=args.port, node_name=args.node_name)
    if args.env == 'wma':
        server.updateModelInfo(640, 480, 5, 0.1)
    else:
        server.updateModelInfo(640, 480, 1, 1)
    return server


def _build_dataset(args: argparse.Namespace):
    if AgileXDataset is None:
        raise ImportError('openpi.training.agilex_dataset_align_zx.AgileXDataset is required for --dataset-mode')
    return AgileXDataset(
        dataset_dir=args.dataset_dir,
        action_horizon=args.dataset_action_horizon,
        action_type=args.dataset_action_type,
        use_history=args.use_history,
        read_from_hdf5=args.read_from_hdf5,
    )


def _infer_action_format(actions: np.ndarray, action_format: str) -> str:
    if action_format != 'auto':
        return action_format
    return 'eef6d' if actions.shape[-1] >= 20 else 'joint'


def _send_actions_to_server(server: Any, obs: Dict[str, Any], actions: np.ndarray, action_rate: int, action_format: str) -> None:
    timestamp = obs['timestamp']
    if action_format == 'eef6d':
        left_state_output = []
        right_state_output = []
        for one in actions:
            left_pos = one[0:3]
            left_rot6d = one[3:9]
            left_gripper = one[9:10]
            right_pos = one[10:13]
            right_rot6d = one[13:19]
            right_gripper = one[19:20]
            left_rotmat_cam = cont6d_to_matrix(left_rot6d)
            right_rotmat_cam = cont6d_to_matrix(right_rot6d)
            left_quat_cam = R.from_matrix(left_rotmat_cam).as_quat()
            right_quat_cam = R.from_matrix(right_rotmat_cam).as_quat()
            left_pose_cam = np.concatenate([left_pos, left_quat_cam], axis=0)
            right_pose_cam = np.concatenate([right_pos, right_quat_cam], axis=0)
            left_pose_base = eef_pose_camera_to_base(left_pose_cam, is_left=True)
            right_pose_base = eef_pose_camera_to_base(right_pose_cam, is_left=False)
            left_end_pose = np.concatenate([left_pose_base, left_gripper], axis=0)
            right_end_pose = np.concatenate([right_pose_base, right_gripper], axis=0)
            left_state_output.append(left_end_pose.tolist())
            right_state_output.append(right_end_pose.tolist())

        server.send_end_pose_action(
            timestamp,
            action_rate,
            left_state_output[:20],
            right_state_output[:20],
            is_euler=False,
        )
        return

    left_state_output = []
    right_state_output = []
    for one in actions:
        left = one[: len(one) // 2].tolist()
        right = one[len(one) // 2 :].tolist()
        left_state_output.append(left)
        right_state_output.append(right)
    server.send_joint_state_action(timestamp, action_rate, left_state_output, right_state_output)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Real-world benchmark runner')
    parser.add_argument('policy', nargs='?', default='real_world_benchmark.example_policy', help='module path or .py file defining Policy')
    parser.add_argument('--mode', default='smoke', choices=['smoke', 'dataset', 'live'], help='benchmark mode: smoke sample, training dataset, or live robot')
    parser.add_argument('--host', default='0.0.0.0', help='server host in live mode')
    parser.add_argument('--port', type=int, default=8886, help='server port in live mode')
    parser.add_argument('--node-name', default='pi0', help='server node name in live mode')
    parser.add_argument('--env', default='wma', choices=['wma', 'aloha', 'fold', 'flat and fold'], help='environment mode for live server setup')
    parser.add_argument('--default-prompt', default='Grab the towel and then wipe the table and put the towel back.', help='fallback text prompt')
    parser.add_argument('--action-format', default='auto', choices=['auto', 'eef6d', 'joint'], help='how to interpret policy actions when sending to robot')
    parser.add_argument('--send-action', action='store_true', help='send predicted actions back to the robot server')
    parser.add_argument('--use-history', action='store_true', help='include cam_high_memory in live new_obs when available')
    parser.add_argument('--max-steps', type=int, default=1, help='number of live iterations; <=0 means infinite loop')
    parser.add_argument('--action-rate', type=int, default=45, help='action rate used when sending actions back')
    parser.add_argument('--dataset-dir', default='/path/to/real_test_data_train/', help='dataset directory for --mode dataset')
    parser.add_argument('--dataset-index', type=int, default=1000, help='starting frame index for --mode dataset')
    parser.add_argument('--dataset-step', type=int, default=30, help='frame stride for --mode dataset')
    parser.add_argument('--dataset-limit', type=int, default=1, help='number of dataset samples to evaluate; <=0 means all available from start index')
    parser.add_argument('--dataset-action-horizon', type=int, default=50, help='action horizon used by AgileXDataset in dataset mode')
    parser.add_argument('--dataset-action-type', default='eef6d', choices=['eef6d', 'joint_angle'], help='action type used by AgileXDataset in dataset mode')
    parser.add_argument('--read-from-hdf5', action='store_true', help='read images from HDF5 in dataset mode')
    return parser.parse_args(list(argv))


def main(argv):
    args = parse_args(argv[1:])

    print(f'Loading policy from: {args.policy}')
    mod = load_policy(args.policy)
    if not hasattr(mod, 'Policy'):
        print('ERROR: module does not define class Policy')
        return 2

    policy_cls = getattr(mod, 'Policy')
    policy = policy_cls()

    if args.mode == 'smoke':
        new_obs = make_sample_obs()
        print('Sample new_obs prepared: images shape', new_obs['images']['cam_high'].shape, 'state shape', new_obs['state'].shape)
        out = policy.infer(new_obs)
        actions = validate_output(out)
        print('Policy returned actions shape:', actions.shape)
        print('Sample actions (first 3 values):', actions.ravel()[:3])
        return 0

    if args.mode == 'dataset':
        if AgileXDataset is None:
            print('ERROR: openpi.training.agilex_dataset_align_zx.AgileXDataset is required for dataset mode')
            return 6

        try:
            dataset = _build_dataset(args)
        except Exception as exc:
            print('ERROR: failed to initialize dataset mode:', exc)
            return 7

        print(f'Dataset loaded: {len(dataset)} samples from {args.dataset_dir}')
        if args.dataset_index >= len(dataset):
            print(f'ERROR: dataset-index {args.dataset_index} >= dataset length {len(dataset)}')
            return 8

        end_index = len(dataset) if args.dataset_limit <= 0 else min(len(dataset), args.dataset_index + args.dataset_limit * args.dataset_step)
        frame_id = args.dataset_index
        step_count = 0
        while frame_id < end_index:
            sample = dataset[frame_id]
            new_obs = _make_dataset_obs(sample, args.use_history)
            print(new_obs.keys())
            print(new_obs['images'].keys())
            output = policy.infer(new_obs)
            actions = validate_output(output)

            print('Dataset new_obs prepared:')
            print('  cam_high:', new_obs['images']['cam_high'].shape)
            print('  state:', new_obs['state'].shape)
            print('  actions:', actions.shape)
            print('  frame_id:', frame_id)

            step_count += 1
            frame_id += args.dataset_step
            if args.dataset_limit > 0 and step_count >= args.dataset_limit:
                break

        return 0

    if R is None:
        print('ERROR: scipy is required for live mode')
        return 4

    try:
        server = _build_server(args)
    except Exception as exc:
        print('ERROR: failed to initialize live server:', exc)
        return 5

    step_count = 0
    while True:
        print('wait obs')
        obs = server.wait_observation()
        print('got obs')
        new_obs = _make_live_new_obs(obs, args.default_prompt)
        if args.use_history and 'img_front' in obs and 'cam_high_memory' not in new_obs['images']:
            new_obs['images']['cam_high_memory'] = _stack_frames(obs['img_front'])

        output = policy.infer(new_obs)
        actions = validate_output(output)
        action_format = _infer_action_format(actions, args.action_format)

        print('Live new_obs prepared:')
        print('  cam_high:', new_obs['images']['cam_high'].shape)
        print('  state:', new_obs['state'].shape)
        print('  actions:', actions.shape)
        print('  action_format:', action_format)

        if args.send_action:
            _send_actions_to_server(server, obs, actions, args.action_rate, action_format)
            print('Action sent to robot server')

        step_count += 1
        if args.max_steps > 0 and step_count >= args.max_steps:
            return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
