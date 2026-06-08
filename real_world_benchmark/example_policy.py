"""
示例 Policy 实现（最小、可运行的示例）。
测试者只需提供同样签名的 `Policy` 类并实现 `infer(new_obs)`。
"""
from typing import Dict, Any
import numpy as np


class Policy:
    """Minimal example Policy.

    - __init__(action_dim=32): 保存输出维度
    - infer(new_obs): 接收 new_obs(dict)，返回 dict 包含 'actions'（numpy.ndarray）
    - 支持三视角输入：`cam_high` + `cam_left_wrist` + `cam_right_wrist`
    """

    def __init__(self, action_dim: int = 32, use_wrist_images: bool = True):
        self.action_dim = action_dim
        self.use_wrist_images = use_wrist_images

    def _image_stat(self, image: Any) -> np.ndarray:
        """Extract a tiny numeric descriptor from one image.

        返回 4 维特征：[mean_r, mean_g, mean_b, overall_mean]
        """
        arr = np.asarray(image)
        if arr.ndim != 3:
            return np.zeros((4,), dtype=np.float32)

        # 兼容 [0,1] float 或 [0,255] uint8
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

        if arr.shape[-1] >= 3:
            rgb = arr[..., :3]
            mean_rgb = rgb.mean(axis=(0, 1))
            overall = np.array([rgb.mean()], dtype=np.float32)
            return np.concatenate([mean_rgb.astype(np.float32), overall], axis=0)

        return np.zeros((4,), dtype=np.float32)

    def infer(self, new_obs: Dict[str, Any]) -> Dict[str, Any]:
        """生成动作：示例返回零动作或基于 state 填充简单值。

        Args:
            new_obs: dict，见 README.md 中的接口说明。
        Returns:
            dict with at least key 'actions' (numpy.ndarray)
        """
        # 尝试从 new_obs 获取 state 来填充动作的一部分
        state = None
        image_feat = np.zeros((12,), dtype=np.float32)  # 3 views x 4 dims
        if isinstance(new_obs, dict):
            state = new_obs.get('state', None)
            images = new_obs.get('images', {})
            # 兼容 first_frame
            if 'first_frame' not in new_obs and 'cam_high' in images:
                new_obs['first_frame'] = images['cam_high']

            # 三视角特征：cam_high + 左腕 + 右腕
            if isinstance(images, dict):
                cam_high = images.get('cam_high', new_obs.get('first_frame', None))
                left_wrist = images.get('cam_left_wrist', None)
                right_wrist = images.get('cam_right_wrist', None)

                f_high = self._image_stat(cam_high) if cam_high is not None else np.zeros((4,), dtype=np.float32)
                f_left = self._image_stat(left_wrist) if left_wrist is not None else np.zeros((4,), dtype=np.float32)
                f_right = self._image_stat(right_wrist) if right_wrist is not None else np.zeros((4,), dtype=np.float32)
                image_feat = np.concatenate([f_high, f_left, f_right], axis=0)

        # 动作形状：(1, action_dim)
        actions = np.zeros((1, self.action_dim), dtype=np.float32)

        # 示例策略：如果 state 存在，复制前 min(len(state), action_dim) 维
        if state is not None:
            try:
                s = np.asarray(state, dtype=np.float32).ravel()
                n = min(s.size, self.action_dim)
                actions[0, :n] = s[:n]
            except Exception:
                pass

        # 可选：把三视角图像特征写入动作尾部，证明策略使用了 wrist image
        if self.use_wrist_images and self.action_dim > 0:
            m = min(image_feat.size, self.action_dim)
            actions[0, self.action_dim - m : self.action_dim] = image_feat[:m]

        return {
            'actions': actions,
            'policy_timing': {'infer_ms': None}
        }


if __name__ == '__main__':
    # quick local test
    p = Policy()
    sample_obs = {
        'images': {'cam_high': np.zeros((240, 320, 3), dtype=np.uint8)},
        'state': np.arange(32, dtype=np.float32),
    }
    out = p.infer(sample_obs)
    print('actions.shape=', out['actions'].shape)
