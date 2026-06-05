import json
import os
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision.transforms as transforms


_worker_hdf5_cache: Dict[int, Dict[str, h5py.File]] = {}


def worker_init_fn(worker_id):
    _worker_hdf5_cache[threading.get_ident()] = {}


def worker_clear_fn():
    """Close HDF5 handles cached by the current worker thread."""
    ident = threading.get_ident()
    if ident in _worker_hdf5_cache:
        for handle in _worker_hdf5_cache[ident].values():
            handle.close()
        del _worker_hdf5_cache[ident]


def _ensure_list(value: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _normalize_task_name(task_name: str) -> str:
    return str(task_name).strip().replace("-", "_").lower()


def _normalize_task_names(value: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    if value is None:
        return None

    if isinstance(value, str):
        raw_values = [value]
    else:
        raw_values = list(value)

    normalized: List[str] = []
    for raw_value in raw_values:
        if raw_value is None:
            continue
        for part in str(raw_value).split(","):
            normalized_part = _normalize_task_name(part)
            if normalized_part:
                normalized.append(normalized_part)

    if not normalized:
        return None

    return list(dict.fromkeys(normalized))


def _resolve_dataset_info_cache_path(
    dataset_info_cache_path: Optional[str], task_names: Optional[Sequence[str]]
) -> Optional[str]:
    if dataset_info_cache_path is None or not task_names:
        return dataset_info_cache_path

    cache_path = Path(dataset_info_cache_path)
    task_suffix = "__".join(task_names)
    safe_suffix = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in task_suffix)
    if cache_path.suffix:
        return str(cache_path.with_name(f"{cache_path.stem}.task_{safe_suffix}{cache_path.suffix}"))
    return str(cache_path.with_name(f"{cache_path.name}.task_{safe_suffix}"))


def _ensure_parent_dir(file_path: str) -> None:
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def _discover_hdf5_files(data_roots: Sequence[str]) -> List[str]:
    hdf5_paths: List[str] = []
    for root in data_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for suffix in ("*.hdf5", "*.h5"):
            hdf5_paths.extend(str(path) for path in root_path.rglob(suffix))
    return sorted(set(hdf5_paths))


def _read_dataset_length(hdf5_file: h5py.File) -> int:
    if "length" in hdf5_file.attrs:
        return int(hdf5_file.attrs["length"])
    if "length" in hdf5_file:
        return int(np.asarray(hdf5_file["length"][()]).item())
    candidates = [
        "tactile/left_gsmini/rgb",
        "tactile/right_gsmini/rgb",
        "tactile/left_gsmini/depth",
        "tactile/right_gsmini/depth",
    ]
    for key in candidates:
        if key in hdf5_file:
            return int(hdf5_file[key].shape[0])
    raise KeyError("Unable to infer episode length from HDF5 file.")


def _to_tensor_image(array: np.ndarray, resize: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    if array.ndim == 2:
        array = array[..., None]
    if array.shape[-1] not in (1, 3):
        raise ValueError(f"Expected last dim to be 1 or 3, got shape {array.shape}")
    image = Image.fromarray(array.astype(np.uint8) if array.dtype != np.uint8 else array)
    if resize is not None:
        image = image.resize((resize[1], resize[0]))
    tensor = transforms.ToTensor()(image)
    return tensor


def _load_hdf5_array(hdf5_file: h5py.File, path: str, index: int) -> np.ndarray:
    if path not in hdf5_file:
        raise KeyError(f"Missing dataset `{path}` in HDF5 file `{hdf5_file.filename}`")
    value = hdf5_file[path][index]
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _resolve_force_npy_path(hdf5_path: str, force_dirname: str = "force") -> str:
    """Resolve virtual-force npy path from an hdf5 path.

    Preferred mapping:
      .../<task>/clean/<demo>.hdf5 -> .../<task>/<force_dirname>/<demo>.npy
    Fallback mapping:
      same directory as hdf5 with suffix changed to .npy
    """
    p = Path(hdf5_path)
    candidate_paths: List[Path] = []

    if p.parent.name == "clean":
        candidate_paths.append(p.parent.parent / force_dirname / f"{p.stem}.npy")

    candidate_paths.append(p.with_suffix(".npy"))

    for cand in candidate_paths:
        if cand.exists():
            return str(cand)

    # Return preferred path for clearer error message downstream.
    if p.parent.name == "clean":
        return str(p.parent.parent / force_dirname / f"{p.stem}.npy")
    return str(p.with_suffix(".npy"))


class TactileHDF5Dataset(Dataset):
    """UniVTAC HDF5 dataset for Genie-Envisioner.

    Return fields are aligned with existing training datasets:
        - video: [C, V, T, H, W], where V=len(valid_cam)
        - actions: [T_action, 7], from embodiment/ee
        - state: [1, 9], from embodiment/joint at the latest memory frame
        - caption: str, generated from task folder name (e.g. insert_HDMI)
        - tactile: [C, V_tac, T, H, W], where V_tac=2 (left, right marker rgb)
    """

    def __init__(
        self,
        data_roots: Optional[Union[str, Sequence[str]]] = None,
        hdf5_paths: Optional[Sequence[str]] = None,
        domains: Optional[Sequence[str]] = None,
        task_names: Optional[Union[str, Sequence[str]]] = None,
        valid_cam: Optional[Sequence[str]] = None,
        samples_per_episode: int = 10,
        sample_size: Tuple[int, int] = (192, 256),
        sample_n_frames: int = 64,
        preprocess: str = "resize",
        chunk: int = 9,
        action_chunk: Optional[int] = None,
        # Legacy temporal params removed: dataset now selects a contiguous chunk.
        dataset_info_cache_path: Optional[str] = None,
        use_unified_prompt: bool = False,
        unified_prompt: str = "The robotic arm performs a precise insertion task with stable contact.",
        fix_epiidx: Optional[int] = None,
        fix_sidx: Optional[int] = None,
        preload_to_memory: bool = False,
        force_dirname: str = "force",
        max_episodes_per_task: Optional[int] = None,
        max_total_episodes: Optional[int] = None,
        fast_index: bool = False,
        rollout_mode: bool = False,
    ):
        super().__init__()

        if (chunk - 1) % 4 != 0:
            raise ValueError(
                f"chunk={chunk} 不满足 4n+1 格式。请使用 5, 9, 13, 17..."
            )

        if hdf5_paths is None:
            if data_roots is None:
                raise ValueError("Please provide either `data_roots` or `hdf5_paths`.")
            data_roots = _ensure_list(data_roots)
            if fast_index:
                hdf5_paths = None
            else:
                hdf5_paths = _discover_hdf5_files(data_roots)
        else:
            hdf5_paths = [str(path) for path in hdf5_paths]
        hdf5_paths = hdf5_paths[:100] #测试用

        self.sample_size = sample_size
        self.sample_n_frames = sample_n_frames
        self.chunk = int(chunk)
        self.action_chunk = int(action_chunk if action_chunk is not None else chunk)
        self.video_temporal_stride = self.action_chunk // self.chunk
        if self.chunk * self.video_temporal_stride != self.action_chunk:
            raise ValueError("action_chunk should be an integer multiple of chunk.")

        # legacy: `n_previous` and `previous_pick_mode` removed
        self.use_unified_prompt = use_unified_prompt
        self.unified_prompt = unified_prompt
        self.samples_per_episode = max(1, int(samples_per_episode))
        self.task_names = _normalize_task_names(task_names if task_names is not None else domains)
        self._task_name_filter = set(self.task_names) if self.task_names is not None else None
        self.dataset_info_cache_path = _resolve_dataset_info_cache_path(dataset_info_cache_path, self.task_names)
        if valid_cam is None:
            self.valid_cam = ["head", "wrist"]
        elif isinstance(valid_cam, (list, tuple)):
            self.valid_cam = list(valid_cam)
        else:
            self.valid_cam = [valid_cam]

        self.fix_epiidx = fix_epiidx
        self.fix_sidx = fix_sidx
        self.preload_to_memory = bool(preload_to_memory)
        self.force_dirname = force_dirname
        # Debug indexing limits: only include up to `max_episodes_per_task` per task
        # and up to `max_total_episodes` in total when building index.
        self.max_episodes_per_task = int(max_episodes_per_task) if max_episodes_per_task is not None else None
        self.max_total_episodes = int(max_total_episodes) if max_total_episodes is not None else None
        self.fast_index = bool(fast_index)
        self.rollout_mode = bool(rollout_mode)
        self._memory_dataset: Dict[str, Dict[str, np.ndarray]] = {}

        if preprocess == "center_crop_resize":
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(min(sample_size)),
                transforms.CenterCrop(sample_size),
            ])
        elif preprocess == "resize":
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(sample_size),
            ])
        else:
            raise NotImplementedError

        self.pixel_transforms_norm = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.dataset: List[Dict] = []
        if self.dataset_info_cache_path is not None and os.path.exists(self.dataset_info_cache_path):
            with open(self.dataset_info_cache_path, "r", encoding="utf-8") as f:
                self.dataset = json.load(f)
            self._filter_dataset_entries_by_task()
            if self._needs_cache_upgrade():
                self._upgrade_cached_dataset_entries()
        else:
            if hdf5_paths is None:
                if fast_index:
                    self._build_index_fast(
                        data_roots,
                        max_episodes_per_task=max_episodes_per_task,
                        max_total_episodes=max_total_episodes,
                    )
                else:
                    raise RuntimeError("Internal error: hdf5_paths is None without fast_index.")
            else:
                self._build_index(hdf5_paths)
            if self.dataset_info_cache_path is not None:
                _ensure_parent_dir(self.dataset_info_cache_path)
                with open(self.dataset_info_cache_path, "w", encoding="utf-8") as f:
                    json.dump(self.dataset, f)

        self.base_length = len(self.dataset)
        self.length = self.base_length * self.samples_per_episode

        if self.preload_to_memory:
            self._preload_all_to_memory()

    # Expose worker init function to DataLoader via class attribute.
    worker_init_fn = staticmethod(worker_init_fn)

    def _needs_cache_upgrade(self) -> bool:
        """Quick check for stale cache schema/content.

        We only trigger full upgrade when required, so normal runs avoid
        rescanning and rewriting cache every initialization.
        """
        if not isinstance(self.dataset, list):
            return True

        for item in self.dataset:
            if not isinstance(item, dict):
                return True
            hdf5_path = item.get("hdf5_path")
            force_npy_path = item.get("force_npy_path")
            length = item.get("length")

            if not hdf5_path or not force_npy_path:
                return True
            if not isinstance(length, int) or length <= 0:
                return True
            if not os.path.exists(hdf5_path) or not os.path.exists(force_npy_path):
                return True

        return False

    def _filter_dataset_entries_by_task(self) -> None:
        if self._task_name_filter is None:
            return

        filtered_dataset: List[Dict] = []
        for item in self.dataset:
            hdf5_path = item.get("hdf5_path") if isinstance(item, dict) else None
            if not hdf5_path:
                continue
            try:
                task_name = self._task_name_from_path(hdf5_path)
            except Exception:
                continue
            if _normalize_task_name(task_name) in self._task_name_filter:
                filtered_dataset.append(item)

        self.dataset = filtered_dataset

    def _task_name_allowed(self, task_name: str) -> bool:
        return self._task_name_filter is None or _normalize_task_name(task_name) in self._task_name_filter

    def _upgrade_cached_dataset_entries(self) -> None:
        """Backfill missing fields in old cache format and write cache back if changed.

        Old cache may only contain: {hdf5_path, length}
        New cache requires:       {hdf5_path, force_npy_path, length}
        """
        if not isinstance(self.dataset, list):
            self.dataset = []
            return

        upgraded: List[Dict] = []
        changed = False

        for item in tqdm(self.dataset, desc="Upgrading tactile cache entries"):
            if not isinstance(item, dict) or "hdf5_path" not in item:
                changed = True
                continue

            hdf5_path = str(item["hdf5_path"])
            if not os.path.exists(hdf5_path):
                changed = True
                continue

            force_npy_path = item.get("force_npy_path", None)
            if not force_npy_path:
                force_npy_path = _resolve_force_npy_path(hdf5_path, self.force_dirname)
                changed = True

            if not os.path.exists(force_npy_path):
                changed = True
                continue

            try:
                force_length = int(np.load(force_npy_path, mmap_mode="r").shape[0])
                hdf5_length = int(item.get("length", 0))
                if hdf5_length <= 0:
                    with h5py.File(hdf5_path, "r") as h5f:
                        hdf5_length = _read_dataset_length(h5f)
                    changed = True
                length = min(hdf5_length, force_length)
            except Exception:
                changed = True
                continue

            if item.get("length") != length:
                changed = True
            if item.get("force_npy_path") != force_npy_path:
                changed = True

            upgraded.append(
                {
                    "hdf5_path": hdf5_path,
                    "force_npy_path": force_npy_path,
                    "length": length,
                }
            )

        self.dataset = upgraded
        self._filter_dataset_entries_by_task()

        if changed and self.dataset_info_cache_path is not None:
            _ensure_parent_dir(self.dataset_info_cache_path)
            with open(self.dataset_info_cache_path, "w", encoding="utf-8") as f:
                json.dump(self.dataset, f)

    def _preload_all_to_memory(self) -> None:
        for item in tqdm(self.dataset, desc="Preloading HDF5 to RAM"):
            hdf5_path = item["hdf5_path"]
            with h5py.File(hdf5_path, "r") as h5f:
                mem: Dict[str, np.ndarray] = {}

                # Video views configured by valid_cam.
                for cam in self.valid_cam:
                    path = f"observation/{cam}/rgb"
                    if path not in h5f:
                        raise KeyError(f"Missing video view `{path}` in HDF5 file `{hdf5_path}`")
                    mem[path] = np.asarray(h5f[path])

                # Tactile marker rgb (left / right).
                for side in ["left", "right"]:
                    path = f"tactile/{side}_gsmini/rgb_marker"
                    if path not in h5f:
                        raise KeyError(f"Missing tactile view `{path}` in HDF5 file `{hdf5_path}`")
                    mem[path] = np.asarray(h5f[path])

                # Proprio/action fields.
                mem["embodiment/ee"] = np.asarray(h5f["embodiment/ee"])
                mem["embodiment/joint"] = np.asarray(h5f["embodiment/joint"])

            force_npy_path = item.get("force_npy_path", None)
            if not force_npy_path:
                force_npy_path = _resolve_force_npy_path(hdf5_path, self.force_dirname)
                item["force_npy_path"] = force_npy_path
            mem["virtual_force"] = np.asarray(np.load(force_npy_path))

            self._memory_dataset[hdf5_path] = mem

    def _build_index(self, hdf5_paths: Sequence[str]) -> None:
        # Support debug indexing limits: cap number of episodes per task
        # and total episodes when requested. This lets debug runs load only
        # a small subset (e.g. 2 per task, 16 total) instead of scanning all files.
        counts: Dict[str, int] = {}
        total_added = 0

        for hdf5_path in tqdm(hdf5_paths, desc="Loading HDF5 metadata"):
            # If global cap reached, stop early
            if self.max_total_episodes is not None and total_added >= self.max_total_episodes:
                break

            try:
                with h5py.File(hdf5_path, "r") as hdf5_file:
                    hdf5_length = _read_dataset_length(hdf5_file)

                force_npy_path = _resolve_force_npy_path(hdf5_path, self.force_dirname)
                if not os.path.exists(force_npy_path):
                    continue

                force_length = int(np.load(force_npy_path, mmap_mode="r").shape[0])
                length = min(hdf5_length, force_length)
            except Exception:
                continue

            # Per-task cap: determine task name from path and skip if exceeded
            try:
                task_name = self._task_name_from_path(hdf5_path)
            except Exception:
                task_name = ""

            if not self._task_name_allowed(task_name):
                continue

            if self.max_episodes_per_task is not None:
                cnt = counts.get(task_name, 0)
                if cnt >= self.max_episodes_per_task:
                    continue

            self.dataset.append({
                "hdf5_path": hdf5_path,
                "force_npy_path": force_npy_path,
                "length": length,
            })

            counts[task_name] = counts.get(task_name, 0) + 1
            total_added += 1

    def _build_index_fast(
        self,
        data_roots: Sequence[str],
        max_episodes_per_task: Optional[int] = None,
        max_total_episodes: Optional[int] = None,
    ) -> None:
        counts: Dict[str, int] = {}
        total_added = 0

        for root in data_roots:
            if max_total_episodes is not None and total_added >= max_total_episodes:
                break

            root_path = Path(root)
            if not root_path.exists():
                continue

            for suffix in ("*.hdf5", "*.h5"):
                for hdf5_path_obj in root_path.rglob(suffix):
                    if max_total_episodes is not None and total_added >= max_total_episodes:
                        return

                    hdf5_path = str(hdf5_path_obj)

                    try:
                        with h5py.File(hdf5_path, "r") as hdf5_file:
                            hdf5_length = _read_dataset_length(hdf5_file)

                        force_npy_path = _resolve_force_npy_path(hdf5_path, self.force_dirname)
                        if not os.path.exists(force_npy_path):
                            continue

                        force_length = int(np.load(force_npy_path, mmap_mode="r").shape[0])
                        length = min(hdf5_length, force_length)
                    except Exception:
                        continue

                    try:
                        task_name = self._task_name_from_path(hdf5_path)
                    except Exception:
                        task_name = ""

                    if not self._task_name_allowed(task_name):
                        continue

                    if max_episodes_per_task is not None:
                        cnt = counts.get(task_name, 0)
                        if cnt >= max_episodes_per_task:
                            continue

                    self.dataset.append({
                        "hdf5_path": hdf5_path,
                        "force_npy_path": force_npy_path,
                        "length": length,
                    })

                    counts[task_name] = counts.get(task_name, 0) + 1
                    total_added += 1

    def __len__(self) -> int:
        return self.length

    def _get_worker_cache(self) -> Dict[str, h5py.File]:
        thread_id = threading.get_ident()
        if thread_id not in _worker_hdf5_cache:
            _worker_hdf5_cache[thread_id] = {}
        return _worker_hdf5_cache[thread_id]

    def _open_hdf5(self, hdf5_path: str) -> h5py.File:
        cache = self._get_worker_cache()
        if hdf5_path not in cache:
            cache[hdf5_path] = h5py.File(hdf5_path, "r")
        return cache[hdf5_path]

    def _decode_jpeg_bytes(self, value: Union[bytes, np.bytes_]) -> np.ndarray:
        arr = np.frombuffer(bytes(value), dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode image bytes from HDF5.")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _get_value_at(self, h5f: Optional[h5py.File], hdf5_path: str, path: str, index: int):
        if self.preload_to_memory:
            return self._memory_dataset[hdf5_path][path][index]
        assert h5f is not None
        return h5f[path][index]

    def _get_array(self, h5f: Optional[h5py.File], hdf5_path: str, path: str, indexes: Sequence[int]) -> np.ndarray:
        if self.preload_to_memory:
            return np.asarray(self._memory_dataset[hdf5_path][path][list(indexes)])
        assert h5f is not None
        return np.asarray(h5f[path][list(indexes)])

    def _get_force_array(self, item: Dict, indexes: Sequence[int]) -> np.ndarray:
        hdf5_path = item["hdf5_path"]
        if self.preload_to_memory:
            return np.asarray(self._memory_dataset[hdf5_path]["virtual_force"][list(indexes)])

        force_npy_path = item.get("force_npy_path", None)
        if not force_npy_path:
            force_npy_path = _resolve_force_npy_path(hdf5_path, self.force_dirname)
            item["force_npy_path"] = force_npy_path
        force = np.load(force_npy_path)
        return np.asarray(force[list(indexes)])

    def _preprocess_rgb(self, img_hwc_uint8: np.ndarray) -> torch.Tensor:
        img = Image.fromarray(img_hwc_uint8)
        img = self.pixel_transforms_resize(img)
        img = transforms.ToTensor()(img)
        img = self.pixel_transforms_norm(img)
        return img

    def _task_name_from_path(self, hdf5_path: str) -> str:
        p = Path(hdf5_path)
        if len(p.parents) >= 2:
            return p.parents[1].name
        return p.stem

    def _task_to_caption(self, task_name: str) -> str:
        if self.use_unified_prompt:
            return self.unified_prompt

        token = task_name.replace("-", "_")
        parts = [x for x in token.split("_") if x]

        task_captions = {
            "collect": "The robotic arm collects contact-rich tactile data for pretraining, deliberately exploring the object and surface with stable, informative contact.",
            "lift_bottle": "The robotic arm grasps a bottle and lifts it off a surface near a wall, maintaining careful contact control to avoid collisions.",
            "lift_can": "The robotic arm grasps a cylindrical can and lifts it smoothly from the surface with precise force and grasp stabilization.",
            "insert_HDMI": "The robotic arm aligns an HDMI connector and inserts it into the port with careful tactile guidance and fine pose correction.",
            "insert_hole": "The robotic arm performs precise peg-in-hole insertion, using tactile feedback to align the peg and complete the insertion accurately.",
            "insert_tube": "The robotic arm inserts a tube into a fixture, adjusting alignment and contact force to achieve a secure fit.",
            "pull_out_key": "The robotic arm extracts a key from a lock by applying controlled pulling force while maintaining stable grasp and alignment.",
            "put_bottle_in_shelf": "The robotic arm places a bottle onto a shelf with careful positioning, controlled release, and stable placement.",
            "grasp_classify": "The robotic arm grasps an object and classifies it using tactile feedback, inferring object properties from the contact signals.",
        }

        normalized_task = token.lower()
        if normalized_task in task_captions:
            return task_captions[normalized_task]

        if len(parts) >= 2 and parts[0].lower() == "insert":
            obj = " ".join(parts[1:])
            return f"The robotic arm aligns and inserts the {obj} with precise tactile guidance and stable contact control."
        if len(parts) >= 2 and parts[0].lower() in {"pick", "pickup"}:
            obj = " ".join(parts[1:])
            return f"The robotic arm picks up the {obj} with careful grasping, force regulation, and stable lifting motion."
        if len(parts) >= 2 and parts[0].lower() == "place":
            obj = " ".join(parts[1:])
            return f"The robotic arm places the {obj} accurately at the target location with controlled motion and gentle release."

        words = " ".join(parts) if len(parts) > 0 else task_name
        return f"The robotic arm performs the task: {words}, using tactile feedback for careful manipulation and precise control."

    def _get_frame_indexes(self, total_frames: int) -> Tuple[List[int], List[int]]:
        # Rollout validation: return the full-length sequence
        if self.rollout_mode:
            indexes = list(range(total_frames))
            return indexes, indexes

        # Simplified: return a continuous block of `action_chunk` frames.
        # If `fix_sidx` and `fix_mem_idx` are provided, respect fixed indices.
        if self.fix_sidx is not None:
            action_indexes = list(range(self.fix_sidx, self.fix_sidx + self.action_chunk))
            frame_indexes = action_indexes[:: self.video_temporal_stride]
            return frame_indexes, action_indexes

        if total_frames < self.action_chunk:
            raise ValueError(f"Not enough frames ({total_frames}) for action_chunk={self.action_chunk}")

        start = int(np.random.randint(0, total_frames - self.action_chunk + 1))
        action_indexes = list(range(start, start + self.action_chunk))
        frame_indexes = action_indexes[:: self.video_temporal_stride]
        return frame_indexes, action_indexes

    def _get_batch(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        item = self.dataset[idx]
        hdf5_path = item["hdf5_path"]
        h5f = None if self.preload_to_memory else self._open_hdf5(hdf5_path)
        total_frames = int(item["length"])

        video_indexes, action_indexes = self._get_frame_indexes(total_frames)

        # video: configurable camera views
        video_views = []
        for cam in self.valid_cam:
            frames = []
            path = f"observation/{cam}/rgb"
            for t in video_indexes:
                img = self._decode_jpeg_bytes(self._get_value_at(h5f, hdf5_path, path, t))
                frames.append(self._preprocess_rgb(img))
            view_tensor = torch.stack(frames, dim=1)  # [C, T, H, W]
            video_views.append(view_tensor)
        video = torch.stack(video_views, dim=1)  # [C, V, T, H, W]

        # tactile: left + right rgb_marker
        tactile_views = []
        for side in ["left", "right"]:
            frames = []
            path = f"tactile/{side}_gsmini/rgb_marker"
            for t in video_indexes:
                img = self._decode_jpeg_bytes(self._get_value_at(h5f, hdf5_path, path, t))
                frames.append(self._preprocess_rgb(img))
            view_tensor = torch.stack(frames, dim=1)  # [C, T, H, W]
            tactile_views.append(view_tensor)
        tactile = torch.stack(tactile_views, dim=1)  # [C, V_tac, T, H, W]

        # actions/state mapping requested by user
        if self.rollout_mode:
            # In rollout validation we only need full video/tactile; keep
            # compatible shapes for actions/state/force in case callers inspect them.
            try:
                actions = torch.as_tensor(self._get_array(h5f, hdf5_path, "embodiment/ee", action_indexes), dtype=torch.float32)
            except Exception:
                actions = torch.zeros(len(action_indexes), 7, dtype=torch.float32)
            try:
                state_seq = torch.as_tensor(self._get_array(h5f, hdf5_path, "embodiment/joint", action_indexes), dtype=torch.float32)
            except Exception:
                state_seq = torch.zeros(len(action_indexes), 9, dtype=torch.float32)
            state = state_seq if len(state_seq) > 0 else torch.zeros(len(action_indexes), 9, dtype=torch.float32)
            try:
                virtual_force = torch.as_tensor(self._get_force_array(item, video_indexes), dtype=torch.float32)
            except Exception:
                virtual_force = torch.zeros(len(video_indexes), 3, dtype=torch.float32)
        else:
            actions = torch.as_tensor(self._get_array(h5f, hdf5_path, "embodiment/ee", action_indexes), dtype=torch.float32)
            state_seq = torch.as_tensor(self._get_array(h5f, hdf5_path, "embodiment/joint", action_indexes), dtype=torch.float32)
            # Use the full joint state sequence within the selected chunk
            state = state_seq
            virtual_force = torch.as_tensor(self._get_force_array(item, video_indexes), dtype=torch.float32)

        task_name = self._task_name_from_path(hdf5_path)
        caption = self._task_to_caption(task_name)

        return {
            "video": video,
            "actions": actions,
            "state": state,
            "caption": caption,
            "tactile": tactile,
            "virtual_force": virtual_force,
            "hdf5_path": hdf5_path,
            "task_name": task_name,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.fix_epiidx is not None:
            return self._get_batch(self.fix_epiidx)

        idx = int(idx) % self.base_length

        # retry logic for occasional broken file/frame
        # NOTE:
        #   之前是无限重试，若数据集整体不可读（路径缺失/字段缺失/解码失败），
        #   DataLoader worker 会一直循环，主进程表现为在 enumerate(...) 处“卡住”。
        #   这里改为有限重试并抛出最后一次异常，便于快速定位根因。
        max_retry = max(1, min(self.base_length, 64))
        last_error: Optional[Exception] = None
        last_hdf5_path = ""

        for _ in range(max_retry):
            try:
                return self._get_batch(idx)
            except Exception as exc:
                last_error = exc
                try:
                    last_hdf5_path = self.dataset[idx].get("hdf5_path", "")
                except Exception:
                    last_hdf5_path = ""
                idx = np.random.randint(0, self.base_length)

        raise RuntimeError(
            f"Failed to fetch a valid sample after {max_retry} retries. "
            f"Last hdf5_path={last_hdf5_path}. Last error: {repr(last_error)}"
        ) from last_error


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--hdf5_path", type=str, default=None)
    parser.add_argument("--task_names", type=str, nargs="+", default=None)
    parser.add_argument("--chunk", type=int, default=9)
    parser.add_argument("--sample_size", type=int, nargs=2, default=[320, 240])
    args = parser.parse_args()

    if args.hdf5_path is not None:
        dataset = TactileHDF5Dataset(
            hdf5_paths=[args.hdf5_path],
            chunk=args.chunk,
            sample_size=tuple(args.sample_size),
            task_names=args.task_names,
        )
    else:
        dataset = TactileHDF5Dataset(
            data_roots=[args.data_root],
            chunk=args.chunk,
            sample_size=tuple(args.sample_size),
            task_names=args.task_names,
        )

    print(f"Dataset length: {len(dataset)}")
    item = dataset[0]
    for key, value in item.items():
        if torch.is_tensor(value):
            print(key, value.shape, value.dtype)
        else:
            print(key, value)
