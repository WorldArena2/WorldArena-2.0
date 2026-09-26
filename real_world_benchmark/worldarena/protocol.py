"""WorldArena 2.0 protocol version and endpoint constants."""

from __future__ import annotations

SCHEMA_VERSION = 'worldarena.v1'

POLICY_PROTOCOL_VERSION = 'wa-policy-v1'
ROBOT_PROTOCOL_VERSION = 'wa-robot-v1'
# Cross-network HTTP Hub: wa-hub-v1 — see worldarena/hub_protocol.py and docs/wa-hub-v1-api.md.

# Policy-side endpoints (wa-policy-v1).
POLICY_ENDPOINT_INFER = 'infer'
POLICY_ENDPOINT_RESET = 'reset'
POLICY_ENDPOINT_HEALTH = 'health'

# Robot-side endpoints (wa-robot-v1).
ROBOT_ENDPOINT_GET_OBSERVATION = 'get_observation'
ROBOT_ENDPOINT_APPLY_ACTION = 'apply_action'
ROBOT_ENDPOINT_RESET = 'reset'
ROBOT_ENDPOINT_HEALTH = 'health'
ROBOT_ENDPOINT_REPORT_EVENT = 'report_event'

# Camera role names (role-based naming).
CAMERA_ROLE_GLOBAL = 'global'
CAMERA_ROLE_LEFT_WRIST = 'left_wrist'
CAMERA_ROLE_RIGHT_WRIST = 'right_wrist'
CAMERA_ROLE_SIDE = 'side'
CAMERA_ROLE_TOP = 'top'
CAMERA_ROLE_FRONT = 'front'
CAMERA_ROLE_ARM = 'arm'
CAMERA_ROLE_HEAD = 'head'

# Action modes.
ACTION_MODE_TASK_SPACE_DELTA = 'task_space_delta'
ACTION_MODE_TASK_SPACE_ABSOLUTE = 'task_space_absolute'
ACTION_MODE_JOINT_DELTA = 'joint_delta'
ACTION_MODE_JOINT_ABSOLUTE = 'joint_absolute'
ACTION_MODE_GRIPPER_ONLY = 'gripper_only'

# Event types.
EVENT_EPISODE_START = 'episode_start'
EVENT_EPISODE_END = 'episode_end'
EVENT_RESET = 'reset'
EVENT_SUCCESS = 'success'
EVENT_FAILURE = 'failure'
EVENT_TIMEOUT = 'timeout'
EVENT_ABORT = 'abort'
EVENT_SAFETY_INTERVENTION = 'safety_intervention'
EVENT_NETWORK_DEGRADED = 'network_degraded'

# Arm identifiers.
ARM_ID_LEFT = 'left'
ARM_ID_RIGHT = 'right'
ARM_ID_MAIN = 'main'

# Tactile roles (role-based naming, aligned with Table30 V2).
TACTILE_ROLE_LEFT_GRIPPER = 'left_gripper'
TACTILE_ROLE_RIGHT_GRIPPER = 'right_gripper'
TACTILE_ROLE_LEFT_WRIST = 'left_wrist'
TACTILE_ROLE_RIGHT_WRIST = 'right_wrist'
TACTILE_ROLE_PALM = 'palm'

# Tactile profiles (see Notion §二十七).
TACTILE_PROFILE_RAW = 'tactile_raw'
TACTILE_PROFILE_DERIVED = 'tactile_derived'
TACTILE_PROFILE_RAW_PLUS_DERIVED = 'tactile_raw+tactile_derived'

# Tactile field types (first release).
TACTILE_FIELD_RECTIFY_BGR = 'rectify_bgr'
TACTILE_FIELD_FORCE_XYZ = 'force_xyz'
TACTILE_FIELD_WRENCH_6D = 'wrench_6d'
TACTILE_FIELD_MESH3DFLOW = 'mesh3dflow'
TACTILE_FIELD_MARKER2D = 'marker2d'

# Vendor identifiers.
TACTILE_VENDOR_XENSE = 'xense'
