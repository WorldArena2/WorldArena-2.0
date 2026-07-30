# New Policy Integration Guide (A-side)

This document describes how to add a new policy worker on machine A for
WorldArena (`wa-policy-v1` / `wa-hub-v1`).

## Goals

- **Isolation**: each policy has its own directory, env file, and launch scripts.
- **Minimal surface**: only A-side protocol code is required; B/C internals are not needed.

## Directory layout

```text
policy/
└── YourPolicyName/
    ├── __init__.py
    ├── policy.py                # Policy class (required)
    ├── env.sh                   # optional env defaults
    └── scripts/
        └── serve.sh             # optional serve wrapper

examples/policy_template/        # zero-weight smoke example
```

## Required `Policy` interface

```python
class Policy:
    def __init__(self, config_path: Optional[str] = None):
        ...

    def reset(self, reset_info: Optional[dict] = None) -> None:
        ...

    def infer(self, new_obs: dict) -> dict:
        return {
            "actions": np.ndarray,          # shape (chunk_size, action_dim)
            "policy_timing": {"infer_ms": float},
            "policy_metadata": {
                "policy_id": str,
                "action_format": "joint" | "eef6d_single" | "eef6d",
                "action_dim": int,
                "control_arm": "right" | "left",  # single-arm formats
                "chunk_size": int,
            },
        }
```

Read the task instruction from `new_obs["prompt"]` (sourced from the B-side
task suite / `ObservationPacket.context.task_instruction`).

## Integration checklist

1. Copy the template:
   ```bash
   cp -r policy/_template policy/YourPolicyName
   ```
2. Implement `policy/YourPolicyName/policy.py`.
3. Smoke-load locally:
   ```bash
   python -c "from real_world_benchmark.policy_loader import load_policy; load_policy('policy/YourPolicyName/policy.py').Policy().infer({'prompt':'demo'})"
   ```
4. Serve (WebSocket):
   ```bash
   python -m real_world_benchmark.serve_policy_worldarena \
     policy/YourPolicyName/policy.py --host 0.0.0.0 --port 8000
   ```
5. Or Hub worker (align `--worker-key` with B-side config):
   ```bash
   python -m real_world_benchmark.serve_policy_worldarena \
     policy/YourPolicyName/policy.py \
     --hub-url https://<hub-host>/policy \
     --worker-key YourPolicyName
   ```

See the root `README.md` for `action_format` conventions and protocol docs under `docs/`.
