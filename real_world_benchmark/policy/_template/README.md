# Template Policy Integration (A-side)

Copy this directory to ``policy/<YourPolicyName>/`` and implement your model.

## Quick start

1. Copy the template:
   ```bash
   cp -r policy/_template policy/YourPolicyName
   ```
2. Edit ``policy/YourPolicyName/policy.py`` — keep class name ``Policy``.
3. Serve:
   ```bash
   python -m real_world_benchmark.serve_policy_worldarena \
     policy/YourPolicyName/policy.py --host 0.0.0.0 --port 8000
   ```

For a zero-weight runnable smoke policy, use ``examples/policy_template``.

See the repo root ``README.md`` for Hub worker mode, ``worker-key`` alignment,
and ``action_format`` conventions.
