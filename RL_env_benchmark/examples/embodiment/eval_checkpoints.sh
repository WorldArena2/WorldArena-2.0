#! /bin/bash
set -uo pipefail

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_DIR="${SCRIPT_DIR}/config"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_embodiment.sh"

usage() {
    echo "Usage:"
    echo "  bash examples/embodiment/eval_checkpoints.sh <checkpoints_dir> [base_config_name] [robot_platform]"
    echo
    echo "Example:"
    echo "  bash examples/embodiment/eval_checkpoints.sh \\"
    echo "    /manifold-obs/tangyinzhou/RLinf/logs/.../checkpoints \\"
    echo "    robotwin_click_bell_grpo_openpi_pi05_single_view_eval ALOHA"
}

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

CHECKPOINTS_DIR="$1"
BASE_CONFIG_NAME="${2:-robotwin_click_bell_grpo_openpi_pi05_single_view_eval}"
ROBOT_PLATFORM_ARG="${3:-${ROBOT_PLATFORM:-ALOHA}}"

BASE_CONFIG_PATH="${CONFIG_DIR}/${BASE_CONFIG_NAME}.yaml"

if [ ! -d "${CHECKPOINTS_DIR}" ]; then
    echo "Error: checkpoints directory not found: ${CHECKPOINTS_DIR}"
    exit 1
fi

if [ ! -f "${BASE_CONFIG_PATH}" ]; then
    echo "Error: base config not found: ${BASE_CONFIG_PATH}"
    exit 1
fi

if [ ! -f "${EVAL_SCRIPT}" ]; then
    echo "Error: eval script not found: ${EVAL_SCRIPT}"
    exit 1
fi

REPO_PATH="$(dirname "$(dirname "${SCRIPT_DIR}")")"
DATASET_ROOT="${REPO_PATH}/datasets/lerobot/rlinf"

TASK_NAME="$(python - "${BASE_CONFIG_PATH}" <<'PY'
import sys
import yaml

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

task_name = (
    cfg.get("env", {})
    .get("eval", {})
    .get("task_config", {})
    .get("task_name")
)
if not task_name:
    task_name = (
        cfg.get("env", {})
        .get("train", {})
        .get("task_config", {})
        .get("task_name")
    )
if not task_name:
    defaults = cfg.get("defaults", [])
    for item in defaults:
        text = str(item)
        if "click_bell" in text:
            task_name = "click_bell"
            break
        if "adjust_bottle" in text:
            task_name = "adjust_bottle"
            break
print(task_name or "")
PY
)"

if [ -z "${TASK_NAME}" ]; then
    case "${BASE_CONFIG_NAME} ${CHECKPOINTS_DIR}" in
        *click_bell*)
            TASK_NAME="click_bell"
            ;;
        *adjust_bottle*)
            TASK_NAME="adjust_bottle"
            ;;
    esac
fi

NORM_STATS_SRC=""
NORM_STATS_REL_DIR=""
if [ "${TASK_NAME}" = "click_bell" ]; then
    NORM_STATS_SRC="${DATASET_ROOT}/robotwin_headcam_click_bell/norm_stats.json"
    NORM_STATS_REL_DIR="robotwin_headcam_click_bell"
elif [ "${TASK_NAME}" = "adjust_bottle" ]; then
    NORM_STATS_SRC="/your/path//RLinf/pi05_jx/rlinf/robotwin_headcam_adjust_bottle/norm_stats.json"
    NORM_STATS_REL_DIR="robotwin_headcam_adjust_bottle"
else
    echo "Error: unsupported task_name '${TASK_NAME}'. Only click_bell and adjust_bottle are supported."
    exit 1
fi

if [ ! -f "${NORM_STATS_SRC}" ]; then
    echo "Error: norm_stats source file not found: ${NORM_STATS_SRC}"
    exit 1
fi

RUN_TAG="$(date +'%Y%m%d-%H%M%S')"
TMP_CONFIGS=()
FAILED_MODELS=()
TOTAL=0
SUCCESS=0
SUMMARY_FILE="${REPO_PATH}/logs/eval_checkpoints_summary_${RUN_TAG}.tsv"
mkdir -p "$(dirname "${SUMMARY_FILE}")"
{
    echo -e "checkpoint\tmodel_path\tstatus\tsuccess_at_end\tsuccess_once"
} > "${SUMMARY_FILE}"

cleanup() {
    for cfg in "${TMP_CONFIGS[@]}"; do
        if [ -f "${cfg}" ]; then
            rm -f "${cfg}"
        fi
    done
}
trap cleanup EXIT

readarray -t CKPT_CANDIDATES < <(
    if compgen -G "${CHECKPOINTS_DIR}/global_step_*" > /dev/null; then
        for d in "${CHECKPOINTS_DIR}"/global_step_*; do
            [ -d "${d}" ] && echo "${d}"
        done | sort -V
    else
        for d in "${CHECKPOINTS_DIR}"/*; do
            [ -d "${d}" ] && echo "${d}"
        done | sort -V
    fi
)

if [ ${#CKPT_CANDIDATES[@]} -eq 0 ]; then
    echo "Error: no checkpoint directories found under ${CHECKPOINTS_DIR}"
    exit 1
fi

echo "Found ${#CKPT_CANDIDATES[@]} checkpoint directories."
echo "Base config: ${BASE_CONFIG_NAME}"
echo "Robot platform: ${ROBOT_PLATFORM_ARG}"
echo "Task name: ${TASK_NAME}"
echo "Norm stats source: ${NORM_STATS_SRC}"
echo

for ckpt_dir in "${CKPT_CANDIDATES[@]}"; do
    TOTAL=$((TOTAL + 1))

    model_path="${ckpt_dir}"
    if [ -d "${ckpt_dir}/actor" ]; then
        model_path="${ckpt_dir}/actor"
    fi

    target_norm_dir="${model_path}/rlinf/${NORM_STATS_REL_DIR}"
    target_norm_file="${target_norm_dir}/norm_stats.json"
    mkdir -p "${target_norm_dir}"
    cp -f "${NORM_STATS_SRC}" "${target_norm_file}"
    echo "[$TOTAL/${#CKPT_CANDIDATES[@]}] Prepared norm_stats: ${target_norm_file}"

    safe_name="$(basename "${ckpt_dir}" | tr '/:' '__')"
    tmp_config_name="${BASE_CONFIG_NAME}__autoeval_${RUN_TAG}_${safe_name}"
    tmp_config_path="${CONFIG_DIR}/${tmp_config_name}.yaml"
    TMP_CONFIGS+=("${tmp_config_path}")

    python - "${BASE_CONFIG_PATH}" "${tmp_config_path}" "${model_path}" <<'PY'
import sys
import yaml

src, dst, model_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("actor", {}).setdefault("model", {})["model_path"] = model_path

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
PY

    echo "[$TOTAL/${#CKPT_CANDIDATES[@]}] Evaluating model_path=${model_path}"
    run_log="$(mktemp "/tmp/eval_ckpt_${RUN_TAG}_${safe_name}.XXXX.log")"
    cmd_ok=0
    if bash "${EVAL_SCRIPT}" "${tmp_config_name}" "${ROBOT_PLATFORM_ARG}" 2>&1 | tee "${run_log}"; then
        cmd_ok=1
    fi

    metrics="$(python - "${run_log}" <<'PY'
import re
import sys

log_path = sys.argv[1]
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

matches = re.findall(
    r"eval/success_at_end': array\(([-+0-9.eE]+).*?eval/success_once': array\(([-+0-9.eE]+)",
    text,
    flags=re.S,
)

if matches:
    end_val, once_val = matches[-1]
    print(f"{end_val} {once_val}")
else:
    print("NA NA")
PY
)"
    success_at_end="$(echo "${metrics}" | awk '{print $1}')"
    success_once="$(echo "${metrics}" | awk '{print $2}')"

    status="FAILED"
    if [ "${cmd_ok}" -eq 1 ] && [ "${success_at_end}" != "NA" ]; then
        status="SUCCESS"
        SUCCESS=$((SUCCESS + 1))
    else
        FAILED_MODELS+=("${model_path}")
    fi

    echo -e "$(basename "${ckpt_dir}")\t${model_path}\t${status}\t${success_at_end}\t${success_once}" >> "${SUMMARY_FILE}"
    echo "Result: ${status} (${model_path})"
    echo "Metrics: success_at_end=${success_at_end}, success_once=${success_once}"
    echo
done

echo "===================="
echo "Evaluation finished."
echo "Success: ${SUCCESS}/${TOTAL}"
echo "Summary file: ${SUMMARY_FILE}"
echo

python - "${SUMMARY_FILE}" <<'PY'
import csv
import math
import sys

summary_file = sys.argv[1]
rows = []
with open(summary_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rows.append(row)

def to_float(s):
    if s in ("", "NA", None):
        return None
    try:
        return float(s)
    except ValueError:
        return None

print("Per-checkpoint summary:")
for row in rows:
    print(
        f"  - {row['checkpoint']}: status={row['status']}, "
        f"success_at_end={row['success_at_end']}, success_once={row['success_once']}"
    )

valid = [r for r in rows if to_float(r["success_at_end"]) is not None]
if not valid:
    print("\nNo valid success metrics found.")
    sys.exit(0)

avg_end = sum(to_float(r["success_at_end"]) for r in valid) / len(valid)
avg_once = sum(to_float(r["success_once"]) for r in valid if to_float(r["success_once"]) is not None)
avg_once_den = sum(1 for r in valid if to_float(r["success_once"]) is not None)
avg_once = avg_once / avg_once_den if avg_once_den > 0 else float("nan")

best = max(valid, key=lambda r: to_float(r["success_at_end"]))

print("\nOverall:")
print(f"  - average success_at_end: {avg_end:.6f}")
if not math.isnan(avg_once):
    print(f"  - average success_once: {avg_once:.6f}")
else:
    print("  - average success_once: NA")
print(
    "  - best checkpoint by success_at_end: "
    f"{best['checkpoint']} ({to_float(best['success_at_end']):.6f})"
)
PY

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "Failed model paths:"
    for p in "${FAILED_MODELS[@]}"; do
        echo "  - ${p}"
    done
    exit 2
fi
