#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))

export MUJOCO_GL=${MUJOCO_GL:-"egl"}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-"egl"}
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export WAN_PATH=${WAN_PATH:-"${REPO_PATH}/diffsynth-studio"}
export PYTHONPATH=${WAN_PATH}:${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME=${CONFIG_NAME:-"env/wan_robotwin_adjust_bottle_http"}
else
    CONFIG_NAME=$1
fi

HOST=${WAN_HTTP_HOST:-"127.0.0.1"}
PORT=${WAN_HTTP_PORT:-"18080"}

echo "Starting Wan HTTP server at http://${HOST}:${PORT}"
echo "Using config ${CONFIG_NAME}"
python -m rlinf.envs.world_model.wan_http_server \
    --config-name "${CONFIG_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    "${@:2}"
