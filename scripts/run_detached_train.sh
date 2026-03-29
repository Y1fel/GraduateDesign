#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_detached_train.sh <teacher|mobile> [extra python args]"
  exit 1
fi

MODE="$1"
shift

case "$MODE" in
  teacher)
    TRAIN_ENTRY="scripts/train.py"
    ;;
  mobile)
    TRAIN_ENTRY="scripts/train_mobile.py"
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash scripts/run_detached_train.sh <teacher|mobile> [extra python args]"
    exit 1
    ;;
esac

LOG_DIR="$ROOT_DIR/outputs/nohup_logs"
mkdir -p "$LOG_DIR"

TS="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/${MODE}_train_${TS}.log"
PID_FILE="$LOG_DIR/${MODE}_train_latest.pid"
CMD_FILE="$LOG_DIR/${MODE}_train_latest.cmd"

CMD=(python -u "$TRAIN_ENTRY" "$@")
printf '%q ' "${CMD[@]}" > "$CMD_FILE"
echo >> "$CMD_FILE"

nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

echo "[INFO] Detached training started"
echo "[INFO] mode: $MODE"
echo "[INFO] pid: $PID"
echo "[INFO] log: $LOG_FILE"
echo "[INFO] pid file: $PID_FILE"
echo "[INFO] cmd file: $CMD_FILE"
echo "[INFO] monitor: tail -f '$LOG_FILE'"
echo "[INFO] check: ps -fp $PID"
echo "[INFO] stop: kill $PID"
