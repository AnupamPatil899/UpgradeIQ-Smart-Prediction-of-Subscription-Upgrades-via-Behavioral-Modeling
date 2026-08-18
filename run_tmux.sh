#!/usr/bin/env bash
# ==============================================================================
# run_tmux.sh — Launch Multi-Day Retraining Inside Detached Tmux Session
# ==============================================================================

set -e

SESSION_NAME="upgradeiq-retrain"
LOG_FILE="retraining.log"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

cd "$DIR"

# Parse CLI arguments or pass through
ARGS="$@"
if [ -z "$ARGS" ]; then
    ARGS="--mode tune --model xgb --n-trials 200 --startup-trials 30 --timeout-hours 72 --cv-folds 5"
fi

echo "=============================================================================="
echo "🚀 UpgradeIQ Multi-Day Retraining Orchestrator"
echo "=============================================================================="
echo "Directory:     $DIR"
echo "Session Name:  $SESSION_NAME"
echo "Log File:      $DIR/$LOG_FILE"
echo "Command Args:  $ARGS"
echo "=============================================================================="

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "⚠️  tmux is not installed on this system. Running directly in foreground..."
    python3 train_master.py $ARGS 2>&1 | tee -a "$LOG_FILE"
    exit 0
fi

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  Tmux session '$SESSION_NAME' is already running!"
    echo ""
    echo "To attach to it:"
    echo "   tmux attach -t $SESSION_NAME"
    echo "To kill existing session:"
    echo "   tmux kill-session -t $SESSION_NAME"
    echo "To view live logs:"
    echo "   tail -f $LOG_FILE"
    exit 1
fi

# Launch in tmux
echo "▶️ Launching training run in background tmux session '$SESSION_NAME'..."
tmux new-session -d -s "$SESSION_NAME" "python3 train_master.py $ARGS 2>&1 | tee -a $LOG_FILE"

echo ""
echo "✅ Retraining successfully started in the background!"
echo "------------------------------------------------------------------------------"
echo "📌 Useful Commands:"
echo "   1. Attach to tmux session:     tmux attach -t $SESSION_NAME"
echo "      (To detach without stopping training, press: Ctrl+b, then d)"
echo "   2. Monitor live logs:          tail -f $DIR/$LOG_FILE"
echo "   3. Launch Optuna Web UI:       optuna-dashboard sqlite:///optuna_study.db"
echo "   4. Stop training anytime:      tmux kill-session -t $SESSION_NAME"
echo "------------------------------------------------------------------------------"
