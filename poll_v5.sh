#!/bin/bash
# Poll ippo_v5.log until training finishes; append a checkpoint line to status file.
LOG=/home/fuddle/git/heist/ippo_v5.log
STATUS=/home/fuddle/git/heist/ippo_v5.status
rm -f "$STATUS"
for i in $(seq 1 600); do
  if grep -q "training done" "$LOG"; then
    echo "V5_DONE" >> "$STATUS"
    echo "JCODE_CHECKPOINT {\"message\":\"v5 training finished\"}" >> "$STATUS"
    exit 0
  fi
  # Emit a progress line ~every 4 min
  if [ $((i % 16)) -eq 0 ]; then
    STEP=$(grep -oP 'step=\K[0-9]+' "$LOG" | tail -1)
    EVAL=$(grep -oP 'eval@[0-9]+: win_rate=\S+ return=\S+' "$LOG" | tail -1)
    echo "JCODE_PROGRESS {\"kind\":\"checkpoint\",\"message\":\"step=${STEP} ${EVAL}\"}" >> "$STATUS"
  fi
  sleep 15
done
echo "V5_POLL_TIMEOUT" >> "$STATUS"
