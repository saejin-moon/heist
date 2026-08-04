#!/bin/bash
LOG=/home/fuddle/git/heist/qmix_v5.log
STATUS=/home/fuddle/git/heist/qmix_v5.status
for i in $(seq 1 900); do
  if grep -q "training done" "$LOG"; then
    echo "QMIX_V5_DONE" >> "$STATUS"
    exit 0
  fi
  sleep 15
done
echo "QMIX_V5_TIMEOUT" >> "$STATUS"
