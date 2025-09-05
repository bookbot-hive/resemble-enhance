#!/bin/bash

# Monitor training script - runs for 15 minutes with checks every 3 minutes
echo "=== Starting monitoring of enhancer stage 2 training ==="
echo "Will check every 3 minutes for the next 5 hour"
echo ""

# Create fresh log file
rm -f enhancer_stage2_training.log
touch enhancer_stage2_training.log
echo "=== Fresh log file created at $(date) ===" > enhancer_stage2_training.log

START_TIME=$(date +%s)
CHECK_NUM=1

while [ $CHECK_NUM -le 100 ]; do
    CURRENT_TIME=$(date "+%H:%M:%S")
    echo "[$CURRENT_TIME] Check #$CHECK_NUM:"
    
    # Check if process is running
    if ps aux | grep -E "resemble_enhance.enhancer.train" | grep -v grep > /dev/null; then
        PID=$(ps aux | grep -E "python -m resemble_enhance.enhancer.train" | grep -v grep | head -1 | awk '{print $2}')
        CPU=$(ps aux | grep -E "python -m resemble_enhance.enhancer.train" | grep -v grep | head -1 | awk '{print $3}')
        MEM=$(ps aux | grep -E "python -m resemble_enhance.enhancer.train" | grep -v grep | head -1 | awk '{print $6}')
        MEM_MB=$((MEM / 1024))
        
        echo "  ✓ Training is RUNNING"
        echo "  - PID: $PID"
        echo "  - CPU Usage: ${CPU}%"
        echo "  - Memory: ${MEM_MB} MB"
        
        # Show last few lines of log
        echo "  - Latest log entries:"
        tail -n 3 enhancer_stage2_training.log | sed 's/^/    /'
    else
        echo "  ✗ Training process NOT FOUND"
        echo "  - Checking log for errors..."
        tail -n 5 enhancer_stage2_training.log | sed 's/^/    /'
    fi
    
    echo ""
    
    
    # Wait 3 minutes before next check
    echo "Waiting 3 minutes until next check..."
    sleep 180
    
    CHECK_NUM=$((CHECK_NUM + 1))
done
