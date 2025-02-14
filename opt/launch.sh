#!/bin/bash

echo Launching experiment $1
echo GPU $2
echo EXTRA ${@:3}

# CKPT_DIR=ckpt/$1
CKPT_DIR=$3/ckpt/$1
# CKPT_DIR=$3/ckpt # Changed to project folder
mkdir -p $CKPT_DIR
NOHUP_FILE=$CKPT_DIR/log
echo CKPT $CKPT_DIR
echo LOGFILE $NOHUP_FILE
echo "TESTING"
CUDA_VISIBLE_DEVICES=$2 nohup python -u opt.py -t $CKPT_DIR ${@:3} --log_depth_map > $NOHUP_FILE 2>&1 &
# CUDA_VISIBLE_DEVICES=$2 nohup python -u opt.py -t $CKPT_DIR ${@:3} > $NOHUP_FILE 2>&1 
echo DETACH
