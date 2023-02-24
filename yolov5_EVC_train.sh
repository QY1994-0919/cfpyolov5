#!/bin/bash 

#SBATCH -J train  # 作业名为 test

#SBATCH -p defq  # 提交到 defq 队列


#SBATCH -N 1     # 使用 1 个节点

#SBATCH --ntasks-per-node=4  # 每个节点开启 6 个进程

#SBATCH --cpus-per-task=2    # 每个进程占用一个 cpu 核心

#SBATCH -t 7-24:00:00 # 任务最大运行时间是 10 分钟 (非必需项) 48:00:00


#SBATCH --gres=gpu:1    # 如果是gpu任务需要在此行定义gpu数量,此处为1

module load cuda11.0/toolkit/11.0.3


python train.py --data data/coco.yaml --cfg models/yolov5s_EVC.yaml --weights weights/yolov5s.pt --batch-size 16 --device 0,1

#python train.py --data data/coco128.yaml --cfg models/yolov5s_EVC.yaml --weights weights/yolov5s.pt --batch-size 16 --device cpu

