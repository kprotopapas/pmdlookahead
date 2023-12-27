#!/bin/bash

python batch.py mcts --alg banditts -s 32 -d $1 --group mcts_v5 --epochs 10000 -n 10 -t 100
