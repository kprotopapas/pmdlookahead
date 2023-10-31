#!/bin/bash


while getopts t:w:e:dch flag
do
    case "${flag}" in
        t) run_type=${OPTARG};;
        w) weights=${OPTARG};;
        d) debug='-m pdb';;
        c) cumulative='--is_cumulative_mode=true';;
        e) env_name=${OPTARG};;
        h) help=true
    esac
done

if [[ $help ]]
then
    echo 'run.sh -t run_type -e env_name [-w weights -d debug -c cumulative]'
    exit 1
elif [[ "$run_type" == "train" ]]
then
    echo 'Training'
    python $debug main.py --env_name=$env_name --tree_depth=2 --run_type=train --total_timesteps=300000 $cumulative
    cp saved_agents/* /workspace/backup
else 
    echo 'Testing'
    if [[ "$weights" == "" ]]
    then
        echo "No weight file specified, missing option -w"
    
    else
        echo "Using weights from $weights"
        python $debug main.py --env_name=$env_name --tree_depth=2 --run_type=evaluate --model_filename=$weights
    fi
fi