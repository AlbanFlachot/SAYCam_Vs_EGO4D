import subprocess
import shlex
import os


for model in ['ego', 'saycam', 'random', 'imagenet', 's', 'a', 'y']:
    for dataset in ['100_faces', '100_objects']:
        gpu = 5
        # command = f"python3 save_activations.py --model {model} --dataset {dataset}" # if don't specify GPU
        command = f"python3 save_activations.py --model {model} --dataset {dataset} --cuda --gpuid {gpu}" # if want to use a GPU
        args = shlex.split(command)
        subprocess.call(args)

