import subprocess
import shlex
import os

model = 'ego'
dataset = 'hmIT_stimuli'
command = f"python3 save_activations.py --model {model} --dataset {dataset}"
args = shlex.split(command)
subprocess.call(args)