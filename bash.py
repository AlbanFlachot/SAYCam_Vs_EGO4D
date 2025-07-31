import subprocess
import shlex
import os


for model in ['ego', 'saycam', 'random', 'imagenet', 's', 'a', 'y']:
#for model in ['supervised']:
    for dataset in ['MEG_face_stim', '100_faces', 'hmIT_stimuli', '100_objects']:
    #for dataset in ['MEG_face_stim']:
        gpu = 3
        # command = f"python3 save_activations.py --model {model} --dataset {dataset}" # if don't specify GPU
        command = f"python3 save_activations.py --model {model} --dataset {dataset} --cuda --gpuid {gpu}" # if want to use a GPU
        args = shlex.split(command)
        subprocess.call(args)

