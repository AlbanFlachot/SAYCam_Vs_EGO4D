import subprocess
import shlex
import os

'''
### Save activations for RSA on specific datasets
for model in ['ego', 'saycam', 'random', 'imagenet', 's', 'a', 'y']:
#for model in ['supervised']:
    for dataset in ['MEG_face_stim', '100_faces', 'hmIT_stimuli', '100_objects']:
    #for dataset in ['MEG_face_stim']:
        gpu = 7
        # command = f"python3 save_activations.py --model {model} --dataset {dataset}" # if don't specify GPU
        command = f"python3 save_activations.py --model {model} --dataset {dataset} --cuda --gpuid {gpu}" # if want to use a GPU
        args = shlex.split(command)
        subprocess.call(args)'''

### Save activations for datadriven approach
for model in ['resnet']:#['ego', 'saycam', 'imagenet', 'supervised', 'random']:
#for model in ['supervised']:
    for dataset in ['ecoVal']:
    #for dataset in ['MEG_face_stim']:
        gpu = 0
        # command = f"python3 save_activations.py --model {model} --dataset {dataset}" # if don't specify GPU
        command = f"python3 save_activations_datadriven.py --model {model} --dataset {dataset} --cuda --gpuid {gpu}" # if want to use a GPU
        args = shlex.split(command)
        subprocess.call(args)