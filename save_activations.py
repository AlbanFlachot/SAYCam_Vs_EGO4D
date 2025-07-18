import torch
import huggingface_hub
from utils import *
import argparse
import glob
from os.path import join, exists
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='path to real images')
parser.add_argument('--model', help='name, either ego or saycam')
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--gpuid', default = 0, help='only when cuda is available')
opt = parser.parse_args()

if opt.cuda:
    device = torch.device(f"cuda:{opt.gpuid}")
else:
    device = torch.device("cpu")


# import model
if opt.model == 'ego':
    model = load_model('dino_ego4d-200h_vitb14')
elif opt.model == 'saycam':
    model = load_model('dino_say_vitb14')
elif opt.model == 'imagenet':
    model = load_model('dino_imagenet100_vitb14')
elif opt.model == "supervised":
    from transformers import ViTFeatureExtractor, ViTModel
    model = ViTModel.from_pretrained('google/vit-huge-patch14-224-in21k')
    print(model)
else:
    model = load_model(f'dino_{opt.model}_vitb14')

print(f"We will save the activations of the model trained on {opt.model} and tested on {opt.dataset}")
# create list of images to run
globimages = glob.glob('testsets/' + opt.dataset + '/*.png')
if len(globimages) < 1:
    globimages = glob.glob('testsets/' + opt.dataset + '/*.jpg')
if len(globimages) < 1:
    globimages = glob.glob('testsets/' + opt.dataset + '/*.tif')
globimages.sort()



# savedir path
root = '/data/alban/activations'
savedir = f'{opt.model}_{opt.dataset}'
savedir_path = join(root, savedir)

if not exists(savedir_path):
    os.makedirs(savedir_path)

for i, imgp in enumerate(globimages):
    image_name = imgp.split('/')[-1][:-4]
    print(f'Retrieving activations for image {i}/{len(globimages)}')
    img = preprocess_image(imgp, 1400)
    with torch.no_grad():
        ### visualize attention over blocks
        cls_token= retrieve_tokens(model, img, device = device)
        np.save(join(savedir_path, f'cls_token_{image_name}.npy'), cls_token)


