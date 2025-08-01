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
    from transformers import ViTImageProcessor, ViTForImageClassification
    from PIL import Image
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model_supervised = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
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
    ### the supervised model is from google and thus different procedure
    if opt.model == 'supervised':
        image = Image.open(imgp)
        if len(image.size) == 2: #greyscale
            image = image.convert("L")
            image = Image.merge("RGB", (image, image, image))
        # Process
        inputs = processor(images=image, return_tensors="pt")
        # Extract CLS token
        model_supervised.eval()
        with torch.no_grad():
            outputs = model_supervised(**inputs, output_hidden_states=True)
            cls_token = [outputs.hidden_states[l][0, 0] for l in range(1, 13)] # ignore 1st input layer to extract only 12 cls_tokens,
            np.save(join(savedir_path, f'cls_token_{image_name}.npy'), np.array(cls_token))
    else:
        img = preprocess_image(imgp, 224)
        with torch.no_grad():
            ### visualize attention over blocks
            cls_token= retrieve_tokens(model, img, device = device)
            np.save(join(savedir_path, f'cls_token_{image_name}.npy'), cls_token)


