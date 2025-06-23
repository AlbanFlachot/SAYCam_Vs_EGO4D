import torch
import huggingface_hub
import utils

device = torch.device("cpu")

print('List of models trained on EGO4D')
indexes_ego = [i for i, x in enumerate(utils.get_available_models()) if 'ego' in x]
print( [utils.get_available_models()[i] for i in indexes_ego])

print('List of models trained on SAYCam')
indexes_say = [i for i, x in enumerate(utils.get_available_models()) if 'say' in x]
print([utils.get_available_models()[i] for i in indexes_say])

print('Only 2 compatible models are thus dino_ego4d-200h_vitb14 and dino_say_vitb14')

from utils import *

model_say = load_model('dino_say_vitb14')
#model_ego = load_model('dino_ego4d-200h_vitb14')

## Print model
print(model_say)
#print(model_ego)

img = preprocess_image("imgs/img_0.jpg", 1400)
with torch.no_grad():
    visualize_attentions(model_say, img, patch_size=14)
