import torch
import torchvision.models as models
import pickle
from os.path import join, exists
import numpy as np

def loadndefine(dataset, cheackpoint_dir= '/home/alban/Documents/checkpoints'):
    if dataset == 'faces':
        # Load the pickle file
        with open(join(cheackpoint_dir, 'resnet50_scratch_weight.pkl'), 'rb') as f:
            state_dict = pickle.load(f)

        # Convert numpy arrays to torch tensors
        converted_state_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, np.ndarray):
                converted_state_dict[key] = torch.from_numpy(value)
            else:
                converted_state_dict[key] = value

        # Remove "module." prefix if present
        if any(key.startswith('module.') for key in converted_state_dict.keys()):
            new_state_dict = {key.replace('module.', ''): value for key, value in converted_state_dict.items()}
            converted_state_dict = new_state_dict

        # Create your custom model
        model = models.resnet50(num_classes=8631)

        # Load the weights
        model.load_state_dict(converted_state_dict)

    elif dataset == 'places':

        model = models.resnet50(num_classes=365)

        checkpoint = torch.load(join(cheackpoint_dir, 'resnet50_places365.pth.tar'), map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']

        # Remove the "module." prefix from all keys (this is from DataParallel)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value

        # Load the corrected state dict
        model.load_state_dict(new_state_dict)
    elif dataset == 'imagenet':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    print("Model loaded successfully!")
    return model