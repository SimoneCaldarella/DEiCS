# Change the falgs and paths according to your system

__MONODEPTH_PATH__ = 'drive/MyDrive/CVProject/monodepth2'
__NON_CONVERTED_DATASET_PATH__ = '/content/drive/MyDrive/CVProject/Dataset/640x360dataset/' # Not needed if your images are already 640x192 png images
__CONVERTED_DATASET_PATH__ = '/content/drive/MyDrive/CVProject/ConvertedDataset/'
__MODEL_CUSTOM_NAME__ = 'last_test' # Additional id for distinguish models
__MODEL_TO_BE_SAVED_PATH__ = {
	'encoder':f'/content/drive/MyDrive/CVProject/encoder_{__MODEL_CUSTOM_NAME__}.pt'
	'decoder':f'/content/drive/MyDrive/CVProject/depth_decoder_{__MODEL_CUSTOM_NAME__}.pt'
}

__SAVED_MODEL_PATH = __MODEL_TO_BE_SAVED_PATH__ # Usually it is used the last model saved

__CONVERT_MODE__ = False
__FINETUNING__ = False
__TEST_MODE__ = True


# -------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from google.colab.patches import cv2_imshow
from tqdm.notebook import tqdm
import random
import time

import torch
from torchvision import transforms, datasets
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch
import torchvision

import sys
sys.path.insert(1, __MONODEPTH_PATH__)

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# --------------------------------------------------------------------------
# Convert a dataset composed of different format 
# images to a dataset composed of 640x192 png images

def read_image_anyformat(image_path, feed_width, feed_height, gray=False):
    input_image = pil.open(image_path).convert('RGB' if not gray else 'L')
    original_width, original_height = input_image.size
    input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image)
    return input_image

def create_converted_dataset(dataset_path=__NON_CONVERTED_DATASET_PATH__, feed_width=640, feed_height=192):
    convDatPath = __CONVERTED_DATASET_PATH__
    for c in os.listdir(dataset_path):
        print(f'C={c}')
        if not os.path.exists(os.path.join(convDatPath, c)):
            os.mkdir(os.path.join(convDatPath, c))

        for i in os.listdir(os.path.join(dataset_path, c)):
            print(f'I={i}')
            if 'depth' in i:
                image = read_image_anyformat(os.path.join(dataset_path, c, i), feed_width, feed_height, gray=True)
            else:
                image = read_image_anyformat(os.path.join(dataset_path, c, i), feed_width, feed_height)
            save_image(image, f'{os.path.join(convDatPath, c, i).split(".")[0]}.png')

# --------------------------------------------------------------------------
# Create Dataset class and build the model

class CustomDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return [read_image(self.samples[idx][0])/255, read_image(self.samples[idx][1], torchvision.io.image.ImageReadMode.GRAY)/255]


def compose_dataset(path, valid_ratio):
    dataset = {int(k):{'samples':{}, 'temp':{'x':{}, 'y':{}}} for k in os.listdir(path)}

    valid_set = {}

    for k in dataset:
        
        img_names_k = os.listdir(os.path.join(path, str(k)))

        for img_name in img_names_k:
            if 'depth' in img_name:
                dataset[k]['temp']['y'][int(img_name.split('_')[1])] = os.path.join(path, str(k), img_name)
            else:
                dataset[k]['temp']['x'][int(img_name.split('_')[1].split('.')[0])] = os.path.join(path, str(k), img_name)

        for id in range(min(dataset[k]['temp']['x']), max(dataset[k]['temp']['x'])+1):

            if id in dataset[k]['temp']['x'] and id in dataset[k]['temp']['y']:
                dataset[k]['samples'][id] = [dataset[k]['temp']['x'][id], dataset[k]['temp']['y'][id]]

            elif (id in dataset[k]['temp']['x'] and not id in dataset[k]['temp']['y']) or (id not in dataset[k]['temp']['x'] and id in dataset[k]['temp']['y']):
                print(id, k)
                print(id in dataset[k]['temp']['x'] and id in dataset[k]['temp']['y'])
                print('-----')

        dataset[k].pop('temp')
        subsample = random.sample(list(dataset[k]['samples']), int(valid_ratio*len(dataset[k]['samples'])))
        valid_set[k] = {r:dataset[k]['samples'][r] for r in subsample}

        for id in valid_set[k]:
            dataset[k]['samples'].pop(id)

        dataset[k] = list(dataset[k]['samples'].values())
        valid_set[k] = list(valid_set[k].values())

    train_set = [img_path for cam in dataset for img_path in dataset[cam]]
    valid_set = [img_path for cam in valid_set for img_path in valid_set[cam]]
    
    return CustomDataset(train_set), CustomDataset(valid_set)

def preliminary_import(dataset_path):
    model_name = 'mono+stereo_640x192'
    pred_metric_depth = True

    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)
    print("-> Loading model from ", model_path)

    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.train()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
    num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.train()

    valid_ratio = 0.2
    bs = 64 # Batchsize

    train_set, valid_set = compose_dataset(dataset_path, valid_ratio)
    print("Dataset created")
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=bs)

    return encoder, depth_decoder, train_loader, valid_loader

# ----------------------------------------------------------------------

def fine_tuning():

    dataset_path = __CONVERTED_DATASET_PATH__
    encoder, depth_decoder, train_loader, valid_loader = preliminary_import(dataset_path) # Test to be added
    loss_fn = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam([{'params':encoder.parameters(), 'params':depth_decoder.parameters()}], lr=3e-4)
    epoch = 20
    epoch_loss_train = {}
    epoch_loss_valid = {}

    # Train-Validation epochs

    for e in range(epoch):

        print(f'Epoch: {e}')
        print('Train')
        epoch_loss_train[e] = 0
        pbar = tqdm(position=0, leave=True)
        pbar.reset(total=len(train_loader))
        encoder.train()
        depth_decoder.train()

        ss = time.time()

        # One train epoch ------------------------------------
        for x, y in train_loader:
            optimizer.zero_grad()
            x_im = x
            y_im = y.to(device)

            input_image = x_im.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)
            disp = outputs[("disp", 0)]

            # Vectorize the scaling
            vmax = disp.max(dim=-1).values.max(dim=-1).values.squeeze()
            vmin = disp.min(dim=-1).values.min(dim=-1).values.squeeze()
            inp_diff = vmax - vmin
            out_max = torch.ones(vmax.shape[0])
            out_min = torch.zeros(vmax.shape[0])
            out_diff = out_max-out_min

            # rescaled = (out_diff/inp_diff)*(disp - vmin)
            rescaled_vect = torch.stack([(out_diff[i]/inp_diff[i])*(disp[i,:,:,:] - vmin[i]) for i in range(vmax.shape[0])])

            loss = loss_fn(rescaled_vect, y_im)
            loss.backward()
            pbar.update()
            epoch_loss_train[e] += loss
            optimizer.step()
        
        epoch_loss_train[e] = epoch_loss_train[e].detach().cpu().numpy()/len(train_loader)
        print(f'Train loss: {epoch_loss_train[e]}')
        pbar.close()

        print('Valid')
        epoch_loss_valid[e] = 0
        pbar = tqdm(position=0, leave=True)
        pbar.reset(total=len(valid_loader))
        encoder.eval()
        depth_decoder.eval()

        # One valid epoch ------------------------------------
        with torch.no_grad():
            for x, y in valid_loader:

                x_im = x
                y_im = y.to(device)

                input_image = x_im.to(device)
                features = encoder(input_image)
                outputs = depth_decoder(features)
                disp = outputs[("disp", 0)]

                # Vectorize the scaling
                vmax = disp.max(dim=-1).values.max(dim=-1).values.squeeze()
                vmin = disp.min(dim=-1).values.min(dim=-1).values.squeeze()
                inp_diff = vmax - vmin
                out_max = torch.ones(vmax.shape[0])
                out_min = torch.zeros(vmax.shape[0])
                out_diff = out_max-out_min

                # rescaled = (out_diff/inp_diff)*(disp - vmin)
                rescaled_vect = torch.stack([(out_diff[i]/inp_diff[i])*(disp[i,:,:,:] - vmin[i]) for i in range(vmax.shape[0])])

                loss = loss_fn(rescaled_vect, y_im)
                pbar.update()
                epoch_loss_valid[e] += loss
            
            epoch_loss_valid[e] = epoch_loss_valid[e].detach().cpu().numpy()/len(valid_loader)
            print(f'Valid loss: {epoch_loss_valid[e]}')
            pbar.close()


    return encoder, depth_decoder, epoch_loss_train, epoch_loss_valid

# -----------------------------------------------------------------------
# Use this function to test your model showing a random image output

def test_mode(encoder, depth_decoder):

	dataset_path = __CONVERTED_DATASET_PATH__
	_, _, train_loader, valid_loader = preliminary_import(dataset_path)

	encoder.eval()
	depth_decoder.eval()

	encoder = encoder.to(device)
	depth_decoder = depth_decoder.to(device)

	idx = random.randrange(len(valid_loader))

	with torch.no_grad():

	    x, y = valid_loader.dataset.__getitem__(idx)

	    x_im = x
	    y_im = y

	    input_image = x_im.to(device)
	    features = encoder(input_image.unsqueeze(0))
	    outputs = depth_decoder(features)
	    disp = outputs[("disp", 0)].squeeze(0)

	    # Vectorize the scaling
	    vmax = disp.max()
	    vmin = disp.min()
	    inp_diff = vmax - vmin
	    out_max = 1
	    out_min = 0
	    out_diff = out_max-out_min

	    # rescaled = (out_diff/inp_diff)*(disp - vmin)
	    rescaled_vect = (out_diff/inp_diff)*(disp.squeeze() - vmin)

	    o_im = rescaled_vect.cpu().detach().numpy()
	    plt.imshow(o_im, cmap='gray')
	    plt.show()


if __name__ == '__main__':
	
	if __CONVERT_MODE__:
		create_converted_dataset()

	if __FINETUNING__:
		encoder, depth_decoder, epoch_loss, epoch_loss_valid = fine_tuning()
		torch.save(encoder, __MODEL_TO_BE_SAVED_PATH__['encoder'])
		torch.save(depth_decoder, __MODEL_TO_BE_SAVED_PATH__['decoder'])

	if __TEST_MODE__:
		encoder = torch.load(__MODEL_TO_BE_SAVED_PATH__['encoder'])
		depth_decoder = torch.load(__MODEL_TO_BE_SAVED_PATH__['decoder'])
		test_mode(encoder, depth_decoder)



