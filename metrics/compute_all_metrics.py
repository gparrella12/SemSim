import os
import cv2
import numpy as np

from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import torch
import torchvision.transforms as transforms
from lpips import LPIPS
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import argparse
from dreamsim import dreamsim

def calculate_dreamsim_distance(img1, img2):
    # Carica e preprocessa le immagini
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')

    model, preprocess = dreamsim(pretrained=True, device=device)
    model = model.to(device)
    # load images as PIL 
    img1 = Image.open(img1)
    img2 = Image.open(img2)
    img1 = preprocess(img1).to(device)
    img2 = preprocess(img2).to(device)

    # Calcola la distanza
    with torch.no_grad():
        distance = model(img1, img2).cpu().item()

    return distance

def calculate_lpips(ori_img, rec_img):
    loss_fn = LPIPS(net='squeeze')  #squeeze
    ori_image = Image.open(ori_img)
    rec_image = Image.open(rec_img)

    transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])

    img1 = transform(ori_image)
    img2 = transform(rec_image)

    # Calculate the LPIPS distance between the two images
    lpips_value = loss_fn(img1, img2)

    return float(lpips_value)

def calculate_psnr(ori_img, rec_img):
    return float(psnr(ori_img, rec_img))

def calculate_ssim(ori_img, rec_img):
    return float(ssim(ori_img, rec_img, channel_axis=-1))

def calculate_mse(ori_img, rec_img):
    return float(mse(ori_img, rec_img))

def compute_metrics(ori_img, rec_img, ori_img_path, rec_img_path):
    mse_score = calculate_mse(ori_img, rec_img)
    ssim_score = calculate_ssim(ori_img, rec_img)
    psnr_score = calculate_psnr(ori_img, rec_img)
    lpips_score = calculate_lpips(ori_img_path, rec_img_path)
    dreamsim_score = calculate_dreamsim_distance(ori_img_path, rec_img_path)   
    return mse_score, ssim_score, psnr_score, lpips_score, dreamsim_score

    
import os

def get_paths(image_directory):
    folders = os.listdir(image_directory)  
    path_dictionary = {}
    
    for folder_name in folders:
        full_path = os.path.join(image_directory, folder_name)
        if not os.path.isdir(full_path):
            continue  # Ignora file non-directory
        
        if '-' in folder_name:
            parts = folder_name.split('-')
            if len(parts) > 1:
                folder_number = parts[1]  # Estrai il numero dopo '-'
                if folder_number not in path_dictionary:
                    path_dictionary[folder_number] = {}
                path_dictionary[folder_number]['rec'] = full_path
        else:
            folder_number = folder_name  # Se non ha '-', usa il nome intero
            if folder_number not in path_dictionary:
                path_dictionary[folder_number] = {}
            path_dictionary[folder_number]['ori'] = full_path
    
    return path_dictionary

def compute_metrics_from_directory(images_dir):
    path_dict= get_paths(images_dir)
    
    metrics_dict = {}
    for folder_number in path_dict.keys():
        
        try:
            ori_img_dir = path_dict[folder_number]['ori']
            rec_img_dir = path_dict[folder_number]['rec']
        except Exception as e:
            print(e)
            print(f"Folder {folder_number} is missing either ori or rec image")
            continue
        print(f"\n\nComputing metrics for folder {folder_number}")
        print("ori_img_dir", ori_img_dir)
        print("rec_img_dir", rec_img_dir)
        # get original image 
        for img in os.listdir(ori_img_dir):
            if "ori.png" in img:
                original_image_path = os.path.join(ori_img_dir, img)
                ori_img = cv2.imread(original_image_path)
                break
        
        metrics_dict[folder_number] = {}
        # compute distance with RECOGNIZABLE reconstructed images
        metrics_dict[folder_number]['recognizable'] = {}
        for img in os.listdir(ori_img_dir):
            if "rec" in img:
                rec_img = cv2.imread(os.path.join(ori_img_dir, img))
                metrics_dict[folder_number]['recognizable'][img] = compute_metrics(ori_img, rec_img, original_image_path, os.path.join(ori_img_dir, img))
        
        metrics_dict[folder_number]['unrecognizable'] = {} 
        for img in os.listdir(rec_img_dir):
            if "rec" in img:
                rec_img = cv2.imread(os.path.join(rec_img_dir, img))
                metrics_dict[folder_number]['unrecognizable'][img] = compute_metrics(ori_img, rec_img, original_image_path, os.path.join(rec_img_dir, img))
        #print(metrics_dict[folder_number])
        #breakù
    return metrics_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Metric computation")
    # use args to get all the paths
    parser.add_argument('--images_dir', type=str, default='/Users/gparrella/Desktop/code/SemSim/data/human_anno_id/train_with_ori', help='Directory of Labeled Images')
    args = parser.parse_args()
    
    metrics_dict = compute_metrics_from_directory(args.images_dir)
    
    # save as pickle
    import pickle
    with open('metrics_dict.pkl', 'wb') as f:
        pickle.dump(metrics_dict, f)
    