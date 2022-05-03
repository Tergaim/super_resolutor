from pyexpat import model
import numpy as np
import cv2
import os
from tqdm import tqdm
import json

source_folder = "."
target_folder = "."
samples_per_image = 4
sample_size = 64
down_factor = 2

def sample(img):
    downsized = []
    downsized32 = []
    original = []
    dim = (int(sample_size/down_factor), int(sample_size/down_factor))
    for _ in range(samples_per_image):
        x = np.random.randint(0, img.shape[1]-sample_size)
        y = np.random.randint(0, img.shape[0]-sample_size)
        crop = img[y:y+sample_size, x:x+sample_size]
        original.append(crop)
        downsized32img = cv2.resize(crop.copy(), dim)
        downsized32.append(downsized32img)
        downsized.append(cv2.resize(downsized32img.copy(), (sample_size, sample_size)))
    return downsized32, downsized, original

def save(downsized32, downsized, original, total):
    for i in range(len(downsized)):
        cv2.imwrite(os.path.join(target_folder, "downsizedsmall", f"{total}.jpg".zfill(13)), downsized32[i])
        cv2.imwrite(os.path.join(target_folder, "downsized", f"{total}.jpg".zfill(13)), downsized[i])
        cv2.imwrite(os.path.join(target_folder, "original", f"{total}.jpg".zfill(13)), original[i])
        total += 1
    return total

def process():
    print("Looking for images...")
    list_pictures = [filename for filename in os.listdir(source_folder) if filename[-3:] in ["jpg", "png", "JPG", "PNG"]]
    if len(list_pictures)==0:
        print("No image found. Closing.")
        return
    print(f"Found {len(list_pictures)} image files. Sampling...")

    os.makedirs(target_folder, exist_ok=True)
    os.makedirs(os.path.join(target_folder, "downsizedsmall"), exist_ok=True)
    os.makedirs(os.path.join(target_folder, "downsized"), exist_ok=True)
    os.makedirs(os.path.join(target_folder, "original"), exist_ok=True)

    total = 0
    for filename in tqdm(list_pictures):
        img = cv2.imread(os.path.join(source_folder,filename))
        downsized32, downsized, original = sample(img)
        total = save(downsized32, downsized, original, total)
        
    print(f"Created {total} data samples. Closing.")

def read_args():
    global args
    global source_folder
    global target_folder
    global samples_per_image
    global sample_size
    global down_factor
    
    with open("parameters.json", "r") as read_file:
        args = json.load(read_file)

    source_folder = args["create_dataset"]["source_folder"]
    target_folder = args["create_dataset"]["target_folder"]
    samples_per_image = args["create_dataset"]["samples_per_image"]
    sample_size = args["model"]["sample_size"]
    down_factor = args["model"]["down_factor"]
    

if __name__ == "__main__":
    read_args()
    process()