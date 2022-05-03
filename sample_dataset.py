from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import torch
import os

class UpscaledDataset(Dataset):
    def __init__(self, data_folder):
        super().__init__()
        self.downsized_folder = os.path.join(data_folder, "downsized")
        self.original_folder = os.path.join(data_folder, "original")
        self.list_pictures = [filename for filename in os.listdir(self.original_folder) if filename[-3:] == "jpg"]

    def __len__(self):
        return len(self.list_pictures)
    
    def __getitem__(self, index):
        tt = transforms.ToTensor()
        downsized = cv2.imread(os.path.join(self.downsized_folder, self.list_pictures[index]))
        downsized = cv2.cvtColor(downsized, cv2.COLOR_BGR2RGB)
        original = cv2.imread(os.path.join(self.original_folder, self.list_pictures[index]))
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        return tt(downsized), tt(original)
    
    def get_loader(self, batch_size, shuffle):
        return DataLoader(self, batch_size, shuffle)

class RegularDataset(Dataset):
    def __init__(self, data_folder):
        super().__init__()
        self.downsized_folder = os.path.join(data_folder, "downsizedsmall")
        self.original_folder = os.path.join(data_folder, "original")
        self.list_pictures = [filename for filename in os.listdir(self.original_folder) if filename[-3:] == "jpg"]

    def __len__(self):
        return len(self.list_pictures)
    
    def __getitem__(self, index):
        tt = transforms.ToTensor()
        downsized = cv2.imread(os.path.join(self.downsized_folder, self.list_pictures[index]))
        downsized = cv2.cvtColor(downsized, cv2.COLOR_BGR2RGB)
        original = cv2.imread(os.path.join(self.original_folder, self.list_pictures[index]))
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        return tt(downsized), tt(original)
    
    def get_loader(self, batch_size, shuffle):
        return DataLoader(self, batch_size, shuffle)