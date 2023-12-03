# Data loader to load images, mesh files
import os 
import sys

# File loaders
import trimesh 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# DL Modules 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# Defined Modules
from utils import *

class Image: 
	def 
	

class SyntheticDataset(Dataset):
	def __init__(self,split_type='train'): 

		if split_type == 'train':
			datapath = TRAIN_PATH
			split_path = 'splits/v2/train.txt'
		elif split_type == 'val':
			datapath = TRAIN_PATH 
			split_path = 'splits/v2/val.txt'
		elif split_type == 'test': 
			datapath = TEST_PATH
			split_path = 'test.txt'
		else: 
			raise KeyError(f'split_type:{split_type} not valid')

		assert os.path.isdir(datapath), f"Folder:{datapath} not a directory"
		assert "objects_v1.csv" in os.listdir(datapath) and "v2.2" in os.listdir(datapath) , f"Folder:{datapath} is not a dataset path"

		self.datapath = datapath
		
		split_path = os.path.join(datapath,split_path)
		with open(split_path,'r') as f: 
			self.samples = f.read().split('\n')

	@staticmethod
	def load_rgbd(sample,datapath):
		color = plt.imread(os.path.join(datapath,sample+'_color_kinect.png'))
		depth = plt.imread(os.path.join(datapath,sample+'_depth_kinect.png'))
		return 

	def __getitem__(self,idx):
		image = SyntheticDataset.load_rgbd(self.samples[idx],datapath)
		if self.split_type != 'test':
			SyntheticDataset.load_pose(self.samples[idx],datapath) 
			return image, self.pose[ind]	
		else: 
			return image, None

	def __len__(self): 
		return len(self.samples)



def analyze_dataset(): 

	for split_type in ['train','val','test']: 
		dataloader = DataLoader(SyntheticDataset(split_type=split_type),batch_size=TRAIN_BATCH_SIZE,shuffle=True)
		for x in dataloader: 
			for k in x: 
				print(k.shape)

	meshes = MeshInfo()



if __name__ == "__main__": 
	analyze_dataset()

		