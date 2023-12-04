# Data loader to load images, mesh files
import os 
import sys

# File loaders
import trimesh 
import pickle 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# DL Modules 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# Defined Modules
from utils import *

class SyntheticDataset(Dataset):
	def __init__(self,split_type='train'): 

		self.split_type = split_type
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
	def load_pickle(filename):
		with open(filename, 'rb') as f:
			return pickle.load(f)

	@staticmethod
	def load_rgbd(sample_path):
		rgb = plt.imread(sample_path+'_color_kinect.png')
		depth = plt.imread(sample_path+'_depth_kinect.png')
		return  rgb, depth
		
	@staticmethod
	def load_mask(mask_path):
		image = plt.imread(mask_path)
		image = (image*255).astype('uint8')
		return  image

	@staticmethod
	def load_pose(sample_path):
		if os.path.isfile(sample_path+'_meta.pkl'):
			return SyntheticDataset.load_pickle(sample_path+'_meta.pkl') 
		else: 
			return np.empty(0)

	@staticmethod
	def load_sample(sample_path,split_type='test',mask_path=""): 
		rgb,depth = SyntheticDataset.load_rgbd(sample_path)
		pose = {} if split_type == 'test' else SyntheticDataset.load_pose(sample_path)
		mask = np.empty(0) if not os.path.isfile(mask_path) else SyntheticDataset.load_mask(mask_path)		

		gt_poses = np.array([ np.zeros((4,4)) if x is None else x for x in pose['poses_world']  ])
		object_ids = np.zeros(NUM_OBJECTS)
		object_ids[pose['object_ids']] = 1
	
		return {'path':sample_path, 'name': os.path.basename(sample_path),		  
		  'rgb':rgb,'depth':depth,'label':mask,
		  'extrinsic':pose['extrinsic'], 'intrinsic': pose['intrinsic'],
		  'gt_poses' : gt_poses, 
		  'object_ids' : object_ids}

	def __getitem__(self,idx):
		sample_path = os.path.join(self.datapath,'v2.2',self.samples[idx])
		return SyntheticDataset.load_sample(sample_path,split_type=self.split_type,mask_path=sample_path+'_label_kinect.png')
		

	def __len__(self): 
		return len(self.samples)


def analyze_dataset(): 

	object_names = ['']*79
	per_object_distribution = [0]*79
	
	for split_type in ['train','val','test']: 
		
		dataset = SyntheticDataset(split_type=split_type)
		for i,sample in range(len(dataset)):
			dataset.__getitem__(i) 
			


	# meshes = MeshInfo()



if __name__ == "__main__": 
	analyze_dataset()

		