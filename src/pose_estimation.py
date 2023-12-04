# Basic modules 
import os 
import sys
import json
import argparse
import numpy as np 
from tqdm import tqdm

# DL Modules 
import torch 
from torch.utils.data import DataLoader

# Local Modules 
from utils import *
from dataloader import SyntheticDataset
from renderer import Visualizer


class ObjectPoseEstimator: 
	def __init__(self): 
		self.device = torch.device('cuda' if CUDA else 'cpu')
		self.segmentor = None

		# Get logger 
		self.logger,self.writer = get_logger(task_name='ICP')

		# Visualizer 
		self.vis = Visualizer()

	@staticmethod
	def depth2pc(depth_im,intrin): 
		
		fx, cx, fy, cy = intrin[:,0,0], intrin[:,0,2], intrin[:,1,1], intrin[:,1,2]
		bs, height, width = depth_im.shape
		u = torch.arange(width) * torch.ones([bs,height, width])
		v = torch.arange(height) * np.ones([bs,width, height])
		v = v.transpose(dim0=1,dim1=2)
		X = (u - cx[:,None,None]) * depth_im / fx[:,None,None]
		Y = (v - cy[:,None,None]) * depth_im / fy[:,None,None]
		Z = depth_im
		return torch.stack([X, Y, Z],dim=3)

	def pose_estimation(self,sample): 
		"""
		Algorithm:
			Input: RGBD image 
			Get point cloud using the depth image 
			Get/Predict Segmentation Mask 
			Load 3D model
			Estimate 6D parameters for each object using either
				Chamfer + ICP
				Pointnet 
				Pointnet + ICP 
				Custom 
			Render Result
		"""


		# 1. Get point cloud from depth image + mask 
		sample['depth_pc'] = self.depth2pc(sample['depth'],sample['intrinsic'])
		print("Depth to pc:",sample['depth_pc'].shape)
		if RENDER:
			self.vis.show_depth_pc(sample)  
		self.logger.info("[Completed] Extract Point cloud from depth images")

		# 2. Get segmentation mask
		if 'label' not in sample or sample['label'].shape[1] == 0:
			label = []
			for file_path in sample['path']:
				if os.path.isfile(file_path + '_label_kinect.png'):  
					mask = SyntheticDataset.load_mask(file_path + '_label_kinect.png') 
					mask = torch.from_numpy(mask).to(self.device)
				else: 
					assert self.segmentor is not None, "No segmentation module passed for pose estimation."
					mask = self.segmentor.predict_mask(sample['image'])
				
				label.append(mask[None])

			sample['label'] = torch.cat(label,dim=0)
		if RENDER:
			self.vis.show_segmentation(sample.copy())  
		self.logger.info("[Completed] Get Segmentation map")

		# 3. 



		# 4. Save results 

		self.vis.clear() # Clearing objects passed to polyscope 

		return sample

	def dataset_pose_estimation(self): 
		# Get visualizer 
		print(LOG_DIR)
		for split_type in ['train','val','test']: 
			dataloader = DataLoader(SyntheticDataset(split_type=split_type),batch_size=TRAIN_BATCH_SIZE,shuffle=False if split_type=='test' else True)
			self.logger.info(f"Predicting 6D pose for the {split_type} dataset")
			pred_pose = {}
			for sample_ind,sample in enumerate(tqdm(dataloader)): 
				sample = self.pose_estimation(sample)
				pred_pose_sample = [ x.cpu().data.numpy().tolist() if x.sum() != 0 else None for x in sample['pred_pose'] ]
				pred_pose[sample['name']] = pred_pose_sample

			# with open(os.path.join(LOG_DIR),split_type + '_results.json','w') as f:
			# 	json.dump(pred_pose,f)


############################# Command line Argument Parser #######################################################
parser = argparse.ArgumentParser(
					prog='6D Pose estimation',
					description='6D Pose estimation',
					epilog='')
parser.add_argument('-f', '--force',
					action='store_true')  # on/off flag

parser.add_argument('--image')


cmd_line_args = parser.parse_args()

if __name__ == "__main__": 

	obe = ObjectPoseEstimator()
	if len(sys.argv) == 1: 
		obe.dataset_pose_estimation()
	else:
		sample_path = sys.argv[1]
		sample = SyntheticDataset.load_sample(sample_path)
		sample = obe.pose_estimation(sample)