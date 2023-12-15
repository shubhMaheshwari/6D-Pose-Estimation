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
from dataloader import SyntheticDataset,MeshInfo
from renderer import Visualizer
from icp import ICP


class ObjectPoseEstimator: 
	def __init__(self): 
		self.device = torch.device('cuda' if CUDA else 'cpu')

		# Get logger 
		self.logger,self.writer = get_logger(task_name='ICP')

		self.segmentor = None

		# Mesh Loader 
		self.meshloader = MeshInfo()

		# ICP 
		self.icp = ICP()

		# Visualizer 
		self.vis = Visualizer()


	def cpu_to_gpu(self,sample): 
		sample_gpu = {}
		for k in sample:
			if type(sample[k]) == torch.Tensor: 
				sample_gpu[k] = sample[k].to(self.device)
			elif type(sample[k]) == np.ndarray: 
				sample_gpu[k] = torch.from_numpy(sample[k]).to(self.device)
			else:
				sample_gpu[k] = sample[k]
		return sample_gpu

	def depth2pc(self,depth_im,intrin): 
		
		fx, cx, fy, cy = intrin[:,0,0], intrin[:,0,2], intrin[:,1,1], intrin[:,1,2]
		bs, height, width = depth_im.shape
		u = torch.arange(width) * torch.ones([bs,height, width])
		v = torch.arange(height) * np.ones([bs,width, height])
		v = v.transpose(dim0=1,dim1=2)
		
		u = u.to(self.device)
		v = v.to(self.device)
		
		X = (u + 0.5- cx[:,None,None]) * depth_im / fx[:,None,None]
		Y = (v + 0.5- cy[:,None,None]) * depth_im / fy[:,None,None]
		Z = depth_im
		return torch.stack([X, Y, Z],dim=3)

	def reshape_image2pcd(self,sample): 
		"""
			Given a batch of XYZ image create a single batch of 3D vertices to for parallelly running ICP across multiple objects
			- Input: 
				depth_pc: BxHxWx3 
				labels: BxHxW 
			- Output 
				pcd: B'x3
				pcd2meta: B'x4 -> Maps to batchsize,h,w,label for each point
				pose2label: Px2 -> Maps p-th pose to the (batchsize,label) to which it belongs 
				pcd2pose: B' -> Maps each b-th point to the p-th pose transformation matrix which needs to be estimated.

		"""

		cond = sample['label'] < NUM_OBJECTS
		points = sample['depth_pc'][cond,:]
		label = sample['label'][cond]
		pcd2meta = list(torch.where(cond)) + [label]  
		pcd2meta = torch.cat([x.unsqueeze(1) for x in  pcd2meta],dim=1)

		pose2label,pcd2pose = torch.unique(pcd2meta[:,[0,3]],return_inverse=True,dim=0)

		assert torch.all(pose2label[pcd2pose] == pcd2meta[:,[0,3]]), "Inverse mapping not unique"		

		mesh_ids = torch.unique(pose2label[:,-1]).cpu().data.numpy() 	

		pcd = {'points': points, 'pcd2meta':pcd2meta,'pose2label':pose2label, 'pcd2pose':pcd2pose, 'mesh_ids':mesh_ids }


		return pcd 


	def pose_estimation(self,sample): 
		"""
		Algorithm:
			For each batch: 
				Input: 
					Load RGBD image (BxHxWx3)
					Load 3D models (M)
				Get/Predict Segmentation Mask 
				Get point cloud using the depth image (Px3) 

				ICP + Variants:	// Estimate 6D parameters for each object
					initialize(rotation and translation)
					For each iteration:
						1. sample points from mesh 
						2. Establish correspondence using either
							Chamfer + ICP
							Pointnet 
							Pointnet + ICP 
							Custom 
						3. Run gradient descent
						4. Update scheduler, loss hyperparameters etc. 
				Render Result
				
				Output: pred-pose
		"""

		if CUDA:
			sample = self.cpu_to_gpu(sample)

		# 1. Get point cloud from depth image + mask 
		sample['depth_pc'] = self.depth2pc(sample['depth'],sample['intrinsic'])
		# print("Depth to pc:",sample['depth_pc'].shape)
		if RENDER:
			self.vis.show_depth_pc(sample,show=False)  
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
		# if RENDER:
		# 	self.vis.show_segmentation(sample.copy())  
		self.logger.info("[Completed] Get Segmentation map")

		# 3. Convert to 3D 
		sample["pcd"] = self.reshape_image2pcd(sample)

		# 4. Get the predicted pose using ICP on meshes
		sample['mesh'] = {}
		sample["mesh"]['index'] = [ MESHID2CLASS[x] for x in  sample['pcd']['mesh_ids'] ]
		sample["mesh"]["meshId2index"] = dict([ (x,i) for i,x in  enumerate(sample['pcd']['mesh_ids']) ])
		sample["mesh"]['pmesh'] = self.meshloader.pmeshes[sample["mesh"]['index']]

		sample["pred_pose"] = self.icp(sample)

		# 4. Save results 
		if RENDER:
			self.vis.compare_poses(sample,show=True)

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
				print(sample['name'])
				sample = self.pose_estimation(sample)
				pred_pose_sample = [ x.cpu().data.numpy().tolist() if x.sum() != 0 else None for x in sample['pred_pose'] ]
				
				# Evalaute
				# pred_pose[sample['name']] = pred_pose_sample

			# with open(os.path.join(LOG_DIR),split_type + '_results.json','w') as f:
			# 	json.dump(pred_pose,f)
if __name__ == "__main__": 

	############################# Command line Argument Parser #######################################################
	parser = argparse.ArgumentParser(
						prog='6D Pose estimation',
						description='6D Pose estimation',
						epilog='')
	parser.add_argument('-f', '--force',
						action='store_true')  # on/off flag

	parser.add_argument('--image')


	cmd_line_args = parser.parse_args()



	obe = ObjectPoseEstimator()
	if len(sys.argv) == 1: 
		obe.dataset_pose_estimation()
	else:
		sample_path = sys.argv[1]
		sample = SyntheticDataset.load_sample(sample_path)
		sample = obe.pose_estimation(sample)