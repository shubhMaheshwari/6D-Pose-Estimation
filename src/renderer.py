# Code to visualize point clouds/meshes using polyscope 
import os 
import sys

import numpy as np 

# DL Modules 
import torch
import pytorch3d 

# Visaualizers
import polyscope as ps
import matplotlib 
matplotlib.use('GTK3Agg') # QT5 Doesn't work on my system 
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# Import custom modules 
from utils import *


class Visualizer:
	def __init__(self):

		# Image plotting 
		cmap = get_cmap('rainbow', NUM_OBJECTS)
		COLOR_PALETTE = np.array([cmap(i)[:3] for i in range(NUM_OBJECTS + 3)])
		COLOR_PALETTE = np.array(COLOR_PALETTE * 255, dtype=np.uint8)
		COLOR_PALETTE[-3] = [119, 135, 150]
		COLOR_PALETTE[-2] = [176, 194, 216]
		COLOR_PALETTE[-1] = [255, 255, 225]
		self.COLOR_PALETTE = COLOR_PALETTE

		ps.init()

		ps.remove_all_structures()
		# Set camera 
		ps.set_automatically_compute_scene_extents(True)
		ps.set_navigation_style("free")
		# ps.set_view_projection_mode("orthographic")
		ps.set_ground_plane_mode('none')

		self.ps_objects = {}

		# Create folder to save images and videos 
		os.makedirs(RENDER_DIR,exist_ok=True)
		os.makedirs(os.path.join(RENDER_DIR,'images'),exist_ok=True)
		os.makedirs(os.path.join(RENDER_DIR,'videos'),exist_ok=True)

	def clear(self): 
		"""
			Remove the polyscope objects 
		"""
		ps.remove_all_structures()
		del self.ps_objects
		self.ps_objects = {}

	def compare(self,pcd1,pcd2): 
		ps_pcd1 = ps.register_point_cloud('PCD1',pcd1)	
		ps_pcd2 = ps.register_point_cloud('PCD2',pcd2)	

		ps.show()
	

	@staticmethod
	def torch2numpy(sample,show_index=0,sample_np={}): 
		"""
			Convert from pytorch tensors to numpy for using polyscope
			if numpy arrays are passed it only copies the data. 			
		"""

		for k in sample:
			if k in sample_np: continue

			if k == 'pcd': 
				assert all([ x in sample['pcd'] for x in ['points', 'pcd2meta', 'pose2label', 'pcd2pose', 'mesh_ids']]), f"sample['pcd'] does not match the required file"
				sample_np['pcd'] = {}
				show_pose_ids = torch.where(sample['pcd']['pose2label'][:,0] == show_index)[0]
				show_pcd = torch.cat([torch.where(sample['pcd']['pcd2pose'] == x)[0] for x in show_pose_ids])

				assert torch.all(torch.unique(sample['pcd']['pcd2pose'][show_pcd]) == show_pose_ids), "Incorrect points selected for showing"
				sample_np['pcd']['points'] = sample['pcd']['points'][show_pcd].cpu().data.numpy()
				sample_np['pcd']['pcd2pose'] = sample['pcd']['pcd2pose'][show_pcd].cpu().data.numpy()
				sample_np['pcd']['pose2label'] = sample['pcd']['pose2label'][show_pose_ids].cpu().data.numpy()
				sample_np['pcd']['pcd2meta'] = sample['pcd']['pcd2meta'][show_pose_ids].cpu().data.numpy()
				sample_np['pcd']['mesh_ids'] = sample_np['pcd']['pose2label'][:,1]

				assert sample_np['pcd'].keys() == sample['pcd'].keys(), "Keys don't match"

			elif k == 'mesh': 
				sample_np['mesh'] = {}
				show_pose_ids = torch.where(sample['pcd']['pose2label'][:,0] == show_index)[0]
				show_mesh_ids = sample['pcd']['pose2label'][show_pose_ids,1] 
				show_class = [ sample['mesh']['meshId2index'][int(x)] for x in show_mesh_ids]
				meshes = sample['mesh']['pmesh'][show_class]
				sample_np['mesh']['verts'] = [ x.cpu().data.numpy() for x in meshes.verts_list()]
				sample_np['mesh']['faces'] = [ x.cpu().data.numpy() for x in meshes.faces_list()]

				sample_np["mesh"]["meshId2index"] = dict([ (x,i) for i,x in enumerate(show_mesh_ids)])

			elif k == 'pred_pose': 
				show_pose_ids = torch.where(sample['pcd']['pose2label'][:,0] == show_index)[0]
				sample_np[k] = sample['pred_pose'][show_pose_ids].cpu().data.numpy()
			elif type(sample[k]) == torch.Tensor: 
				sample_np[k] = sample[k][show_index].cpu().data.numpy()
			elif type(sample[k]) == list: 
				sample_np[k] = sample[k][show_index]
			else: 
				sample_np[k] = sample[k]
		return sample_np 
	
	@staticmethod
	def reflect_opengl(points):
		# apply 180deg rotation around 'x' axis to transform the mesh into OpenGL coordinates
		
		assert len(points.shape) == 2 and points.shape[1] == 3, "Points should be of the shape Nx3"
		points[:,0] *= -1 
		points[:,1] *= -1 
		points[:,2] *= 1
		return points 



	def show_segmentation(self,sample,show_index=0):
		"""
			Shows the prediction for the 0th index of a batch 
		"""

		sample = self.torch2numpy(sample,show_index=show_index)
		
		plt.subplot(1, 3, 1)
		plt.imshow(sample['rgb'])
		plt.subplot(1, 3, 2)
		plt.imshow(sample['depth'])
		plt.subplot(1, 3, 3)
		plt.imshow(self.COLOR_PALETTE[sample['label']])  # draw colorful segmentation

		save_path = os.path.join(RENDER_DIR,"images", sample['name'] + '_segmentation.png')
		plt.savefig(save_path)

	def show_depth_pc(self,sample,show_index=0,stride=10,show=False): 
		"""
			Visualize the depth point cloud

		"""
		
		assert 'depth_pc' in sample, KeyError(f"depth_pc not in sample. Depth image was not converted to point cloud.") 


		# Set camera and bbounding box
		if 'world_camera' not in self.ps_objects: 
			sample = self.torch2numpy(sample,show_index=show_index)
			world_bbox = sample['depth_pc'].max(axis=(0,1)) - sample['depth_pc'].min(axis=(0,1))
			world_center = sample['depth_pc'].mean(axis=(0,1))


		# Check if point cloud is already registered into polyscope 
		if 'depth_pc' not in self.ps_objects: 
			sample = self.torch2numpy(sample,show_index=show_index)
		
			pcd = sample['depth_pc'][::stride,::stride,:].reshape((-1,3))
			pcd = self.reflect_opengl(pcd)
			pcd_rgb = sample['rgb'][::stride,::stride,:].reshape((-1,3))
			pcd_label = sample['label'][::stride,::stride].reshape(-1)

			self.ps_objects['depth_pc'] = ps.register_point_cloud('Depth Point cloud', pcd,radius=0.001)
			self.ps_objects['depth_pc'].add_color_quantity('Label', self.COLOR_PALETTE[pcd_label]/255,enabled=True)
			self.ps_objects['depth_pc'].add_color_quantity('Color', pcd_rgb,enabled=False)

		self.ps_objects['depth_pc'].set_enabled(True)	
		
		if show:
			ps.show()

	@staticmethod
	def warp_np(points,bsize,pose,ext):
		assert len(points.shape) == 2 and points.shape[1] == 3, f"Points should be a numpy array of size:Nx3"
		assert pose.shape == (4,4), f"Pose should be 4x4 matrix"

		orig_size = points.max(axis=0)-points.min(axis=0)

		warp_points = points * bsize/orig_size
		warp_points = warp_points @ pose[:3,:3].T  + pose[:3,3]
		warp_points = warp_points @  ext[:3,:3].T  +  ext[:3,3]
		warp_points = Visualizer.reflect_opengl(warp_points)

		return warp_points

	def compare_poses(self,sample,show_index=0,stride=1,show=False):

		if 'mesh' not in self.ps_objects:
			self.ps_objects['mesh'] = {'gt':{},'pred':{}}

			sample_np = self.torch2numpy(sample,show_index=show_index)
			if len(sample_np['gt_poses']) > 0:
				gt_mesh_ids = np.where(sample_np['object_ids'])[0]
				gt_pose = sample_np['gt_poses'][gt_mesh_ids]
				for idx,mesh_id in enumerate(gt_mesh_ids): 
					# print(sample_np['box_sizes'][mesh_id])
					# print(gt_pose[idx])
					verts = self.warp_np(sample_np['mesh']['verts'][idx],sample_np['box_sizes'][mesh_id],gt_pose[idx],sample_np['extrinsic'])
					self.ps_objects['mesh']['gt'][mesh_id] = ps.register_surface_mesh(f'gt-{mesh_id}', vertices=verts, faces=sample_np['mesh']['faces'][idx],enabled=True,color=np.array([0,1,0]))


			for mesh_id in sample_np['mesh']['meshId2index']:
				idx =  sample_np['mesh']['meshId2index'][mesh_id]	
				verts = self.warp_np(sample_np['mesh']['verts'][idx],sample_np['box_sizes'][mesh_id],sample_np['pred_pose'][idx],sample_np['extrinsic'])
				self.ps_objects['mesh']['pred'][mesh_id] = ps.register_surface_mesh(f'pred-{mesh_id}', vertices=verts, faces=sample_np['mesh']['faces'][idx],enabled=True,color=np.array([1,0,0]))

			P = torch.where(sample['pcd']['pose2label'] == 0 )[0]
			centroid_pcd = torch.stack([sample['pcd']['points'][sample['pcd']['pcd2pose'] == p,:].mean(dim=0) for p in P ])
			centroid_pcd = centroid_pcd.cpu().data.numpy()
			self.ps_objects['centroid'] = ps.register_point_cloud(f'centroid_pcds',self.reflect_opengl(centroid_pcd))
		if show:
			ps.show()
		

		return 

if __name__ == "__main__":

	import trimesh 
	source_pcd = trimesh.load(os.path.join(DATASET_DIR,"banana/banana.source.ply")).vertices
	target_pcd = trimesh.load(os.path.join(DATASET_DIR,"banana/banana.target.ply")).vertices
	gt_T = np.loadtxt(os.path.join(DATASET_DIR,"banana/banana.pose.txt"))

	vis = Visualizer()
	vis.compare(source_pcd @ gt_T[:3,:3].T + gt_T[:3,3],target_pcd)