# Code to visualize point clouds/meshes using polyscope 
import os 
import sys
import numpy as np 


import polyscope as ps

# Import custom modules 
from utils import *


class Visualizer:
	def __init__(self):
		ps.init()
		ps.init()

		ps.remove_all_structures()
		# Set camera 
		ps.set_automatically_compute_scene_extents(True)
		ps.set_navigation_style("free")
		# ps.set_view_projection_mode("orthographic")
		ps.set_ground_plane_mode('none')

	def compare(self,pcd1,pcd2): 
		ps_pcd1 = ps.register_point_cloud('PCD1',pcd1)	
		ps_pcd2 = ps.register_point_cloud('PCD2',pcd2)	

		ps.show()


if __name__ == "__main__":

	import trimesh 
	source_pcd = trimesh.load(os.path.join(DATASET_PATH,"banana/banana.source.ply")).vertices
	target_pcd = trimesh.load(os.path.join(DATASET_PATH,"banana/banana.target.ply")).vertices
	gt_T = np.loadtxt(os.path.join(DATASET_PATH,"banana/banana.pose.txt"))

	vis = Visualizer()
	vis.compare(source_pcd @ gt_T[:3,:3].T + gt_T[:3,3],target_pcd)