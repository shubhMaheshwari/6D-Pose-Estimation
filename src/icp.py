# Basic modules 
import os 
import numpy as np 

# Mesh loaders
import trimesh

# DL Modules 
import torch
from torch import optim 
from lietorch import SO3, SE3, LieGroupParameter 

# For mesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import (
    RasterizationSettings,
    MeshRasterizer
)
from pytorch3d.ops import sample_points_from_meshes

# Local modules 
from utils import *
from loss3D import chamfer_dist



# Local Modules
from utils import *
from dataloader import MeshInfo

class ICP: 
    def __init__(self):
        pass
        image_size = (480,1080)
        # self.renderer = PCDRender(K, img_size=image_size)
        
        # Consants
        # self.max_iterations = 200
        self.max_iterations = 0
        self.lr = 0.01
        self.device = torch.device('cuda' if CUDA else 'cpu')    
        self.w_chamfer = 1
        self.batch_size = 10

       
    def __call__(self,sample,sample_points=10000):
        """
            Sample is a dictionary containing various information about the points
        """ 
        N = len(sample['pcd']['mesh_ids'])
        M = sample['pcd']['points'].shape[0]
        P = sample['pcd']['pose2label'].shape[0]

        
        """translations"""
        node_translations = torch.zeros((P,3)).to(self.device)
        node_translations = torch.nn.Parameter(node_translations)
        node_translations.requires_grad = True

        """rotations"""
        phi = torch.zeros((P,3)).to(self.device)
        phi = torch.nn.Parameter(phi)
        phi.requires_grad = True
        node_rotations_so3 = SO3.exp(phi)
        
        node_rotations = LieGroupParameter(node_rotations_so3)    

        """optimizer setup"""
        optimizer = optim.Adam([node_rotations, node_translations], lr= self.lr )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

        # ICP Implementation    

        for iteration in range(self.max_iterations):
            
            # Get self.batchsize x self.
            mesh_sample_pts = sample["mesh"]["pmesh"].sample() 


            # Get rotation & translations
            iter_rotation = node_rotations[sample['pcd']['pcd2pose']]
            iter_translations = node_translations[sample['pcd']['pcd2pose']]

            # warped_pcd = 

            # Compute loss
            cd = chamfer_dist(warped_pcd, self.tgt_pcd) if self.w_chamfer > 0 else 0

            # Log info     
            if iteration % 100: 
                print(f"Rotvec:{node_rotations_so3} Translation:{node_translations}")            

        pred_pose = node_rotations_so3.matrix() 
        pred_pose[:,:3,3] = node_translations

        return pred_pose

            