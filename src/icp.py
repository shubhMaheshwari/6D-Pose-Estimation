# Basic modules 
import os 
import numpy as np 
from tqdm import tqdm

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
from loss3D import compute_truncated_chamfer_distance



# Local Modules
from utils import *
from dataloader import MeshInfo

class ICP: 
    def __init__(self,vis=None):
        pass
        image_size = (480,1080)
        # self.renderer = PCDRender(K, img_size=image_size)
        
        # Consants
        self.max_iterations = 200
        # self.max_iterations = 0
        self.lr = 0.0005
        self.device = torch.device('cuda' if CUDA else 'cpu')    
        self.w_chamfer = 1
        self.batch_size = 10

        self.vis = vis

    
    def warp(self,sample,sample_pts,node_rotations,node_translations):
        
        pose2mesh_id = [sample["mesh"]["meshId2index"][int(i)] for i in sample["pcd"]["pose2label"][:,-1]]
        pose2mesh_id = torch.LongTensor(np.array(pose2mesh_id)).to(self.device)

        sample_pts = sample_pts[pose2mesh_id]

        orig_size = sample_pts.max(dim=1)[0] - sample_pts.min(dim=1)[0] # update to mesh size 
        bsize = sample['box_sizes'][sample['pcd']['pose2label'][:,0],sample['pcd']['pose2label'][:,1]]

        warped_points = sample_pts * (bsize/orig_size)[:,None,:]
        # warped_points = sample_pts

        # Pose
        warped_points = (node_rotations[:,None,:] * warped_points) + node_translations[:,None,:]

        # Ext
        ext = sample['extrinsic'][sample['pcd']['pose2label'][:,0]] 

        warped_points = torch.bmm(warped_points, ext[:,:3,:3].transpose(1,2)) + ext[:,None,:3,3]

        return warped_points


    def pcd2batch(self,sample): 
        _,pcd_length = torch.unique(sample['pcd']['pcd2pose'],return_counts=True)
        P = len(pcd_length)

        source_pts = torch.zeros((P,pcd_length.max(),3),dtype=torch.float64).to(self.device)
        for p in range(P): 
            source_pts[p,:pcd_length[p]] = sample['pcd']['points'][ sample['pcd']['pcd2pose'] == p ]

        return source_pts,pcd_length


    def __call__(self,sample,sample_points=10000):
        """
            Sample is a dictionary containing various information about the points
        """ 
        N = len(sample['pcd']['mesh_ids'])
        M = sample['pcd']['points'].shape[0]
        P = sample['pcd']['pose2label'].shape[0]

        
        """Initialze translations"""
        node_translations = torch.zeros((P,3)).to(self.device)

        centroid_pcd = torch.stack([sample['pcd']['points'][sample['pcd']['pcd2pose'] == p].mean(dim=0) for p in range(P)])
        extrinsic_batch = sample['extrinsic'][sample['pcd']['pose2label'][:,0]]  

        node_translations[:] = torch.bmm((centroid_pcd - extrinsic_batch[:,:3,3])[:,None,:], extrinsic_batch[:,:3,:3]).squeeze()

        node_translations = torch.nn.Parameter(node_translations)
        node_translations.requires_grad = True

        """rotations"""
        phi = torch.zeros((P,3),dtype=torch.double).to(self.device)
        node_rotations_so3 = SO3.exp(phi)
        node_rotations = LieGroupParameter(node_rotations_so3)    

        """optimizer setup"""
        # optimizer = optim.Adam([node_rotations, node_translations], lr= self.lr )
        optimizer = optim.Adam([
            {"params": node_rotations, "lr": self.lr*100},
            {"params": node_translations,"lr": self.lr}],lr=self.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

        # ICP Implementation    
        target_pts,target_length = self.pcd2batch(sample)

        loss_list = {'loss':[], 'cd':[]}

        for iteration in tqdm(range(self.max_iterations)):
            optimizer.zero_grad()
            
            # Get self.batchsize x self.
            mesh_sample_pts =  sample_points_from_meshes(sample["mesh"]["pmesh"],num_samples=sample_points).to(self.device) 

            warped_pcd = self.warp(sample,mesh_sample_pts,node_rotations,node_translations)
            warped_length = (sample_points*torch.ones(warped_pcd.shape[0],dtype=torch.long)).to(self.device)


            # Compute loss
            cd,corresp,valid_target_pts = compute_truncated_chamfer_distance(target_pts,warped_pcd, x_lengths=target_length,y_lengths=warped_length,trunc=0.01)

            loss = cd                

            # Log info     
            if iteration % 10 == 0: 
                print(f"Loss:{[ (x,round(loss_list[x][-1],6)) for x in loss_list if len(loss_list[x]) > 0]} Rotvec:{node_rotations.data[0]} Translation:{node_translations[0]}")            


            if RENDER and self.vis is not None:  

                import polyscope as ps 
                show_pose = 0

                ps.remove_all_structures()
                ps.register_point_cloud('Warped',warped_pcd[show_pose,:warped_length[show_pose]].cpu().data.numpy())
                ps.register_point_cloud('Target',target_pts[show_pose,:target_length[show_pose]].cpu().data.numpy())


                corresp_target = torch.where(valid_target_pts[show_pose])[0]
                if len(corresp_target) > 0: 
                    corresp_source = corresp[show_pose][corresp_target]

                    corresp_points = torch.cat([  target_pts[show_pose,corresp_target],  warped_pcd[show_pose,corresp_source]  ],dim=0)
                    corresp_points = corresp_points.cpu().data.numpy()
                    corresp_edges = np.array([ [i,i+len(corresp_target)] for i in range(len(corresp_target))])

                    ps.register_curve_network('Corresp', corresp_points,corresp_edges,radius=0.001)

                
                if iteration == 0: 
                    ps.show()

                os.makedirs(os.path.join(RENDER_DIR,sample['name'][show_pose],"images"),exist_ok=True)
                image_path = os.path.join(RENDER_DIR,sample['name'][show_pose],"images",f"icp_iter_{iteration}.png")
                print(f"Saving plot to :{image_path}")	
                ps.set_screenshot_extension(".png");
                ps.screenshot(image_path,transparent_bg=False)



            loss.backward()

            if (iteration//10) % 2 == 0: 
                node_translations.grad[:] = 0 
            else: 
                node_rotations.grad[:] = 0

            optimizer.step()
            scheduler.step()

            loss_list['loss'].append(loss.item())
            loss_list['cd'].append(cd.item())




        print(f"Loss list:{loss_list}")
        if RENDER and self.vis is not None:
            image_path = os.path.join(RENDER_DIR,sample['name'][show_pose],"images",f"icp_iter_\%d.png")
            video_path = os.path.join(RENDER_DIR,'videos',sample['name'][show_pose]+ f"-icp.mp4")
            palette_path = os.path.join(RENDER_DIR,sample['name'][show_pose],"images",f"palette.png") 
            framerate = 24
            os.system(f"ffmpeg -y -framerate {framerate} -i {image_path} -vf palettegen {palette_path}")
            os.system(f"ffmpeg -y -framerate {framerate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path}")

        pred_pose = torch.stack([ node_rotations[p].matrix() for p in range(P)])  
        pred_pose[:,:3,3] = node_translations
        return pred_pose

            