import torch
import torch.nn as nn

import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import numpy as np

# For point cloud
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)

# For mesh
from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights,
    SoftPhongShader
)

from pytorch3d.renderer.mesh import (
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,

)
 


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

from skimage import io
import os
# from .geometry import  depth_2_pc



def opencv_to_pytorch3d(T):
    ''' ajust axis
    :param T: 4x4 mat
    :return:
    '''
    origin = np.array(((-1, 0, 0, 0), (0, -1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    origin = torch.from_numpy(origin).float().to(T)
    return T @ origin


class PCDRender(nn.Module):

    def __init__(self, K, img_size=(500,512), device=torch.device("cuda:0")):
        super().__init__()

        self.camera = None
        self.device = device

        self.img_size = img_size # (width, height) of the image

        self.camera = self.init_camera(K)

        raster_settings = PointsRasterizationSettings( image_size=self.img_size, radius = 0.005, points_per_pixel = 10)

        self.rasterizer = PointsRasterizer(cameras=self.camera, raster_settings=raster_settings).to(device)

        self.compositor= AlphaCompositor(background_color=(0, 0, 0)).to(device)


    def load_pcd (self, pcd) :
        feature = torch.ones_like(pcd)
        point_cloud = Pointclouds(points=[pcd], features=[feature]).to(self.device)
        return point_cloud

    def init_camera(self, K, T=torch.eye(4),  ):

        # T = T.to( self.device)
        T_py3d =  opencv_to_pytorch3d(T).to(device)
        R = T_py3d[:3, :3]
        t = T_py3d[:3, 3:]

        """creat camera"""
        fx = K[0, 0]
        fy = K[1, 1]
        px = K[0, 2]
        py = K[1, 2]
        f = torch.tensor((fx, fy), dtype=torch.float32).unsqueeze(0) # dim = (1, 2)
        p = torch.tensor((px, py), dtype=torch.float32).unsqueeze(0) # dim = (1, 2)
        img_size = self.img_size

        camera = PerspectiveCameras( R=R[None], T=t.T, focal_length=f, principal_point=p, in_ndc=False, image_size=(img_size,)).to(device)

        return camera



    def forward(self, point_clouds) -> torch.Tensor:

        point_clouds = self.load_pcd(point_clouds)

        fragments = self.rasterizer(point_clouds, gamma=(1e-5,))

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius


        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            gamma=(1e-5,),
        )

        images = images.permute(0, 2, 3, 1)

        return images, fragments.zbuf


class MeshRender(nn.Module):

    def __init__(self, K, img_size=(500,512), device=torch.device("cuda:0")):
        super().__init__()

        self.camera = None
        self.device = device

        self.img_size = img_size # (width, height) of the image

        self.camera = self.init_camera(K)

        raster_settings = RasterizationSettings(image_size=self.img_size,
                                                cull_to_frustum=True,
                                                faces_per_pixel=4)

        self.rasterizer = MeshRasterizer(cameras=self.camera, raster_settings=raster_settings).to(device)

        self.compositor= AlphaCompositor(background_color=(0, 0, 0)).to(device)

        self.lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])    

        self.renderer = MeshRenderer(rasterizer=self.rasterizer,shader=SoftPhongShader(device=device, cameras=self.camera,lights=self.lights))



    def init_camera(self, K, T=torch.eye(4),  ):

        # T = T.to( self.device)
        T_py3d =  opencv_to_pytorch3d(T).to(device)
        R = T_py3d[:3, :3]
        t = T_py3d[:3, 3:]

        """creat camera"""
        fx = K[0, 0]
        fy = K[1, 1]
        px = K[0, 2]
        py = K[1, 2]
        f = torch.tensor((fx, fy), dtype=torch.float32).unsqueeze(0) # dim = (1, 2)
        p = torch.tensor((px, py), dtype=torch.float32).unsqueeze(0) # dim = (1, 2)
        img_size = self.img_size

        camera = PerspectiveCameras( R=R[None], T=t.T, focal_length=f, principal_point=p, image_size=(img_size,)).to(device)

        return camera



    def forward(self, meshes) -> torch.Tensor:

        # meshes = self.load_mesh(verts_list,face_list)

        fragments = self.rasterizer(meshes.to(device))

        return fragments.zbuf



if __name__ == '__main__':


    # Local modules
    from utils import *
    from dataloader import MeshInfo

    meshinfo = MeshInfo()

    """ 
        Check mesh render
    """
    # datapath = "/media/srialien/Elements/AT-Datasets/DeepDeform/new_test/tigerD8H_Swim0_cam2/"
    K = np.array([[514.13902522,   0.,         640.        ],
 [  0.,         514.13902522, 360.        ],
 [  0.,           0.,           1.        ]])
    print(K)
    renderer = MeshRender(K)

    # gr_img = plt.imread(datapath+"/depth/0003.png")


    # source_frame_id = 3
    # data_path = "/media/srialien/Elements/AT-Datasets/DeepDeform/new_test/tigerD8H_Swim0_cam2/results/NRR/passing_source_skinning_update_1/"
    # # Load trajectory
    # trajectory = np.load(os.path.join(data_path,"trajectory",f"trajectory_{source_frame_id}.npy"))
    # # TODO fix for multiple source frame
    # face_filename = os.listdir(os.path.join(data_path,"updated_graph",str(source_frame_id),"face_path"))[0] 
    # faces = np.load(os.path.join(data_path,"updated_graph",str(source_frame_id),"face_path",face_filename))

    # verts = trajectory[0]
    # verts[:,1] *= -1
    # verts[:,2] *= -1

    # o3d.visualization.draw_geometries([mesh])

    img = renderer(meshinfo.pmeshes)
    img = img[0, ..., 0].cpu().numpy()

    img[img <0] = 0

    # img *= gr_img.max()/img.max()

    # depth_inverse = img.copy()
    # depth_inverse[depth_inverse > 0] = 1/depth_inverse[depth_inverse>0]


    # gr_depth_inverse = gr_img.copy()
    # gr_depth_inverse[gr_depth_inverse > 0] = 1/gr_depth_inverse[gr_depth_inverse>0]

    print("Printing:")
    # print("IOU:",np.sum(np.logical_and(gr_img > 0, img > 0))/np.sum(np.logical_or(gr_img > 0, img > 0)))
    # print("Depth Inverse:",np.sum(np.logical_and(gr_img > 0, img > 0))/np.sum(np.logical_or(gr_img > 0, img > 0)))
    print(img.min(), img.max())
    # print(gr_img.min(), gr_img.max())

    plt.imshow(img)
    plt.show()

    fig = plt.figure(figsize=(16,4))
    title_list = ["Warped Mask","Warped Depth", "Target Mask", "Target Depth", "Depth Error(MLE)"]
    depth_image_list = [img > 0 , img, gr_img>0, gr_img, np.abs(img-gr_img)]

    for i,im in enumerate(depth_image_list):
        # print(100 + (i+1)*10 + len(depth_image_list)) 
        ax = fig.add_subplot(100 + 10*len(depth_image_list) + i+1)
        print(i,im.shape)

        # im = im.detach().cpu().numpy()
        if im.dtype == np.float32 and im.max() > 1:
            im /= im.max()
        # if im.shape[-1] > 3:
        #     im = im[...,0]
        print(im.shape)
        ax.imshow(im)
        ax.set_title(title_list[i])
        ax.axis('off')
    plt.axis('off')
    plt.show()




    fig = plt.figure()
    ax1 = fig.add_subplot(113)
    plt.imshow(img[0, ..., 0].cpu().numpy())
    plt.show()



    # """data path"""
    # seq_name = "moose6OK9_AttackTrotRM"
    # seq_dir = "/home/liyang/workspace/NeuralTracking/GlobalReg/example/" + seq_name
    # depth_name = "cam1_0015.png"
    # intr_name = "cam1intr.txt"
    # tgt_depth_image_path = os.path.join( seq_dir,"depth", "cam1_0009.png")
    # intrinsics_path = os.path.join(seq_dir, intr_name)
    # K = np.loadtxt(intrinsics_path)
    # tgt_depth = io.imread( tgt_depth_image_path )/1000.
    # tgt_pcd = depth_2_pc(tgt_depth,K).transpose(1,2,0)
    # tgt_pcd = torch.from_numpy( tgt_pcd[ tgt_depth >0 ] ).float()
    # # tgt_pcd = tgt_pcd - tgt_pcd.mean(dim=0, keepdim=True)


    # renderer = PCDRender(K)

    # img = renderer.render_pcd(tgt_pcd)

    # plt.imshow(img[0, ..., :3].cpu().numpy())
    # plt.show()




