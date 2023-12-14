# Data loader to load images, mesh files
import os 
import sys

# File loaders
import PIL
import trimesh 
import pickle 
import numpy as np 
import pandas as pd
import collada 
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt

# DL Modules 
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch3d.structures import Meshes,join_meshes_as_batch
from pytorch3d.renderer import TexturesUV

# Defined Modules
from utils import *
import plotter

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
			self.samples = [ x for x in  f.read().split('\n') if len(x) > 0]

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
		mask = SyntheticDataset.load_mask(mask_path) if os.path.isfile(mask_path) else np.empty(0)		

		gt_poses = np.array([ np.zeros((4,4)) if x is None else x for x in pose['poses_world']  ]) if split_type != 'test' else np.empty(0) 
		object_ids = np.zeros(NUM_OBJECTS)
		object_ids[pose['object_ids']] = 1

		# print(pose['intrinsic'])    

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

# Class maintaining all infromation regarding the meshes, texture, symmetry.
class MeshInfo(Dataset): 
    """
        Load the mesh for each 3D model and other information provided.
    """
    def __init__(self,logger=None): 

        self.mesh_info = self.load_mesh_info()
        self.pmeshes = self.load_mesh() 
        self.logger = logger
    
    def load_mesh_info(self):
        
        # Load model details 
        mesh_info = pd.read_csv(os.path.join(TRAIN_PATH,'objects_v1.csv'))

        meshes = {}
        for i in VALID_CLASS_INDICES: 
            mesh = {}
            mesh['name'] = LABELS[i]
            mesh['path'] = os.path.join(MESH_PATH,mesh['name'],'visual_meshes','visual.dae')

            pd_index = list(mesh_info.index[mesh_info['object'] == mesh['name']])
            assert len(pd_index) == 1, f"Unable to find mesh:{mesh['name']} in objects.csv"
            mesh['info'] = mesh_info.iloc[pd_index[0]].T.to_dict()
            meshes[i] = mesh
        return meshes

    ##### Copied from Trimesh for convience: https://github.com/MPEGGroup/trimesh/blob/8f8c807843645c4bbc731b451a9bc1038d34cede/trimesh/exchange/dae.py#L262
    def _load_texture(self,image_path):
        """
        Load a texture from a file into a PIL image.
        """
        assert os.path.isfile(image_path), f"{image_path} does not exist"
        image = PIL.open(image_path)
        return image
    
    def _parse_material(self,effect,idx):
        """
        Turn a COLLADA effect into a trimesh material.
        params: 
            effect: Collada effect
            idx: mesh_id
        """

        # Compute base color
        baseColorFactor = np.ones(4)
        baseColorTexture = None
        if isinstance(effect.diffuse, collada.material.Map):
            try:

                image_path = os.path.join(
                    MESH_PATH,self.mesh_info[idx]['name'],'visual_meshes',
                    effect.diffuse.sampler.surface.image.path)
                
                baseColorTexture = self._load_texture(image_path)
            except BaseException:
                self.logger.warning('unable to load base texture',exc_info=True)
        elif effect.diffuse is not None:
            baseColorFactor = effect.diffuse

        # Compute emission color
        emissiveFactor = np.zeros(3)
        emissiveTexture = None
        if isinstance(effect.emission, collada.material.Map):
            try:
                image_path = os.path.join(
                    MESH_PATH,self.mesh_info[idx]['name'],'visual_meshes',
                    effect.diffuse.sampler.surface.image.path)
                emissiveTexture = self._load_texture(image_path)
            except BaseException:
                self.logger.warning('unable to load emissive texture',
                            exc_info=True)
        elif effect.emission is not None:
            emissiveFactor = effect.emission[:3]

        # Compute roughness
        roughnessFactor = 1.0
        if (not isinstance(effect.shininess, collada.material.Map)
                and effect.shininess is not None):
            roughnessFactor = np.sqrt(2.0 / (2.0 + effect.shininess))

        # Compute metallic factor
        metallicFactor = 0.0

        # Compute normal texture
        normalTexture = None
        if effect.bumpmap is not None:
            try:
                image_path = os.path.join(
                    MESH_PATH,self.mesh_info[idx]['name'],'visual_meshes',
                    effect.bumpmap.sampler.surface.image.path)
                normalTexture = self._load_texture(image_path)
            except BaseException:
                self.logger.warning('unable to load bumpmap',
                            exc_info=True)

        return trimesh.visual.material.PBRMaterial(
            emissiveFactor=emissiveFactor,
            emissiveTexture=emissiveTexture,
            normalTexture=normalTexture,
            baseColorTexture=baseColorTexture,
            baseColorFactor=baseColorFactor,
            metallicFactor=metallicFactor,
            roughnessFactor=roughnessFactor
        )


    def trimesh_material_to_pytorch3d(trimesh_material):
        """
        Convert a trimesh.visual.PBRMaterial to a pytorch3d.structures.Materials.

        Args:
        - trimesh_material (trimesh.visual.PBRMaterial): The trimesh PBRMaterial.

        Returns:
        - pytorch3d.structures.Materials: The corresponding PyTorch3D Materials object.
        """
        if not isinstance(trimesh_material, trimesh.visual.material.PBRMaterial):
            raise ValueError("Input must be a trimesh.visual.PBRMaterial")

        # Extract material properties from trimesh PBRMaterial
        diffuse_color = torch.tensor(trimesh_material.baseColorFactor, dtype=torch.float32)
        # specular_color = torch.tensor(trimesh_material.specular_color, dtype=torch.float32)
        roughness = torch.tensor(trimesh_material.roughnessFactor, dtype=torch.float32)
        metallic = torch.tensor(trimesh_material.metallicFactor, dtype=torch.float32)
        specular_texture = torch.tensor(trimesh_material.baseColorTexture.image, dtype=torch.float32)
        diffuse_texture = torch.tensor(trimesh_material.metallicRoughnessTexture.image, dtype=torch.float32)

        # Create PyTorch3D Materials object
        materials = Materials(
            diffuse_color=diffuse_color.unsqueeze(0),  # unsqueeze to add batch dimension
            # specular_color=specular_color.unsqueeze(0),  # unsqueeze to add batch dimension
            shininess=1.0 / roughness.unsqueeze(0),  # convert roughness to shininess
            metallic=metallic.unsqueeze(0),  # unsqueeze to add batch dimension
            diffuse_texture=diffuse_texture.unsqueeze(0),  # unsqueeze to add batch dimension
            specular_texture=specular_texture.unsqueeze(0),  # unsqueeze to add batch dimension
        )

        return materials

    
    def load_mesh(self):


        # 
        # c = collada.Collada(filepath)
        # material = self._parse_material(c.materials[0].effect,idx)

        # primitive=c.geometries[0].primitives[0]
        # texcoord=primitive.texcoordset[0]
        # texcoord_index = primitive.texcoord_indexset[0]
        # uv = texcoord[texcoord_index].reshape((len(texcoord_index) * 3, 2))

        # tvisual = trimesh.visual.TextureVisuals(uv=uv,material=material)
        # data = trimesh.load(filepath,process=False)
        # mesh_key = next(iter(data.geometry)) 
        # tmesh = data.geometry[mesh_key]
        # tmesh = trimesh.Trimesh(vertices=tmesh.vertices,
        #   faces=tmesh.faces,
        #   face_normals=tmesh.face_normals,
        #   visual=tvisual,
        #   validate=True,
        #   process=False
        # )
        
        # pmeshes = {'verts':[],'faces':[],'textures':[]}
        pmeshes = []
        for idx in self.mesh_info:
            filepath = self.mesh_info[idx]['path']
            # Load using Tmesh
            cur_dir = os.getcwd()
            ch_dir = os.path.dirname(filepath)
            os.chdir(ch_dir)
            tmesh = trimesh.load(filepath)
            tmesh = tmesh.geometry[next(iter(tmesh.geometry))]
            os.chdir(cur_dir)
        
            # Transfer to Pytorch3D 
            
            verts, faces = torch.from_numpy(tmesh.vertices).float(), torch.from_numpy(tmesh.faces)
            
            texture_uv = torch.from_numpy(np.array(tmesh.visual.uv))
            if self.mesh_info[idx]['name'] == 'prism': 
                texture_map = torch.from_numpy(np.ones((4096,4096,3)).astype(np.float32)) # NULL for prissim    
            else:
                texture_map = torch.from_numpy(np.array(tmesh.visual.material.baseColorTexture).astype(np.float32))
            
            # Downsample textures by a factor of 64 :(
            texture_map = texture_map[::8,::8] 
            
            texture = TexturesUV(maps=texture_map[None,:,:,:3], faces_uvs=faces[None], verts_uvs=[texture_uv])

            
            

            mesh = Meshes(
                verts=verts[None],   
                faces=faces[None], 
                textures=texture
            )

            # verts_rgb = torch.ones_like(verts)[None]
            
            # pmeshes['verts'].append(verts)
            # pmeshes['faces'].append(faces)
            # pmeshes['textures'].append(texture)
            pmeshes.append(mesh)


        pmeshes = join_meshes_as_batch(pmeshes)
        # tmesh.show()
        return pmeshes
    
    def load_sample(self,mesh_ids):
        return [ self.__getitem__(int(i)) for i in mesh_ids ]


def analyze_dataset(): 

	logger, writer = get_logger(task_name='Dataset-Analysis')

	# Analysis
	object_names = ['']*NUM_OBJECTS
	per_object_distribution = {}
	num_object_distribution = {}
	per_object_num_points = {}
	max_objects_per_sample = 0
	valid_classes = 0


	for split_type in ['test','train','val']: 
		
		dataset = SyntheticDataset(split_type=split_type)
		per_object_distribution[split_type] = np.zeros(NUM_OBJECTS) 
		num_object_distribution[split_type] = [] 
		per_object_num_points[split_type] = {}

		for idx in range(len(dataset)):
			sample_path = os.path.join(dataset.datapath,'v2.2',dataset.samples[idx])
			print(sample_path)
			# try:
			# For each sample, we check  
			sample_meta = dataset.load_pose(sample_path)
			sample_mask = dataset.load_mask(sample_path + '_label_kinect.png')
			
			# Make sure correct mask corresponds to meta file
			assert set(np.unique(sample_mask)) == set(np.unique(sample_mask)), f"Segmentation mask:{np.unique(sample_mask)} and objects ids:{np.unique(sample_mask)} do not match"
			
			for i,x in enumerate(sample_meta['object_ids']):
				if sample_meta['object_names'][i] != object_names[x]: 
					logger.debug(f"{split_type} {idx}/{len(dataset)} Replacing string for label-{x}={object_names[x]} to {sample_meta['object_names'][i]}") 
					object_names[x] = sample_meta['object_names'][i]

				num_points = (sample_mask == x).sum()
				if x not in per_object_num_points[split_type]: 
					per_object_num_points[split_type][x] = []
				per_object_num_points[split_type][x].append(num_points)		

			per_object_distribution[split_type][sample_meta['object_ids']] += 1
			num_object_distribution[split_type].append(len(sample_meta['object_ids']))
			# except Exception as e: 
			# 	logger.warning(f"Unble to load meta data:{sample_meta},{sample_path}\n{e}")

		num_object_distribution[split_type] = np.array(num_object_distribution[split_type])

	valid_classes = [i for i,x in enumerate(object_names) if len(x) > 0 ]

	logger.info(f"NUM VALID Classes: {len(valid_classes)}  Indices:{valid_classes} Object Names:{object_names}")
	logger.info(f"Max Objects : {[(x,num_object_distribution[x].max()) for x in num_object_distribution]}")
	logger.info(f"Per Object Distribution:{[per_object_distribution]}")


	plotter.plot_num_object_distribution(num_object_distribution,save_path=os.path.join(RENDER_DIR,'NumberObjectDistribution.png'))
	plotter.plot_per_object_distribution(per_object_distribution,object_names,save_path=os.path.join(RENDER_DIR,'PerObjectDistribution.png'))
	plotter.plot_points_object_distribution(per_object_num_points,object_names,save_path=os.path.join(RENDER_DIR,'NumPointsObjectDistribution'))


	# meshes = MeshInfo()





if __name__ == "__main__": 
	analyze_dataset()

		