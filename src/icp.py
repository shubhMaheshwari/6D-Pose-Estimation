# Basic modules 
import os 
import numpy as np 

# Mesh loaders
import trimesh

# DL Modules 
import torch 
from lietorch import SO3, SE3, LieGroupParameter 

# Local Modules
from utils import *


class ICP: 
    def __init__(self):
        pass
    
    def __call__(self,pcd): 
        N,_ = pcd['points'].shape

        
