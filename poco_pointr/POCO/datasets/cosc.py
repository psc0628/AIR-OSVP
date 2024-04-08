from torch_geometric.data import Dataset
from lightconvpoint.datasets.data import Data
import os
import numpy as np
import torch
import logging

class COSC(Dataset):

    def __init__(self, root, scene_file=None, split="surface", transform=None, dataset_size=None, obj_list=[], **kwargs):
            
        super().__init__(root, transform, None)

        self.filenames = [scene_file]
        self.obj_list = obj_list
        
        logging.info(self.filenames)
        logging.info(f"Dataset - len {len(self.filenames)}")

    def get_category(self, f_id):
        return self.filenames[f_id].split("/")[-2]

    def get_object_name(self, f_id):
        return self.filenames[f_id].split("/")[-3]

    def get_class_name(self, f_id):
        return self.metadata[self.get_category(f_id)]["name"]
    
    def get_rotate_name(self, f_id):
        return self.filenames[f_id].split("/")[-2]
    
    def get_scene_name(self, f_id):
        return self.filenames[f_id].split("/")[-1]

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def _download(self): # override _download to remove makedirs
        pass

    def download(self):
        pass

    def process(self):
        pass

    def _process(self):
        pass

    def len(self):
        return len(self.filenames)

    def get(self, idx):
        """Get item."""

        filename = self.filenames[idx]

        manifold_data = np.loadtxt(filename, dtype=np.float32)
        points = np.loadtxt(filename.replace("scene", "surface"))
        # points = np.loadtxt(filename.replace("scene_view", "surface_view"))

        pts_shp = torch.tensor(points, dtype=torch.float) 
        points_space = torch.tensor(manifold_data[:, :3], dtype=torch.float) 
        occupancies = torch.tensor(manifold_data[:, 3], dtype=torch.long)

        data = Data(x = torch.ones_like(pts_shp),
                    shape_id=idx, 
                    pos=pts_shp,
                    normal=None,
                    pos_non_manifold=points_space, occupancies=occupancies, #
                    )

        return data
