"""
Full plot dataset loader for sliding window evaluation.
"""
# =============================================================================
# Imports
# =============================================================================
import os
from PIL import Image
import torch
from torch.utils.data import Dataset


# =============================================================================
class FullPlotDataset(Dataset):
    def __init__(self, data_dir: str, transform) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self._check_data_dir()
        self.data_paths = self._load_data_paths()
    
    def _check_data_dir(self) -> None:
        if not os.path.isdir(self.data_dir):
            raise ValueError(f"Data directory '{self.data_dir}' does not exist.")
    
    def _load_data_paths(self) -> None:
        data_paths = []      
        for path in os.listdir(self.data_dir):
            data_paths.append(os.path.join(self.data_dir, path))
        
        return data_paths
    
    def __len__(self) -> int:
        return len(self.data_paths)
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        return Image.open(image_path)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.data_paths[idx]
        image = self._load_image(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image