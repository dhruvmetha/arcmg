import torch
from torch.utils.data import Dataset, TensorDataset
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Callable
from tqdm import tqdm

class SequenceDataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int, mask_ratio_range: Tuple[float, float], normalize):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.mask_ratio_range = mask_ratio_range
        self.data_max, self.data_min = 0.0, 0.0
        self.normalize = normalize
        self.datafiles = self._load_data()

        
    def _load_data(self) -> np.ndarray:
        datafiles = list(self.data_dir.glob('*.txt'))
        full_data = []

        # the files are valid if they can be loaded as a numpy array
        for file in datafiles:
            try:
                data = np.loadtxt(file, delimiter=',').tolist()
                full_data.extend(data)
                # max_sequence_length = max(max_sequence_length, data.shape[0])
            except:
                # remove the file from the list
                datafiles.remove(file)
                print(f"Invalid file: {file}")
            
        full_data = np.array(full_data)

        # padded_array = np.full((len(full_data), max_sequence_length, data.shape[1]), np.nan)
        # for i, data in enumerate(full_data):
        #     padded_array[i, :data.shape[0], :] = data
        # full_data = padded_array.copy()

        # setup column min, max for normalization
        self.data_max, self.data_min = np.max(full_data, axis=0), np.min(full_data, axis=0)
        return datafiles
    
    def __len__(self) -> int:
        return len(self.datafiles)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        trajectory = np.loadtxt(self.datafiles[idx], delimiter=',')
        trajectory = self.normalize(trajectory, self.data_min, self.data_max)
        len_trajectory = trajectory.shape[0]
        trajectory = trajectory[:min(len_trajectory, self.seq_len), :]
        
        # Create masked version
        mask_ratio = np.random.uniform(*self.mask_ratio_range)
        mask = torch.zeros(self.seq_len, dtype=torch.bool)
        mask_indices = torch.randperm(self.seq_len)[:int(mask_ratio * self.seq_len)]
        mask[mask_indices] = True
        masked_trajectory = trajectory.copy()
        masked_trajectory[mask] = 0

        
        if len_trajectory < self.seq_len:   
            padding = np.zeros((self.seq_len - len_trajectory, trajectory.shape[1]))
            trajectory = np.concatenate([trajectory, padding], axis=0)
        
        padding_mask = np.zeros(self.seq_len, dtype=bool)
        padding_mask[len_trajectory:] = True
        
        return (
            torch.FloatTensor(trajectory),
            torch.FloatTensor(mask * 1.0),
            torch.FloatTensor(masked_trajectory),
            torch.FloatTensor((~mask) * 1.0),
            torch.FloatTensor(~padding_mask * 1.0)
        )

class DistillationDataset(Dataset):
    """Dataset for training the distillation model."""
    def __init__(self, data_path: str, max_samples: int = 100000):
        data = np.load(data_path)

        if max_samples and len(data['data_points']) > max_samples:
            random_indices = np.random.choice(
                len(data['data_points']), 
                max_samples, 
                replace=False
            )
            self.data_points = torch.FloatTensor(data['data_points'][random_indices])
            self.representations = torch.FloatTensor(data['representations'][random_indices])
        else:
            self.data_points = torch.FloatTensor(data['data_points'])
            self.representations = torch.FloatTensor(data['representations'])

    def __len__(self) -> int:
        return len(self.data_points)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data_points[idx], self.representations[idx]