# A torch dataset for representation learning

import torch
from torch.utils.data import Dataset
import os
import numpy as np
import glob

def process_data(data):
    d = np.zeros((data.shape[0], 2))

    d[:, 0] = data[:, 0]/np.pi
    d[:, 1] = data[:, 1]/(2*np.pi)
    
    return d

class SequenceDataset(Dataset):
    def __init__(self, data_dir, seq_len, transform=None):
        self.seq_len = seq_len
        self.data_dir = sorted(glob.glob(os.path.join(data_dir, '*.txt')))

    def __len__(self):
        return len(self.data_dir)
    
    def __getitem__(self, idx):
        # load data from file
        traj = np.loadtxt(self.data_dir[idx], delimiter=',')
        traj = process_data(traj)
        traj = torch.tensor(traj, dtype=torch.float32)
        
        # sample a random subsequence of length seq_len with padding either in front or in the back if necessary
        start_idx = 0 # np.random.randint(0, traj.shape[0])
        end_idx = start_idx + self.seq_len
        traj = traj[start_idx:end_idx]
        
    
        padding_length = 0
        # pad the trajectory with zeros if necessary
        # if traj.shape[0] < self.seq_len:
        #     padding_length = self.seq_len - traj.shape[0]
        #     traj = np.concatenate([traj, np.zeros((self.seq_len - traj.shape[0], 2))])
        #     traj = torch.tensor(traj, dtype=torch.float32)


        # Store the original trajectory before masking
        traj_orig = traj.clone()

        padding_mask = torch.zeros(traj.shape[0], 1).bool()
        padding_mask[traj.shape[0] - padding_length:] = True


        # randomly mask 80% of the trajectory
        # the mask is a boolean tensor of shape (seq_len, 1)
        mask = torch.rand(traj.shape[0], 1) < np.random.uniform(0.5, 0.75)

        traj = traj * ((mask) * 1.0)

        # return the masked trajectory, the mask and the original trajectory
        return traj, mask, traj_orig, padding_mask

if __name__ == "__main__":
    cwd = "/media/dhruv/a7519aee-b272-44ae-a117-1f1ea1796db6/2024/arcmg/data/pendulum_lqr" 
    data_dir = os.path.join(cwd, 'pendulum_lqr1k', 'pendulum_lqr1k')

    dataset = SequenceDataset(data_dir, seq_len=10)
    for i in range(5):
        traj, mask, traj_orig = dataset[i]
        print(traj.shape, mask.shape, traj_orig.shape)
        
