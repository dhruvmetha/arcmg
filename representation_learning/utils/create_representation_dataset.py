import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse
from representation_learning.models import create_transformer
from representation_learning.data import SequenceDataset

def create_representation_dataset(config: dict, model: torch.nn.Module, output_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained model
    model.eval()
    # Load the original dataset
    print(f"Loading dataset from {config['data']['data_dir']}")
    dataset = SequenceDataset(config['data']['data_dir'], config['data']['seq_len'], config['data']['mask_ratio_range'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Prepare lists to store data
    all_data_points = []
    all_representations = []
    
    # Generate representations
    with torch.no_grad():
        for traj, _, _, _ in tqdm(dataloader, desc="Generating representations"):
            traj = traj.to(device)
            representation = model.get_representation(traj)
            
            traj = traj.cpu().numpy()
            representation = representation.cpu().numpy()
            
            # Reshape to pair each point with its corresponding representation
            traj = traj.reshape(-1, traj.shape[-1])
            representation = representation.reshape(-1, representation.shape[-1])
            
            all_data_points.append(traj)
            all_representations.append(representation)
    
    # Save the dataset
    np.savez(
        Path(output_path) / "dataset.npz",
        data_points=np.concatenate(all_data_points),
        representations=np.concatenate(all_representations)
    )

    print(np.concatenate(all_data_points).shape)
    print(np.concatenate(all_representations).shape)
