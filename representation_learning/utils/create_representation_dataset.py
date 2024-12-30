import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
from representation_learning.data import SequenceDataset
from experiments.systems import system_factory

def create_representation_dataset(config: dict, system_name: str, model: torch.nn.Module, output_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    system_model = system_factory[system_name]()
    # Load the trained model
    model.eval()
    # Load the original dataset
    print(f"Loading dataset from {config['data']['data_dir']}")
    dataset = SequenceDataset(config['data']['data_dir'], config['data']['seq_len'], config['data']['mask_ratio_range'], lambda traj, data_min, data_max: system_model.normalize(traj, data_min, data_max))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Prepare lists to store data
    all_data_points = []
    all_representations = []
    all_padding_masks = []
    
    # Generate representations
    with torch.no_grad():
        for traj, _, _, _, padding_mask in tqdm(dataloader, desc="Generating representations"):
            traj = traj.to(device)
            representation = model.get_representation(traj)
            
            first_padding_index = padding_mask.squeeze().nonzero()[-1].item() + 1
            
            
            traj = traj.cpu().numpy()[:, :first_padding_index, :]
            representation = representation.cpu().numpy()[:, :first_padding_index, :]
            
            # Reshape to pair each point with its corresponding representation
            traj = traj.reshape(-1, traj.shape[-1])
            representation = representation.reshape(-1, representation.shape[-1])
            
            all_data_points.append(traj)
            all_representations.append(representation)
    # Save the dataset

    if not Path(output_path).exists():
        Path(output_path).mkdir(parents=True, exist_ok=True)

    np.savez(
        Path(output_path) / "dataset.npz",
        data_points=np.concatenate(all_data_points),
        representations=np.concatenate(all_representations),
    )

    print(np.concatenate(all_data_points).shape)
    print(np.concatenate(all_representations).shape)