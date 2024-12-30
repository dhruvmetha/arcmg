import torch
import os
from tqdm import tqdm
import numpy as np

from data import SequenceDataset
from transformer import create_transformer
from runner import load_config

def create_representation_dataset(config, model_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained model
    model = create_transformer(
        input_dim=config['model']['input_dim'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dim_feedforward=config['model']['dim_feedforward']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load the original dataset
    dataset = SequenceDataset(config['data']['data_dir'], config['data']['seq_len'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Prepare lists to store data
    all_data_points = []
    all_representations = []
    
    # Generate representations
    with torch.no_grad():
        for _, _, traj, _ in tqdm(dataloader, desc="Generating representations"):
            traj = traj.to(device).transpose(0, 1)
            representation = model.get_representation(traj)
            
            # traj shape: [seq_len, batch_size, input_dim]
            # representation shape: [seq_len, batch_size, d_model]
            traj = traj.cpu().numpy()
            representation = representation.cpu().numpy()
            
            # Reshape to pair each point with its corresponding representation
            traj = traj.reshape(-1, traj.shape[-1])
            representation = representation.reshape(-1, representation.shape[-1])
            
            all_data_points.append(traj)
            all_representations.append(representation)
    
    # Concatenate all data
    all_data_points = np.concatenate(all_data_points, axis=0)
    all_representations = np.concatenate(all_representations, axis=0)
    
    # Save the dataset
    np.savez_compressed(output_path, 
                        data_points=all_data_points, 
                        representations=all_representations)
    
    print(f"Dataset saved to {output_path}")
    print(f"Data points shape: {all_data_points.shape}")
    print(f"Representations shape: {all_representations.shape}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Representation Dataset")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output dataset')
    args = parser.parse_args()

    config = load_config(args.config)
    create_representation_dataset(config, args.model, args.output)

if __name__ == "__main__":
    main()
