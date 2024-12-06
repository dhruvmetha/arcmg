import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
import yaml
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

from data import SequenceDataset
from transformer import create_transformer

class TransformerRunner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = create_transformer(
            input_dim=config['model']['input_dim'],
            d_model=config['model']['d_model'],
            nhead=config['model']['nhead'],
            num_layers=config['model']['num_layers'],
            dim_feedforward=config['model']['dim_feedforward']
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['learning_rate'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.clip_value = 1.0  # Add gradient clipping
        
        self.train_loader, self.val_loader = self._create_data_loaders()

        self.train_losses = []
        self.val_losses = []
        self.unmasked_losses = []

    def _create_data_loaders(self):
        full_dataset = SequenceDataset(self.config['data']['data_dir'], self.config['data']['seq_len'])
        
        # Calculate the size of each split
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size

        print(train_size, val_size)

        # Split the dataset
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        
        return train_loader, val_loader

    def train(self):
        self.model.train()
        total_loss = 0
        for traj, mask, traj_orig, padding_mask in tqdm(self.train_loader, desc="Training"):
            traj, mask, traj_orig, padding_mask = traj.to(self.device), mask.to(self.device), traj_orig.to(self.device), padding_mask.to(self.device)
            
            self.optimizer.zero_grad()
            traj = traj.transpose(0, 1)
            mask = mask.squeeze(-1).bool()  # Remove transpose
            padding_mask = padding_mask.squeeze(-1).bool()  # Remove transpose
            
            output = self.model(traj, src_key_padding_mask=padding_mask)

            
            traj_orig = traj_orig.transpose(0, 1)
            
            # Calculate loss only for masked (filled) elements, excluding padding
            loss = self.criterion(output, traj_orig)

            # loss = loss * (~mask).transpose(0, 1).unsqueeze(-1).float()

            # valid_mask = (~padding_mask)
            # loss = loss * valid_mask.transpose(0, 1).unsqueeze(-1).float()
            # loss = loss.sum() / valid_mask.sum()  # Average over masked elements

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for traj, mask, traj_orig, padding_mask in tqdm(self.val_loader, desc="Validating"):
                traj, mask, traj_orig, padding_mask = traj.to(self.device), mask.to(self.device), traj_orig.to(self.device), padding_mask.to(self.device)
                
                traj = traj.transpose(0, 1)
                mask = mask.squeeze(-1).bool()  # Remove transpose
                padding_mask = padding_mask.squeeze(-1).bool()  # Remove transpose
                output = self.model(traj, src_key_padding_mask=padding_mask)
                traj_orig = traj_orig.transpose(0, 1)
                
                # Calculate loss only for masked (filled) elements, excluding padding
                loss = self.criterion(output, traj_orig)

                # valid_mask = (~mask) & (~padding_mask)
                # loss = loss * valid_mask.transpose(0, 1).unsqueeze(-1).float()
                # loss = loss.sum() / valid_mask.sum()  # Average over masked elements
                
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def evaluate_unmasked(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for traj, _, traj_orig, padding_mask in tqdm(self.val_loader, desc="Evaluating Unmasked"):
                traj, traj_orig, padding_mask = traj.to(self.device), traj_orig.to(self.device), padding_mask.to(self.device)
                traj_orig = traj_orig.transpose(0, 1)
                padding_mask = padding_mask.squeeze(-1).bool()
                output = self.model(traj_orig, src_key_padding_mask=padding_mask)  # No masking applied
                
                # Calculate MSE loss for the entire sequence
                loss = F.mse_loss(output, traj_orig)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def run_training(self):
        for epoch in range(self.config['training']['num_epochs']):
            train_loss = self.train()
            val_loss = self.validate()
            unmasked_loss = self.evaluate_unmasked()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.unmasked_losses.append(unmasked_loss)
            
            print(f"Epoch {epoch+1}/{self.config['training']['num_epochs']}, "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Unmasked Loss: {unmasked_loss:.4f}")
            
            self.scheduler.step(val_loss)
            
            if not torch.isfinite(torch.tensor([train_loss, val_loss, unmasked_loss])).all():
                print("Stopping training due to infinite loss")
                break
        
        # Save the trained model
        torch.save(self.model.state_dict(), self.config['model']['save_path'])
        
        # Plot and save the loss graph
        self.plot_losses()
        
        # Plot a full batch of trajectories
        self.plot_batch_trajectories()

    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.plot(self.unmasked_losses, label='Unmasked Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training, Validation, and Unmasked Losses')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plot_path = os.path.join(os.path.dirname(self.config['model']['save_path']), 'loss_plot.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Loss plot saved to {plot_path}")

    def plot_random_sample(self):
        self.model.eval()
        # Randomly select a batch from the validation set
        random_batch = random.choice(list(self.val_loader))
        traj, _, traj_orig, _ = random_batch
        
        # Randomly select a sample from the batch
        sample_idx = random.randint(0, traj.shape[0] - 1)
        input_traj = traj_orig[sample_idx].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            input_traj = input_traj.transpose(0, 1)
            output_traj = self.model(input_traj)
            output_traj = output_traj.squeeze().cpu().numpy()
        
        input_traj = input_traj.squeeze().cpu().numpy()

        
        # Plot input and output trajectories
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.scatter(input_traj[:, 0], input_traj[:, 1], label='Input')
        ax1.set_title('Input Trajectory')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
        
        ax2.scatter(output_traj[:, 0], output_traj[:, 1], label='Output')
        ax2.set_title('Model Output')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.legend()
        
        plt.tight_layout()
        sample_plot_path = os.path.join(os.path.dirname(self.config['model']['save_path']), 'sample_trajectory_plot.png')
        plt.savefig(sample_plot_path)
        plt.close()
        print(f"Sample trajectory plot saved to {sample_plot_path}")

    def run_inference(self, input_sequence):
        self.model.eval()
        with torch.no_grad():
            input_sequence = input_sequence.unsqueeze(1).to(self.device)  # Add batch dimension
            output = self.model(input_sequence)
            return output.squeeze(1).cpu()  # Remove batch dimension

    def plot_batch_trajectories(self):
        self.model.eval()
        # Get a batch from the validation set
        batch = next(iter(self.val_loader))
        traj_orig, _, traj, _ = batch
        
        # Create a folder to save the plots
        plot_folder = os.path.join(os.path.dirname(self.config['model']['save_path']), 'batch_trajectory_plots')
        os.makedirs(plot_folder, exist_ok=True)
        
        with torch.no_grad():
            traj = traj.to(self.device).transpose(0, 1)
            output = self.model(traj)
            output = output.transpose(0, 1).cpu().numpy()
        
        traj = traj.transpose(0, 1).cpu().numpy()

        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        for i in range(traj.shape[0]):
            ax1.plot(traj[i, :, 0], traj[i, :, 1],  label=f'Trajectory {i+1}' if i == 0 else None)
            ax2.plot(output[i, :, 0], output[i, :, 1], label=f'Trajectory {i+1}' if i == 0 else None)
        
        ax1.set_title('Input Trajectories')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_xlim(-1.0, 1.0)
        ax1.set_ylim(-1.0, 1.0)
        ax1.legend()
        
        ax2.set_title('Model Output Trajectories')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_xlim(-1.0, 1.0)
        ax2.set_ylim(-1.0, 1.0)
        ax2.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(plot_folder, 'batch_trajectory_plots.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Batch trajectory plots saved to {plot_folder}")

def load_config(config_path):
    def numeric_constructor(loader, node):
        value = loader.construct_scalar(node)
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    yaml.add_constructor('tag:yaml.org,2002:str', numeric_constructor)
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    parser = argparse.ArgumentParser(description="Transformer Runner")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    config = load_config(args.config)
    
    runner = TransformerRunner(config)
    
    if config['mode'] == "train":
        runner.run_training()
    elif config['mode'] == "inference":
        # Example inference
        input_sequence = torch.randn(config['data']['seq_len'], config['model']['input_dim'])
        output = runner.run_inference(input_sequence)
        print("Inference output shape:", output.shape)
        print("Inference output:", output)

if __name__ == "__main__":
    main()
