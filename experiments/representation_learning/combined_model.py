import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse
import yaml
import matplotlib.pyplot as plt

from transformer import create_transformer
from distillation.distillation import DistillationModel
from data import SequenceDataset  # Import SequenceDataset from data.py

class CombinedModel(torch.nn.Module):
    def __init__(self, distillation_model, transformer_model):
        super().__init__()
        self.distillation_model = distillation_model
        self.transformer_output_proj = transformer_model.output_proj

    def forward(self, x):
        distilled_output = self.distillation_model(x)
        final_output = self.transformer_output_proj(distilled_output)
        return final_output

class CombinedModelRunner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.load_data()
        self.setup_models()
        self.setup_training()

    def load_data(self):
        print(f"Loading data from {self.config['data']['data_dir']}")
        full_dataset = SequenceDataset(self.config['data']['data_dir'], self.config['data']['seq_len'])
        
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.config['training']['batch_size'], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config['training']['batch_size'])
        
        print(f"Data loaded. Total samples: {len(full_dataset)}, Train samples: {train_size}, Validation samples: {val_size}")

    def setup_models(self):
        # Load the distillation model
        input_dim = self.config['model']['input_dim']
        distillation_model = DistillationModel(input_dim=input_dim, output_dim=self.config['model']['d_model'])
        distillation_model.load_state_dict(torch.load(self.config['model']['distillation_path']))
        
        # Load the transformer model
        transformer_model = create_transformer(
            input_dim=self.config['model']['input_dim'],
            d_model=self.config['model']['d_model'],
            nhead=self.config['model']['nhead'],
            num_layers=self.config['model']['num_layers'],
            dim_feedforward=self.config['model']['dim_feedforward']
        )
        transformer_model.load_state_dict(torch.load(self.config['model']['transformer_path']))
        
        # Create the combined model
        self.model = CombinedModel(distillation_model, transformer_model).to(self.device)
        print("Combined model created")

    def setup_training(self):
        self.criterion = torch.nn.MSELoss()

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_originals = []
        with torch.no_grad():
            for traj_orig, mask, traj, padding_mask in tqdm(self.train_loader, desc="Evaluating"):
                traj, traj_orig = traj.to(self.device), traj_orig.to(self.device)
                output = self.model(traj)
                loss = self.criterion(output, traj)
                total_loss += loss.item()
                all_outputs.append(output.cpu())
                all_originals.append(traj.cpu())
        
        return total_loss / len(self.val_loader), all_outputs, all_originals

    def plot_trajectories(self, outputs, originals, num_plots=100):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot all original trajectories
        for i in range(num_plots):
            ax[0].plot(originals[i][:, 0], originals[i][:, 1], label=f'Original {i+1}')
        ax[0].set_title('Original Trajectories')
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')
        ax[0].set_xlim(-1.0, 1.0)
        ax[0].set_ylim(-1.0, 1.0)
        
        # Plot all model output trajectories
        for i in range(num_plots):
            ax[1].plot(outputs[i][:, 0], outputs[i][:, 1], label=f'Model Output {i+1}')
        ax[1].set_title('Model Output Trajectories')
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Y')
        ax[0].set_xlim(-1.0, 1.0)
        ax[0].set_ylim(-1.0, 1.0)

        plt.tight_layout()
        plt.savefig('trajectory_comparison.png')
        print("Trajectory comparison plot saved as trajectory_comparison.png")

    def run_evaluation(self):
        print("Starting evaluation")
        loss, outputs, originals = self.evaluate()
        print(f"Evaluation Loss: {loss:.4f}")
        
        # Concatenate all batches
        all_outputs = torch.cat(outputs, dim=0)
        all_originals = torch.cat(originals, dim=0)
        
        # Plot trajectories
        self.plot_trajectories(all_outputs, all_originals)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    parser = argparse.ArgumentParser(description="Combined Model Evaluator")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    config = load_config(args.config)
    runner = CombinedModelRunner(config)
    runner.run_evaluation()

if __name__ == "__main__":
    main()
