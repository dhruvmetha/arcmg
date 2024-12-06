from .base_trainer import BaseTrainer
from ..models.combined_model import CombinedModel
from ..data.dataset import SequenceDataset
import torch
from torch.utils.data import DataLoader
from typing import Tuple, List
import matplotlib.pyplot as plt
from tqdm import tqdm

class CombinedModelTrainer(BaseTrainer):
    def _create_model(self) -> torch.nn.Module:
        model = CombinedModel.from_pretrained(self.config, self.device)
        return model.to(self.device)

    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        full_dataset = SequenceDataset(
            self.config['data']['data_dir'], 
            self.config['data']['seq_len']
        )
        
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['training']['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['training']['batch_size']
        )
        
        self.logger.info(
            f"Data loaded. Total samples: {len(full_dataset)}, "
            f"Train samples: {train_size}, "
            f"Validation samples: {val_size}"
        )
        
        return train_loader, val_loader 

    def evaluate(self) -> Tuple[float, List[torch.Tensor], List[torch.Tensor]]:
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_originals = []
        
        with torch.no_grad():
            for traj_orig, mask, traj, padding_mask in tqdm(self.train_loader, desc="Evaluating"):
                traj = traj.to(self.device)
                traj_orig = traj_orig.to(self.device)
                
                output = self.model(traj)
                loss = self.criterion(output, traj)
                total_loss += loss.item()
                
                all_outputs.append(output.cpu())
                all_originals.append(traj.cpu())
        
        return total_loss / len(self.val_loader), all_outputs, all_originals

    def plot_trajectories(self, outputs: List[torch.Tensor], originals: List[torch.Tensor], 
                         num_plots: int = 100, save_path: str = 'trajectory_comparison.png'):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot original trajectories
        for i in range(min(num_plots, len(originals))):
            ax[0].plot(originals[i][:, 0], originals[i][:, 1], label=f'Original {i+1}')
        ax[0].set_title('Original Trajectories')
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')
        ax[0].set_xlim(-1.0, 1.0)
        ax[0].set_ylim(-1.0, 1.0)
        
        # Plot model output trajectories
        for i in range(min(num_plots, len(outputs))):
            ax[1].plot(outputs[i][:, 0], outputs[i][:, 1], label=f'Model Output {i+1}')
        ax[1].set_title('Model Output Trajectories')
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Y')
        ax[1].set_xlim(-1.0, 1.0)
        ax[1].set_ylim(-1.0, 1.0)
        
        plt.tight_layout()
        plt.savefig(save_path)
        self.logger.info(f"Trajectory comparison plot saved as {save_path}")