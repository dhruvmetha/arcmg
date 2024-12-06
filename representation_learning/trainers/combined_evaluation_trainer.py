from typing import Dict, Tuple
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from .base_trainer import BaseTrainer
from ..models.distillation import create_distillation_model
from ..models.transformer import create_transformer
from ..utils.logging_utils import log_epoch, register_validation
from ..utils.visualization import plot_trajectory_comparison
from ..utils.config import ConfigManager
from ..data import SequenceDataset


class CombinedEvaluationTrainer(BaseTrainer):
    def _create_model(self) -> torch.nn.Module:
        # Load distillation model
        self.distillation_config = None
        distillation_dir = Path(self.config['distillation_model']['model_dir']) 
        self.distillation_model = self._create_distillation_model(distillation_dir)
        
        # Load transformer model
        self.transformer_config = None
        transformer_dir = Path(self.config['transformer_model']['model_dir'])
        self.transformer_model = self._create_transformer_model(transformer_dir)
        """Dummy model creation as we're not training"""
        return torch.nn.Identity()  # dummy model
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        full_dataset = SequenceDataset(
            self.transformer_config['data']['data_dir'], 
            self.transformer_config['data']['seq_len'],
            self.transformer_config['data']['mask_ratio_range']
        )
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['training']['batch_size'], 
            shuffle=True
        )
        
        val_loader = DataLoader(val_dataset, batch_size=64)
        
        return train_loader, val_loader
        
    def _create_distillation_model(self, distillation_dir: str) -> torch.nn.Module:
        """Load pretrained distillation model"""
        self.distillation_config = ConfigManager.load_specific_config   (distillation_dir / 'config.yaml')
        model = create_distillation_model(**(self.distillation_config['model'])).to(self.device)
        checkpoint = torch.load(distillation_dir / 'checkpoints/best_model.pth', map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
        
    def _create_transformer_model(self, transformer_dir: str) -> torch.nn.Module:
        """Load pretrained transformer model"""
        self.transformer_config = ConfigManager.load_specific_config(transformer_dir / 'config.yaml')
        
        model = create_transformer(**(self.transformer_config['model'])).to(self.device)
        checkpoint = torch.load(transformer_dir / 'checkpoints/best_model.pth', map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    @register_validation('reconstruction')
    @log_epoch("Reconstruction Validation")
    def validate_reconstruction(self) -> float:
        """Validate the full pipeline: input -> representation -> reconstruction"""
        total_loss = 0
        self.num_batches = len(self.val_loader)
        
        pbar = self.create_progress_bar(self.val_loader, "Validating Reconstruction")
        trajectories = []
        reconstructions = []
        
        with torch.no_grad():
            for batch_idx, (traj, _, _, _) in enumerate(pbar):
                # Move to device
                traj = traj.to(self.device)
                
                # Get representation using distillation model
                representation = self.distillation_model(traj)
                
                # Get reconstruction using transformer's output projection
                reconstruction = self.transformer_model.output_proj(representation)
                
                # Calculate reconstruction loss
                loss = self.criterion(reconstruction, traj)
                total_loss += loss.item()
                
                # Store for visualization
                trajectories.append(traj.cpu())
                reconstructions.append(reconstruction.cpu())
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}', 
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                })
        
        # Save visualization
        self._save_reconstruction_plot(trajectories, reconstructions, num_samples=100)
        
        return total_loss / self.num_batches
    
    def _save_reconstruction_plot(self, trajectories: list, reconstructions: list, num_samples: int = 10):
        """Save visualization of original vs reconstructed trajectories"""
        # Concatenate all batches
        all_trajectories = torch.cat(trajectories, dim=0)
        all_reconstructions = torch.cat(reconstructions, dim=0)
        
        # Select random samples
        indices = torch.randperm(len(all_trajectories))[:num_samples]
        sample_trajectories = all_trajectories[indices]
        sample_reconstructions = all_reconstructions[indices]
        
        # Create visualization
        save_path = Path(self.config['output_dir']) / 'reconstruction_comparison.png'
        plot_trajectory_comparison(
            inputs=sample_trajectories,
            outputs=sample_reconstructions,
            save_path=save_path,
            num_samples=num_samples,
            title="Original vs Reconstructed Trajectories"
        )
        
        self.logger.info(f"Reconstruction comparison plot saved to {save_path}")

    def validate(self) -> float:
        """Override default validate to use reconstruction validation"""
        return self.validate_reconstruction()