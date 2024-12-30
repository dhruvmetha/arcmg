from .base_trainer import BaseTrainer
from ..models import create_transformer
from ..data import SequenceDataset
import torch
from torch.utils.data import DataLoader, random_split
from typing import Dict, Any, Tuple
from ..utils.logging_utils import log_epoch, log_step, register_validation
from ..utils.visualization import plot_trajectory_comparison, evaluate_full_trajectories
from pathlib import Path
import numpy as np

class TransformerTrainer(BaseTrainer):
    def _create_model(self) -> torch.nn.Module:
        model = create_transformer(**self.config['model'])
        return model.to(self.device)
        
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        full_dataset = SequenceDataset(
            self.config['data']['data_dir'],  
            self.config['data']['seq_len'],
            self.config['data']['mask_ratio_range'],
            lambda traj, data_min, data_max: self.system.normalize(traj, data_min, data_max)
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
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        return train_loader, val_loader
        
    @log_epoch("Training Epoch")
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        self.num_batches = len(self.train_loader)
        
        pbar = self.create_progress_bar(self.train_loader, "Training")
        for batch_idx, (full_traj, _, traj, _, padding_mask) in enumerate(pbar):
            batch = {'full_traj': full_traj, 'traj': traj, 'padding_mask': padding_mask}
            loss = self._train_step(batch=batch, batch_idx=batch_idx)
            total_loss += loss
            pbar.set_postfix({'loss': f'{loss:.4f}', 
                             'avg_loss': f'{total_loss/(batch_idx+1):.4f}'})
        
        return total_loss / self.num_batches

    def _create_criterion(self) -> torch.nn.Module:
        return torch.nn.MSELoss(reduction='none')
        # return torch.nn.MSELoss(reduction='mean')
        
    def _train_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> float:
        traj, full_traj, padding_mask = batch['traj'].to(self.device), batch['full_traj'].to(self.device), batch['padding_mask'].to(self.device)
        self.optimizer.zero_grad()
        output = self.model(traj)
        loss = self.criterion(output, full_traj) 
        
        loss = loss * padding_mask.unsqueeze(-1)
        num_valid = padding_mask.sum() * 2
        
        if num_valid > 0:
            loss = loss.sum() / num_valid
        else:
            loss = loss.sum() * 0.0

        loss.backward()
        if self.config['training'].get('clip_value', None):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['clip_value']
            )
        self.optimizer.step()
        return loss.item()
        
    @register_validation('default')
    @log_epoch("Validation")
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0
        self.num_batches = len(self.val_loader)
        
        pbar = self.create_progress_bar(self.val_loader, "Validating")
        with torch.no_grad():
            for batch_idx, (full_traj, _, traj, _, padding_mask) in enumerate(pbar):
                batch = {'full_traj': full_traj, 'traj': traj, 'padding_mask': padding_mask}
                loss = self._validation_step(batch=batch, batch_idx=batch_idx)
                total_loss += loss
                pbar.set_postfix({'loss': f'{loss:.4f}', 
                                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'})
        
        return total_loss / self.num_batches

    def _validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> float:    
        traj, full_traj, padding_mask = batch['traj'].to(self.device), batch['full_traj'].to(self.device), batch['padding_mask'].to(self.device)
        output = self.model(traj)
        loss = self.criterion(output, full_traj) 
        loss = loss * padding_mask.unsqueeze(-1)
        num_valid = padding_mask.sum() * 2
        if num_valid > 0:
            loss = loss.sum() / num_valid
        else:
            loss = loss.sum() * 0.0
        return loss.item()
    
    @register_validation('full_trajectory')
    @log_epoch("Validation Full Trajectory")
    def validate_full_traj(self) -> float:
        self.model.eval()
        total_loss = 0
        self.num_batches = len(self.val_loader)

        pbar = self.create_progress_bar(self.val_loader, "Validating Full Trajectory")
        with torch.no_grad():
            for batch_idx, (full_traj, _, traj, _, padding_mask) in enumerate(pbar):
                batch = {'full_traj': full_traj, 'traj': full_traj, 'padding_mask': padding_mask}
                loss = self._validation_full_traj_step(batch=batch, batch_idx=batch_idx)
                total_loss += loss
                pbar.set_postfix({'loss': f'{loss:.4f}', 
                                 'avg_loss': f'{total_loss/(batch_idx+1):.4f}'})
        
        return total_loss / self.num_batches
    
    def _validation_full_traj_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> float:
        traj, full_traj, padding_mask = batch['traj'].to(self.device), batch['full_traj'].to(self.device), batch['padding_mask'].to(self.device)
        output = self.model(traj)
        loss = self.criterion(output, full_traj)
        loss = loss * padding_mask.unsqueeze(-1)
        num_valid = padding_mask.sum() * 2
        if num_valid > 0:
            loss = loss.sum() / num_valid
        else:
            loss = loss.sum() * 0.0
        return loss.item()

    def _register_validation_functions(self):
        """Register validation functions specific to TransformerTrainer."""
        super()._register_validation_functions()
        self._validation_functions['full_trajectory'] = self.validate_full_traj

    def visualize_full_trajectories(self, checkpoint_path: Path, output_dir: Path, num_samples: int = 10):
        """Evaluate and visualize model performance on full trajectories."""
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get predictions
        inputs, outputs = evaluate_full_trajectories(
            model=self.model,
            val_loader=self.val_loader,
            device=self.device,
            num_samples=num_samples
        )
        
        # Create visualization
        save_path = output_dir / 'full_trajectory_comparison.png'
        plot_trajectory_comparison(
            inputs=inputs,
            outputs=outputs,
            save_path=save_path,
            num_samples=num_samples,
            title=f"Full Trajectory Comparison - {self.__class__.__name__}"
        )
        
        self.logger.info(f"Full trajectory comparison plot saved to {save_path}")