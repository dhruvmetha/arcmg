from typing import Dict, Tuple
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from representation_learning.trainers.base_trainer import BaseTrainer
from representation_learning.models.distillation import create_distillation_model
from representation_learning.data.dataset import DistillationDataset
from ..utils.logging_utils import log_epoch, log_step, register_validation

class DistillationTrainer(BaseTrainer):
    def _create_model(self) -> torch.nn.Module:
        return create_distillation_model(**self.config['model']).to(self.device)
    
    def _create_data_loaders(self) -> Tuple[DataLoader]:    
        full_dataset = DistillationDataset(self.config['data']['path'])
        
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size

        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])    

        train_loader = DataLoader(train_dataset, batch_size=self.config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['training']['batch_size'], shuffle=False)

        return train_loader, val_loader
    
    @log_epoch("Training Epoch")
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        self.num_batches = len(self.train_loader)

        pbar = self.create_progress_bar(self.train_loader, "Training")
        for batch_idx, (raw_input, representation) in enumerate(pbar):
            batch = {'raw_input': raw_input, 'representation': representation}
            loss = self._train_step(batch=batch, batch_idx=batch_idx)
            total_loss += loss

            pbar.set_postfix({'loss': f'{loss:.4f}', 
                             'avg_loss': f'{total_loss/(batch_idx+1):.4f}'})

        return total_loss / self.num_batches
    
    def _train_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> float:
        raw_input, representation = batch['raw_input'], batch['representation']
        raw_input, representation = raw_input.to(self.device), representation.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(raw_input)
        loss = self.criterion(outputs, representation)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    @register_validation('default')
    @log_epoch("Validation Epoch")
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0
        self.num_batches = len(self.val_loader)

        pbar = self.create_progress_bar(self.val_loader, "Validation")
    
        for batch_idx, (raw_input, representation) in enumerate(pbar):
            batch = {'raw_input': raw_input, 'representation': representation}
            loss = self._validation_step(batch=batch, batch_idx=batch_idx)
            total_loss += loss

            pbar.set_postfix({'loss': f'{loss:.4f}', 
                            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'})

        return total_loss / self.num_batches

    def _validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> float:
        raw_input, representation = batch['raw_input'], batch['representation']
        raw_input, representation = raw_input.to(self.device), representation.to(self.device)
        with torch.no_grad():
            outputs = self.model(raw_input)
            loss = self.criterion(outputs, representation)
        return loss.item()