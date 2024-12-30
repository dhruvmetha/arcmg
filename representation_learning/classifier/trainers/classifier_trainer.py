from typing import Dict, Tuple
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from ..data.dataset import ReachabilityDataset
from ..models.classifier import create_classifier
from ...trainers.base_trainer import BaseTrainer
from ...utils.logging_utils import log_epoch, register_validation, log_step
from ...models.distillation import create_distillation_model
from representation_learning.utils.config import ConfigManager

class ClassifierTrainer(BaseTrainer):
    def __init__(self, config: Dict, system_name: str):

        self.distillation_config = None
        self.distillation_model_dir = Path(config['distillation_model'])

        # Load distillation model if using representations
        if config['training'].get('use_representations', True):
            self.distillation_model = self._create_distillation_model(self.distillation_model_dir)
        else:
            self.distillation_model = None
            
        super().__init__(config, system_name)
        # self.criterion = torch.nn.BCEWithLogitsLoss() 
        # self.distillation_model = self.distillation_model.to(self.device)

        self.model = self.model.to(self.device)

    def _create_model(self) -> torch.nn.Module:
        """Create the classifier model."""
        return create_classifier(self.config['model']).to(self.device)
    
    def _create_distillation_model(self, model_dir: str) -> torch.nn.Module:
        """Create and load pretrained distillation model."""
        self.distillation_config = ConfigManager.load_specific_config(model_dir / 'config.yaml')
        model = create_distillation_model(**self.distillation_config['model'])
       
        checkpoint = torch.load(
            model_dir / 'checkpoints/best_model.pth', 
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation data loaders."""
        dataset = ReachabilityDataset(
            data_path=self.config['data']['path'],
            distillation_model=self.distillation_model,
            use_representations=self.config['training'].get('use_representations', True),
            normalize=lambda data, data_min, data_max: self.system.normalize(data, data_min, data_max)
        )
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )
        
        return train_loader, val_loader
    
    @log_epoch("Training Loss")
    def train_epoch(self) -> None:
        """Single training epoch."""
        self.model.train()
        total_loss = 0
        self.num_batches = len(self.train_loader)

        pbar = self.create_progress_bar(self.train_loader, "Training")
        for batch_idx, (features, labels) in enumerate(pbar):
            batch = {'features': features, 'labels': labels}
            loss = self._train_step(batch=batch, batch_idx=batch_idx)
            total_loss += loss
            pbar.set_postfix({'loss': f'{loss:.4f}', 
                             'avg_loss': f'{total_loss/(batch_idx+1):.4f}'})
        
        return total_loss / self.num_batches
    
    
    def _train_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> float:
        """Single training step."""
        features, labels = batch['features'].to(self.device), batch['labels'].to(self.device)
        
        self.optimizer.zero_grad()
        predictions = self.model(features)
        loss = self.criterion(predictions, labels.unsqueeze(1))
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    @register_validation('default')
    @log_epoch("Validation")
    def validate(self) -> None:
        """Single validation epoch."""
        self.model.eval()
        total_loss = 0
        self.num_batches = len(self.val_loader)

        pbar = self.create_progress_bar(self.val_loader, "Validation")
        for batch_idx, (features, labels) in enumerate(pbar):
            batch = {'features': features, 'labels': labels}
            loss = self._validation_step(batch=batch, batch_idx=batch_idx)
            total_loss += loss
            pbar.set_postfix({'loss': f'{loss:.4f}', 
                             'avg_loss': f'{total_loss/(batch_idx+1):.4f}'})
        
        return total_loss / self.num_batches
    
    def _validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> float:
        """Single validation step."""
        features, labels = batch['features'].to(self.device), batch['labels'].to(self.device)
        with torch.no_grad():   
            predictions = self.model(features)
            loss = self.criterion(predictions, labels.unsqueeze(1))
        return loss.item()
    
    
    def evaluate(self, dataset) -> None:
        """Evaluate the model on a dataset."""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(dataset['features'])
            return predictions
        
    def _create_scheduler(self) -> _LRScheduler:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
    
    def _create_criterion(self) -> torch.nn.Module:
        return torch.nn.BCEWithLogitsLoss()