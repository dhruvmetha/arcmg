from typing import Dict, Tuple
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from representation_learning.classifier.data.dataset import ProbabilityDataset
from representation_learning.classifier.models.probability import create_probability_classifier
from representation_learning.classifier.models.classifier import create_classifier
from representation_learning.trainers.base_trainer import BaseTrainer
from representation_learning.utils.logging_utils import log_epoch, register_validation, log_step
from representation_learning.models.distillation import create_distillation_model
from representation_learning.utils.config import ConfigManager

class ProbabilityTrainer(BaseTrainer):
    def __init__(self, config: Dict):

        self.distillation_config = None
        self.distillation_model_dir = Path(config['distillation_model']['model_dir'])

        self.classifier_config = None
        self.classifier_model_dir = Path(config['classifier_model']['model_dir'])

        if config['training'].get('use_representations', True):
            self.distillation_model = self._create_distillation_model(self.distillation_model_dir)
        else:
            self.distillation_model = None

        self.classifier_model = self._create_classifier_model(self.classifier_model_dir)

        
            
        super().__init__(config)
        self.model = self.model.to(self.device)

    def _create_model(self) -> torch.nn.Module:
        """Create the classifier model."""
        return create_probability_classifier(self.config['model']).to(self.device)
    
    def _create_distillation_model(self, model_dir: str) -> torch.nn.Module:
        """Create and load pretrained distillation model."""
        self.distillation_config = ConfigManager.load_specific_config(model_dir / 'config.yaml')
        model = create_distillation_model(**self.distillation_config['model'])
       
        checkpoint = torch.load(
            "outputs/distillation"/Path(self.distillation_config['training']['save_dir']) / 'best_model.pth', 
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def _create_classifier_model(self, model_dir: str) -> torch.nn.Module:
        """Create and load pretrained classifier model."""
        self.classifier_config = ConfigManager.load_specific_config(model_dir / 'config.yaml')
        print(self.classifier_config['model'])
        model = create_classifier(self.classifier_config['model'])
        checkpoint = torch.load(model_dir / 'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation data loaders."""
        dataset = ProbabilityDataset(
            data_path=self.config['data']['path'],
            distillation_model=self.distillation_model,
            use_representations=self.config['training'].get('use_representations', True),
            classifier_model=self.classifier_model
        )
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # import numpy as np
        # for _ in range(10):
        #     random_idx = np.random.randint(len(train_dataset))
        #     print(train_dataset[random_idx][0][-3:], train_dataset[random_idx][1])


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
            # print(features[:, -3:], labels)
            # print(torch.argmax(features[:, -3:], dim=1) == torch.argmax(labels, dim=1))
            # exit()
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
        loss = self.criterion(predictions, labels)
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
            loss = self.criterion(predictions, labels)
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
        return torch.nn.CrossEntropyLoss()