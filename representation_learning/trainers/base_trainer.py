from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from ..utils.logger import setup_logger
from ..utils.logging_utils import TrainerLoggingMixin, log_epoch, log_step
import json
from datetime import datetime
from experiments.systems import system_factory


class BaseTrainer(ABC, TrainerLoggingMixin):
    def __init__(self, config: Dict[str, Any], system_name=None):
        self.config = config
        if system_name is None:
            print("Attach a physics system to the trainer")
            exit()
        self.system = system_factory[system_name]()
        self.device = torch.device(config['training']['device'])
        self.logger = setup_logger(self.__class__.__name__)
        
        # Initialize components
        self.model = self._create_model()
        self.criterion = self._create_criterion()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup data
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # Metrics tracking
        self.metrics = {
            'train_losses': [],
            'best_val_loss': float('inf'),
            'epochs_without_improvement': 0
        }

        # Initialize validation functions
        self._validation_functions = {}
        self._register_validation_functions()
        
        # Initialize metrics for all validation types
        for val_type in self._validation_functions.keys():
            self.metrics[f'{val_type}_val_losses'] = []

    def _register_validation_functions(self):
        """Register all validation functions. Override this in subclasses."""
        # Store the bound method directly
        self._validation_functions['default'] = self.validate

    def run_validation(self, validation_type: str = 'default') -> float:
        """Run a specific type of validation."""
        if validation_type not in self._validation_functions:
            raise ValueError(f"Unknown validation type: {validation_type}")
        validation_method = self._validation_functions[validation_type].__get__(self, self.__class__)
        return validation_method()
        
    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save a checkpoint of the model and training state."""
        checkpoint = {
            'epoch': len(self.metrics['train_losses']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, path)
        
        # Save best model if this is the best performance
        if is_best:
            best_path = path.parent / 'best_model.pth'
            torch.save(checkpoint, best_path)
            
        # Save metrics history
        metrics_path = path.parent / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
            
        # self.logger.info(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, path: Path):
        """Load a checkpoint and restore the training state."""
        self.logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.metrics = checkpoint['metrics']
        
    def train(self, checkpoint_dir: Optional[Path] = None):
        """Main training loop with improved logging and checkpointing."""
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
        validation_losses = {}
        for epoch in range(self.config['training']['num_epochs']):
            # Training phase
            train_loss = self.train_epoch()
            self.metrics['train_losses'].append(train_loss)
            
            # Run all validations
            for validation_type in self._validation_functions.keys():
                validation_losses[validation_type] = self.run_validation(validation_type)
                self.metrics[f'{validation_type}_val_losses'].append(validation_losses[validation_type])
            
            # Learning rate scheduling
            val_loss = self.metrics['default_val_losses'][-1]  # Use default validation loss
            self.scheduler.step(val_loss)
            
            # Checkpointing
            is_best = val_loss < self.metrics['best_val_loss']
            if is_best:
                self.metrics['best_val_loss'] = val_loss
                self.metrics['epochs_without_improvement'] = 0
            else:
                self.metrics['epochs_without_improvement'] += 1
                
            # Logging
            log_msg = f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            for val_type, loss in validation_losses.items():
                log_msg += f"{val_type}_val_loss={loss:.4f}, "
            log_msg += f"lr={self.optimizer.param_groups[0]['lr']:.6f}"
            self.logger.info(log_msg)
            
            # Save checkpoint
            if checkpoint_dir:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
                self.save_checkpoint(checkpoint_path, is_best=is_best)
                
            # Early stopping
            if self.metrics['epochs_without_improvement'] >= self.config['training'].get('patience', 10):
                self.logger.info("Early stopping triggered")
                break
        
    @abstractmethod
    def _create_model(self) -> torch.nn.Module:
        pass
        
    def _create_criterion(self) -> torch.nn.Module:
        return torch.nn.MSELoss()
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        if len(list(self.model.parameters())) == 0:
            return None
        return torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['training']['learning_rate']
        )
        
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        if self.optimizer is None:
            return None
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
    @abstractmethod
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        pass
        
    @log_epoch("Training Epoch")
    def train_epoch(self) -> float:
        pass

    @log_epoch("Validation")
    def validate(self) -> float:
        pass