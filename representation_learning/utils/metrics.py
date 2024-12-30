from typing import Dict, List, Any
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsTracker:

    @property
    def metrics(self) -> Dict[str, List[float]]:
        return self._metrics    
    
    # operator overloading
    def __getitem__(self, key: str) -> List[float]:
        return self._metrics[key]
    
    def __setitem__(self, key: str, value: List[float]) -> None:
        if key not in self._metrics:
            self._metrics[key] = []
    
    def __append__(self, key: str, value: float) -> None:
        if key not in self._metrics:
            self._metrics[key] = []
        self._metrics[key].append(value)
    
    def __extend__(self, key: str, values: List[float]) -> None:
        if key not in self._metrics:
            self._metrics[key] = []
        self._metrics[key].extend(values)
    
    def __init__(self):
        self._metrics: Dict[str, List[float]] = {}
        
    def update(self, metric_name: str, value: float):
        if metric_name not in self._metrics:
            self._metrics[metric_name] = []
        self._metrics[metric_name].append(value)
        
    def get_best(self, metric_name: str, mode: str = 'min') -> float:
        values = self._metrics[metric_name]
        return min(values) if mode == 'min' else max(values)
        
    def get_latest(self, metric_name: str) -> float:
        """Get the most recent value for a metric."""
        if metric_name in self._metrics and len(self._metrics[metric_name]) > 0:
            return self._metrics[metric_name][-1]
        return float('inf')
    
    def plot_losses(self, save_path: Path) -> None:
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 6))
        if self._metrics['train_losses']:
            plt.plot(self._metrics['train_losses'], label='Train Loss')
        if self._metrics['val_losses']:
            plt.plot(self._metrics['val_losses'], label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig(save_path)
        plt.close()

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for saving."""
        return self._metrics
    
    def load_dict(self, metrics_dict: Dict[str, Any]) -> None:
        """Load metrics from dictionary."""
        self._metrics = metrics_dict