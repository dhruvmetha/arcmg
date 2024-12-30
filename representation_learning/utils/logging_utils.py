from functools import wraps
from typing import Callable, Any
from tqdm import tqdm
import logging
import time

class TrainerLoggingMixin:
    """Mixin class to add logging capabilities to trainers."""
    
    def log_info(self, message: str) -> None:
        """Log info message if logger exists."""
        if hasattr(self, 'logger'):
            self.logger.info(message)

    def create_progress_bar(self, iterable, desc: str, leave: bool = True) -> tqdm:
        """Create a progress bar with consistent styling."""
        return tqdm(
            iterable,
            desc=desc,
            leave=leave,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

def log_epoch(desc: str = ""):
    """Decorator for logging epoch-level functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            start_time = time.time()
            # self.log_info(f"Starting {desc}")
            result = func(self, *args, **kwargs)
            duration = time.time() - start_time
            # self.log_info(f"Finished {desc} in {duration:.2f}s - Loss: {result:.4f}")
            return result
        return wrapper
    return decorator

def log_step(interval: int = 100):
    """Decorator for logging training steps."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, batch, batch_idx: int, *args, **kwargs) -> Any:
            loss = func(self, batch, batch_idx, *args, **kwargs)
            if batch_idx % interval == 0:
                self.log_info(
                    f"Step [{batch_idx}/{self.num_batches}] Loss: {loss:.4f}"
                )
            return loss
        return wrapper
    return decorator

def register_validation(validation_type: str):
    """Decorator to register a validation function."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self._validation_functions[validation_type] = func
            return result
        return wrapper
    return decorator 