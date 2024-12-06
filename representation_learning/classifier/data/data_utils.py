import numpy as np
from typing import Tuple

def create_batch_around_attractor(
    attractor: np.ndarray,
    radius: float,
    num_samples: int
) -> np.ndarray:
    """Create a batch of points around an attractor."""
    samples = np.random.normal(0, 0.1, (num_samples, 2))
    samples = samples / np.linalg.norm(samples, axis=1).reshape(-1, 1)
    samples = samples * radius
    samples = samples + attractor
    return samples

def sample_trajectory_pairs(
    trajectory: np.ndarray,
    num_pairs: int,
    min_separation: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample pairs of points from a trajectory that are reachable."""
    trajectory_length = len(trajectory)
    source_indices = np.random.randint(0, trajectory_length - min_separation, num_pairs)
    target_indices = source_indices + np.random.randint(min_separation, min_separation * 2, num_pairs)
    target_indices = np.clip(target_indices, 0, trajectory_length - 1)
    
    sources = trajectory[source_indices]
    targets = trajectory[target_indices]
    
    return sources, targets 