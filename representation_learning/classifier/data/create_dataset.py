import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List
import glob
from tqdm import tqdm
from experiments.systems.pendulum import Pendulum
import random
import pickle

def sample_positive_pairs(
    trajectory: np.ndarray, 
    first_attractor_idx: int,

) -> Tuple[np.ndarray, np.ndarray]:
    """Sample pairs of points from the same trajectory (positive examples)."""

    idx1, idx2 = tuple(sorted(np.random.randint(0, first_attractor_idx, 2).tolist()))
    return trajectory[idx1], trajectory[idx2]

def sample_negative_pairs(
    all_trajectories: List[np.ndarray],
    current_traj_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample pairs of points from different trajectories (negative examples)."""
    # Pick a different trajectory
    other_traj_idx = current_traj_idx
    while other_traj_idx == current_traj_idx:
        other_traj_idx = np.random.randint(0, len(all_trajectories))
    
    # Sample random points from each trajectory
    source_traj = all_trajectories[current_traj_idx]
    target_traj = all_trajectories[other_traj_idx]
    
    source_idx = np.random.randint(0, len(source_traj))
    target_idx = np.random.randint(0, len(target_traj))
    
    return source_traj[source_idx], target_traj[target_idx]

def create_reachability_dataset(
    trajectory_path: Path,
    output_path: Path,
    samples_per_trajectory: int = 10,
    attractor_data: bool = True
) -> None:
    """
    Create a dataset for reachability classification.
    
    Args:
        trajectory_path: Path to trajectory data (HDF5 format)
        output_path: Path to save the classification dataset
        samples_per_trajectory: Number of samples to generate per trajectory
        min_separation: Minimum separation between points in positive pairs
        max_separation: Maximum separation between points in positive pairs
    """
    # Load trajectories
    print(f"Loading trajectories from {trajectory_path}")
    trajectories = glob.glob(str(trajectory_path / '*.txt'))
    trajectories = [np.loadtxt(traj_path, delimiter=',') for traj_path in trajectories]
    
    # Initialize lists for dataset
    sources = []
    targets = []
    labels = []


    complete_trajectories = []
    incomplete_trajectories = []
    all_trajectories = []

    system = Pendulum()
    rad = 0.05
    
    # Generate positive and negative samples
    for traj_idx, trajectory in enumerate(tqdm(trajectories, desc="Processing trajectories")):
        attractor_class, attractor = system.which_attracting_region(trajectory[-1, :], rad=rad)
        all_trajectories.append(trajectory)
        if attractor_class == -1:
            first_attractor_idx = trajectory.shape[0]
            incomplete_trajectories.append(trajectory)
        else:
            # first point that enters the attractor's neighborhood
            first_attractor_idx = np.where(np.linalg.norm(trajectory - attractor, axis=1) < rad)[0][0] + 1
            complete_trajectories.append((trajectory[:first_attractor_idx], attractor_class))
        
        new_samples_per_trajectory = samples_per_trajectory
        first_point = trajectory[0]
        if abs(first_point[1]) > np.pi:
            new_samples_per_trajectory *= 2
        for _ in range(new_samples_per_trajectory):
            source, target = sample_positive_pairs(
                trajectory, 
                first_attractor_idx
            )
            sources.append(source)
            targets.append(target)
            labels.append(1)  # Positive label
            
            # Negative samples (from different trajectories)
            # source, target = sample_negative_pairs(trajectories, traj_idx)
            # sources.append(source)
            # targets.append(target)
            # labels.append(0)  # Negative label
    

    # positive samples that have attractor information. 
    if attractor_data:
        for traj, _ in tqdm(complete_trajectories, desc="Processing complete trajectories"):
            first_point = traj[0]
            new_samples_per_trajectory = samples_per_trajectory
            if abs(first_point[1]) > np.pi:
                new_samples_per_trajectory *= 2 
            for _ in range(new_samples_per_trajectory):
                if len(traj) > 2:
                    idx1 = np.random.randint(0, len(traj) - 1)
                    sources.append(traj[idx1])
                    targets.append(traj[-1])
                    labels.append(1)
        
    full_dataset = np.concatenate(all_trajectories.copy(), axis=0).tolist()
    random.shuffle(full_dataset)

    # sample negative pairs from the full dataset
    for _ in range(len(sources)):
        idx1, idx2 = np.random.randint(0, len(full_dataset), 2)
        sources.append(full_dataset[idx1])
        targets.append(full_dataset[idx2])
        labels.append(0)  # Negative label

    sources = np.array(sources)
    targets = np.array(targets)
    labels = np.array(labels)
    
    
    # Save dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        sources=sources,
        targets=targets,
        labels=labels
    )

    print(f"Dataset created with {len(labels)} samples")
    print(f"Positive samples: {np.sum(labels == 1)}")
    print(f"Negative samples: {np.sum(labels == 0)}")

    sources = []
    labels = []
    for traj, attractor_class in tqdm(complete_trajectories, desc="Processing complete trajectories"):
        one_hot = np.zeros((1, len(system.attractors())))
        one_hot[0, attractor_class] = 1
        
        for _ in range(samples_per_trajectory):
            idx = np.random.randint(0, len(traj))
            sources.append(traj[idx])
            labels.append(one_hot)
    
    sources = np.array(sources)
    labels = np.array(labels)
    # Save complete trajectories
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path.parent / 'probability_dataset.npz',
        sources=sources,
        labels=labels
    )

    print(f"Probability dataset created with {len(labels)} samples")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing trajectory data")
    parser.add_argument("--output_file", type=Path, required=True, help="Full path to the output file, for example: outputs/classifier/reachability_dataset.npz")
    parser.add_argument("--samples_per_trajectory", type=int, default=10, help="Number of samples to generate per trajectory")
    parser.add_argument("--attractor_data", type=bool, default=True, help="Whether to use attractor data")
    
    args = parser.parse_args()
    
    create_reachability_dataset(
        trajectory_path=args.data_dir,
        output_path=args.output_file,
        samples_per_trajectory=args.samples_per_trajectory,
        attractor_data=args.attractor_data
    )