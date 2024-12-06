import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import DataLoader
import logging

def plot_trajectory_comparison(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    save_path: Path,
    num_samples: int = 10,
    title: str = "Trajectory Comparison"
) -> None:
    """
    Plot input and output trajectories side by side.
    
    Args:
        inputs: Input trajectories tensor [N, seq_len, 2]
        outputs: Output trajectories tensor [N, seq_len, 2]
        save_path: Path to save the plot
        num_samples: Number of trajectories to plot
        title: Title for the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot input trajectories
    for i in range(min(num_samples, inputs.size(0))):
        ax1.plot(inputs[i, :, 0], inputs[i, :, 1], 
                 label=f'Input {i+1}' if i == 0 else None)
    ax1.set_title('Input Trajectories')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_xlim(-1.0, 1.0)
    ax1.set_ylim(-1.0, 1.0)
    if num_samples > 1:
        ax1.legend()
    
    # Plot output trajectories
    for i in range(min(num_samples, outputs.size(0))):
        ax2.plot(outputs[i, :, 0], outputs[i, :, 1], 
                 label=f'Output {i+1}' if i == 0 else None)
    ax2.set_title('Model Output Trajectories')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_xlim(-1.0, 1.0)
    ax2.set_ylim(-1.0, 1.0)
    if num_samples > 1:
        ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def evaluate_full_trajectories(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_samples: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate model on full trajectories from validation set.
    
    Args:
        model: The trained model
        val_loader: Validation data loader
        device: Device to run evaluation on
        num_samples: Number of samples to evaluate
        
    Returns:
        Tuple of (inputs, outputs) tensors
    """
    model.eval()
    batch_inputs = []
    batch_outputs = []
    
    with torch.no_grad():
        for full_traj, _, _, _ in val_loader:
            # Use full trajectory as input (no masking)
            full_traj = full_traj.to(device)
            output = model(full_traj)
            
            # Store results
            batch_inputs.append(full_traj.cpu())
            batch_outputs.append(output.cpu())
            
            if len(batch_inputs) * full_traj.size(0) >= num_samples:
                break
    
    # Concatenate all batches
    all_inputs = torch.cat(batch_inputs, dim=0)
    all_outputs = torch.cat(batch_outputs, dim=0)
    
    return all_inputs, all_outputs