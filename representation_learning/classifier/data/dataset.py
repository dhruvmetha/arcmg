from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from experiments.systems.pendulum import Pendulum


from representation_learning.models.distillation import create_distillation_model

class ReachabilityDataset(Dataset):
    def __init__(
        self, 
        data_path: Path,
        distillation_model: Optional[torch.nn.Module] = None,
        use_representations: bool = True,
        size: Optional[int] = None
    ):
        """
        Dataset for reachability classification.
        
        Args:
            data_path: Path to trajectory data
            distillation_model: Pre-trained distillation model for getting representations
            use_representations: Whether to use learned representations or raw coordinates
        """
        self.data = self._load_and_process_data(data_path)
        self.distillation_model = distillation_model
        self.use_representations = use_representations
        self.size = size
    def _load_and_process_data(self, data_path: Path) -> np.ndarray:
        """Load and normalize data."""
        # load csv file
        print(data_path)
        data = np.load(data_path)
        sources = data['sources']
        targets = data['targets']
        labels = data['labels']

        processed = np.zeros((sources.shape[0], 5))
        processed[:, 0] = sources[:, 0] / np.pi
        processed[:, 1] = sources[:, 1] / (2 * np.pi)
        processed[:, 2] = targets[:, 0] / np.pi
        processed[:, 3] = targets[:, 1] / (2 * np.pi)
        processed[:, 4] = labels
        return processed
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get source and target states
        source = torch.tensor(self.data[idx, :2]).float()
        target = torch.tensor(self.data[idx, 2:4]).float()
        
        if self.use_representations and self.distillation_model is not None:
            with torch.no_grad():
                source_repr = self.distillation_model(source.unsqueeze(0)).squeeze(0)
                target_repr = self.distillation_model(target.unsqueeze(0)).squeeze(0)
            features = torch.cat([source_repr, target_repr], dim=0)
        else:
            features = torch.cat([source, target], dim=0)
        
        label = torch.tensor(self.data[idx, 4]).float()
        return features, label


class ProbabilityDataset(Dataset):
    def __init__(self, data_path: Path, distillation_model: Optional[torch.nn.Module] = None, use_representations: bool = True, classifier_model: Optional[torch.nn.Module] = None):

        system = Pendulum()
        self.attractors = system.attractors()

        self.data, self.labels = self._load_and_process_data(data_path)
        self.distillation_model = distillation_model
        self.use_representations = use_representations
        self.classifier_model = classifier_model


    def _load_and_process_data(self, data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load and normalize data."""
        data = np.load(data_path)
        sources = data['sources']
        labels = data['labels']

        processed = np.zeros((sources.shape[0], 2))
        processed[:, 0] = sources[:, 0] / np.pi
        processed[:, 1] = sources[:, 1] / (2 * np.pi)
        return processed, labels
    

    def create_batch_around_attractor(self, attractor, radius, num_samples):
        samples = np.random.normal(0, 0.1, (num_samples, 2))
        samples = samples / np.linalg.norm(samples, axis=1).reshape(-1, 1)
        samples = samples * radius
        samples = samples + attractor
        return samples

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        source = torch.tensor(self.data[idx, :2]).float().unsqueeze(0)
        raw_source = source.clone()
        labels_inp = torch.zeros((1, len(self.attractors)))
        labels_out = torch.tensor(self.labels[idx]).float()

        if self.use_representations and self.distillation_model is not None:
            with torch.no_grad():
                source_repr = self.distillation_model(source.unsqueeze(0)).squeeze(0)
            source = source_repr.clone()
        source_repr = source.repeat(64, 1)
        source_input = source.clone()

        for i, attractor in enumerate(self.attractors):
            # sample 64 points around the attractor at a radius of 0.05
            points = self.create_batch_around_attractor(attractor, 0.05, 64)
            points[-1, :] = attractor
            points[:, 0] /= np.pi
            points[:, 1] /= (2*np.pi)
            target_repr = self.distillation_model(torch.tensor(points, dtype=torch.float32))
            target = target_repr.clone()
            inp = torch.cat([source_repr, target], dim=1)
            with torch.no_grad():
                output = torch.max(torch.sigmoid(self.classifier_model(inp)))
                labels_inp[0, i] = output

        return raw_source.squeeze(0), torch.cat([source_input, labels_inp], dim=1).squeeze(0), labels_out.squeeze(0)

if __name__ == "__main__":
    import numpy as np
    
    # dataset = ReachabilityDataset(Path('/media/dhruv/a7519aee-b272-44ae-a117-1f1ea1796db6/2024/arcmg/data/pendulum_clf_100/5k/dataset_5k.csv'))
    dataset = ReachabilityDataset(Path('/media/dhruv/a7519aee-b272-44ae-a117-1f1ea1796db6/2024/arcmg/data/pendulum_lqr/reachability_data_5k.npz'))

    idx = np.random.randint(0, len(dataset))
    print(dataset[idx])