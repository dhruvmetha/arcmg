import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
import argparse
import yaml

class DistillationModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DistillationRunner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.load_data()
        self.setup_model()
        self.setup_training()

    def load_data(self):
        print(f"Loading data from {self.config['data']['path']}")
        data = np.load(self.config['data']['path'])
        random_indices = np.random.choice(len(data['data_points']), 100000, replace=False)
        data_points = data['data_points'][random_indices]
        representations = data['representations'][random_indices]
        # use only 100000 samples
        data_points = torch.FloatTensor(data_points)
        representations = torch.FloatTensor(representations)
        
        dataset = TensorDataset(data_points, representations)
        
        # Split the dataset into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.config['training']['batch_size'], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config['training']['batch_size'])
        
        print(f"Data loaded. Total samples: {len(dataset)}, Train samples: {train_size}, Validation samples: {val_size}")

    def setup_model(self):
        input_dim = self.train_loader.dataset[0][0].shape[0]
        output_dim = self.train_loader.dataset[0][1].shape[0]
        self.model = DistillationModel(input_dim=input_dim, output_dim=output_dim).to(self.device)
        print(f"Model created with input dim {input_dim} and output dim {output_dim}")

    def setup_training(self):
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['training']['learning_rate'])

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch_data, batch_repr in self.train_loader:
            batch_data, batch_repr = batch_data.to(self.device), batch_repr.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(batch_data)
            loss = self.criterion(output, batch_repr)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_data, batch_repr in self.val_loader:
                batch_data, batch_repr = batch_data.to(self.device), batch_repr.to(self.device)
                output = self.model(batch_data)
                loss = self.criterion(output, batch_repr)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def train(self):
        print("Starting training")
        for epoch in range(self.config['training']['num_epochs']):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            print(f"Epoch {epoch+1}/{self.config['training']['num_epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        print("Training completed")
        self.save_model()

    def save_model(self):
        save_path = self.config['model']['save_path']
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    parser = argparse.ArgumentParser(description="Distillation Runner")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    config = load_config(args.config)
    runner = DistillationRunner(config)
    runner.train()

if __name__ == "__main__":
    main()