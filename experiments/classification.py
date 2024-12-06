import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import os
from systems.pendulum import Pendulum
from matplotlib import pyplot as plt
from tqdm import tqdm
from representation_learning.transformer import create_transformer  # Import the transformer model
from representation_learning.distillation.distillation import DistillationModel  # Import the distillation model
torch.manual_seed(42)
np.random.seed(42)


def create_batch_around_attractor(attractor, radius, num_samples):
    samples = np.random.normal(0, 0.1, (num_samples, 2))
    samples = samples / np.linalg.norm(samples, axis=1).reshape(-1, 1)
    samples = samples * radius
    samples = samples + attractor
    return samples

def process_data(data):
    d = np.zeros((data.shape[0], 5))

    d[:, 0] = data[:, 0]/np.pi
    d[:, 1] = data[:, 1]/(2*np.pi)
    d[:, 2] = data[:, 2]/np.pi
    d[:, 3] = data[:, 3]/(2*np.pi)
    d[:, 4] = data[:, 4]
    return d

class RepresentationDataset(Dataset):
    def __init__(self, data, distillation_model, use_distillation=True):
        self.data = process_data(data)
        self.distillation_model = distillation_model
        self.use_distillation = use_distillation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source = torch.tensor(self.data[idx, :2]).float()
        target = torch.tensor(self.data[idx, 2:4]).float()
        
        if self.use_distillation:
            with torch.no_grad():
                source_representation = self.distillation_model(source.unsqueeze(0)).squeeze(0)
                target_representation = self.distillation_model(target.unsqueeze(0)).squeeze(0)
            representation = torch.cat([source_representation, target_representation], dim=0)
        else:
            representation = torch.cat([source, target], dim=0)
        
        label = torch.tensor(self.data[idx, 4]).float()
        return representation, label

class ClassifierNet(nn.Module):
    def __init__(self, input_dim):
        super(ClassifierNet, self).__init__()
        width = 512
        self.fc1 = nn.Linear(input_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, width)
        self.fc4 = nn.Linear(width, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        # x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        # x = self.dropout(x)
        x = self.fc4(x)
        return x

def train_classifier(net, criterion, optimizer, train_loader, val_loader, device='cpu', epochs=10):
    print("Starting classifier training...")
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.2f}%")
    
    print("Classifier training completed.")
    return net

def main(args, kwargs):
    print(f"Loading data from {kwargs['save_dir']}/dataset_{kwargs['dataset_size']}.csv")
    data = np.loadtxt(f"{kwargs['save_dir']}/dataset_{kwargs['dataset_size']}.csv", delimiter=',')

    np.random.shuffle(data)
    train_data = data[:int(0.8*len(data))]
    val_data = data[int(0.8*len(data)):]
    print(f"Data loaded. Total samples: {len(data)}, Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    distillation_model = None
    if kwargs['use_distillation']:
        print(f"Loading distillation model from {kwargs['distillation_model_path']}")
        distillation_model = DistillationModel(input_dim=2, output_dim=kwargs['d_model'])
        distillation_model.load_state_dict(torch.load(kwargs['distillation_model_path']))
        distillation_model.eval()
        print("Distillation model loaded successfully.")

    print("Preparing datasets and dataloaders...")
    train_dataset = RepresentationDataset(train_data, distillation_model, use_distillation=kwargs['use_distillation'])
    val_dataset = RepresentationDataset(val_data, distillation_model, use_distillation=kwargs['use_distillation'])

    train_loader = DataLoader(train_dataset, batch_size=kwargs['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=kwargs['batch_size'], shuffle=False)
    print(f"Dataloaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing classifier network...")
    input_dim = kwargs['d_model']*2 if kwargs['use_distillation'] else 4
    net = ClassifierNet(input_dim=input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    net = train_classifier(net, criterion, optimizer, train_loader, val_loader, device=device, epochs=kwargs['epochs'])

    model_type = "representation" if kwargs['use_distillation'] else "raw"
    save_path = f"{kwargs['save_dir']}/pendulum_{kwargs['dataset_size']}_{model_type}_classifier.pth"
    print(f"Saving trained classifier to {save_path}")
    torch.save(net.state_dict(), save_path)
    print("Classifier saved successfully.")

def plot_clf_results(kwargs, combine_attractors_2_3=True):
    print("Starting classification results plotting...")
    
    model_type = "representation" if kwargs['use_distillation'] else "raw"
    prob_data_path = f"{kwargs['save_dir']}/prob_data_mean_{kwargs['dataset_size']}_{model_type}.pkl"
    
    if True or not os.path.exists(prob_data_path):
        # discretize the state space
        x = np.linspace(-3.14, 3.14, 200) 
        y = np.linspace(-6.28, 6.28, 400)

        system = Pendulum()
        attractors = system.attractors()
        clf = ClassifierNet(input_dim=kwargs['d_model']*2 if kwargs['use_distillation'] else 4)
        clf = clf.to('cuda')
        clf.load_state_dict(torch.load(f"{kwargs['save_dir']}/pendulum_{kwargs['dataset_size']}_{model_type}_classifier.pth"))
        clf.eval()

        distillation_model = None
        if kwargs['use_distillation']:
            distillation_model = DistillationModel(input_dim=2, output_dim=kwargs['d_model']).to('cuda')
            distillation_model.load_state_dict(torch.load(kwargs['distillation_model_path']))
            distillation_model.eval()

        # Initialize probability matrices for each attractor
        prob_data = np.zeros((len(x), len(y), 3))  # 3 for three attractors

        with torch.no_grad():
            for i, theta in enumerate(tqdm(x, desc="Processing x-axis")):
                for j, theta_dot in enumerate(tqdm(y, desc=f"Processing y-axis (x={theta:.2f})", leave=False)):
                    probs = []
                    for att in attractors:
                        samples = create_batch_around_attractor(att, 0.05, 32)
                        batch = np.zeros((samples.shape[0]+1, 5))
                        batch[:, 0] = theta
                        batch[:, 1] = theta_dot
                        batch[:-1, 2:4] = samples
                        batch[-1, 2:4] = att

                        state = process_data(batch)[:, :4]
                        state = torch.tensor(state).float().to('cuda')
                        
                        if kwargs['use_distillation']:
                            source_repr = distillation_model(state[:, :2])
                            target_repr = distillation_model(state[:, 2:])
                            combined_repr = torch.cat([source_repr, target_repr], dim=1)
                        else:
                            combined_repr = state
                        
                        # Get raw probabilities instead of binary classification
                        probs.append(torch.sigmoid(clf(combined_repr)).max().cpu().numpy())

                    prob_data[i, j] = probs

        print(f"Saving probability data to {prob_data_path}")
        with open(prob_data_path, "wb") as f:
            pickle.dump(prob_data, f)
    
    print(f"Loading probability data from {prob_data_path}")
    with open(prob_data_path, "rb") as f:
        prob_data = pickle.load(f)

    # Create three separate heatmaps
    fig, axes = plt.subplots(1, 3 if not combine_attractors_2_3 else 2, figsize=(18 if not combine_attractors_2_3 else 12, 6))
    titles = ['Attractor 1', 'Attractors 2&3' if combine_attractors_2_3 else 'Attractor 2', 'Attractor 3']

    print(prob_data.shape)

    mean_prob_data = np.mean(prob_data, axis=2)
    print(mean_prob_data.shape)

    # plot the mean probability data
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mean_prob_data.T, extent=[-3.14, 3.14, -6.28, 6.28], origin='lower', aspect='auto')
    ax.set_xlabel('θ')
    ax.set_ylabel('ω')
    ax.set_title('Mean Probability Heatmap')
    plt.savefig(f"clf_prob_mean_heatmap_non_batched_{kwargs['dataset_size']}_{model_type}.png")
    plt.close()
    
    for idx in range(3 if not combine_attractors_2_3 else 2):
        plot_data = prob_data[:, :, idx]
        if combine_attractors_2_3 and idx == 1:
            # Combine probabilities for attractors 2 and 3
            plot_data = np.maximum(prob_data[:, :, 1], prob_data[:, :, 2])
        
        im = axes[idx].imshow(
            plot_data.T,
            extent=[-3.14, 3.14, -6.28, 6.28],
            origin='lower',
            aspect='auto',
            cmap='hot'
        )
        axes[idx].set_title(titles[idx])
        axes[idx].set_xlabel('θ')
        axes[idx].set_ylabel('ω')
        plt.colorbar(im, ax=axes[idx])

    plt.tight_layout()
    model_type = "representation" if kwargs['use_distillation'] else "raw"
    combined_suffix = "_combined23" if combine_attractors_2_3 else ""
    plt.savefig(f"clf_mean_heatmaps_{kwargs['dataset_size']}_{model_type}{combined_suffix}.png")
    plt.close()

    print(f"Heatmaps saved as clf_heatmaps_{kwargs['dataset_size']}_{model_type}{combined_suffix}.png")

    # Fix the RGB heatmap plotting
    if combine_attractors_2_3:
        # For combined case, use red for attractor 1 and green for attractors 2&3
        rgb_data = np.zeros((prob_data.shape[0], prob_data.shape[1], 3))
        rgb_data[:, :, 0] = prob_data[:, :, 0].T  # Red channel for attractor 1
        rgb_data[:, :, 1] = np.maximum(prob_data[:, :, 1].T, prob_data[:, :, 2].T) # Green channel for combined attractors 2&3
    else:
        rgb_data = np.stack([prob_data[:,:,i].T for i in range(3)], axis=-1)

    
    
    # prob values for all above 0.75 in each layer, 0 otherwise
    rgb_data[rgb_data < 0.8] = 0
    
    # # Normalize the RGB values to be between 0 and 1
    # rgb_data = (rgb_data - rgb_data.min()) / (rgb_data.max() - rgb_data.min())
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb_data, extent=[-3.14, 3.14, -6.28, 6.28], origin='lower', aspect='auto')
    ax.set_xlabel('θ')
    ax.set_ylabel('ω')
    ax.set_title('Combined RGB Heatmap')
    plt.savefig(f"clf_mean_heatmap_rgb_{kwargs['dataset_size']}_{model_type}{combined_suffix}.png")
    plt.close()


def plot_clf_results_non_batched(kwargs):
    print("Starting non-batched classification results plotting...")
    
    model_type = "representation" if kwargs['use_distillation'] else "raw"
    if not os.path.exists(f"{kwargs['save_dir']}/prob_data_{kwargs['dataset_size']}_{model_type}.pkl"):
        # Discretize the state space
        x = np.linspace(-3.14, 3.14, 200)
        y = np.linspace(-6.28, 6.28, 400)

        system = Pendulum()
        attractors = system.attractors()
        clf = ClassifierNet(input_dim=kwargs['d_model']*2 if kwargs['use_distillation'] else 4)
        clf = clf.to('cuda')
        clf.load_state_dict(torch.load(f"{kwargs['save_dir']}/pendulum_{kwargs['dataset_size']}_{model_type}_classifier.pth"))
        clf.eval()

        distillation_model = None
        if kwargs['use_distillation']:
            distillation_model = DistillationModel(input_dim=2, output_dim=kwargs['d_model']).to('cuda')
            distillation_model.load_state_dict(torch.load(kwargs['distillation_model_path']))
            distillation_model.eval()

        # Initialize probability matrices for each attractor
        prob_data = np.zeros((len(x), len(y), 3))  # 3 for three attractors

        with torch.no_grad():
            for i, theta in enumerate(tqdm(x, desc="Processing x-axis")):
                for j, theta_dot in enumerate(tqdm(y, desc=f"Processing y-axis (x={theta:.2f})", leave=False)):
                    # Create input state
                    state = np.array([[theta, theta_dot]])
                    
                    # Process state for each attractor
                    for k, att in enumerate(attractors):
                        # Combine current state with attractor
                        combined_state = np.zeros((1, 5))
                        combined_state[0, :2] = state
                        combined_state[0, 2:4] = att
                        combined_state[0, 4] = 1
                        # Process the data
                        processed_state = process_data(combined_state)[:, :4]
                        processed_state = torch.tensor(processed_state).float().to('cuda')
                        
                        if kwargs['use_distillation']:
                            source_repr = distillation_model(processed_state[:, :2])
                            target_repr = distillation_model(processed_state[:, 2:])
                            combined_repr = torch.cat([source_repr, target_repr], dim=1)
                        else:
                            combined_repr = processed_state
                        
                        # Get probability for this attractor
                        prob_data[i, j, k] = torch.sigmoid(clf(combined_repr)).cpu().numpy()

        print(f"{kwargs['save_dir']}/prob_data_{kwargs['dataset_size']}_{model_type}.pkl")
        with open(f"{kwargs['save_dir']}/prob_data_{kwargs['dataset_size']}_{model_type}.pkl", "wb") as f:
            pickle.dump(prob_data, f)

    print(f"Loading probability data from {kwargs['save_dir']}/prob_data_{kwargs['dataset_size']}_{model_type}.pkl")

    with open(f"{kwargs['save_dir']}/prob_data_{kwargs['dataset_size']}_{model_type}.pkl", "rb") as f:
        prob_data = pickle.load(f)

    print(prob_data.shape)

    mean_prob_data = np.mean(prob_data, axis=2)
    print(mean_prob_data.shape)

    # plot the mean probability data
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mean_prob_data.T, extent=[-3.14, 3.14, -6.28, 6.28], origin='lower', aspect='auto')
    ax.set_xlabel('θ')
    ax.set_ylabel('ω')
    ax.set_title('Mean Probability Heatmap')
    plt.savefig(f"clf_prob_mean_heatmap_non_batched_{kwargs['dataset_size']}_{model_type}.png")
    plt.close()


    
    # Create three separate heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ['Attractor 1', 'Attractor 2', 'Attractor 3']

    for idx in range(3):
        im = axes[idx].imshow(
            prob_data[:, :, idx].T,
            extent=[-3.14, 3.14, -6.28, 6.28],
            origin='lower',
            aspect='auto',
            cmap='hot'
        )
        axes[idx].set_title(titles[idx])
        axes[idx].set_xlabel('θ')
        axes[idx].set_ylabel('ω')
        plt.colorbar(im, ax=axes[idx])

    plt.tight_layout()
    model_type = "representation" if kwargs['use_distillation'] else "raw"
    plt.savefig(f"clf_heatmaps_non_batched_{kwargs['dataset_size']}_{model_type}.png")
    plt.close()

    print(f"Non-batched heatmaps saved as clf_heatmaps_non_batched_{kwargs['dataset_size']}_{model_type}.png")

if __name__ == "__main__":
    train = False
    use_distillation = True  # Toggle this to switch between distillation and raw input
    kwargs = dict(
        dataset_size='5k',
        batch_size=512,
        epochs=10,
        cwd="/media/dhruv/a7519aee-b272-44ae-a117-1f1ea1796db6/2024/arcmg",
        d_model=64,
        distillation_model_path="representation_learning/distillation/distillation_model.pth",
        use_distillation=use_distillation  # Add this to kwargs
    )
    kwargs['save_dir'] = f"{kwargs['cwd']}/data/pendulum/clf_{kwargs['dataset_size']}"
    
    if train:
        print("Starting training process...")
        main([], kwargs)
        print("Training completed.")
    
    print("Starting plotting process...")
    plot_clf_results(kwargs, combine_attractors_2_3=False)  # Set to True to combine attractors 2 and 3
    # plot_clf_results_non_batched(kwargs)
    print("Plotting completed.")
