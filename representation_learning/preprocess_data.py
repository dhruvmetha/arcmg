import argparse
import yaml
from pathlib import Path
from glob import glob
from datetime import datetime
import random
from typing import Optional, Type
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from representation_learning.utils.config import ConfigManager


def parse_args():
    parser = argparse.ArgumentParser(description='Representation Learning Training')
    parser.add_argument('--config', type=Path, required=True, help='Path to config file')
    parser.add_argument('--system', type=str, help='System name')
    return parser.parse_args()

def preprocess_data(config: dict):
    
    labels = str(Path(config['data_dir']).parent / "labels.txt")
    with open(labels, 'r') as f:
        data = f.readlines()
    success_data = []
    failure_data = []   
    for line in data:

        l = line.strip().split(',')
        if int(l[1]) == 1:
            success_data.append(l)
        else:
            failure_data.append(l)

    # split train_test and create two txt files, one for train and one for test. Test should be 10% of the data
    # train should be 90% of the data
    # save the files in the data_dir
    # Get indices for train/test split
    success_indices = list(range(len(success_data)))
    failure_indices = list(range(len(failure_data)))
    
    # Sample train indices
    train_success_indices = random.sample(success_indices, int(0.9 * len(success_data)))
    train_failure_indices = random.sample(failure_indices, int(0.9 * len(failure_data)))
    
    # Get remaining indices for test set
    test_success_indices = list(set(success_indices) - set(train_success_indices))
    test_failure_indices = list(set(failure_indices) - set(train_failure_indices))
    
    # Create train and test sets using the indices
    train_data = [success_data[i] for i in train_success_indices] + [failure_data[i] for i in train_failure_indices]
    test_data = [success_data[i] for i in test_success_indices] + [failure_data[i] for i in test_failure_indices]
    with open(Path(config['data_dir']).parent / 'train.txt', 'w') as f:
        for l in train_data:
            f.write(','.join(l) + '\n')
    with open(Path(config['data_dir']).parent / 'test.txt', 'w') as f:
        for l in test_data:
            f.write(','.join(l) + '\n') 

    return success_data, failure_data

def get_num_clusters(data, min_clusters=2, max_clusters=10):
    silhouette_scores = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(data)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    print(silhouette_scores)
    return np.argmax(silhouette_scores) + min_clusters

def find_attractor(data_dir: str, data: list, name: str):
    _dir = Path(data_dir)
    new_data = []
    file_names = []
    for file in data: 
        try:
            file_names.append(file[0])
            new_data.append(np.loadtxt(_dir / file[0], delimiter=',')[-1])
        except:
            pass
    new_data = np.array(new_data)
    
    num_clusters = get_num_clusters(new_data)
    print("Found number of clusters: ", num_clusters)
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    kmeans.fit(new_data)

    pred_labels = kmeans.predict(new_data)
    

    with open(Path(data_dir).parent / f'{name}_attractor.txt', 'w') as f:
        for i in range(num_clusters):
            f.write(f'{i}, {np.round(np.array(kmeans.cluster_centers_[i]), 2)}\n')

    labels = kmeans.labels_
    print("Cluster centers:")
    for i, center in enumerate(kmeans.cluster_centers_):
        print(f"Cluster {i}", np.bincount(labels)[i])
    

    with open(Path(data_dir).parent / f'{name}_labels.txt', 'w') as f:
        for i in range(len(pred_labels)):
            f.write(f'{file_names[i]}, {pred_labels[i]}\n')

if __name__ == '__main__':
    args = parse_args()
    config = ConfigManager.load_config(args.config)
    success_data, failure_data = preprocess_data(config)
    find_attractor(config['data_dir'], success_data, name='success')
    find_attractor(config['data_dir'], failure_data, name='failure')