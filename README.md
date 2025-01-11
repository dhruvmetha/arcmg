# RoA Estimation using Reachability Classifiers

## Installation
```
git clone https://github.com/dhruvmetha/arcmg.git roa-estimation
cd roa-estimation
pip install -e .
```

## Usage
We use the `representation_learning` module to setup this project.

We need the following files to run the project:
1. A system .py file in `experiments/` that will get registered in `experiments/__init__.py` as a part of the `system_factory`. For example, `pendulum.py` is a system file.
2. Register the system into the `system_factory` in `experiments/__init__.py`
3. A dataset of trajectories for that system: a folder with a list of `.txt` files, each containing a trajectory. In the parent folder of the dataset, there should be a `labels.txt` file that contains the labels for each trajectory (success or failure).
4. A config file for the RoA prediction like `representation_learning/configs/pendulum_config.yaml`

### Latent Representation Learning

#### Transformer Training
We will first train a latent representation for datapoints in the dataset:
```
./scripts/train_transformer.sh SYSTEM_NAME
```

This will train a transformer based on the config file: `representation_learning/configs/SYSTEM_NAME_config.yaml`. Edit the `transformer` section to make sure the training is done correctly.

#### Distillation
 This distills what the transformer has learned from a sequence input to a sequence output to a projection from the original datapoint to it's latent representation without the sequence information. To do this we first need to build a representation dataset from the learned transformer model:
```
./scripts/create_representation_dataset.sh SYSTEM_NAME
```
Use the `distillation` section in the config file to specify the output directory for the representation dataset.

Once the representation dataset is created, we train the distillation model on the dataset.
```
./scripts/train_distillation.sh SYSTEM_NAME
```
Use the `distillation` section in the config file to tune hyperparameters for the distillation model.

### Reachability Classifier Training

To learn a reachability classifier, we first need to create a reachability dataset. This dataset contains datapoints and their corresponding reachability labels. Datapoints are sampled from the dataset of trajectories. We can create this dataset by running the following script:
```
./scripts/create_reachability_dataset.sh SYSTEM_NAME
```

<b>Note</b>: We can choose to not use the representations for the reachability classifier by setting `use_representations: false` in the config file.

Once the reachability dataset is created, we train the reachability classifier on the dataset.
```
./scripts/train_classifier.sh SYSTEM_NAME
```
Use the `reachability_classifier` section in the config file to tune hyperparameters for the reachability classifier.

### TODO
#### Generate ROA estimate
To generate an RoA estimate using the reachability classifier, we first need to determine the attractors. We do this by clustering the end points of trajectories. To do so, we run the following script:

```
./scripts/preprocess_data.sh SYSTEM_NAME
```

This will generate a `success_attractors.txt` and `failure_attractors.txt` file in the parent directory of the dataset. This will also generate `success_labels.txt` and `failure_labels.txt` files in the same directory, where each trajectory is labeled with the attractor it belongs to. 

Each dataset has labels for successful and unsuccessful trajectories.
Generate RoA estimate under the assumption that the system has bistable dynamics. We have to first determine successful and unsuccessful attractors using clustering or other techniques.

#### Evaluate ROA estimate
