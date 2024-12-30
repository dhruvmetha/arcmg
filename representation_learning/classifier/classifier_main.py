import argparse
from pathlib import Path
from representation_learning.classifier.trainers.classifier_trainer import ClassifierTrainer
from representation_learning.utils.config import ConfigManager
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier model')
    parser.add_argument('--system', type=str, help='System to train the classifier for')
    parser.add_argument('--config', type=str, 
                      default='representation_learning/configs/classifier_config.yaml',
                      help='Path to the config file')
    parser.add_argument('--output_dir', type=str, help='Directory to save outputs')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                      help='Mode to run the model in')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file')
    return parser.parse_args()

def main():
    args = parse_args()
    system_name = args.system
    
    # Load configuration
    full_config = ConfigManager.load_config(args.config)
    config = full_config['reachability_classifier']
    # Override output directory if specified
    # if args.output_dir:
    #     config['output_dir'] = args.output_dir
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Initialize trainer
    trainer = ClassifierTrainer(config, system_name)
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Run training or evaluation
    if args.mode == 'train':
        trainer.train(checkpoint_dir=output_dir)
    else:
        val_loss = trainer.validate()
        print(f'Validation Loss: {val_loss:.4f}')

if __name__ == '__main__':
    main()
