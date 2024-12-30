import argparse
from pathlib import Path
from representation_learning.classifier.trainers.probability_trainer import ProbabilityTrainer
from representation_learning.utils.config import ConfigManager
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier model')
    parser.add_argument('--config', type=str, 
                      default='representation_learning/configs/probability_config.yaml',
                      help='Path to the config file')
    parser.add_argument('--output_dir', type=str, help='Directory to save outputs')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                      help='Mode to run the model in')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    config = ConfigManager.load_config(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = ProbabilityTrainer(config)

    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Run training or evaluation
    if args.mode == 'train':
        trainer.train(checkpoint_dir=output_dir)
    # else:
    #     val_loss = trainer.validate()
    #     print(f'Validation Loss: {val_loss:.4f}')

if __name__ == '__main__':
    main()
