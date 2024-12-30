import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, Type

from representation_learning.utils.config import ConfigManager
from representation_learning.utils.logger import setup_logger
from representation_learning.trainers import TransformerTrainer, BaseTrainer, DistillationTrainer
from representation_learning.utils.create_representation_dataset import create_representation_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Representation Learning Training')
    parser.add_argument('--config', type=Path, required=True, help='Path to config file')
    parser.add_argument('--trainer', choices=['transformer', 'distillation', 'combined', 'evaluation'], 
                       required=True, help='Type of trainer to use')
    parser.add_argument('--mode', choices=['train', 'eval', 'visualize', 'create_dataset'], default='train',
                       help='Training or evaluation mode')
    parser.add_argument('--checkpoint', type=Path, help='Path to checkpoint for resuming training')
    parser.add_argument('--output-dir', type=Path, help='Directory for outputs')
    parser.add_argument('--system', type=str, help='System name')
    return parser.parse_args()

def setup_output_dir(base_dir: Optional[Path] = None, trainer: Optional[str] = None, date=False) -> Path:
    """Create and return output directory with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if base_dir is None:
        base_dir = Path('outputs')
    output_dir = base_dir
    if trainer is not None:
        output_dir = base_dir / trainer
    if date:
        output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def main():
    args = parse_args()
    system_name = args.system
    
    print(args.checkpoint)
    # Setup output directory and logging
    
    if args.mode == 'train':
        output_dir = setup_output_dir(args.output_dir, args.trainer)
    else:
        output_dir = setup_output_dir(args.output_dir)

    print(output_dir)

    logger = setup_logger('main', output_dir / 'training.log')
    logger.info(f"Starting {args.trainer} in {args.mode} mode")
    
    trainer_cls: Type[BaseTrainer] = {
        'transformer': TransformerTrainer,
        'distillation': DistillationTrainer,
        # 'combined': CombinedModelTrainer,
        # 'evaluation': CombinedEvaluationTrainer
    }[args.trainer]

    # Load and validate configuration
    full_config = ConfigManager.load_specific_config(args.config)
    # print(full_config)
    # exit()
    
    config = full_config[args.trainer]
    # Save configuration
    with open(output_dir / 'config.yaml', 'w') as f: 
        yaml.dump(config, f)
    
    # Create trainer based on argument
    trainer = trainer_cls(config, system_name=args.system)
    
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    if args.mode == 'train':
        trainer.train(checkpoint_dir=output_dir / 'checkpoints')
        
    if args.mode == 'eval':
        val_loss = trainer.validate()
        logger.info(f'Validation Loss: {val_loss:.4f}')
        
    if args.mode == 'visualize':
        if args.trainer == 'transformer':
            # TODO: incase no checkpoint is provided, use the best checkpoint
            trainer.visualize_full_trajectories(args.checkpoint, output_dir, num_samples=50)

    if args.mode == 'create_dataset':
        create_representation_dataset(config, args.system, trainer.model, output_dir)

if __name__ == "__main__":
    main()