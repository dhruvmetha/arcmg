from pathlib import Path
from typing import Dict, Any
import yaml
from copy import deepcopy

class ConfigManager:
    @staticmethod
    def load_config(config_path: Path) -> Dict[str, Any]:
        
        print(config_path)
        # Load base config
        base_config_path = Path(config_path).parent / 'base_config.yaml'
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Load specific config
        with open(config_path, 'r') as f:
            specific_config = yaml.safe_load(f)
            
        # Merge configs
        merged_config = ConfigManager._deep_merge(deepcopy(config), specific_config)
        
        # Handle special cases
        # if 'distillation' in config_path.stem:
        #     if 'data_dir' in merged_config.get('data', {}):
        #         del merged_config['data']['data_dir']
        
        # ConfigManager._validate_config(merged_config, config_path.stem)
        return merged_config
    
    @staticmethod
    def load_specific_config(config_path: Path) -> Dict[str, Any]:
        print(config_path)
        with open(config_path, 'r') as f:
            specific_config = yaml.safe_load(f)
        # ConfigManager._validate_config(specific_config, config_path.stem)
        return specific_config
    
    @staticmethod
    def _validate_config(config: Dict[str, Any], config_type: str) -> None:
        """Validate config based on model type."""
        pass
        # common_required = {
        #     'training': ['batch_size', 'learning_rate', 'num_epochs', 'device', 'save_path'],
        #     'logging': ['log_interval', 'save_interval']
        # }
        
        # model_specific = {
        #     'distillation': {
        #         'data': ['path'],
        #         'model': ['input_dim', 'output_dim']
        #     },
        #     'combined': {
        #         'data': ['data_dir', 'seq_len'],
        #         'model': ['distillation_model_path', 'transformer_model_path']
        #     },
        #     'transformer': {
        #         'data': ['data_dir', 'seq_len'],
        #         'model': ['input_dim', 'd_model', 'nhead', 'num_layers']
        #     }
        # }
        
        # # Get the correct validation schema
        # for model_type, required in model_specific.items():
        #     if model_type in config_type:
        #         required.update(common_required)
        #         ConfigManager._check_required_fields(config, required)
        #         break

    @staticmethod
    def _check_required_fields(config: Dict[str, Any], required: Dict[str, list]) -> None:
        for section, fields in required.items():
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
            for field in fields:
                if field not in config[section]:
                    raise ValueError(f"Missing required field '{field}' in section '{section}'")

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """Recursively merge two dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                ConfigManager._deep_merge(base[key], value)
            else:
                base[key] = value
        return base