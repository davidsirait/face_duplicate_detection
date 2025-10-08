import yaml
from pathlib import Path

class Config:
    """Practical configuration loader"""
    
    def __init__(self, config_path: str = "./config.yaml"):
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        self._create_directories()
    
    def _create_directories(self):
        """Automatically create the essential directories"""
        for key in ['db_dir', 'logs_dir']:
            Path(self._config['paths'][key]).mkdir(parents=True, exist_ok=True)
    
    def get(self, key_path: str, default=None):
        """Get value using dot notation, example :config.get('path.db_dir')"""
        keys = key_path.split('.')
        value = self._config
        for key in keys:
            value = value.get(key, default) if isinstance(value, dict) else default
        return value