"""
Configuration system for Data Integrity Agent
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json

@dataclass
class ValidationConfig:
    """Configuration for validation settings"""
    
    # Missing values configuration
    missing_values: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'threshold_percentage': 5.0,
        'critical_threshold': 20.0
    })
    
    # Outlier detection configuration
    outliers: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'method': 'iqr',  # 'iqr', 'zscore', 'isolation_forest'
        'iqr_multiplier': 1.5,
        'zscore_threshold': 3.0,
        'threshold_percentage': 5.0
    })
    
    # Distribution analysis configuration
    distributions: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'include_skewness': True,
        'include_kurtosis': True,
        'histogram_bins': 20
    })
    
    # Format validation configuration
    format_validation: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'email_pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'date_format': '%Y-%m-%d',
        'phone_pattern': r'^\+?1?\d{9,15}$'
    })
    
    # Range validation configuration
    range_validation: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'age_range': {'min': 0, 'max': 150},
        'score_range': {'min': 0, 'max': 100},
        'percentage_range': {'min': 0, 'max': 100}
    })
    
    # Consistency checks configuration
    consistency_checks: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'check_duplicates': True,
        'check_constants': True,
        'check_logical_consistency': True
    })
    
    # LLM configuration
    llm: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'model': 'gemini-1.5-flash',
        'max_tokens': 1000,
        'temperature': 0.1,
        'timeout': 30
    })
    
    # Output configuration
    output: Dict[str, Any] = field(default_factory=lambda: {
        'format': 'json',
        'include_detailed_results': True,
        'include_recommendations': True,
        'include_visualizations': False
    })
    
    # General configuration (no domain-specific settings)

class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config = self._load_config()
    
    def _find_config_file(self) -> str:
        """Find configuration file in common locations"""
        possible_paths = [
            'config.yaml',
            'config.yml',
            'data_integrity_config.yaml',
            os.path.expanduser('~/.data_integrity/config.yaml'),
            '/etc/data_integrity/config.yaml'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Return default path if no config found
        return 'config.yaml'
    
    def _load_config(self) -> ValidationConfig:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                return self._create_config_from_dict(config_data)
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
                print("Using default configuration")
        
        # Create default config
        config = ValidationConfig()
        self._save_config(config)
        return config
    
    def _create_config_from_dict(self, config_data: Dict[str, Any]) -> ValidationConfig:
        """Create ValidationConfig from dictionary"""
        config = ValidationConfig()
        
        # Update config with loaded data
        for section, values in config_data.items():
            if hasattr(config, section):
                current_section = getattr(config, section)
                if isinstance(current_section, dict):
                    current_section.update(values)
                else:
                    setattr(config, section, values)
        
        return config
    
    def _save_config(self, config: ValidationConfig) -> None:
        """Save configuration to file"""
        # Ensure directory exists
        config_dir = os.path.dirname(self.config_path)
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)
        
        # Convert config to dictionary
        config_dict = self._config_to_dict(config)
        
        # Save as YAML
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def _config_to_dict(self, config: ValidationConfig) -> Dict[str, Any]:
        """Convert ValidationConfig to dictionary"""
        config_dict = {}
        
        for field_name in config.__dataclass_fields__:
            value = getattr(config, field_name)
            config_dict[field_name] = value
        
        return config_dict
    
    def get_config(self) -> ValidationConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        config_dict = self._config_to_dict(self.config)
        config_dict.update(updates)
        
        self.config = self._create_config_from_dict(config_dict)
        self._save_config(self.config)
    

    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate missing values config
        if self.config.missing_values['threshold_percentage'] < 0 or self.config.missing_values['threshold_percentage'] > 100:
            issues.append("Missing values threshold percentage must be between 0 and 100")
        
        # Validate outlier config
        if self.config.outliers['iqr_multiplier'] <= 0:
            issues.append("IQR multiplier must be positive")
        
        if self.config.outliers['zscore_threshold'] <= 0:
            issues.append("Z-score threshold must be positive")
        
        # Validate range config
        for range_name, range_config in self.config.range_validation.items():
            if isinstance(range_config, dict) and 'min' in range_config and 'max' in range_config:
                if range_config['min'] >= range_config['max']:
                    issues.append(f"Range {range_name}: min must be less than max")
        
        return issues
    

    
    def export_config(self, format: str = 'yaml') -> str:
        """Export configuration in specified format"""
        config_dict = self._config_to_dict(self.config)
        
        if format.lower() == 'json':
            return json.dumps(config_dict, indent=2)
        elif format.lower() == 'yaml':
            return yaml.dump(config_dict, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

# Global configuration instance
_config_manager: Optional[ConfigManager] = None

def get_config() -> ValidationConfig:
    """Get global configuration instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.get_config()

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def update_config(updates: Dict[str, Any]) -> None:
    """Update global configuration"""
    config_manager = get_config_manager()
    config_manager.update_config(updates)

def create_default_config() -> None:
    """Create default configuration file"""
    config_manager = ConfigManager()
    config = config_manager.get_config()
    print(f"Default configuration created at: {config_manager.config_path}")
