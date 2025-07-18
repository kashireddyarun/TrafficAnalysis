"""
Configuration management utilities for Traffic Analysis System
"""

import yaml
import os
from typing import Dict, Any
from pathlib import Path

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    
    # Make path absolute if it's relative
    if not os.path.isabs(config_path):
        # Get the project root directory (assuming this file is in src/utils/)
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / config_path
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            return config if config is not None else {}
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        return get_default_config()
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration when config file is not available
    
    Returns:
        Default configuration dictionary
    """
    
    return {
        'model': {
            'name': 'yolov8n',
            'confidence_threshold': 0.5,
            'iou_threshold': 0.5,
            'device': 'auto'
        },
        'classes': {
            'vehicles': [2, 3, 5, 7],  # car, motorcycle, bus, truck
            'pedestrians': [0],        # person
            'traffic_lights': [9],     # traffic light
            'road_signs': [11, 12]     # stop sign, etc.
        },
        'tracking': {
            'max_age': 30,
            'min_hits': 3,
            'iou_threshold': 0.3,
            'enable_kalman_filter': True
        },
        'traffic_analysis': {
            'speed_estimation': {
                'enabled': True,
                'fps': 30,
                'pixel_to_meter_ratio': 10
            },
            'flow_analysis': {
                'count_lines': [[[100, 300], [500, 300]]],
                'direction_analysis': True
            },
            'congestion': {
                'density_threshold': 0.7,
                'speed_threshold': 5,
                'time_window': 30
            }
        },
        'traffic_management': {
            'signal_optimization': {
                'enabled': True,
                'min_green_time': 10,
                'max_green_time': 60
            }
        },
        'display': {
            'enabled': False,
            'show_bboxes': True,
            'show_tracks': True,
            'show_flow_lines': True
        },
        'output': {
            'enabled': True,
            'save_detections': False,
            'save_analytics': True
        },
        'processing': {
            'target_fps': 30,
            'frame_skip': 1
        }
    }

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary to save
        config_path: Path to save the configuration file
        
    Returns:
        True if successful, False otherwise
    """
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False

def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values (deep merge)
    
    Args:
        config: Original configuration
        updates: Updates to apply
        
    Returns:
        Updated configuration
    """
    
    def deep_merge(base_dict: Dict, update_dict: Dict) -> Dict:
        """Recursively merge dictionaries"""
        result = base_dict.copy()
        
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    return deep_merge(config, updates)

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure and values
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    
    required_sections = ['model', 'classes', 'tracking', 'traffic_analysis']
    
    for section in required_sections:
        if section not in config:
            print(f"Missing required configuration section: {section}")
            return False
    
    # Validate model section
    model_config = config.get('model', {})
    if 'confidence_threshold' in model_config:
        threshold = model_config['confidence_threshold']
        if not (0.0 <= threshold <= 1.0):
            print(f"Invalid confidence threshold: {threshold}. Must be between 0.0 and 1.0")
            return False
    
    # Validate tracking section
    tracking_config = config.get('tracking', {})
    if 'max_age' in tracking_config:
        max_age = tracking_config['max_age']
        if not isinstance(max_age, int) or max_age <= 0:
            print(f"Invalid max_age: {max_age}. Must be a positive integer")
            return False
    
    return True

def get_config_info(config: Dict[str, Any]) -> str:
    """
    Get human-readable information about the configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration summary string
    """
    
    info_lines = [
        "=== Traffic Analysis System Configuration ===",
        "",
        f"Model: {config.get('model', {}).get('name', 'unknown')}",
        f"Confidence Threshold: {config.get('model', {}).get('confidence_threshold', 'unknown')}",
        f"Device: {config.get('model', {}).get('device', 'unknown')}",
        "",
        f"Vehicle Classes: {config.get('classes', {}).get('vehicles', [])}",
        f"Tracking Max Age: {config.get('tracking', {}).get('max_age', 'unknown')}",
        f"Speed Estimation: {config.get('traffic_analysis', {}).get('speed_estimation', {}).get('enabled', False)}",
        f"Signal Optimization: {config.get('traffic_management', {}).get('signal_optimization', {}).get('enabled', False)}",
        "",
        "=== End Configuration ==="
    ]
    
    return "\n".join(info_lines)
