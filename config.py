"""
Configuration management for DriveSafe Drowsiness Detection System
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('FLASK_PORT', 5001))
    
    # Model Configuration
    MODEL_FILE_ID = os.environ.get('MODEL_FILE_ID', '1UInMiIbaHChmI-KSQ7VRMp_53RZpSDd4')
    MODEL_PATH = os.environ.get('MODEL_PATH', './model/resnet50v2_model.keras')
    
    # Detection Parameters
    DEFAULT_EYE_THRESHOLD = float(os.environ.get('DEFAULT_EYE_THRESHOLD', 0.25))
    EYES_CLOSED_DURATION = int(os.environ.get('EYES_CLOSED_DURATION', 1))
    YAWNING_DURATION = int(os.environ.get('YAWNING_DURATION', 3))
    
    # Camera Configuration
    CAMERA_INDICES = [0, 1, 2]  # Try multiple camera indices
    MAX_CAMERA_FAILURES = int(os.environ.get('MAX_CAMERA_FAILURES', 10))
    
    # API Configuration
    MAX_FRAMES_PER_CALIBRATION = int(os.environ.get('MAX_FRAMES_PER_CALIBRATION', 100))
    FRAME_PROCESSING_INTERVAL = int(os.environ.get('FRAME_PROCESSING_INTERVAL', 200))
    
    # Security Configuration
    CORS_ORIGINS = [
        "http://localhost:5001",
        "http://127.0.0.1:5001",
        "*"
    ]
    RATE_LIMIT_ENABLED = os.environ.get('RATE_LIMIT_ENABLED', 'True').lower() == 'true'
    RATE_LIMIT_REQUESTS = int(os.environ.get('RATE_LIMIT_REQUESTS', 100))
    RATE_LIMIT_WINDOW = int(os.environ.get('RATE_LIMIT_WINDOW', 60))
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'app.log')
    
    # Performance Configuration
    MAX_MEMORY_VALUES = int(os.environ.get('MAX_MEMORY_VALUES', 100))
    STATISTICS_CLEANUP_INTERVAL = int(os.environ.get('STATISTICS_CLEANUP_INTERVAL', 300))  # 5 minutes

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'production-secret-key-must-be-set'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name: str = None) -> Config:
    """Get configuration based on environment"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    return config.get(config_name, config['default'])
