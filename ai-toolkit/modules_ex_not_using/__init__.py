"""
Advanced FLUX LoRA Trainer - Modular Training System

A flexible, modular system for training FLUX LoRA models with advanced features.
"""

from .core import ConfigManager, GPUManager, DatasetProcessor
from .training import FluxLoRATrainer, ConfigBuilder  
from .ui import GradioInterface
from .utils import recursive_update, MAX_IMAGES, SUPPORTED_EXTENSIONS

__version__ = "2.0.0"
__author__ = "Advanced FLUX LoRA Trainer Team"

__all__ = [
    # Core components
    'ConfigManager',
    'GPUManager', 
    'DatasetProcessor',
    
    # Training components
    'FluxLoRATrainer',
    'ConfigBuilder',
    
    # UI components
    'GradioInterface',
    
    # Utilities
    'recursive_update',
    'MAX_IMAGES',
    'SUPPORTED_EXTENSIONS'
]


def create_trainer():
    """Factory function to create a trainer instance"""
    return FluxLoRATrainer()


"""
Modular FLUX LoRA Trainer - Main Module

This package provides a modular, maintainable architecture for FLUX LoRA training.
All components are properly separated with clean interfaces and dependency injection.

Features:
- World-class interface with research-based defaults
- Live training sample gallery
- Professional monitoring and analytics
- Configuration presets for different scenarios
"""

from .ui.interface import GradioInterface
from .ui.world_class_interface import WorldClassGradioInterface, create_world_class_interface
from .ui.world_class_simple_interface import WorldClassSimpleInterface, create_world_class_simple_interface

def create_interface():
    """Factory function to create the main interface"""
    return GradioInterface().create_interface()

def create_world_class_trainer():
    """Factory function to create the world-class trainer interface"""
    return create_world_class_simple_interface()

def create_world_class_trainer_advanced():
    """Factory function to create the advanced world-class trainer interface"""
    return create_world_class_interface()


def main():
    """Main entry point for the application"""
    import sys
    import os
    
    # Add current directory to path
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, "../")
    
    # Create and launch interface
    interface = create_interface()
    interface.launch()


if __name__ == "__main__":
    main()