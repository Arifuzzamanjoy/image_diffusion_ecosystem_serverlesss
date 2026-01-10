"""
World-Class Configuration Manager

Enhanced configuration management with research-based defaults,
professional presets, and advanced validation for optimal training results.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
import yaml
import json
from pathlib import Path


@dataclass
class WorldClassLoRAConfig:
    """World-class LoRA configuration with research-based defaults"""
    
    # Core LoRA Architecture (Research-Optimized)
    rank: int = 32  # Optimal balance of capacity and efficiency
    alpha: int = 32  # Usually equal to rank for balanced scaling
    target_modules: List[str] = field(default_factory=lambda: [
        "to_q", "to_k", "to_v", "to_out"  # Core attention modules
    ])
    dropout: float = 0.1  # Prevent overfitting
    
    # Advanced Training Parameters
    learning_rate: float = 1e-4  # Proven optimal for most cases
    optimizer: str = "adamw"  # Professional standard optimizer
    lr_scheduler: str = "cosine_with_restarts"  # Best convergence pattern
    warmup_steps: int = 100  # Gradual learning rate increase
    max_train_steps: int = 1500  # Sweet spot for quality vs time
    
    # Memory and Performance Optimization
    batch_size: int = 1  # Safe for most GPUs
    gradient_accumulation_steps: int = 4  # Effective batch size of 4
    gradient_checkpointing: bool = True  # Memory efficiency
    mixed_precision: str = "bf16"  # Optimal for modern GPUs
    
    # Quality and Sampling Settings
    save_precision: str = "float16"  # Compact but high quality
    guidance_scale: float = 3.5  # FLUX optimal guidance
    sample_steps: int = 28  # Good quality/speed balance
    sample_every: int = 250  # Frequent monitoring
    save_every: int = 500  # Regular checkpoints
    
    # Advanced Features
    use_ema: bool = True  # Exponential moving average for stability
    ema_decay: float = 0.99  # Smoothing factor
    noise_scheduler: str = "flowmatch"  # FLUX native scheduler
    
    # Memory Management
    low_vram_mode: bool = False
    cpu_offload: bool = False
    quantization: bool = False


@dataclass
class WorldClassTrainingConfig:
    """Complete world-class training configuration"""
    
    # Project Information
    project_name: str = "world-class-lora"
    model_name: str = "my-amazing-lora"
    trigger_word: str = "TOK"
    description: str = "A world-class LoRA trained with advanced techniques"
    
    # Model Selection
    flux_model: str = "dev"  # dev or schnell
    model_path: str = ""  # Auto-filled
    
    # Dataset Configuration
    dataset_path: str = ""
    resolution: int = 1024  # FLUX native resolution
    aspect_ratio_buckets: bool = True  # Handle different aspect ratios
    
    # LoRA Configuration
    lora: WorldClassLoRAConfig = field(default_factory=WorldClassLoRAConfig)
    
    # Sample Generation
    sample_prompts: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "prompt": "A portrait of TOK, high quality, detailed",
            "negative": "blurry, low quality, deformed",
            "weight": 1.0
        },
        {
            "prompt": "TOK in a beautiful landscape, cinematic lighting", 
            "negative": "dark, unclear, poor composition",
            "weight": 1.0
        }
    ])
    
    # Advanced Features
    enable_sample_gallery: bool = True
    gallery_update_frequency: int = 250
    enable_wandb: bool = False  # Weights & Biases logging
    wandb_project: str = "flux-lora-training"
    
    # Output Configuration
    output_dir: str = "output"
    logging_level: str = "INFO"
    save_optimizer_state: bool = True
    
    def to_ai_toolkit_config(self) -> Dict[str, Any]:
        """Convert to ai-toolkit compatible configuration"""
        
        config = {
            "job": "extension",
            "config": {
                "name": f"{self.project_name}_{self.model_name}",
                "process": [
                    {
                        "type": "sd_trainer",
                        "training_folder": f"{self.output_dir}/{self.model_name}",
                        "device": "cuda:0",
                        "trigger_word": self.trigger_word,
                        "network": {
                            "type": "lora",
                            "linear": self.lora.rank,
                            "linear_alpha": self.lora.alpha,
                            "linear_dropout": self.lora.dropout
                        },
                        "save": {
                            "dtype": self.lora.save_precision,
                            "save_every": self.lora.steps,  # Save only at end of training
                            "max_step_saves_to_keep": 1,     # Keep only final model
                            "push_to_hub": False,            # Disable auto-upload
                            "use_safetensors": True
                        },
                        "datasets": [
                            {
                                "folder_path": self.dataset_path,
                                "caption_ext": "txt",
                                "caption_dropout_rate": 0.05,
                                "shuffle_tokens": False,
                                "cache_latents_to_disk": True,
                                "resolution": [self.resolution, self.resolution]
                            }
                        ],
                        "train": {
                            "batch_size": self.lora.batch_size,
                            "steps": self.lora.max_train_steps,
                            "gradient_accumulation_steps": self.lora.gradient_accumulation_steps,
                            "train_unet": True,
                            "train_text_encoder": False,
                            "gradient_checkpointing": self.lora.gradient_checkpointing,
                            "noise_scheduler": self.lora.noise_scheduler,
                            "optimizer": self.lora.optimizer,
                            "lr": self.lora.learning_rate,
                            "ema_config": {
                                "use_ema": self.lora.use_ema,
                                "ema_decay": self.lora.ema_decay
                            } if self.lora.use_ema else None,
                            "dtype": self.lora.mixed_precision
                        },
                        "model": {
                            "name_or_path": f"black-forest-labs/FLUX.1-{self.flux_model}",
                            "is_flux": True,
                            "quantize": self.lora.quantization
                        },
                        "sample": {
                            "sampler": "flowmatch",
                            "sample_every": self.lora.sample_every,
                            "width": self.resolution,
                            "height": self.resolution,
                            "prompts": [
                                {
                                    "prompt": p["prompt"] if isinstance(p, dict) else str(p),
                                    "seed": p.get("seed", 42 + i) if isinstance(p, dict) else (42 + i)
                                }
                                for i, p in enumerate(self.sample_prompts)
                            ],
                            "neg": "",
                            "seed": 42,
                            "walk_seed": True,
                            "guidance_scale": self.lora.guidance_scale,
                            "sample_steps": self.lora.sample_steps
                        }
                    }
                ]
            }
        }
        
        # Add memory optimizations
        if self.lora.low_vram_mode:
            config["config"]["process"][0]["model"]["low_vram"] = True
            config["config"]["process"][0]["train"]["gradient_accumulation_steps"] = max(
                config["config"]["process"][0]["train"]["gradient_accumulation_steps"], 8
            )
        
        return config
    
    def save_config(self, path: Union[str, Path]):
        """Save configuration to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config = self.to_ai_toolkit_config()
        
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    @classmethod
    def from_preset(cls, preset_name: str) -> 'WorldClassTrainingConfig':
        """Create configuration from preset"""
        presets = cls.get_presets()
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        config = cls()
        preset_config = presets[preset_name]
        
        # Apply preset to LoRA config
        for key, value in preset_config.items():
            if hasattr(config.lora, key):
                setattr(config.lora, key, value)
            elif hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    @staticmethod
    def get_presets() -> Dict[str, Dict[str, Any]]:
        """Get all available configuration presets"""
        return {
            "🎨 Style/Concept (Recommended)": {
                "rank": 32, "alpha": 32, "learning_rate": 1e-4, "max_train_steps": 1500,
                "optimizer": "adamw", "lr_scheduler": "cosine_with_restarts",
                "batch_size": 1, "gradient_accumulation_steps": 4, "sample_every": 250,
                "warmup_steps": 100, "use_ema": True, "ema_decay": 0.9999
            },
            "👤 Character/Person": {
                "rank": 64, "alpha": 64, "learning_rate": 8e-5, "max_train_steps": 2000,
                "optimizer": "adamw", "lr_scheduler": "cosine_with_restarts",
                "batch_size": 1, "gradient_accumulation_steps": 2, "sample_every": 200,
                "warmup_steps": 150, "use_ema": True, "ema_decay": 0.9999,
                "target_modules": ["to_q", "to_k", "to_v", "to_out", "ff.net.0", "ff.net.2"]
            },
            "🏃 Quick Test": {
                "rank": 16, "alpha": 16, "learning_rate": 2e-4, "max_train_steps": 500,
                "optimizer": "adamw", "lr_scheduler": "linear",
                "batch_size": 1, "gradient_accumulation_steps": 2, "sample_every": 100,
                "warmup_steps": 50, "use_ema": False
            },
            "🔬 Research/Experimental": {
                "rank": 128, "alpha": 128, "learning_rate": 5e-5, "max_train_steps": 3000,
                "optimizer": "adamw", "lr_scheduler": "cosine_with_restarts",
                "batch_size": 1, "gradient_accumulation_steps": 8, "sample_every": 500,
                "warmup_steps": 200, "use_ema": True, "ema_decay": 0.9999,
                "target_modules": ["to_q", "to_k", "to_v", "to_out", "ff.net.0", "ff.net.2"]
            },
            "💾 Low VRAM (<12GB)": {
                "rank": 16, "alpha": 16, "learning_rate": 1e-4, "max_train_steps": 1000,
                "optimizer": "adamw", "lr_scheduler": "cosine",
                "batch_size": 1, "gradient_accumulation_steps": 8, "sample_every": 200,
                "low_vram_mode": True, "gradient_checkpointing": True, "cpu_offload": True
            },
            "⚡ Speed Optimized": {
                "rank": 16, "alpha": 16, "learning_rate": 3e-4, "max_train_steps": 800,
                "optimizer": "lion", "lr_scheduler": "linear",
                "batch_size": 2, "gradient_accumulation_steps": 2, "sample_every": 200,
                "warmup_steps": 50, "use_ema": False, "gradient_checkpointing": False
            }
        }


class WorldClassValidation:
    """Advanced validation for world-class training configurations"""
    
    @staticmethod
    def validate_config(config: WorldClassTrainingConfig) -> List[str]:
        """Validate configuration and return warnings/errors"""
        warnings = []
        
        # LoRA parameter validation
        if config.lora.rank > config.lora.alpha * 2:
            warnings.append("⚠️ LoRA rank is much higher than alpha - consider increasing alpha")
        
        if config.lora.learning_rate > 5e-4:
            warnings.append("⚠️ Learning rate is quite high - may cause unstable training")
        
        if config.lora.batch_size * config.lora.gradient_accumulation_steps > 8:
            warnings.append("⚠️ Effective batch size is very large - may need more training data")
        
        # Memory validation
        estimated_vram = WorldClassValidation.estimate_vram_usage(config)
        if estimated_vram > 24:
            warnings.append(f"⚠️ Estimated VRAM usage: {estimated_vram:.1f}GB - consider low VRAM optimizations")
        
        # Training duration validation
        if config.lora.max_train_steps > 5000:
            warnings.append("⚠️ Very long training - may lead to overfitting")
        
        if config.lora.max_train_steps < 500:
            warnings.append("⚠️ Short training - may not achieve good results")
        
        # Dataset validation
        if not config.dataset_path:
            warnings.append("❌ Dataset path is required")
        
        if not config.trigger_word:
            warnings.append("❌ Trigger word is required")
        
        return warnings
    
    @staticmethod
    def estimate_vram_usage(config: WorldClassTrainingConfig) -> float:
        """Estimate VRAM usage in GB"""
        base_usage = 12.0  # FLUX base model
        
        # LoRA parameters
        lora_usage = (config.lora.rank ** 2) * 0.001  # Rough estimate
        
        # Batch size impact
        batch_usage = config.lora.batch_size * 2.0
        
        # Gradient checkpointing saves ~30%
        if config.lora.gradient_checkpointing:
            total = (base_usage + lora_usage + batch_usage) * 0.7
        else:
            total = base_usage + lora_usage + batch_usage
        
        # Low VRAM mode saves additional 20%
        if config.lora.low_vram_mode:
            total *= 0.8
        
        return total
    
    @staticmethod
    def suggest_optimizations(config: WorldClassTrainingConfig) -> List[str]:
        """Suggest optimizations based on configuration"""
        suggestions = []
        
        vram_usage = WorldClassValidation.estimate_vram_usage(config)
        
        if vram_usage > 16 and not config.lora.low_vram_mode:
            suggestions.append("💡 Enable Low VRAM mode to reduce memory usage")
        
        if not config.lora.gradient_checkpointing:
            suggestions.append("💡 Enable gradient checkpointing to save memory")
        
        if config.lora.optimizer == "adamw" and config.lora.learning_rate == 1e-4:
            suggestions.append("💡 Consider using 'adamw' optimizer for professional results")
        
        if config.lora.rank < 32 and "character" in config.project_name.lower():
            suggestions.append("💡 Consider increasing rank to 64 for character training")
        
        if config.lora.sample_every > 500:
            suggestions.append("💡 Consider more frequent sampling for better monitoring")
        
        return suggestions


class ConfigurationExporter:
    """Export configurations in various formats"""
    
    @staticmethod
    def export_to_huggingface(config: WorldClassTrainingConfig) -> Dict[str, Any]:
        """Export configuration for Hugging Face format"""
        return {
            "peft_config": {
                "base_model_name_or_path": f"black-forest-labs/FLUX.1-{config.flux_model}",
                "bias": "none",
                "fan_in_fan_out": False,
                "inference_mode": False,
                "init_lora_weights": True,
                "layers_pattern": None,
                "layers_to_transform": None,
                "lora_alpha": config.lora.alpha,
                "lora_dropout": config.lora.dropout,
                "modules_to_save": None,
                "peft_type": "LORA",
                "r": config.lora.rank,
                "revision": None,
                "target_modules": config.lora.target_modules,
                "task_type": "DIFFUSION"
            },
            "training_args": {
                "learning_rate": config.lora.learning_rate,
                "num_train_epochs": 1,
                "per_device_train_batch_size": config.lora.batch_size,
                "gradient_accumulation_steps": config.lora.gradient_accumulation_steps,
                "max_steps": config.lora.max_train_steps,
                "warmup_steps": config.lora.warmup_steps,
                "lr_scheduler_type": config.lora.lr_scheduler,
                "optim": config.lora.optimizer,
                "bf16": config.lora.mixed_precision == "bf16",
                "fp16": config.lora.mixed_precision == "fp16",
                "gradient_checkpointing": config.lora.gradient_checkpointing
            }
        }
    
    @staticmethod
    def export_to_diffusers(config: WorldClassTrainingConfig) -> Dict[str, Any]:
        """Export configuration for Diffusers format"""
        return {
            "model_name": f"black-forest-labs/FLUX.1-{config.flux_model}",
            "dataset": {
                "path": config.dataset_path,
                "image_column": "image",
                "caption_column": "text"
            },
            "lora_config": {
                "r": config.lora.rank,
                "lora_alpha": config.lora.alpha,
                "target_modules": config.lora.target_modules,
                "lora_dropout": config.lora.dropout
            },
            "training_config": {
                "learning_rate": config.lora.learning_rate,
                "max_train_steps": config.lora.max_train_steps,
                "train_batch_size": config.lora.batch_size,
                "gradient_accumulation_steps": config.lora.gradient_accumulation_steps,
                "lr_scheduler": config.lora.lr_scheduler,
                "optimizer": config.lora.optimizer,
                "mixed_precision": config.lora.mixed_precision
            }
        }