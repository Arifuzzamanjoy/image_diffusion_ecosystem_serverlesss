"""
Configuration management for FLUX LoRA training
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration settings"""
    name_or_path: str = "black-forest-labs/FLUX.1-dev"
    model_type: str = "dev"  # "dev" or "schnell"
    assistant_lora_path: Optional[str] = None
    low_vram: bool = False
    quantize: bool = False
    text_encoder_path: str = "openai/clip-vit-large-patch14-336"
    text_encoder_2_path: str = "google/t5-v1_1-xxl" 
    text_encoder_offload: bool = True
    text_encoder_2_offload: bool = True
    text_encoder_attention_mask: bool = True
    
    def __post_init__(self):
        """Set model-specific defaults"""
        if self.model_type == "schnell":
            self.name_or_path = "black-forest-labs/FLUX.1-schnell"
            self.assistant_lora_path = "ostris/FLUX.1-schnell-training-adapter"
        elif self.model_type == "dev":
            self.name_or_path = "black-forest-labs/FLUX.1-dev"


@dataclass
class NetworkConfig:
    """LoRA network configuration"""
    linear: int = 16
    linear_alpha: int = 16
    network_type: str = "lora"
    init_weights: str = "xavier_uniform"
    use_bias: bool = False
    dropout: float = 0.0
    use_tucker: bool = False
    use_cp: bool = False
    
    def __post_init__(self):
        """Set rank-based optimizations"""
        if self.linear >= 64:
            self.use_tucker = True
            self.use_cp = True
            self.dropout = 0.15
        elif self.linear >= 32:
            self.use_tucker = True
            self.dropout = 0.1


@dataclass
class TrainingConfig:
    """Training configuration settings"""
    steps: int = 1000
    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    optimizer: str = "adamw8bit"
    train_dtype: str = "bf16"
    save_dtype: str = "float16"
    gradient_checkpointing: bool = True
    noise_scheduler: str = "flowmatch"
    
    # Advanced optimization settings
    min_snr_gamma: Optional[float] = None
    snr_scale: Optional[float] = None
    linear_timesteps: bool = False
    timestep_sampling: str = "uniform"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    
    # Loss configuration
    loss_type: str = "huber"
    huber_c: float = 0.001
    wavelet_loss_weight: float = 0.1
    wavelet_type: str = "db4"
    perceptual_loss_weight: float = 0.05
    lpips_loss_weight: float = 0.02
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine_with_restarts"
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Regularization
    gradient_surgery: bool = True
    gradient_clipping: float = 1.0
    weight_decay: float = 0.01
    
    # Memory optimization
    use_xformers: bool = True
    attention_slicing: str = "auto"
    cpu_offload_params: bool = False
    sequential_cpu_offload: bool = False
    enable_model_cpu_offload: bool = False
    
    def __post_init__(self):
        """Set scheduler-specific optimizations"""
        if self.noise_scheduler == "flowmatch":
            self.linear_timesteps = True
            self.timestep_sampling = "logit_normal"
            # Remove SNR-related settings for flow matching
            self.min_snr_gamma = None
            self.snr_scale = None
        else:
            # DDPM-style schedulers
            self.min_snr_gamma = 5.0
            self.snr_scale = 2.5
        
        # Set LR scheduler defaults
        if not self.lr_scheduler_kwargs:
            self.lr_scheduler_kwargs = {
                "first_cycle_steps": self.steps // 4,
                "cycle_mult": 1.5,
                "max_lr": self.learning_rate * 2.0,
                "min_lr": self.learning_rate * 0.1,
                "warmup_steps": self.steps // 20,
                "gamma": 0.9
            }
        
        # Set memory optimization based on low VRAM
        if hasattr(self, '_low_vram') and self._low_vram:
            self.cpu_offload_params = True
            self.sequential_cpu_offload = True
            self.enable_model_cpu_offload = True


@dataclass
class DatasetConfig:
    """Dataset configuration settings"""
    folder_path: str = ""
    caption_dropout_rate: float = 0.1
    cache_latents_to_disk: bool = True
    shuffle_tokens: bool = False
    
    # Augmentation settings
    random_crop: bool = True
    random_flip: bool = True
    color_jitter: float = 0.1
    brightness_jitter: float = 0.05
    contrast_jitter: float = 0.05
    
    # Caption processing
    tag_separator: str = ","
    keep_tokens: int = 2
    dream_booth_class_tokens: bool = True
    pad_tokens: bool = True
    truncate_pad: bool = True
    
    # Memory management
    pin_memory: bool = True
    persistent_workers: bool = True
    num_workers: int = 2
    
    # Resolution settings
    resolutions: List[int] = field(default_factory=lambda: [1024])
    bucket_step_size: int = 64
    enable_bucket_debug: bool = False


@dataclass
class SamplingConfig:
    """Sampling configuration for validation"""
    guidance_scale: float = 3.5
    sample_steps: int = 28
    sample_every: int = 1000
    sampler: str = "flowmatch"
    seed: int = 42
    walk_seed: bool = True
    height: int = 1024
    width: int = 1024
    neg: str = ""
    sample_prompts: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Set model-specific sampling defaults"""
        if hasattr(self, '_model_type') and self._model_type == "schnell":
            self.guidance_scale = 1.0
            self.sample_steps = 4


@dataclass
class SaveConfig:
    """Save configuration settings"""
    dtype: str = "float16"
    push_to_hub: bool = False  # Default: no auto-upload to HuggingFace
    hf_private: bool = True
    save_every: int = None  # Will be set to match training steps
    max_step_saves_to_keep: int = 1  # Keep only final checkpoint by default
    
    # Advanced archiving
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_random_states: bool = True
    use_safetensors: bool = True
    create_model_card: bool = True
    
    # Logging
    log_with: List[str] = field(default_factory=lambda: ["tensorboard", "wandb"])
    log_predictions: bool = True
    log_model_architecture: bool = True


@dataclass
class EMAConfig:
    """Exponential Moving Average configuration"""
    use_ema: bool = False
    ema_decay: float = 0.99
    ema_update_after_step: int = 100
    ema_update_every: int = 10
    ema_power: float = 0.75
    ema_use_half_precision: bool = True
    ema_device: str = "cuda"
    ema_cpu_only: bool = False


class ConfigManager:
    """Manages all configuration components"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.network = NetworkConfig()
        self.training = TrainingConfig()
        self.dataset = DatasetConfig()
        self.sampling = SamplingConfig()
        self.save = SaveConfig()
        self.ema = EMAConfig()
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section_name, section_config in config_dict.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_config.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
    
    def to_yaml_config(self, lora_name: str) -> Dict[str, Any]:
        """Convert to ai-toolkit YAML configuration format"""
        config = {
            "job": "extension",
            "config": {
                "name": lora_name,
                "process": [{
                    "type": "sd_trainer",
                    "training_folder": "output",
                    "performance_log_every": 100,
                    
                    # Model configuration
                    "model": {
                        "name_or_path": self.model.name_or_path,
                        "is_flux": True,
                        "quantize": self.model.quantize,
                        "low_vram": self.model.low_vram,
                        "text_encoder_path": self.model.text_encoder_path,
                        "text_encoder_2_path": self.model.text_encoder_2_path,
                        "text_encoder_offload": self.model.text_encoder_offload,
                        "text_encoder_2_offload": self.model.text_encoder_2_offload,
                        "text_encoder_attention_mask": self.model.text_encoder_attention_mask,
                    },
                    
                    # Network configuration
                    "network": {
                        "type": self.network.network_type,
                        "linear": self.network.linear,
                        "linear_alpha": self.network.linear_alpha,
                        "init_weights": self.network.init_weights,
                        "use_bias": self.network.use_bias,
                        "dropout": self.network.dropout,
                    },
                    
                    # Training configuration
                    "train": {
                        "steps": self.training.steps,
                        "lr": self.training.learning_rate,
                        "batch_size": self.training.batch_size,
                        "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
                        "optimizer": self.training.optimizer,
                        "dtype": self.training.train_dtype,
                        "gradient_checkpointing": self.training.gradient_checkpointing,
                        "noise_scheduler": self.training.noise_scheduler,
                        "skip_first_sample": True,
                        
                        # Loss configuration
                        "loss_type": self.training.loss_type,
                        "huber_c": self.training.huber_c,
                        "wavelet_loss_weight": self.training.wavelet_loss_weight,
                        "wavelet_type": self.training.wavelet_type,
                        "perceptual_loss_weight": self.training.perceptual_loss_weight,
                        "lpips_loss_weight": self.training.lpips_loss_weight,
                        
                        # Learning rate scheduling
                        "lr_scheduler": self.training.lr_scheduler,
                        "lr_scheduler_kwargs": self.training.lr_scheduler_kwargs,
                        
                        # Regularization
                        "gradient_surgery": self.training.gradient_surgery,
                        "gradient_clipping": self.training.gradient_clipping,
                        "weight_decay": self.training.weight_decay,
                        
                        # Memory optimization
                        "use_xformers": self.training.use_xformers,
                        "attention_slicing": self.training.attention_slicing,
                        "cpu_offload_params": self.training.cpu_offload_params,
                        "sequential_cpu_offload": self.training.sequential_cpu_offload,
                        "enable_model_cpu_offload": self.training.enable_model_cpu_offload,
                    },
                    
                    # Dataset configuration
                    "datasets": [{
                        "folder_path": self.dataset.folder_path,
                        "caption_ext": ".txt",
                        "caption_dropout_rate": self.dataset.caption_dropout_rate,
                        "cache_latents_to_disk": self.dataset.cache_latents_to_disk,
                        "shuffle_tokens": self.dataset.shuffle_tokens,
                        
                        # Augmentation
                        "random_crop": self.dataset.random_crop,
                        "random_flip": self.dataset.random_flip,
                        "color_jitter": self.dataset.color_jitter,
                        "brightness_jitter": self.dataset.brightness_jitter,
                        "contrast_jitter": self.dataset.contrast_jitter,
                        
                        # Caption processing
                        "tag_separator": self.dataset.tag_separator,
                        "keep_tokens": self.dataset.keep_tokens,
                        "dream_booth_class_tokens": self.dataset.dream_booth_class_tokens,
                        "pad_tokens": self.dataset.pad_tokens,
                        "truncate_pad": self.dataset.truncate_pad,
                        
                        # Memory management
                        "pin_memory": self.dataset.pin_memory,
                        "persistent_workers": self.dataset.persistent_workers,
                        "num_workers": self.dataset.num_workers,
                        
                        # Resolution bucketing
                        "resolution": self.dataset.resolutions,
                    }],
                    
                    # Sampling configuration
                    "sample": {
                        "guidance_scale": self.sampling.guidance_scale,
                        "sample_steps": self.sampling.sample_steps,
                        "sample_every": self.sampling.sample_every,
                        "sampler": self.sampling.sampler,
                        "seed": self.sampling.seed,
                        "walk_seed": self.sampling.walk_seed,
                        "height": self.sampling.height,
                        "width": self.sampling.width,
                        "neg": self.sampling.neg,
                    },
                    
                    # Save configuration
                    "save": {
                        "dtype": self.save.dtype,
                        "push_to_hub": self.save.push_to_hub,
                        "hf_private": self.save.hf_private,
                        "save_every": self.save.save_every,
                        "max_step_saves_to_keep": self.save.max_step_saves_to_keep,
                        "save_optimizer": self.save.save_optimizer,
                        "save_scheduler": self.save.save_scheduler,
                        "save_random_states": self.save.save_random_states,
                        "use_safetensors": self.save.use_safetensors,
                        "create_model_card": self.save.create_model_card,
                        "log_with": self.save.log_with,
                        "log_predictions": self.save.log_predictions,
                        "log_model_architecture": self.save.log_model_architecture,
                    }
                }]
            }
        }
        
        # Add model-specific settings
        if self.model.assistant_lora_path:
            config["config"]["process"][0]["model"]["assistant_lora_path"] = self.model.assistant_lora_path
        
        # Add scheduler-specific settings
        if self.training.noise_scheduler == "flowmatch":
            config["config"]["process"][0]["train"]["linear_timesteps"] = self.training.linear_timesteps
            config["config"]["process"][0]["train"]["timestep_sampling"] = self.training.timestep_sampling
            config["config"]["process"][0]["train"]["logit_mean"] = self.training.logit_mean
            config["config"]["process"][0]["train"]["logit_std"] = self.training.logit_std
        else:
            if self.training.min_snr_gamma:
                config["config"]["process"][0]["train"]["min_snr_gamma"] = self.training.min_snr_gamma
            if self.training.snr_scale:
                config["config"]["process"][0]["train"]["snr_scale"] = self.training.snr_scale
        
        # Add network-specific optimizations
        if self.network.use_tucker:
            config["config"]["process"][0]["network"]["use_tucker"] = True
        if self.network.use_cp:
            config["config"]["process"][0]["network"]["use_cp"] = True
        
        # Add EMA configuration if enabled
        if self.ema.use_ema:
            config["config"]["process"][0]["train"]["ema_config"] = {
                "use_ema": True,
                "ema_decay": self.ema.ema_decay,
                "ema_update_after_step": self.ema.ema_update_after_step,
                "ema_update_every": self.ema.ema_update_every,
                "ema_power": self.ema.ema_power,
                "ema_use_half_precision": self.ema.ema_use_half_precision,
                "ema_device": self.ema.ema_device,
                "ema_cpu_only": self.ema.ema_cpu_only,
            }
        
        # Add sample prompts if available
        if self.sampling.sample_prompts:
            config["config"]["process"][0]["sample"]["prompts"] = [
                {
                    "prompt": prompt["prompt"] if isinstance(prompt, dict) else str(prompt),
                    "seed": prompt.get("seed", self.sampling.seed + i) if isinstance(prompt, dict) else (self.sampling.seed + i)
                }
                for i, prompt in enumerate(self.sampling.sample_prompts)
            ]
        
        return config
    
    def save_config(self, filepath: str, lora_name: str):
        """Save configuration to YAML file"""
        config = self.to_yaml_config(lora_name)
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    def load_config(self, filepath: str):
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        # Parse configuration back to dataclasses
        # This is a simplified version - you might want to make it more robust
        process_config = config.get("config", {}).get("process", [{}])[0]
        
        # Update each section
        if "model" in process_config:
            self.update_from_dict({"model": process_config["model"]})
        if "network" in process_config:
            self.update_from_dict({"network": process_config["network"]})
        if "train" in process_config:
            self.update_from_dict({"training": process_config["train"]})
        # ... continue for other sections