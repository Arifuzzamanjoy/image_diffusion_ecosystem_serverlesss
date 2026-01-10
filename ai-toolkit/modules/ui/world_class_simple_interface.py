"""
FLUX LoRA Trainer Interface - Simplified & User-Friendly

Clean, intuitive interface with optimized presets and progressive disclosure.
"""

import gradio as gr
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Import existing working modules
from .interface import GradioInterface
from .enhanced_components import WorldClassUIComponents
from ..core.world_class_config import WorldClassTrainingConfig
from ..core.dataset_processor import DatasetProcessor
from ..training.trainer import FluxLoRATrainer
import tempfile
import yaml


class WorldClassSimpleInterface:
    """Simplified FLUX LoRA trainer interface with clean UX"""
    
    def __init__(self):
        self.base_interface = GradioInterface()  # Use existing working interface
        self.world_class_config = WorldClassTrainingConfig()
        
        print("🌟 Initializing FLUX LoRA Trainer...")
        print("✓ Optimized templates ready")
    
    def create_interface(self) -> gr.Blocks:
        """Create the simplified professional interface"""
        
        # Get world-class CSS
        css = WorldClassUIComponents.create_world_class_css()
        
        with gr.Blocks(
            css=css,
            title="FLUX LoRA Trainer"
        ) as app:
            
            # Simplified header
            gr.HTML('''
            <div style="text-align: center; padding: 1.5em; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 1.5em;">
                <h1 style="font-size: 2em; margin: 0;">
                    🌟 FLUX LoRA Trainer
                </h1>
                <p style="font-size: 1em; margin: 0.5em 0 0 0; opacity: 0.9;">
                    Professional LoRA training with optimized defaults
                </p>
            </div>
            ''')
            
            # Configuration presets section
            preset_components = self._create_presets_section()
            
            # Use existing working interface components
            with gr.Row():
                with gr.Column(scale=2):
                    # File upload (from existing interface)
                    (images_zip, captions_json, load_dataset_btn) = self.base_interface.ui_components.create_file_upload_section()
                    
                    # Dataset preview (from existing interface)
                    (dataset_info, preview_gallery, matching_details) = self.base_interface.ui_components.create_dataset_preview_section()
                    
                    # Training parameters with world-class enhancements
                    with gr.Column(visible=False, elem_classes="training-section enhanced") as training_section:
                        
                        # Basic parameters (enhanced with research defaults)
                        basic_params = self._create_enhanced_basic_params()
                        
                        # Model settings (from existing)
                        model_settings = self.base_interface.ui_components.create_model_settings()
                        
                        # Advanced settings
                        with gr.Accordion("⚙️ Advanced Professional Settings", open=False):
                            advanced_settings = self.base_interface.ui_components.create_advanced_settings()
                            sampling_settings = self.base_interface.ui_components.create_sampling_settings()
                            dataset_settings = self.base_interface.ui_components.create_dataset_settings()
                            ema_settings = self.base_interface.ui_components.create_ema_settings()
                        
                        # Sample prompts
                        sample_prompts = self.base_interface.ui_components.create_sample_prompts()
                        
                        # Expert mode (from existing)
                        expert_mode = self.base_interface.ui_components.create_expert_mode()
                        
                        # Action buttons with world-class enhancements
                        action_buttons = self._create_world_class_action_buttons()
                
                with gr.Column(scale=1):
                    # World-class status section
                    status_components = self._create_world_class_status()
                    
                    # Verification section (from existing)
                    verification_section = self.base_interface.ui_components.create_verification_section()
            
            # Setup state management
            matched_data_state = gr.State([])
            matching_report_state = gr.State({})
            
            # Setup event handlers using existing working patterns
            self._setup_world_class_handlers(
                # Upload components
                images_zip, captions_json, load_dataset_btn,
                # Display components
                dataset_info, preview_gallery, matching_details,
                # Parameter components
                basic_params, model_settings, advanced_settings, sampling_settings, 
                dataset_settings, ema_settings, sample_prompts, expert_mode,
                # Action components
                action_buttons, status_components, verification_section,
                training_section,
                # States
                matched_data_state, matching_report_state,
                # Presets
                preset_components
            )
        
        return app
    
    def _create_presets_section(self) -> Dict[str, gr.Component]:
        """Create simplified quick-start presets section"""
        components = {}
        
        with gr.Column(elem_classes="preset-section"):
            gr.HTML('''
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5em; border-radius: 10px; margin-bottom: 1.5em;">
                <h3 style="color: white; margin: 0 0 0.5em 0; font-size: 1.3em;">⚡ Quick Start Templates</h3>
                <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 0.95em;">Choose a template optimized for your training type</p>
            </div>
            ''')
            
            preset_descriptions = {
                "👤 Character/Person": "Best for faces and people • 1500 steps • Rank 32",
                "🎨 Art Style": "Best for artistic styles • 1000 steps • Rank 16",
                "🏃 Quick Test": "Fast testing • 500 steps • Low resources",
                "� Maximum Quality": "Best quality, slower • 3000 steps • Rank 64"
            }
            
            components['preset_selector'] = gr.Radio(
                label="Training Type",
                choices=list(preset_descriptions.keys()),
                value="👤 Character/Person",
                info="Select based on what you're training"
            )
            
            components['preset_description'] = gr.Markdown(
                value=f"✅ **{preset_descriptions['👤 Character/Person']}**",
                elem_classes="preset-description"
            )
            
            # Hidden button - auto-apply on selection
            components['apply_preset_btn'] = gr.Button("Apply", visible=False)
        
        return components
    
    def _create_enhanced_basic_params(self) -> Dict[str, gr.Component]:
        """Create simplified essential parameters"""
        components = {}
        
        with gr.Column(elem_classes="parameter-section enhanced"):
            gr.HTML('''
            <div style="background: rgba(102, 126, 234, 0.15); padding: 1em; border-left: 4px solid #667eea; border-radius: 6px; margin-bottom: 1em;">
                <h3 style="margin: 0 0 0.3em 0; color: #e5e7eb; font-size: 1.1em;">📝 Essential Settings</h3>
                <p style="margin: 0; color: #9ca3af; font-size: 0.9em;">Configure your model name and trigger word</p>
            </div>
            ''')
            
            with gr.Row():
                components['lora_name'] = gr.Textbox(
                    label="📦 LoRA Model Name",
                    placeholder="poco",
                    info="Unique identifier for your LoRA model"
                )
                components['concept_sentence'] = gr.Textbox(
                    label="🎯 Trigger Word/Phrase",
                    placeholder="pxco",
                    info="Word(s) to activate your LoRA in prompts"
                )
            
            # Auto-fix trigger formatting (enabled by default, less prominent)
            components['auto_fix_trigger_formatting'] = gr.Checkbox(
                label="✓ Auto-add commas after trigger words",
                value=True,
                info="Automatically formats: 'poco A man' → 'poco, A man'"
            )
            
            gr.HTML('''
            <div style="background: rgba(102, 126, 234, 0.15); padding: 1em; border-left: 4px solid #667eea; border-radius: 6px; margin: 1.5em 0 1em 0;">
                <h3 style="margin: 0 0 0.3em 0; color: #e5e7eb; font-size: 1.1em;">⚙️ Training Configuration</h3>
                <p style="margin: 0; color: #9ca3af; font-size: 0.9em;">Fine-tune these if needed (template values are optimized)</p>
            </div>
            ''')
            
            with gr.Row():
                components['steps'] = gr.Slider(
                    label="🔄 Training Steps",
                    minimum=100,
                    maximum=5000,
                    value=1500,
                    step=50,
                    info="Higher = better quality but longer training"
                )
                components['lr'] = gr.Number(
                    label="📈 Learning Rate",
                    value=0.0004,
                    precision=6,
                    info="Default: 0.0004 (recommended)"
                )
            
            with gr.Row():
                components['rank'] = gr.Slider(
                    label="🎚️ LoRA Rank",
                    minimum=4,
                    maximum=128,
                    value=32,
                    step=4,
                    info="Model capacity (32 = balanced)"
                )
                components['linear_alpha'] = gr.Slider(
                    label="⚖️ LoRA Alpha",
                    minimum=1,
                    maximum=128,
                    value=32,
                    step=1,
                    info="Should match Rank (auto-adjusted)"
                )
            
            # Simplified EMA toggle
            with gr.Row():
                components['use_ema'] = gr.Checkbox(
                    label="✓ Use EMA (Recommended for better quality)",
                    value=True,
                    info="Keeps smoother model weights during training"
                )
                components['ema_decay'] = gr.Number(
                    label="EMA Decay",
                    value=0.9999,
                    precision=4,
                    visible=False,  # Hidden - use default
                    info="Default: 0.9999 (optimal)"
                )
        
        return components
    
    def _create_world_class_action_buttons(self) -> Dict[str, gr.Component]:
        """Create simplified action buttons"""
        components = {}
        
        with gr.Column():
            with gr.Row():
                components['train_btn'] = gr.Button(
                    "🚀 Start Training",
                    variant="primary", 
                    size="lg",
                    elem_classes="start-training-btn"
                )
            
            gr.HTML('''
            <div style="margin-top: 0.5em; padding: 0.8em; background: rgba(33, 150, 243, 0.15); border-radius: 6px; border-left: 3px solid #2196f3;">
                <p style="margin: 0; color: #90caf9; font-size: 0.9em;">
                    💡 <strong>Tip:</strong> Use "Advanced Settings" below for custom configurations, or enable "Expert Mode" for full YAML control
                </p>
            </div>
            ''')
            
            components['yaml_section'] = gr.Column(visible=False)
        
        return components
    
    def _create_world_class_status(self) -> Dict[str, gr.Component]:
        """Create simplified status section"""
        components = {}
        
        with gr.Column(elem_classes="monitoring-section"):
            gr.HTML('''
            <div style="background: rgba(40, 167, 69, 0.15); padding: 1em; border-left: 4px solid #28a745; border-radius: 6px; margin-bottom: 1em;">
                <h3 style="margin: 0 0 0.3em 0; color: #e5e7eb; font-size: 1.1em;">📊 Training Status</h3>
            </div>
            ''')
            
            components['training_status'] = gr.Markdown(
                value="""
**Status**: Ready  
**Template**: Character/Person  

Upload your dataset to begin training.
                """,
                elem_classes="status-display"
            )
            
            components['validation_feedback'] = gr.Markdown(
                value="",
                elem_classes="validation-box",
                visible=False  # Hidden until needed
            )
        
        return components
    
    def _setup_world_class_handlers(self, *args):
        """Setup simplified event handlers"""
        # Unpack components
        (images_zip, captions_json, load_dataset_btn, dataset_info, preview_gallery, 
         matching_details, basic_params, model_settings, advanced_settings, 
         sampling_settings, dataset_settings, ema_settings, sample_prompts, 
         expert_mode, action_buttons, status_components, verification_section,
         training_section, matched_data_state, matching_report_state, preset_components) = args
        
        # Auto-apply preset on selection (no button needed)
        preset_components['preset_selector'].change(
            fn=self._apply_world_class_preset,
            inputs=[preset_components['preset_selector']],
            outputs=[
                basic_params['steps'], basic_params['lr'], basic_params['rank'], 
                basic_params['linear_alpha'], preset_components['preset_description'],
                status_components['validation_feedback']
            ]
        )
        
        # Auto-sync alpha with rank for simplicity
        basic_params['rank'].change(
            fn=lambda x: x,  # Match alpha to rank
            inputs=[basic_params['rank']],
            outputs=[basic_params['linear_alpha']]
        )
        
        # Dataset loading (use existing working method)
        load_dataset_btn.click(
            fn=self.base_interface.dataset_processor.extract_captioning_data,
            inputs=[images_zip, captions_json],
            outputs=[matched_data_state, dataset_info, preview_gallery, matching_report_state],
            show_progress=True
        ).then(
            fn=lambda x: gr.update(visible=len(x) > 0),
            inputs=[matched_data_state],
            outputs=[training_section]
        )
        
        # Training start (use existing working trainer with expert YAML support)  
        action_buttons['train_btn'].click(
            fn=self._start_world_class_training_wrapper,
            inputs=[
                basic_params['lora_name'], basic_params['concept_sentence'], matched_data_state,
                basic_params['steps'], basic_params['lr'], basic_params['rank'], 
                basic_params['linear_alpha'], model_settings['model_to_train'],
                model_settings['batch_size'], model_settings['gradient_accumulation_steps'],
                advanced_settings['optimizer'], basic_params['use_ema'], basic_params['ema_decay'],
                sampling_settings['guidance_scale'], sampling_settings['sample_steps'], sampling_settings['sample_every'],
                expert_mode['use_more_advanced_options'], expert_mode['more_advanced_options'], 
                status_components['training_status']
            ],
            outputs=[status_components['training_status']],
            show_progress=True
        )
        
        # Expert YAML override toggle
        expert_mode['use_more_advanced_options'].change(
            fn=self._toggle_expert_yaml_mode,
            inputs=[expert_mode['use_more_advanced_options']],
            outputs=[expert_mode['more_advanced_options']]
        )
    
    def _apply_world_class_preset(self, preset_name: str) -> tuple:
        """Apply simplified preset configuration"""
        # Simplified presets mapping
        presets_config = {
            "👤 Character/Person": {
                "steps": 1500,
                "lr": 0.0004,
                "rank": 32,
                "alpha": 32,
                "description": "Best for faces and people • 1500 steps • Rank 32"
            },
            "🎨 Art Style": {
                "steps": 1000,
                "lr": 0.0004,
                "rank": 16,
                "alpha": 16,
                "description": "Best for artistic styles • 1000 steps • Rank 16"
            },
            "🏃 Quick Test": {
                "steps": 500,
                "lr": 0.0005,
                "rank": 16,
                "alpha": 16,
                "description": "Fast testing • 500 steps • Low resources"
            },
            "💎 Maximum Quality": {
                "steps": 3000,
                "lr": 0.0002,
                "rank": 64,
                "alpha": 64,
                "description": "Best quality, slower • 3000 steps • Rank 64"
            }
        }
        
        if preset_name in presets_config:
            preset = presets_config[preset_name]
            description = f"✅ **{preset['description']}**"
            validation = ""  # Empty for cleaner UI
            
            return (
                preset["steps"],      # steps
                preset["lr"],         # lr
                preset["rank"],       # rank
                preset["alpha"],      # alpha
                description,          # preset description
                validation            # validation feedback (hidden)
            )
        
        # Default to Character
        return (1500, 0.0004, 32, 32, "✅ **Character/Person template**", "")
    
    def _toggle_expert_yaml_mode(self, use_expert_mode: bool):
        """Toggle expert YAML override visibility and populate with default config"""
        if use_expert_mode:
            default_yaml = self._get_default_yaml_config()
            return gr.update(visible=True, value=default_yaml)
        else:
            return gr.update(visible=False, value="")
    
    def _get_default_yaml_config(self) -> str:
        """Get simplified default YAML configuration for expert mode"""
        return """# Expert YAML Override
# This will merge with your UI settings
# Modify any parameter below or uncomment alternatives

---
job: extension
config:
  name: "my_lora_model"
  process:
    - type: 'sd_trainer'
      training_folder: "output"
      device: cuda:0
      # device: cuda:1  # Alternative: use second GPU
      
      # Network (LoRA) Settings
      network:
        type: "lora"
        linear: 32          # LoRA rank (4, 8, 16, 32, 64, 128)
        linear_alpha: 32    # LoRA alpha (usually matches rank)
        
      # Advanced: Target specific model layers (for focused training)
      network_kwargs:
        only_if_contains:
          - "transformer.single_transformer_blocks."  # Focus on transformer blocks
          # - "attn"                                  # Alternative: attention layers only
          # - "transformer"                           # Alternative: all transformer layers
      
      # Training Parameters
      train:
        batch_size: 1       # Increase if you have VRAM (2, 4, 8)
        steps: 1500
        gradient_accumulation_steps: 1  # Increase for effective larger batch (2, 4, 8)
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        
        noise_scheduler: "flowmatch"  # FLUX default
        # noise_scheduler: "ddpm"     # Alternative: for compatibility
        
        optimizer: "adamw"            # Default optimizer
        # optimizer: "adamw8bit"      # Alternative: save VRAM
        # optimizer: "prodigy"        # Alternative: adaptive LR (experimental)
        # optimizer: "lion"           # Alternative: faster convergence
        
        lr: 0.0004          # Learning rate (0.0001 - 0.001)
        
        dtype: bf16         # Training precision
        # dtype: fp16       # Alternative: if bf16 not supported
        # dtype: fp32       # Alternative: maximum quality (slow)
        
        # Learning Rate Scheduler (uncomment to use)
        # lr_scheduler: "constant"    # No schedule (default)
        # lr_scheduler: "cosine"      # Cosine annealing
        # lr_scheduler: "linear"      # Linear decay
        # warmup_steps: 100           # LR warmup steps
        
        # EMA (Recommended - Enabled by default)
        ema_config:
          use_ema: true
          ema_decay: 0.9999  # Higher = more smoothing (0.99 - 0.9999)
          
      # Model Configuration
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"  # Best quality
        # name_or_path: "black-forest-labs/FLUX.1-schnell"  # Alternative: faster
        is_flux: true
        quantize: false     # False = better quality
        # quantize: true    # True = save VRAM (8-bit)
        low_vram: false     # False = faster training
        # low_vram: true    # True = <16GB VRAM optimization
        
      # Dataset (folder_path auto-managed by UI)
      datasets:
        - caption_ext: "txt"
          # caption_ext: "caption"  # Alternative extension
          
          caption_dropout_rate: 0.05  # 5% no caption (0.0 - 0.3)
          shuffle_tokens: false       # False for characters
          # shuffle_tokens: true      # True for styles (randomize caption words)
          
          cache_latents_to_disk: true  # Cache for speed (recommended)
          resolution: [512, 768, 1024]  # Multi-resolution training
          # resolution: [1024]         # Single resolution (faster)
          # resolution: [512, 640, 768, 896, 1024, 1152]  # More resolutions
          
      # Sampling/Preview Configuration
      sample:
        sampler: "flowmatch"  # FLUX sampler
        # sampler: "euler"    # Alternative sampler
        
        sample_every: 250     # Generate preview every N steps
        width: 1024
        height: 1024
        # width: 512          # Alternative: faster previews
        # height: 512
        
        guidance_scale: 4     # Prompt adherence (1-20)
        sample_steps: 20      # Quality vs speed (10-50)
        seed: 42
        walk_seed: true       # Vary seed each sample
        # walk_seed: false    # Use same seed
        
        prompts:
          - "your custom prompt 1"
          - "your custom prompt 2"
          # Add more prompts as needed
      
      # Save Configuration
      save:
        dtype: float16        # Save precision
        # dtype: float32      # Alternative: higher quality, larger files
        
        save_every: 500       # Save checkpoint every N steps
        max_step_saves_to_keep: 3  # Keep last N checkpoints
        
        # push_to_hub: false  # Don't upload to Hugging Face
        # push_to_hub: true   # Auto-upload to Hugging Face

# Advanced Training Options (uncomment as needed):
# train:
#   skip_first_sample: true        # Don't generate sample at step 0
#   linear_timesteps: true         # Experimental: curved noise weighting
#   min_snr_gamma: 5.0            # Min-SNR weighting (research: 5.0)
#   max_grad_norm: 1.0            # Gradient clipping
#   noise_offset: 0.0357          # Add noise offset (0.03-0.1)

# Validation Set (uncomment to use separate validation images):
# validation:
#   folder_path: "path/to/validation/images"
#   validate_every: 500
#   num_validation_images: 4

# Advanced Memory Optimization (for very low VRAM):
# model:
#   cpu_offload_checkpointing: true
# train:
#   attention_slicing: true
#   vae_slicing: true
"""

    def _start_world_class_training_wrapper(self, lora_name: str, concept_sentence: str, 
                                          matched_data: List, steps: int, lr: float, 
                                          rank: int, alpha: int, model_type: str,
                                          batch_size: int, grad_acc: int, optimizer: str, 
                                          use_ema: bool, ema_decay: float,
                                          guidance_scale: float, sample_steps: int, sample_every: int,
                                          use_expert_mode: bool, expert_yaml: str,
                                          current_status: str) -> str:
        """Start training with simplified status messages"""
        try:
            if not matched_data:
                return "❌ **Error**: Please load a dataset first!"
            
            if not lora_name or not lora_name.strip():
                return "❌ **Error**: Please provide a LoRA model name!"
            
            # Use existing working trainer
            trainer = FluxLoRATrainer()
            
            if use_expert_mode and expert_yaml.strip():
                # Use expert YAML configuration
                try:
                    import yaml
                    expert_config = yaml.safe_load(expert_yaml)
                    
                    # Extract configuration from ai-toolkit structure
                    process_config = expert_config.get('config', {}).get('process', [{}])[0] if expert_config.get('config', {}).get('process') else {}
                    train_config = process_config.get('train', {})
                    network_config = process_config.get('network', {})
                    model_config = process_config.get('model', {})
                    sample_config = process_config.get('sample', {})
                    
                    # YAML overrides UI parameters
                    kwargs = {
                        'steps': train_config.get('steps', steps),
                        'lr': train_config.get('lr', lr),
                        'rank': network_config.get('linear', rank),
                        'linear_alpha': network_config.get('linear_alpha', alpha),
                        'model_to_train': model_type,
                        'batch_size': train_config.get('batch_size', batch_size),
                        'gradient_accumulation_steps': train_config.get('gradient_accumulation_steps', grad_acc),
                        'optimizer': train_config.get('optimizer', optimizer),
                        'gradient_checkpointing': train_config.get('gradient_checkpointing', True),
                        'noise_scheduler': train_config.get('noise_scheduler', 'flowmatch'),
                        'train_dtype': train_config.get('dtype', 'bf16'),
                        'quantize': model_config.get('quantize', False),
                        'low_vram': model_config.get('low_vram', False),
                        'use_ema': train_config.get('ema_config', {}).get('use_ema', use_ema),
                        'ema_decay': train_config.get('ema_config', {}).get('ema_decay', ema_decay),
                        'guidance_scale': sample_config.get('guidance_scale', guidance_scale),
                        'sample_steps': sample_config.get('sample_steps', sample_steps),
                        'sample_every': sample_config.get('sample_every', sample_every),
                        'expert_yaml_config': expert_config
                    }
                    
                    config_mode = "Expert YAML"
                    
                except yaml.YAMLError as e:
                    return f"❌ **Invalid YAML**: {str(e)}\n\nPlease check your YAML syntax."
                except Exception as e:
                    return f"❌ **Error parsing configuration**: {str(e)}"
            else:
                # Use standard UI parameters
                kwargs = {
                    'steps': steps,
                    'lr': lr, 
                    'rank': rank,
                    'linear_alpha': alpha,
                    'model_to_train': model_type,
                    'batch_size': batch_size,
                    'gradient_accumulation_steps': grad_acc,
                    'optimizer': optimizer,
                    'gradient_checkpointing': True,
                    'noise_scheduler': 'flowmatch',
                    'quantize': False,
                    'use_ema': use_ema,
                    'ema_decay': ema_decay,
                    'guidance_scale': guidance_scale,
                    'sample_steps': sample_steps,
                    'sample_every': sample_every,
                }
                
                config_mode = "UI Settings"
            
            # Start training
            result = trainer.start_training(
                lora_name=lora_name,
                concept_sentence=concept_sentence,
                matched_data=matched_data,
                **kwargs
            )
            
            return f"""
**Status**: ✅ Training Started

**Model**: {lora_name}  
**Trigger**: {concept_sentence}  
**Mode**: {config_mode}  
**Steps**: {kwargs['steps']} | **Rank**: {kwargs['rank']} | **LR**: {kwargs['lr']}  
**Sample Every**: {kwargs['sample_every']} steps  
**EMA**: {'✓ Enabled' if kwargs['use_ema'] else '✗ Disabled'}

{result}
            """
            
        except Exception as e:
            return f"❌ **Training failed**: {str(e)}"
    
    def launch(self, **kwargs):
        """Launch the interface"""
        interface = self.create_interface()
        
        # Default launch settings
        launch_kwargs = {
            "server_name": "0.0.0.0",
            "share": True,
            "show_error": True,
            "quiet": False,
            **kwargs
        }
        
        print("\n🌟 Launching FLUX LoRA Trainer...")
        print("🎯 Ready to train!")
        
        return interface.launch(**launch_kwargs)


# Factory function for easy import
def create_world_class_simple_interface() -> WorldClassSimpleInterface:
    """Create the FLUX LoRA trainer interface"""
    return WorldClassSimpleInterface()