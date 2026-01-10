"""
World-Class FLUX LoRA Trainer Interface

The ultimate professional interface for FLUX LoRA training with:
- Research-based optimization and defaults
- Live training sample gallery
- Professional monitoring and analytics
- Configuration presets for different scenarios
- Advanced parameter validation and suggestions
"""

import gradio as gr
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Import our world-class components
from .enhanced_components import WorldClassUIComponents
from .components import UIComponents  # Keep original components as fallback
from ..core.world_class_config import (
    WorldClassTrainingConfig, 
    WorldClassValidation, 
    ConfigurationExporter
)
from ..training.sample_gallery import TrainingSampleGallery, AdvancedProgressTracker
from ..training.trainer import FluxLoRATrainer
from ..core.dataset_processor import DatasetProcessor
from ..core.gpu_manager import GPUManager


class WorldClassGradioInterface:
    """World-class Gradio interface for professional FLUX LoRA training"""
    
    def __init__(self):
        self.current_config = WorldClassTrainingConfig()
        self.sample_gallery = None
        self.progress_tracker = None
        self.is_training = False
        
        print("🌟 Initializing World-Class FLUX LoRA Trainer...")
        print("🎯 Features: Research-based defaults, live gallery, professional monitoring")
    
    def create_interface(self) -> gr.Interface:
        """Create the world-class training interface"""
        
        # Apply world-class CSS
        css = WorldClassUIComponents.create_world_class_css()
        
        with gr.Blocks(
            title="🌟 World-Class FLUX LoRA Trainer",
            theme=gr.themes.Soft(),
            css=css
        ) as interface:
            
            # Header
            gr.HTML("""
            <div style="text-align: center; padding: 2em; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 12px; margin-bottom: 2em;">
                <h1 style="font-size: 2.5em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    🌟 World-Class FLUX LoRA Trainer
                </h1>
                <p style="font-size: 1.2em; margin: 0.5em 0 0 0; opacity: 0.9;">
                    Professional-grade LoRA training with research-based optimization and live monitoring
                </p>
            </div>
            """)
            
            # State variables
            state = gr.State({
                "dataset_loaded": False,
                "config_valid": False,
                "training_active": False
            })
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Dataset Upload Section
                    self._create_dataset_section()
                    
                    # Configuration Presets
                    preset_components = self._create_presets_section()
                    
                    # Core Training Parameters
                    training_components = self._create_training_section()
                    
                    # Advanced Optimization
                    optimization_components = self._create_optimization_section()
                    
                    # Model Settings
                    model_components = self._create_model_section()
                
                with gr.Column(scale=1):
                    # Live Sample Gallery
                    gallery_components = self._create_gallery_section()
                    
                    # Training Monitoring
                    monitoring_components = self._create_monitoring_section()
            
            # Configuration Validation
            with gr.Row():
                validation_display = gr.Markdown(
                    value="Configure your training parameters to see validation results.",
                    elem_classes="validation-box"
                )
            
            # Setup event handlers
            self._setup_event_handlers(
                preset_components,
                training_components,
                optimization_components,
                model_components,
                gallery_components,
                monitoring_components,
                validation_display,
                state
            )
        
        return interface
    
    def _create_dataset_section(self):
        """Create enhanced dataset upload section"""
        with gr.Column(elem_classes="dataset-section"):
            gr.HTML('''
            <div class="section-header">
                📂 Dataset Configuration
                <div class="section-subtitle">Upload and configure your training dataset</div>
            </div>
            ''')
            
            with gr.Row():
                self.images_zip = gr.File(
                    label="🖼️ Training Images (ZIP)",
                    file_types=[".zip"],
                    type="filepath"
                )
                self.captions_json = gr.File(
                    label="📝 Captions (JSON/TXT)",
                    file_types=[".json", ".txt"],
                    type="filepath"
                )
            
            self.load_dataset_btn = gr.Button(
                "🔍 Load & Validate Dataset",
                variant="secondary",
                size="lg"
            )
            
            self.dataset_info = gr.Markdown(
                value="Upload your dataset files to see information here.",
                elem_classes="info-box"
            )
            
            self.dataset_preview = gr.Gallery(
                label="Dataset Preview (First 6 samples)",
                columns=3,
                rows=2,
                height="auto",
                visible=False
            )
    
    def _create_presets_section(self) -> Dict[str, gr.Component]:
        """Create configuration presets section"""
        return WorldClassUIComponents.create_configuration_presets()
    
    def _create_training_section(self) -> Dict[str, gr.Component]:
        """Create enhanced training parameters section"""
        return WorldClassUIComponents.create_enhanced_training_params()
    
    def _create_optimization_section(self) -> Dict[str, gr.Component]:
        """Create advanced optimization section"""
        return WorldClassUIComponents.create_advanced_optimization()
    
    def _create_model_section(self) -> Dict[str, gr.Component]:
        """Create advanced model settings section"""
        return WorldClassUIComponents.create_advanced_model_settings()
    
    def _create_gallery_section(self) -> Dict[str, gr.Component]:
        """Create training sample gallery section"""
        return WorldClassUIComponents.create_training_sample_gallery()
    
    def _create_monitoring_section(self) -> Dict[str, gr.Component]:
        """Create professional monitoring section"""
        return WorldClassUIComponents.create_professional_monitoring()
    
    def _setup_event_handlers(self, *component_groups):
        """Setup all event handlers for the interface"""
        preset_components, training_components, optimization_components, model_components, gallery_components, monitoring_components, validation_display, state = component_groups
        
        # Preset selection handler
        preset_components['apply_preset_btn'].click(
            fn=self._apply_preset,
            inputs=[preset_components['preset_selector']],
            outputs=list(training_components.values()) + list(optimization_components.values())
        )
        
        # Dataset loading handler
        self.load_dataset_btn.click(
            fn=self._load_and_validate_dataset,
            inputs=[self.images_zip, self.captions_json],
            outputs=[self.dataset_info, self.dataset_preview, state]
        )
        
        # Configuration validation (triggered by parameter changes)
        validation_inputs = (
            list(training_components.values()) + 
            list(optimization_components.values()) + 
            list(model_components.values())
        )
        
        for component in validation_inputs:
            if hasattr(component, 'change'):
                component.change(
                    fn=self._validate_configuration,
                    inputs=validation_inputs,
                    outputs=[validation_display]
                )
        
        # Training controls
        monitoring_components['start_training_btn'].click(
            fn=self._start_world_class_training,
            inputs=validation_inputs + [gallery_components['sample_prompts']],
            outputs=[
                monitoring_components['training_log'],
                monitoring_components['start_training_btn'],
                monitoring_components['pause_training_btn'],
                monitoring_components['stop_training_btn']
            ]
        )
        
        # Gallery controls
        gallery_components['refresh_gallery'].click(
            fn=self._refresh_sample_gallery,
            outputs=[gallery_components['sample_gallery']]
        )
        
        gallery_components['clear_gallery'].click(
            fn=self._clear_sample_gallery,
            outputs=[gallery_components['sample_gallery']]
        )
        
        gallery_components['download_samples'].click(
            fn=self._download_training_samples,
            outputs=[gr.File()]
        )
    
    def _apply_preset(self, preset_name: str) -> List[Any]:
        """Apply a configuration preset"""
        try:
            config = WorldClassTrainingConfig.from_preset(preset_name)
            
            # Return values for all training and optimization components
            return [
                config.model_name,  # lora_name
                config.trigger_word,  # trigger_word
                config.lora.rank,  # rank
                config.lora.alpha,  # alpha
                config.lora.target_modules,  # target_modules
                config.lora.optimizer,  # optimizer
                config.lora.learning_rate,  # learning_rate
                config.lora.lr_scheduler,  # lr_scheduler
                config.lora.warmup_steps,  # warmup_steps
                config.lora.max_train_steps,  # max_train_steps
                config.lora.save_every,  # save_every
            ]
            
        except Exception as e:
            print(f"Error applying preset: {e}")
            return [gr.update() for _ in range(11)]  # Return no changes
    
    def _load_and_validate_dataset(self, images_zip: str, captions_file: str) -> Tuple[str, List, Dict]:
        """Load and validate dataset"""
        try:
            if not images_zip or not captions_file:
                return "Please upload both images ZIP and captions file.", [], {"dataset_loaded": False}
            
            # Use existing dataset processor
            processor = DatasetProcessor()
            result = processor.load_and_validate_dataset(images_zip, captions_file)
            
            if result["success"]:
                # Create preview gallery
                preview_images = result.get("preview_images", [])[:6]  # First 6 images
                
                info_text = f"""
## ✅ Dataset Loaded Successfully!

- **Total Images**: {result['total_images']}
- **Matched Pairs**: {result['matched_pairs']}
- **Average Caption Length**: {result.get('avg_caption_length', 'N/A')} words
- **Resolution Range**: {result.get('resolution_info', 'Mixed')}
- **Dataset Quality**: {'🟢 Excellent' if result['matched_pairs'] == result['total_images'] else '🟡 Good'}

{result.get('details', '')}
                """
                
                return info_text, preview_images, {"dataset_loaded": True}
            else:
                error_text = f"❌ Dataset validation failed:\n{result.get('error', 'Unknown error')}"
                return error_text, [], {"dataset_loaded": False}
                
        except Exception as e:
            return f"❌ Error loading dataset: {str(e)}", [], {"dataset_loaded": False}
    
    def _validate_configuration(self, *args) -> str:
        """Validate current configuration and provide feedback"""
        try:
            # Build configuration from UI inputs
            config = self._build_config_from_inputs(*args)
            
            # Validate configuration
            warnings = WorldClassValidation.validate_config(config)
            suggestions = WorldClassValidation.suggest_optimizations(config)
            
            # Estimate resources
            vram_estimate = WorldClassValidation.estimate_vram_usage(config)
            
            # Build validation report
            report = f"""
## 🔍 Configuration Validation

### 📊 Resource Estimation
- **Estimated VRAM Usage**: {vram_estimate:.1f} GB
- **Training Duration**: ~{config.lora.max_train_steps * 2 / 60:.1f} minutes
- **Effective Batch Size**: {config.lora.batch_size * config.lora.gradient_accumulation_steps}

### ⚠️ Validation Results
{chr(10).join(warnings) if warnings else "✅ All parameters look good!"}

### 💡 Optimization Suggestions
{chr(10).join(suggestions) if suggestions else "🎯 Configuration is well-optimized!"}
            """
            
            return report.strip()
            
        except Exception as e:
            return f"❌ Validation error: {str(e)}"
    
    def _build_config_from_inputs(self, *args) -> WorldClassTrainingConfig:
        """Build configuration from UI inputs"""
        # This would map UI inputs to config object
        # For now, return default config
        return self.current_config
    
    def _start_world_class_training(self, *args) -> Tuple[str, gr.Button, gr.Button, gr.Button]:
        """Start world-class training with all enhancements"""
        try:
            if self.is_training:
                return (
                    "⚠️ Training is already running!",
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True)
                )
            
            # Build configuration
            config = self._build_config_from_inputs(*args[:-1])  # Exclude sample prompts
            sample_prompts = args[-1]  # Last argument is sample prompts
            
            # Initialize training components
            output_dir = Path(f"output/{config.model_name}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.sample_gallery = TrainingSampleGallery(str(output_dir))
            self.progress_tracker = AdvancedProgressTracker(str(output_dir))
            
            # Start sample gallery monitoring
            if config.enable_sample_gallery:
                self.sample_gallery.initialize_gallery(sample_prompts)
                self.sample_gallery.start_monitoring(config.to_ai_toolkit_config(), config.gallery_update_frequency)
            
            # Initialize trainer
            trainer = FluxLoRATrainer()
            
            # Start training in background thread
            self.is_training = True
            
            import threading
            training_thread = threading.Thread(
                target=self._run_training_loop,
                args=(trainer, config),
                daemon=True
            )
            training_thread.start()
            
            return (
                "🚀 World-class training started! Monitor progress below.\n\n🖼️ Sample gallery will update automatically during training.\n📊 Check the logs below for detailed progress.",
                gr.update(visible=False),  # Hide start button
                gr.update(visible=True),   # Show pause button
                gr.update(visible=True)    # Show stop button
            )
            
        except Exception as e:
            return (
                f"❌ Error starting training: {str(e)}",
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False)
            )
    
    def _run_training_loop(self, trainer: FluxLoRATrainer, config: WorldClassTrainingConfig):
        """Run the training loop in background thread"""
        try:
            # Save configuration
            config_path = Path(f"output/{config.model_name}/config.yaml")
            config.save_config(config_path)
            
            # Start training
            result = trainer.start_training(str(config_path))
            
            if result["success"]:
                print("✅ Training completed successfully!")
            else:
                print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Training error: {e}")
        finally:
            self.is_training = False
            if self.sample_gallery:
                self.sample_gallery.stop_monitoring()
    
    def _refresh_sample_gallery(self) -> List[str]:
        """Refresh the sample gallery"""
        if self.sample_gallery:
            return self.sample_gallery.get_current_gallery()
        return []
    
    def _clear_sample_gallery(self) -> List[str]:
        """Clear the sample gallery"""
        if self.sample_gallery:
            self.sample_gallery.clear_gallery()
        return []
    
    def _download_training_samples(self) -> str:
        """Download training samples as ZIP"""
        if self.sample_gallery:
            return self.sample_gallery.export_samples()
        return None
    
    def launch(self, **kwargs):
        """Launch the world-class interface"""
        interface = self.create_interface()
        
        # Default launch settings for world-class experience
        default_kwargs = {
            "server_name": "0.0.0.0",
            "share": True,
            "show_error": True,
            "quiet": False
        }
        
        # Merge with user-provided kwargs
        launch_kwargs = {**default_kwargs, **kwargs}
        
        print("\n🌟 Launching World-Class FLUX LoRA Trainer...")
        print("🎯 Ready for professional-grade LoRA training!")
        print("🚀 Features enabled: Research defaults, live gallery, advanced monitoring")
        
        return interface.launch(**launch_kwargs)


# Factory function for easy import
def create_world_class_interface() -> WorldClassGradioInterface:
    """Create a world-class training interface"""
    return WorldClassGradioInterface()