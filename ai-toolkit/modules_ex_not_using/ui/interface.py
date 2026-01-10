"""
Main Gradio interface for Advanced FLUX LoRA Trainer
"""

import gradio as gr
from typing import Dict, Any

from .components import UIComponents
from .styles import UIStyles
from ..core.dataset_processor import DatasetProcessor
from ..training.trainer import FluxLoRATrainer


class GradioInterface:
    """Main Gradio interface class"""
    
    def __init__(self):
        self.trainer = FluxLoRATrainer()
        self.dataset_processor = DatasetProcessor()
        self.ui_components = UIComponents()
        self.ui_styles = UIStyles()
        
        # State variables
        self.matched_data_state = None
        self.matching_report_state = None
        
        # Component references
        self.components = {}
    
    def create_interface(self) -> gr.Blocks:
        """Create the complete Gradio interface"""
        
        with gr.Blocks(
            css=self.ui_styles.get_custom_css(),
            title="Advanced FLUX LoRA Trainer Pro"
        ) as app:
            
            # Header
            gr.HTML(self.ui_styles.get_header_html())
            
            # State components
            self.matched_data_state = gr.State([])
            self.matching_report_state = gr.State({})
            
            # Main interface layout
            with gr.Row():
                with gr.Column(scale=2):
                    self._create_left_column()
                with gr.Column(scale=1):
                    self._create_right_column()
            
            # Information section
            gr.HTML(self.ui_styles.get_info_section_html())
            
            # Setup event handlers
            self._setup_event_handlers()
        
        return app
    
    def _create_left_column(self):
        """Create left column with main controls"""
        
        # File upload section
        (self.components['images_zip'], 
         self.components['captions_json'], 
         self.components['load_dataset_btn']) = self.ui_components.create_file_upload_section()
        
        # Dataset preview section
        (self.components['dataset_info'], 
         self.components['preview_gallery'], 
         self.components['matching_details']) = self.ui_components.create_dataset_preview_section()
        
        # Training parameters
        with gr.Column(visible=False, elem_classes="training-section") as training_section:
            self.components['training_section'] = training_section
            
            # Basic parameters
            basic_params = self.ui_components.create_basic_training_params()
            self.components.update(basic_params)
            
            # Model settings
            model_settings = self.ui_components.create_model_settings()
            self.components.update(model_settings)
            
            # Advanced settings accordion
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                advanced_settings = self.ui_components.create_advanced_settings()
                self.components.update(advanced_settings)
                
                sampling_settings = self.ui_components.create_sampling_settings()
                self.components.update(sampling_settings)
                
                dataset_settings = self.ui_components.create_dataset_settings()
                self.components.update(dataset_settings)
                
                ema_settings = self.ui_components.create_ema_settings()
                self.components.update(ema_settings)
            
            # Sample prompts
            sample_prompts = self.ui_components.create_sample_prompts()
            self.components.update(sample_prompts)
            
            # Expert mode accordion
            with gr.Accordion("🧠 Expert Mode", open=False):
                expert_mode = self.ui_components.create_expert_mode()
                self.components.update(expert_mode)
            
            # Action buttons
            action_buttons = self.ui_components.create_action_buttons()
            self.components.update(action_buttons)
    
    def _create_right_column(self):
        """Create right column with status and verification"""
        
        # Status section
        status_section = self.ui_components.create_status_section()
        self.components.update(status_section)
        
        # Verification section
        verification_section = self.ui_components.create_verification_section()
        self.components.update(verification_section)
    
    def _setup_event_handlers(self):
        """Setup all event handlers"""
        
        # Dataset loading
        self.components['load_dataset_btn'].click(
            fn=self.dataset_processor.extract_captioning_data,
            inputs=[
                self.components['images_zip'],
                self.components['captions_json']
            ],
            outputs=[
                self.matched_data_state,
                self.components['dataset_info'],
                self.components['training_section'],
                self.matching_report_state
            ]
        ).then(
            fn=self.dataset_processor.create_dataset_preview,
            inputs=[self.matched_data_state, self.matching_report_state],
            outputs=[
                self.components['preview_gallery'],
                self.components['matching_details']
            ]
        ).then(
            fn=self._enable_training_components,
            outputs=[
                self.components['train_btn'],
                self.components['verification_section']
            ]
        ).then(
            fn=self.dataset_processor.preview_caption_verification,
            inputs=[self.matched_data_state],
            outputs=[self.components['caption_verification_report']]
        )
        
        # Training start
        self.components['train_btn'].click(
            fn=self._start_training_wrapper,
            inputs=self._get_training_inputs(),
            outputs=[self.components['training_status']]
        )
        
        # Expert mode toggle
        self.components['use_more_advanced_options'].change(
            fn=self._toggle_expert_mode,
            inputs=[self.components['use_more_advanced_options']],
            outputs=[self.components['more_advanced_options']]
        )
    
    def _get_training_inputs(self) -> list:
        """Get all training input components"""
        return [
            # Basic parameters
            self.components['lora_name'],
            self.components['concept_sentence'],
            self.components['steps'],
            self.components['lr'],
            self.components['rank'],
            self.components['linear_alpha'],
            
            # Model settings
            self.components['model_to_train'],
            self.components['low_vram'],
            self.components['batch_size'],
            self.components['gradient_accumulation_steps'],
            
            # Advanced settings
            self.components['optimizer'],
            self.components['save_dtype'],
            self.components['train_dtype'],
            self.components['guidance_scale'],
            self.components['sample_steps'],
            self.components['sample_every'],
            self.components['caption_dropout_rate'],
            self.components['resolution_512'],
            self.components['resolution_768'],
            self.components['resolution_1024'],
            self.components['quantize'],
            self.components['gradient_checkpointing'],
            self.components['noise_scheduler'],
            
            # EMA settings
            self.components['use_ema'],
            self.components['ema_decay'],
            
            # State
            self.matched_data_state,
            
            # Sample prompts
            self.components['sample_1'],
            self.components['sample_2'],
            self.components['sample_3'],
            
            # Dataset settings
            self.components['trigger_position'],
            
            # Expert mode
            self.components['use_more_advanced_options'],
            self.components['more_advanced_options']
        ]
    
    def _start_training_wrapper(self, *args):
        """Wrapper for training start with proper parameter mapping"""
        
        # Map arguments to parameter names
        param_names = [
            'lora_name', 'concept_sentence', 'steps', 'lr', 'rank', 'linear_alpha',
            'model_to_train', 'low_vram', 'batch_size', 'gradient_accumulation_steps',
            'optimizer', 'save_dtype', 'train_dtype', 'guidance_scale', 'sample_steps',
            'sample_every', 'caption_dropout_rate', 'resolution_512', 'resolution_768',
            'resolution_1024', 'quantize', 'gradient_checkpointing', 'noise_scheduler',
            'use_ema', 'ema_decay', 'matched_data', 'sample_1', 'sample_2', 'sample_3',
            'trigger_position', 'use_more_advanced_options', 'more_advanced_options'
        ]
        
        # Create kwargs from args
        kwargs = dict(zip(param_names, args))
        
        # Extract special parameters from kwargs
        matched_data = kwargs.pop('matched_data')
        lora_name = kwargs.pop('lora_name')
        concept_sentence = kwargs.pop('concept_sentence')
        
        # Start training
        return self.trainer.start_training(
            lora_name=lora_name,
            concept_sentence=concept_sentence,
            matched_data=matched_data,
            **kwargs
        )
    
    def _enable_training_components(self):
        """Enable training components after dataset load"""
        return [
            gr.update(visible=True),  # train_btn
            gr.update(visible=True)   # verification_section
        ]
    
    def _toggle_expert_mode(self, use_expert_mode: bool):
        """Toggle expert mode visibility"""
        return gr.update(visible=use_expert_mode)
    
    def launch(self, **kwargs):
        """Launch the interface"""
        app = self.create_interface()
        
        # Default launch parameters
        launch_params = {
            'server_name': '0.0.0.0',
            'server_port': 7861,
            'share': False,
            'show_error': True,
            'quiet': False
        }
        
        # Update with provided parameters
        launch_params.update(kwargs)
        
        # Try multiple ports if the default is taken
        for port in [7861, 7862, 7863, 7864, 7865]:
            try:
                launch_params['server_port'] = port
                print(f"🚀 Launching Advanced FLUX LoRA Trainer on port {port}...")
                print(f"🌐 Open in browser: http://localhost:{port}")
                app.launch(**launch_params)
                break
            except OSError as e:
                if "port" in str(e).lower() or "address" in str(e).lower():
                    print(f"⚠️ Port {port} is not available, trying next port...")
                    continue
                else:
                    raise e
        else:
            print("❌ Could not find an available port. Please close other applications and try again.")