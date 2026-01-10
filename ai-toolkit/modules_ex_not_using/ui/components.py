"""
UI components for the Advanced FLUX LoRA Trainer interface
"""

import gradio as gr
from typing import Dict, Any, List, Tuple


class UIComponents:
    """UI component factory for the trainer interface"""
    
    @staticmethod
    def create_file_upload_section() -> Tuple[gr.File, gr.File, gr.Button]:
        """Create file upload section"""
        with gr.Column(elem_classes="upload-section"):
            gr.HTML('<div class="section-header">📁 Dataset Files</div>')
            
            with gr.Row():
                images_zip = gr.File(
                    label="🖼️ Images ZIP File",
                    file_types=[".zip"],
                    type="filepath"
                )
                captions_json = gr.File(
                    label="📝 Captions JSON File", 
                    file_types=[".json"],
                    type="filepath"
                )
            
            load_dataset_btn = gr.Button(
                "🔄 Load Dataset & Verify Matching",
                variant="secondary",
                size="lg"
            )
        
        return images_zip, captions_json, load_dataset_btn
    
    @staticmethod
    def create_dataset_preview_section() -> Tuple[gr.Markdown, gr.Gallery, gr.Markdown]:
        """Create dataset preview section"""
        with gr.Column(visible=False, elem_classes="dataset-info") as dataset_section:
            gr.HTML('<div class="section-header">📊 Dataset Information</div>')
            
            dataset_info = gr.Markdown(
                value="Upload your dataset files to see information here.",
                elem_classes="info-box"
            )
            
            gr.HTML('<div class="section-header">🖼️ Preview Gallery</div>')
            preview_gallery = gr.Gallery(
                label="First 6 Image-Caption Matches",
                columns=3,
                rows=2,
                height="auto",
                show_label=True
            )
            
            matching_details = gr.Markdown(
                value="Detailed matching report will appear here.",
                elem_classes="info-box"
            )
        
        return dataset_info, preview_gallery, matching_details
    
    @staticmethod
    def create_basic_training_params() -> Dict[str, gr.Component]:
        """Create basic training parameters section"""
        components = {}
        
        with gr.Column(elem_classes="parameter-section"):
            gr.HTML('<div class="section-header">🎯 Basic Training Parameters</div>')
            
            with gr.Row():
                components['lora_name'] = gr.Textbox(
                    label="🏷️ LoRA Name",
                    placeholder="e.g., my-awesome-style",
                    info="Unique name for your LoRA model"
                )
                components['concept_sentence'] = gr.Textbox(
                    label="🎯 Trigger Word/Phrase",
                  #  placeholder="e.g., TOK, mycharacter, mystyle",
                    info="Word/phrase to trigger your LoRA"
                )
            
            with gr.Row():
                components['steps'] = gr.Slider(
                    label="🔄 Training Steps",
                    minimum=100,
                    maximum=10000,
                    value=1000,
                    step=50,
                    interactive=True,
                    info="Number of training iterations"
                )
                components['lr'] = gr.Number(
                    label="📈 Learning Rate",
                    value=0.0004,
                    precision=6,
                    interactive=True,
                    info="Learning rate (1e-4 recommended)"
                )
            
            with gr.Row():
                components['rank'] = gr.Slider(
                    label="🧬 LoRA Rank",
                    minimum=4,
                    maximum=128,
                    value=16,
                    step=4,
                    interactive=True,
                    info="Higher = more capacity, slower training"
                )
                components['linear_alpha'] = gr.Slider(
                    label="🎯 LoRA Alpha",
                    minimum=1,
                    maximum=128,
                    value=16,
                    step=1,
                    interactive=True,
                    info="Controls LoRA strength (usually = rank)"
                )
        
        return components
    
    @staticmethod
    def create_model_settings() -> Dict[str, gr.Component]:
        """Create model settings section"""
        components = {}
        
        with gr.Column(elem_classes="parameter-section"):
            gr.HTML('<div class="section-header">🤖 Model Settings</div>')
            
            with gr.Row():
                components['model_to_train'] = gr.Radio(
                    label="🧠 FLUX Model",
                    choices=["dev", "schnell"],
                    value="dev",
                    info="dev: Higher quality, schnell: Faster generation"
                )
                components['low_vram'] = gr.Checkbox(
                    label="💾 Low VRAM Mode",
                    value=False,
                    info="Enable for GPUs with <24GB VRAM"
                )
            
            with gr.Row():
                components['batch_size'] = gr.Slider(
                    label="📦 Batch Size",
                    minimum=1,
                    maximum=8,
                    value=1,
                    step=1,
                    interactive=True,
                    info="Images per training step"
                )
                components['gradient_accumulation_steps'] = gr.Slider(
                    label="📈 Gradient Accumulation",
                    minimum=1,
                    maximum=16,
                    value=1,
                    step=1,
                    interactive=True,
                    info="Effective batch size multiplier"
                )
        
        return components
    
    @staticmethod
    def create_advanced_settings() -> Dict[str, gr.Component]:
        """Create advanced settings section"""
        components = {}
        
        with gr.Column(elem_classes="advanced-section"):
            gr.HTML('<div class="section-header">⚙️ Advanced Settings</div>')
            
            with gr.Row():
                components['optimizer'] = gr.Dropdown(
                    label="🔧 Optimizer",
                    choices=["adamw", "adamw8bit", "lion", "adafactor"],
                    value="adamw",
                    info="Training optimizer"
                )
                components['noise_scheduler'] = gr.Dropdown(
                    label="🔬 Noise Scheduler",
                    choices=["flowmatch", "ddpm", "dpm", "euler"],
                    value="flowmatch",
                    info="FLUX uses flowmatch"
                )
            
            with gr.Row():
                components['train_dtype'] = gr.Dropdown(
                    label="💾 Training Precision",
                    choices=["bf16", "fp16", "fp32"],
                    value="bf16",
                    info="Lower = faster, less memory"
                )
                components['save_dtype'] = gr.Dropdown(
                    label="💿 Save Precision",
                    choices=["float16", "float32", "bf16"],
                    value="float16",
                    info="Model save format"
                )
            
            with gr.Row():
                components['quantize'] = gr.Checkbox(
                    label="📊 Quantization",
                    value=False,
                    info="Reduce memory usage"
                )
                components['gradient_checkpointing'] = gr.Checkbox(
                    label="🔄 Gradient Checkpointing",
                    value=True,
                    info="Save memory at cost of speed"
                )
        
        return components
    
    @staticmethod
    def create_sampling_settings() -> Dict[str, gr.Component]:
        """Create sampling settings section"""
        components = {}
        
        with gr.Column(elem_classes="parameter-section"):
            gr.HTML('<div class="section-header">🎨 Sampling Settings</div>')
            
            with gr.Row():
                components['guidance_scale'] = gr.Slider(
                    label="🎯 Guidance Scale",
                    minimum=1.0,
                    maximum=20.0,
                    value=3.5,
                    step=0.5,
                    interactive=True,
                    info="How closely to follow prompts"
                )
                components['sample_steps'] = gr.Slider(
                    label="🔄 Sample Steps",
                    minimum=4,
                    maximum=100,
                    value=28,
                    step=2,
                    interactive=True,
                    info="Quality vs speed tradeoff"
                )
            
            components['sample_every'] = gr.Slider(
                label="📸 Sample Every N Steps",
                minimum=50,
                maximum=1000,
                value=150,
                step=50,
                interactive=True,
                info="How often to generate samples"
            )
        
        return components
    
    @staticmethod
    def create_dataset_settings() -> Dict[str, gr.Component]:
        """Create dataset settings section"""
        components = {}
        
        with gr.Column(elem_classes="parameter-section"):
            gr.HTML('<div class="section-header">📊 Dataset Settings</div>')
            
            components['caption_dropout_rate'] = gr.Slider(
                label="📝 Caption Dropout Rate",
                minimum=0.0,
                maximum=0.5,
                value=0.05,
                step=0.05,
                interactive=True,
                info="Randomly drop captions for unconditional training"
            )
            
            gr.HTML('<div style="margin-top: 15px;"><strong>📐 Training Resolutions</strong></div>')
            with gr.Row():
                components['resolution_512'] = gr.Checkbox(
                    label="512px",
                    value=True,  # ✅ Default enabled for multi-resolution training
                    info="Include 512x512 resolution"
                )
                components['resolution_768'] = gr.Checkbox(
                    label="768px",
                    value=True,  # ✅ Default enabled for multi-resolution training
                    info="Include 768x768 resolution"
                )
                components['resolution_1024'] = gr.Checkbox(
                    label="1024px",
                    value=True,  # ✅ Default enabled for multi-resolution training
                    info="Include 1024x1024 resolution"
                )
            
            components['trigger_position'] = gr.Radio(
                label="🎯 Trigger Word Position",
                choices=["beginning", "end", "both"],
                value="beginning",
                info="Where to place trigger word in captions"
            )
        
        return components
    
    @staticmethod
    def create_ema_settings() -> Dict[str, gr.Component]:
        """Create EMA settings section"""
        components = {}
        
        with gr.Column(elem_classes="advanced-section"):
            gr.HTML('<div class="section-header">📈 EMA (Exponential Moving Average)</div>')
            
            with gr.Row():
                components['use_ema'] = gr.Checkbox(
                    label="✅ Enable EMA",
                    value=True,
                    interactive=True,
                    info="Smoother training, better quality"
                )
                components['ema_decay'] = gr.Slider(
                    label="📉 EMA Decay",
                    minimum=0.9,
                    maximum=0.9999,
                    value=0.99,
                    step=0.0001,
                    interactive=True,
                    info="Higher = smoother averaging"
                )
        
        return components
    
    @staticmethod
    def create_sample_prompts() -> Dict[str, gr.Component]:
        """Create sample prompts section"""
        components = {}
        
        with gr.Column(elem_classes="parameter-section"):
            gr.HTML('<div class="section-header">🎨 Sample Prompts</div>')
            
            components['sample_1'] = gr.Textbox(
                label="📝 Sample Prompt 1",
                placeholder="e.g., A photo of TOK in a beautiful landscape",
                info="First validation prompt"
            )
            components['sample_2'] = gr.Textbox(
                label="📝 Sample Prompt 2", 
                placeholder="e.g., Portrait of TOK, high quality",
                info="Second validation prompt"
            )
            components['sample_3'] = gr.Textbox(
                label="📝 Sample Prompt 3",
                placeholder="e.g., TOK in an artistic style",
                info="Third validation prompt"
            )
        
        return components
    
    @staticmethod
    def create_expert_mode() -> Dict[str, gr.Component]:
        """Create expert mode section"""
        components = {}
        
        with gr.Column(elem_classes="advanced-section"):
            gr.HTML('<div class="section-header">🧠 Expert Mode</div>')
            
            components['use_more_advanced_options'] = gr.Checkbox(
                label="🔬 Enable Expert YAML Override",
                value=False,
                info="Override config with custom YAML"
            )
            
            components['more_advanced_options'] = gr.Code(
                label="📝 Advanced Configuration (YAML)",
                language="yaml",
                lines=20,
                visible=False,
                value="# Advanced YAML configuration overrides\n# This will be merged with the generated config\n"
            )
        
        return components
    
    @staticmethod
    def create_action_buttons() -> Dict[str, gr.Component]:
        """Create action buttons section"""
        components = {}
        
        with gr.Column():
            components['train_btn'] = gr.Button(
                "🚀 Start Advanced FLUX Training",
                variant="primary",
                size="lg",
                elem_classes="train-btn",
                visible=False
            )
        
        return components
    
    @staticmethod
    def create_status_section() -> Dict[str, gr.Component]:
        """Create status and results section"""
        components = {}
        
        with gr.Column(elem_classes="status-container"):
            gr.HTML('<div class="section-header">📊 Training Status</div>')
            
            components['training_status'] = gr.Markdown(
                value="Upload dataset and configure parameters to start training.",
                elem_classes="info-box"
            )
        
        return components
    
    @staticmethod
    def create_verification_section() -> Dict[str, gr.Component]:
        """Create verification section"""
        components = {}
        
        with gr.Column(visible=False, elem_classes="dataset-info") as verification_section:
            gr.HTML('<div class="section-header">🔍 Caption Integration Verification</div>')
            
            components['caption_verification_report'] = gr.Markdown(
                value="Dataset verification will appear here.",
                elem_classes="info-box"
            )
        
        components['verification_section'] = verification_section
        
        return components