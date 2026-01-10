"""
Enhanced World-Class UI Components for Advanced FLUX LoRA Trainer

This module provides professional-grade UI components with research-based defaults,
advanced features, and a polished interface for world-class LoRA training.

Features:
- Research-based parameter defaults optimized for FLUX
- Configuration presets for different training scenarios
- Live training sample gallery with progress monitoring
- Professional parameter organization and tooltips
- Advanced scheduling and optimization options
- Real-time validation and feedback
"""

import gradio as gr
from typing import Dict, Any, List, Tuple, Optional
import os
from pathlib import Path


class WorldClassUIComponents:
    """World-class UI component factory with research-based enhancements"""
    
    # Research-based default values for FLUX LoRA training
    RESEARCH_DEFAULTS = {
        # Core LoRA parameters based on research and best practices
        "rank": 32,  # Higher than basic for better capacity
        "alpha": 32,  # Match rank for balanced scaling
        "learning_rate": 1e-4,  # Proven optimal for most cases
        "steps": 1500,  # Sweet spot for quality vs time
        
        # Advanced optimization
        "optimizer": "prodigy",  # Self-adaptive learning rate
        "lr_scheduler": "cosine_with_restarts",  # Better convergence
        "warmup_steps": 100,  # Gradual learning rate ramp
        
        # Memory and performance
        "batch_size": 1,  # Safe default for most GPUs
        "gradient_accumulation_steps": 4,  # Effective batch size of 4
        "gradient_checkpointing": True,  # Memory efficiency
        
        # Quality settings
        "mixed_precision": "bf16",  # Best for modern GPUs
        "save_precision": "float16",  # Compact but quality
        "guidance_scale": 3.5,  # FLUX optimal
        "sample_steps": 28,  # Good quality/speed balance
        
        # Sampling and validation
        "sample_every": 250,  # More frequent for monitoring
        "save_every": 500,  # Regular checkpoints
        "max_train_steps": 1500,  # Research-backed sweet spot
    }
    
    @staticmethod
    def create_configuration_presets() -> Dict[str, gr.Component]:
        """Create configuration preset selection"""
        components = {}
        
        with gr.Column(elem_classes="preset-section"):
            gr.HTML('''
            <div class="section-header">
                🎯 Configuration Presets
                <div class="section-subtitle">Research-optimized presets for different training scenarios</div>
            </div>
            ''')
            
            preset_descriptions = {
                "🎨 Style/Concept (Recommended)": "Optimized for style transfer and concept learning. Balanced quality and speed.",
                "👤 Character/Person": "Fine-tuned for character consistency and facial features. Higher rank for detail.",
                "🏃 Quick Test": "Fast training for testing datasets and concepts. Lower quality but rapid results.",
                "🔬 Research/Experimental": "Maximum quality settings for research and professional work. Slower but best results.",
                "💾 Low VRAM (<12GB)": "Memory-optimized settings for systems with limited GPU memory.",
                "⚡ Speed Optimized": "Fastest possible training while maintaining decent quality."
            }
            
            with gr.Row():
                components['preset_selector'] = gr.Dropdown(
                    label="🎯 Training Preset",
                    choices=list(preset_descriptions.keys()),
                    value="🎨 Style/Concept (Recommended)",
                    info="Choose a preset based on your training goal and hardware"
                )
                
                components['apply_preset_btn'] = gr.Button(
                    "Apply Preset",
                    variant="secondary",
                    size="sm"
                )
            
            components['preset_description'] = gr.Markdown(
                value=preset_descriptions["🎨 Style/Concept (Recommended)"],
                elem_classes="preset-description"
            )
        
        return components
    
    @staticmethod
    def create_enhanced_training_params() -> Dict[str, gr.Component]:
        """Create enhanced training parameters with research-based defaults"""
        components = {}
        
        with gr.Column(elem_classes="parameter-section enhanced"):
            gr.HTML('''
            <div class="section-header">
                🚀 Core Training Parameters
                <div class="section-subtitle">Research-optimized settings for professional results</div>
            </div>
            ''')
            
            with gr.Row():
                components['lora_name'] = gr.Textbox(
                    label="🏷️ LoRA Model Name",
                    placeholder="e.g., my-style-v1, character-name",
                    info="Unique identifier for your LoRA model",
                    scale=2
                )
                components['trigger_word'] = gr.Textbox(
                    label="🎯 Trigger Word/Phrase",
                    placeholder="e.g., TOK, [character_name], mystyle",
                    info="Word(s) to activate your LoRA in prompts",
                    scale=2
                )
            
            # Advanced LoRA architecture settings
            with gr.Accordion("🧬 LoRA Architecture", open=True):
                with gr.Row():
                    components['rank'] = gr.Slider(
                        label="🧠 LoRA Rank",
                        minimum=4,
                        maximum=128,
                        value=WorldClassUIComponents.RESEARCH_DEFAULTS["rank"],
                        step=4,
                        info="Model capacity: Higher = more detail, slower training. 32 optimal for most cases."
                    )
                    components['alpha'] = gr.Slider(
                        label="⚖️ LoRA Alpha",
                        minimum=1,
                        maximum=128,
                        value=WorldClassUIComponents.RESEARCH_DEFAULTS["alpha"],
                        step=1,
                        info="Scaling factor: Usually equal to rank for balanced training."
                    )
                
                with gr.Row():
                    components['target_modules'] = gr.CheckboxGroup(
                        label="🎯 Target Modules",
                        choices=["to_q", "to_k", "to_v", "to_out", "ff.net.0", "ff.net.2"],
                        value=["to_q", "to_k", "to_v", "to_out"],
                        info="Which attention layers to train. More = better quality, slower training."
                    )
        
        return components
    
    @staticmethod
    def create_advanced_optimization() -> Dict[str, gr.Component]:
        """Create advanced optimization settings"""
        components = {}
        
        with gr.Column(elem_classes="parameter-section"):
            gr.HTML('''
            <div class="section-header">
                ⚡ Advanced Optimization
                <div class="section-subtitle">Cutting-edge optimization techniques for superior results</div>
            </div>
            ''')
            
            with gr.Accordion("🎛️ Learning Configuration", open=True):
                with gr.Row():
                    components['optimizer'] = gr.Dropdown(
                        label="🔧 Optimizer",
                        choices=[
                            "prodigy",      # Self-adaptive, research favorite
                            "adamw8bit",    # Memory efficient
                            "adamw",        # Standard reliable
                            "lion",         # Fast convergence
                            "adafactor",    # Memory efficient
                            "dadaptation"   # Adaptive learning rate
                        ],
                        value="prodigy",
                        info="Prodigy: Self-adaptive learning rate (recommended)"
                    )
                    
                    components['learning_rate'] = gr.Number(
                        label="📈 Learning Rate",
                        value=WorldClassUIComponents.RESEARCH_DEFAULTS["learning_rate"],
                        precision=6,
                        info="1e-4 optimal for most cases. Prodigy auto-adjusts."
                    )
                
                with gr.Row():
                    components['lr_scheduler'] = gr.Dropdown(
                        label="📊 LR Scheduler",
                        choices=[
                            "cosine_with_restarts",  # Best convergence
                            "cosine",                # Smooth decay
                            "linear",                # Simple decay
                            "constant",              # No scheduling
                            "polynomial"             # Smooth polynomial
                        ],
                        value="cosine_with_restarts",
                        info="Cosine with restarts: Best for stable convergence"
                    )
                    
                    components['warmup_steps'] = gr.Slider(
                        label="🔥 Warmup Steps",
                        minimum=0,
                        maximum=500,
                        value=100,
                        step=10,
                        info="Gradual learning rate increase at start"
                    )
            
            with gr.Accordion("🎯 Training Dynamics", open=False):
                with gr.Row():
                    components['max_train_steps'] = gr.Slider(
                        label="🔄 Total Training Steps",
                        minimum=100,
                        maximum=5000,
                        value=WorldClassUIComponents.RESEARCH_DEFAULTS["steps"],
                        step=50,
                        info="Total training iterations. 1500 is the sweet spot for most cases."
                    )
                    
                    components['save_every'] = gr.Slider(
                        label="💾 Save Every N Steps",
                        minimum=100,
                        maximum=1000,
                        value=500,
                        step=50,
                        info="How often to save model checkpoints"
                    )
        
        return components
    
    @staticmethod
    def create_training_sample_gallery() -> Dict[str, gr.Component]:
        """Create live training sample gallery"""
        components = {}
        
        with gr.Column(elem_classes="sample-gallery-section"):
            gr.HTML('''
            <div class="section-header">
                🖼️ Live Training Gallery
                <div class="section-subtitle">Monitor your training progress with generated samples</div>
            </div>
            ''')
            
            with gr.Row():
                components['enable_gallery'] = gr.Checkbox(
                    label="📸 Enable Live Sample Generation",
                    value=True,
                    info="Generate samples during training to monitor progress"
                )
                
                components['gallery_update_freq'] = gr.Slider(
                    label="🔄 Update Every N Steps",
                    minimum=100,
                    maximum=1000,
                    value=250,
                    step=50,
                    info="How often to generate new samples"
                )
            
            # Sample prompts configuration
            with gr.Accordion("📝 Sample Prompts", open=True):
                components['sample_prompts'] = gr.DataFrame(
                    headers=["Prompt", "Weight", "Negative"],
                    datatype=["str", "number", "str"],
                    value=[
                        [f"A portrait of TOK, high quality, detailed", 1.0, "blurry, low quality"],
                        [f"TOK in a beautiful landscape, cinematic lighting", 1.0, "dark, unclear"],
                        [f"Close-up of TOK, professional photography", 1.0, "amateur, poor quality"],
                        [f"TOK, artistic style, creative composition", 1.0, "boring, generic"]
                    ],
                    label="Training Sample Prompts",
                    interactive=True,
                    wrap=True
                )
            
            # Gallery display
            components['sample_gallery'] = gr.Gallery(
                label="Training Progress Gallery",
                columns=4,
                rows=2,
                height="auto",
                show_label=True,
                elem_classes="training-gallery"
            )
            
            # Gallery controls
            with gr.Row():
                components['refresh_gallery'] = gr.Button(
                    "🔄 Refresh Gallery",
                    variant="secondary",
                    size="sm"
                )
                components['clear_gallery'] = gr.Button(
                    "🗑️ Clear Gallery",
                    variant="secondary", 
                    size="sm"
                )
                components['download_samples'] = gr.Button(
                    "📥 Download Samples",
                    variant="secondary",
                    size="sm"
                )
        
        return components
    
    @staticmethod
    def create_advanced_model_settings() -> Dict[str, gr.Component]:
        """Create advanced model and memory settings"""
        components = {}
        
        with gr.Column(elem_classes="parameter-section"):
            gr.HTML('''
            <div class="section-header">
                🤖 Advanced Model Configuration
                <div class="section-subtitle">Professional model settings and memory optimization</div>
            </div>
            ''')
            
            with gr.Accordion("🧠 Model Selection", open=True):
                with gr.Row():
                    components['flux_model'] = gr.Radio(
                        label="🚀 FLUX Model Version",
                        choices=[
                            ("FLUX.1 Dev (Recommended)", "dev"),
                            ("FLUX.1 Schnell (Faster)", "schnell")
                        ],
                        value="dev",
                        info="Dev: Higher quality. Schnell: Faster inference."
                    )
                    
                    components['precision_mode'] = gr.Radio(
                        label="🎯 Training Precision",
                        choices=[
                            ("BF16 (Recommended)", "bf16"),
                            ("FP16 (Memory Efficient)", "fp16"),
                            ("FP32 (Highest Quality)", "fp32")
                        ],
                        value="bf16",
                        info="BF16 optimal for modern GPUs"
                    )
            
            with gr.Accordion("💾 Memory Optimization", open=False):
                with gr.Row():
                    components['batch_size'] = gr.Slider(
                        label="📦 Batch Size",
                        minimum=1,
                        maximum=8,
                        value=1,
                        step=1,
                        info="Images per training step. Keep at 1 for most GPUs."
                    )
                    
                    components['gradient_accumulation'] = gr.Slider(
                        label="📈 Gradient Accumulation",
                        minimum=1,
                        maximum=16,
                        value=4,
                        step=1,
                        info="Effective batch size = batch_size × this value"
                    )
                
                with gr.Row():
                    components['gradient_checkpointing'] = gr.Checkbox(
                        label="✅ Gradient Checkpointing",
                        value=True,
                        info="Reduce memory usage (recommended)"
                    )
                    
                    components['low_vram_mode'] = gr.Checkbox(
                        label="💾 Low VRAM Mode",
                        value=False,
                        info="Enable for GPUs with <16GB VRAM"
                    )
                    
                    components['cpu_offload'] = gr.Checkbox(
                        label="🖥️ CPU Offloading",
                        value=False,
                        info="Offload to CPU when not in use"
                    )
        
        return components
    
    @staticmethod
    def create_professional_monitoring() -> Dict[str, gr.Component]:
        """Create professional training monitoring section"""
        components = {}
        
        with gr.Column(elem_classes="monitoring-section"):
            gr.HTML('''
            <div class="section-header">
                📊 Training Monitoring & Analytics
                <div class="section-subtitle">Professional-grade training supervision and analysis</div>
            </div>
            ''')
            
            # Live training status
            with gr.Accordion("🔴 Live Training Status", open=True):
                components['training_status'] = gr.Markdown(
                    value="**Status**: Ready to start training\n**Progress**: 0%\n**ETA**: Not started",
                    elem_classes="status-display"
                )
                
                with gr.Row():
                    components['current_step'] = gr.Number(
                        label="Current Step",
                        value=0,
                        interactive=False
                    )
                    components['current_loss'] = gr.Number(
                        label="Current Loss",
                        value=0.0,
                        precision=6,
                        interactive=False
                    )
                    components['learning_rate_display'] = gr.Number(
                        label="Learning Rate",
                        value=0.0,
                        precision=8,
                        interactive=False
                    )
                
                components['training_log'] = gr.Textbox(
                    label="Training Log",
                    lines=8,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True
                )
            
            # Training controls
            with gr.Row():
                components['start_training_btn'] = gr.Button(
                    "🚀 Start World-Class Training",
                    variant="primary",
                    size="lg",
                    elem_classes="start-training-btn"
                )
                components['pause_training_btn'] = gr.Button(
                    "⏸️ Pause Training",
                    variant="secondary",
                    visible=False
                )
                components['stop_training_btn'] = gr.Button(
                    "⏹️ Stop Training",
                    variant="stop",
                    visible=False
                )
            
            # Results and export
            with gr.Accordion("📋 Results & Export", open=False):
                components['training_results'] = gr.Markdown(
                    value="Training results will appear here after completion.",
                    elem_classes="results-box"
                )
                
                with gr.Row():
                    components['download_model'] = gr.Button(
                        "📥 Download LoRA",
                        variant="secondary",
                        visible=False
                    )
                    components['test_model'] = gr.Button(
                        "🧪 Test Model",
                        variant="secondary",
                        visible=False
                    )
                    components['share_model'] = gr.Button(
                        "🌐 Share to Hub",
                        variant="secondary",
                        visible=False
                    )
        
        return components
    
    @staticmethod
    def create_world_class_css() -> str:
        """Create world-class CSS styling with dark theme"""
        return """
        <style>
        /* World-class professional dark theme styling */
        .gradio-container {
            max-width: 1400px !important;
            margin: 0 auto;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e5e7eb !important;
            min-height: 100vh;
        }
        
        /* Override Gradio's default backgrounds */
        .block, .form, .wrap {
            background: rgba(31, 41, 55, 0.8) !important;
            color: #e5e7eb !important;
            border: 1px solid #374151 !important;
        }
        
        /* Text inputs and selects */
        input, textarea, select {
            background: #374151 !important;
            color: #e5e7eb !important;
            border: 1px solid #4b5563 !important;
        }
        
        /* Labels and text */
        label, .label, p, span, div {
            color: #e5e7eb !important;
        }
        
        .section-header {
            font-size: 1.5em;
            font-weight: bold;
            background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5em;
            padding: 0.5em 0;
            border-bottom: 2px solid #4b5563;
        }
        
        .section-subtitle {
            font-size: 0.9em;
            color: #9ca3af !important;
            font-weight: normal;
            -webkit-text-fill-color: #9ca3af !important;
            margin-top: 0.25em;
        }
        
        .preset-section {
            background: linear-gradient(135deg, rgba(96, 165, 250, 0.1), rgba(167, 139, 250, 0.1));
            border: 2px solid rgba(96, 165, 250, 0.3);
            border-radius: 12px;
            padding: 1.5em;
            margin-bottom: 1.5em;
        }
        
        .parameter-section {
            background: rgba(31, 41, 55, 0.9) !important;
            border: 1px solid #4b5563 !important;
            border-radius: 8px;
            padding: 1.5em;
            margin-bottom: 1em;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .parameter-section.enhanced {
            background: linear-gradient(135deg, rgba(31, 41, 55, 0.95), rgba(55, 65, 81, 0.95)) !important;
            border: 2px solid #3b82f6 !important;
        }
        
        .sample-gallery-section {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(168, 85, 247, 0.1));
            border: 2px solid #a855f7;
            border-radius: 12px;
            padding: 1.5em;
            margin: 1em 0;
        }
        
        .monitoring-section {
            background: linear-gradient(135deg, rgba(8, 145, 178, 0.1), rgba(14, 165, 233, 0.1));
            border: 2px solid #0891b2;
            border-radius: 12px;
            padding: 1.5em;
            margin: 1em 0;
        }
        
        .training-gallery img {
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            transition: transform 0.2s;
        }
        
        .training-gallery img:hover {
            transform: scale(1.05);
        }
        
        .start-training-btn {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
            color: white !important;
            font-weight: bold;
            font-size: 1.1em;
            padding: 0.75em 2em;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(16, 185, 129, 0.3);
        }
        
        .preset-description {
            background: rgba(31, 41, 55, 0.8) !important;
            border-left: 4px solid #60a5fa;
            padding: 1em;
            margin-top: 1em;
            border-radius: 0 8px 8px 0;
            color: #e5e7eb !important;
        }
        
        .results-box {
            background: rgba(31, 41, 55, 0.9) !important;
            border: 1px solid #0ea5e9;
            border-radius: 8px;
            padding: 1em;
            color: #e5e7eb !important;
        }
        
        /* Buttons */
        button {
            background: rgba(59, 130, 246, 0.9) !important;
            color: white !important;
            border: 1px solid #3b82f6 !important;
        }
        
        button:hover {
            background: rgba(59, 130, 246, 1) !important;
        }
        
        /* Dropdown and select styling */
        .dropdown {
            background: #374151 !important;
            color: #e5e7eb !important;
        }
        
        /* File upload areas */
        .file-upload {
            background: rgba(31, 41, 55, 0.8) !important;
            border: 2px dashed #4b5563 !important;
            color: #e5e7eb !important;
        }
        
        /* Additional dark theme overrides */
        .markdown, .prose {
            color: #e5e7eb !important;
        }
        
        /* Code blocks */
        pre, code {
            background: rgba(17, 24, 39, 0.9) !important;
            color: #f3f4f6 !important;
            border: 1px solid #374151 !important;
        }
        
        /* Gallery and image components */
        .gallery {
            background: rgba(31, 41, 55, 0.8) !important;
        }
        
        /* Tabs */
        .tab-nav {
            background: rgba(31, 41, 55, 0.9) !important;
            border-bottom: 1px solid #4b5563 !important;
        }
        
        /* Accordion headers */
        .accordion .label {
            background: rgba(55, 65, 81, 0.8) !important;
            color: #e5e7eb !important;
        }
        
        /* Progress bars */
        .progress {
            background: rgba(31, 41, 55, 0.8) !important;
        }
        
        /* Slider tracks */
        input[type="range"] {
            background: #374151 !important;
        }
        
        /* Checkbox and radio buttons - Enhanced visibility */
        input[type="checkbox"], input[type="radio"] {
            background: #374151 !important;
            border: 2px solid #60a5fa !important;
            width: 18px !important;
            height: 18px !important;
            border-radius: 4px !important;
            cursor: pointer !important;
        }
        
        input[type="checkbox"]:checked, input[type="radio"]:checked {
            background: #3b82f6 !important;
            border-color: #1d4ed8 !important;
        }
        
        input[type="checkbox"]:hover, input[type="radio"]:hover {
            border-color: #93c5fd !important;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
        }
        
        /* Ensure checkbox labels are clickable */
        .checkbox label, .radio label {
            cursor: pointer !important;
            color: #e5e7eb !important;
            padding-left: 8px !important;
        }
        
        /* Fix Gradio checkbox containers */
        .form > label, .checkbox > label {
            display: flex !important;
            align-items: center !important;
            cursor: pointer !important;
            color: #e5e7eb !important;
        }
        
        /* Ensure checkbox wrapper is properly styled */
        .checkbox, .radio {
            margin: 8px 0 !important;
        }
        
        /* Override any z-index issues */
        input[type="checkbox"], input[type="radio"] {
            z-index: 1 !important;
            position: relative !important;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .section-header {
                font-size: 1.2em;
            }
            
            .parameter-section {
                padding: 1em;
            }
        }
        </style>
        """
    
    @staticmethod
    def get_preset_configurations() -> Dict[str, Dict[str, Any]]:
        """Get configuration presets for different training scenarios"""
        return {
            "🎨 Style/Concept (Recommended)": {
                "rank": 32, "alpha": 32, "learning_rate": 1e-4, "max_train_steps": 1500,
                "optimizer": "prodigy", "lr_scheduler": "cosine_with_restarts",
                "batch_size": 1, "gradient_accumulation": 4, "sample_every": 250
            },
            "👤 Character/Person": {
                "rank": 64, "alpha": 64, "learning_rate": 8e-5, "max_train_steps": 2000,
                "optimizer": "prodigy", "lr_scheduler": "cosine_with_restarts",
                "batch_size": 1, "gradient_accumulation": 2, "sample_every": 200
            },
            "🏃 Quick Test": {
                "rank": 16, "alpha": 16, "learning_rate": 2e-4, "max_train_steps": 500,
                "optimizer": "adamw8bit", "lr_scheduler": "linear",
                "batch_size": 1, "gradient_accumulation": 2, "sample_every": 100
            },
            "🔬 Research/Experimental": {
                "rank": 128, "alpha": 128, "learning_rate": 5e-5, "max_train_steps": 3000,
                "optimizer": "prodigy", "lr_scheduler": "cosine_with_restarts",
                "batch_size": 1, "gradient_accumulation": 8, "sample_every": 500
            },
            "💾 Low VRAM (<12GB)": {
                "rank": 16, "alpha": 16, "learning_rate": 1e-4, "max_train_steps": 1000,
                "optimizer": "adamw8bit", "lr_scheduler": "cosine",
                "batch_size": 1, "gradient_accumulation": 2, "low_vram_mode": True
            },
            "⚡ Speed Optimized": {
                "rank": 16, "alpha": 16, "learning_rate": 3e-4, "max_train_steps": 800,
                "optimizer": "lion", "lr_scheduler": "linear",
                "batch_size": 2, "gradient_accumulation": 2, "sample_every": 200
            }
        }