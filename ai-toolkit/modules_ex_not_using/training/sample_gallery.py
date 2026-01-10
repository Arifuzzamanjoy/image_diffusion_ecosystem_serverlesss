"""
Training Sample Gallery Manager

Manages live sample generation and gallery updates during training.
Provides real-time visual feedback on training progress with professional monitoring.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import gradio as gr
from PIL import Image
import threading
import queue
import subprocess
import asyncio


class TrainingSampleGallery:
    """Manages live sample generation and gallery updates during training"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.samples_dir = self.output_dir / "training_samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        self.sample_queue = queue.Queue()
        self.is_monitoring = False
        self.current_step = 0
        
        # Gallery state
        self.gallery_images = []
        self.sample_history = []
        
        # Default sample prompts
        self.default_prompts = [
            {
                "prompt": "A portrait of TOK, high quality, detailed",
                "negative": "blurry, low quality, deformed",
                "weight": 1.0
            },
            {
                "prompt": "TOK in a beautiful landscape, cinematic lighting",
                "negative": "dark, unclear, poor composition",
                "weight": 1.0
            },
            {
                "prompt": "Close-up of TOK, professional photography",
                "negative": "amateur, poor quality, blurry",
                "weight": 1.0
            },
            {
                "prompt": "TOK, artistic style, creative composition",
                "negative": "boring, generic, low quality",
                "weight": 1.0
            }
        ]
        
    def initialize_gallery(self, sample_prompts: List[List[Any]] = None) -> List[str]:
        """Initialize the sample gallery"""
        if sample_prompts:
            self.update_sample_prompts(sample_prompts)
        
        # Clear previous samples
        self.gallery_images.clear()
        self.sample_history.clear()
        
        # Create initial placeholder
        placeholder_path = self.samples_dir / "placeholder.png"
        if not placeholder_path.exists():
            # Create a simple placeholder image
            placeholder_img = Image.new('RGB', (512, 512), color='lightgray')
            placeholder_img.save(placeholder_path)
        
        return [str(placeholder_path)]
    
    def update_sample_prompts(self, prompts_data: List[List[Any]]):
        """Update sample prompts from UI input"""
        self.sample_prompts = []
        for row in prompts_data:
            if len(row) >= 3 and row[0].strip():  # Ensure we have prompt data
                self.sample_prompts.append({
                    "prompt": row[0].strip(),
                    "weight": float(row[1]) if row[1] else 1.0,
                    "negative": row[2].strip() if row[2] else ""
                })
        
        # If no valid prompts, use defaults
        if not self.sample_prompts:
            self.sample_prompts = self.default_prompts.copy()
    
    def start_monitoring(self, config: Dict[str, Any], update_frequency: int = 250):
        """Start monitoring training and generating samples"""
        self.is_monitoring = True
        self.current_step = 0
        self.update_frequency = update_frequency
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_training_loop,
            args=(config,),
            daemon=True
        )
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring training"""
        self.is_monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5.0)
    
    def _monitor_training_loop(self, config: Dict[str, Any]):
        """Main monitoring loop running in separate thread"""
        step_file = self.output_dir / "current_step.txt"
        loss_file = self.output_dir / "current_loss.txt"
        
        last_sample_step = 0
        
        while self.is_monitoring:
            try:
                # Check current training step
                if step_file.exists():
                    with open(step_file, 'r') as f:
                        self.current_step = int(f.read().strip())
                
                # Generate samples if needed
                if (self.current_step > 0 and 
                    self.current_step - last_sample_step >= self.update_frequency):
                    
                    self._generate_training_samples(config, self.current_step)
                    last_sample_step = self.current_step
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _generate_training_samples(self, config: Dict[str, Any], step: int):
        """Generate training samples at current step"""
        try:
            model_path = self.output_dir / f"checkpoint-{step}"
            if not model_path.exists():
                # Try finding the latest checkpoint
                checkpoints = list(self.output_dir.glob("checkpoint-*"))
                if checkpoints:
                    model_path = max(checkpoints, key=lambda x: int(x.name.split('-')[-1]))
                else:
                    return
            
            # Generate samples for each prompt
            step_samples = []
            for i, prompt_config in enumerate(self.sample_prompts):
                sample_path = self._generate_single_sample(
                    model_path, prompt_config, step, i
                )
                if sample_path:
                    step_samples.append(sample_path)
            
            if step_samples:
                self.gallery_images.extend(step_samples)
                self.sample_history.append({
                    "step": step,
                    "samples": step_samples,
                    "timestamp": time.time()
                })
                
                # Keep only last 20 samples for memory
                if len(self.gallery_images) > 20:
                    self.gallery_images = self.gallery_images[-20:]
                    
        except Exception as e:
            print(f"Error generating training samples: {e}")
    
    def _generate_single_sample(self, model_path: Path, prompt_config: Dict, 
                               step: int, prompt_idx: int) -> Optional[str]:
        """Generate a single sample image"""
        try:
            output_file = self.samples_dir / f"step_{step:06d}_prompt_{prompt_idx}.png"
            
            # Construct generation command
            cmd = [
                "python", "-m", "toolkit.generate",
                "--config_file", str(self.output_dir / "config.yaml"),
                "--model_path", str(model_path),
                "--prompt", prompt_config["prompt"],
                "--negative_prompt", prompt_config.get("negative", ""),
                "--output", str(output_file),
                "--steps", "28",
                "--guidance_scale", "3.5",
                "--width", "1024",
                "--height", "1024"
            ]
            
            # Run generation
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=120,
                cwd=self.output_dir.parent
            )
            
            if result.returncode == 0 and output_file.exists():
                return str(output_file)
            else:
                print(f"Generation failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Error in sample generation: {e}")
            return None
    
    def get_current_gallery(self) -> List[str]:
        """Get current gallery images"""
        return self.gallery_images.copy()
    
    def clear_gallery(self):
        """Clear the gallery"""
        self.gallery_images.clear()
        self.sample_history.clear()
        
        # Remove sample files
        for sample_file in self.samples_dir.glob("step_*.png"):
            try:
                sample_file.unlink()
            except:
                pass
    
    def export_samples(self) -> str:
        """Export all samples as a ZIP file"""
        import zipfile
        
        zip_path = self.output_dir / f"training_samples_{int(time.time())}.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for sample_file in self.samples_dir.glob("step_*.png"):
                zf.write(sample_file, sample_file.name)
            
            # Add metadata
            metadata = {
                "total_samples": len(self.gallery_images),
                "sample_history": self.sample_history,
                "export_time": time.time()
            }
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))
        
        return str(zip_path)
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics for monitoring"""
        stats = {
            "current_step": self.current_step,
            "total_samples": len(self.gallery_images),
            "sample_batches": len(self.sample_history),
            "is_monitoring": self.is_monitoring
        }
        
        if self.sample_history:
            latest_batch = self.sample_history[-1]
            stats["latest_sample_step"] = latest_batch["step"]
            stats["latest_sample_time"] = latest_batch["timestamp"]
        
        return stats


class AdvancedProgressTracker:
    """Advanced progress tracking for world-class training monitoring"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.metrics_file = self.output_dir / "training_metrics.json"
        self.metrics_history = []
        
        # Real-time metrics
        self.current_metrics = {
            "step": 0,
            "loss": 0.0,
            "learning_rate": 0.0,
            "grad_norm": 0.0,
            "gpu_memory": 0.0,
            "training_speed": 0.0
        }
        
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update current training metrics"""
        self.current_metrics.update(metrics)
        self.metrics_history.append({
            "timestamp": time.time(),
            **self.current_metrics
        })
        
        # Save to file
        self._save_metrics()
        
        # Keep only last 1000 entries for memory
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def _save_metrics(self):
        """Save metrics to file"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            print(f"Error saving metrics: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current training metrics"""
        return self.current_metrics.copy()
    
    def get_training_curves(self) -> Dict[str, List]:
        """Get training curves for plotting"""
        if not self.metrics_history:
            return {}
        
        curves = {
            "steps": [m["step"] for m in self.metrics_history],
            "loss": [m["loss"] for m in self.metrics_history],
            "learning_rate": [m["learning_rate"] for m in self.metrics_history],
            "grad_norm": [m.get("grad_norm", 0.0) for m in self.metrics_history]
        }
        
        return curves
    
    def generate_training_report(self) -> str:
        """Generate a comprehensive training report"""
        if not self.metrics_history:
            return "No training data available."
        
        latest = self.metrics_history[-1]
        total_steps = len(self.metrics_history)
        
        # Calculate statistics
        losses = [m["loss"] for m in self.metrics_history if m["loss"] > 0]
        avg_loss = sum(losses) / len(losses) if losses else 0
        min_loss = min(losses) if losses else 0
        
        report = f"""
# Training Report

## Current Status
- **Step**: {latest['step']:,}
- **Current Loss**: {latest['loss']:.6f}
- **Learning Rate**: {latest['learning_rate']:.2e}
- **Training Speed**: {latest.get('training_speed', 0):.2f} steps/sec

## Statistics
- **Total Training Steps**: {total_steps:,}
- **Average Loss**: {avg_loss:.6f}
- **Minimum Loss**: {min_loss:.6f}
- **GPU Memory Usage**: {latest.get('gpu_memory', 0):.1f} GB

## Performance Analysis
{"🟢 Training is progressing well!" if avg_loss > min_loss * 1.5 else "🟡 Loss has stabilized."}
{"🚀 Good training speed!" if latest.get('training_speed', 0) > 1.0 else "⏱️ Consider optimizing batch size."}
        """
        
        return report.strip()