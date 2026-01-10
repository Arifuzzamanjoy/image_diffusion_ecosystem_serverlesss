"""
GPU management and optimization for FLUX LoRA training
"""

import torch
import gc
import time
from typing import List, Dict, Optional, Tuple


class GPUManager:
    """Manages GPU initialization, monitoring, and optimization"""
    
    def __init__(self):
        self.device_count = 0
        self.devices = []
        self.is_initialized = False
    
    def initialize_pod_gpu_environment(self) -> bool:
        """
        🚀 POD GPU INITIALIZATION SYSTEM
        Comprehensive GPU initialization for pod environments with full utilization
        """
        print("🚀 Initializing pod GPU environment for maximum performance...")
        
        try:
            # Clear any existing CUDA context
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(2)
            
            # Set optimal PyTorch settings for pod environment
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for speed
            torch.backends.cudnn.allow_tf32 = True
            
            # Set memory fraction to prevent OOM while maximizing utilization
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available memory
                
            # Initialize CUDA properly
            self.device_count = torch.cuda.device_count()
            if self.device_count == 0:
                raise RuntimeError("No CUDA devices available. Please check your pod configuration.")
                
            print(f"🎯 Found {self.device_count} CUDA device(s)")
            
            # Get device info and store
            self.devices = []
            for i in range(self.device_count):
                device_name = torch.cuda.get_device_name(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                device_info = {
                    'id': i,
                    'name': device_name,
                    'memory_total': memory_total,
                    'device': torch.device(f"cuda:{i}")
                }
                self.devices.append(device_info)
                
                print(f"   GPU {i}: {device_name} ({memory_total:.1f} GB)")
                
            # Test CUDA functionality
            device = torch.device("cuda:0")
            test_tensor = torch.randn(100, 100, device=device, dtype=torch.float16)
            test_result = torch.matmul(test_tensor, test_tensor.T)
            del test_tensor, test_result
            torch.cuda.empty_cache()
            
            self.is_initialized = True
            print("✅ Pod GPU environment initialized successfully with full utilization enabled!")
            return True
            
        except Exception as e:
            print(f"❌ Pod GPU initialization failed: {e}")
            return False
    
    def handle_pod_cuda_errors(self) -> bool:
        """
        🛠️ POD CUDA ERROR HANDLER
        Robust error handling for pod-specific CUDA issues
        """
        try:
            import torch
            import gc
            import time
            
            print("🔄 Attempting pod CUDA recovery...")
            
            # Clear memory and cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
            gc.collect()
            time.sleep(3)
            
            # Reset CUDA context
            torch.cuda.reset_peak_memory_stats()
            
            # Test if CUDA is working
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                test_tensor = torch.ones(10, device=device)
                result = test_tensor.sum()
                del test_tensor, result
                torch.cuda.empty_cache()
                
                print("🎉 Pod CUDA recovery successful!")
                return True
            else:
                print("❌ CUDA not available after recovery attempt")
                return False
                
        except Exception as e:
            print(f"❌ Pod CUDA recovery failed: {e}")
            return False
    
    def monitor_gpu_utilization(self) -> List[Dict]:
        """
        📊 GPU UTILIZATION MONITOR
        Monitors GPU usage to ensure full utilization during training
        """
        try:
            import torch
            
            if not torch.cuda.is_available():
                print("❌ No CUDA devices available for monitoring")
                return []
            
            utilization_info = []
            
            for i in range(self.device_count):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                utilization_info.append({
                    'device_id': i,
                    'device_name': torch.cuda.get_device_name(i),
                    'memory_allocated': memory_allocated,
                    'memory_reserved': memory_reserved,
                    'memory_total': memory_total,
                    'memory_free': memory_total - memory_reserved,
                    'utilization_percent': (memory_reserved / memory_total) * 100
                })
            
            return utilization_info
        
        except Exception as e:
            print(f"⚠️ GPU monitoring error: {e}")
            return []
    
    def get_optimal_device(self) -> torch.device:
        """Get the optimal device for training"""
        if not self.is_initialized:
            self.initialize_pod_gpu_environment()
        
        if self.devices:
            return self.devices[0]['device']
        else:
            return torch.device("cpu")
    
    def get_memory_info(self, device_id: int = 0) -> Dict[str, float]:
        """Get detailed memory information for a specific device"""
        if not torch.cuda.is_available() or device_id >= self.device_count:
            return {}
        
        return {
            'allocated': torch.cuda.memory_allocated(device_id) / 1024**3,
            'reserved': torch.cuda.memory_reserved(device_id) / 1024**3,
            'total': torch.cuda.get_device_properties(device_id).total_memory / 1024**3,
            'free': (torch.cuda.get_device_properties(device_id).total_memory - 
                    torch.cuda.memory_reserved(device_id)) / 1024**3
        }
    
    def optimize_for_training(self, low_vram: bool = False):
        """Apply GPU optimizations for training"""
        if not torch.cuda.is_available():
            return
        
        # Set memory management
        if low_vram:
            torch.cuda.set_per_process_memory_fraction(0.8)
        else:
            torch.cuda.set_per_process_memory_fraction(0.95)
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
    
    def cleanup(self):
        """Cleanup GPU resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
    
    def get_device_info(self) -> List[Dict]:
        """Get information about all available devices"""
        if not self.is_initialized:
            self.initialize_pod_gpu_environment()
        
        return self.devices.copy()
    
    def print_utilization_report(self):
        """Print a detailed GPU utilization report"""
        utilization_info = self.monitor_gpu_utilization()
        
        if not utilization_info:
            print("❌ No GPU utilization data available")
            return
        
        print("\n📊 GPU UTILIZATION REPORT:")
        print("=" * 60)
        
        for info in utilization_info:
            print(f"🎮 GPU {info['device_id']}: {info['device_name']}")
            print(f"   💾 Memory: {info['memory_allocated']:.1f}G allocated / "
                  f"{info['memory_reserved']:.1f}G reserved / {info['memory_total']:.1f}G total")
            print(f"   📈 Utilization: {info['utilization_percent']:.1f}%")
            print(f"   🆓 Free: {info['memory_free']:.1f}G")
            print()
    
    @staticmethod
    def get_system_info() -> Dict[str, any]:
        """Get system information relevant to GPU training"""
        info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'pytorch_version': torch.__version__,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            info['devices'] = []
            for i in range(torch.cuda.device_count()):
                device_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'compute_capability': torch.cuda.get_device_capability(i),
                    'total_memory': torch.cuda.get_device_properties(i).total_memory / 1024**3,
                    'multi_processor_count': torch.cuda.get_device_properties(i).multi_processor_count
                }
                info['devices'].append(device_info)
        
        return info


# Global GPU manager instance
gpu_manager = GPUManager()