"""
Helper functions for the Advanced FLUX LoRA Trainer
"""

from typing import Dict, Any, List


def recursive_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update dictionary"""
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def format_verification_report(verification_report: Dict[str, Any]) -> str:
    """Format the verification report for display"""
    
    if not verification_report:
        return "No verification data available."
    
    report = f"""
## 🔍 **Caption Integration Verification**

### Status: {verification_report['status']}

"""
    
    # Add details if available
    if verification_report.get('details'):
        details = verification_report['details']
        
        if 'jsonl_entries' in details:
            report += f"""
### 📋 **Dataset Structure:**
- **JSONL Entries**: {details['jsonl_entries']} 
- **Expected Entries**: {details['expected_entries']}
- **Image Files Found**: {details.get('image_files_found', 'Unknown')}
"""
        
        if 'caption_stats' in details:
            stats = details['caption_stats']
            report += f"""
### 📝 **Caption Quality Analysis:**
- **Total Captions**: {stats['total_captions']}
- **Empty Captions**: {stats['empty_captions']}
- **Average Length**: {stats['avg_length']:.1f} characters
- **Length Range**: {stats['min_length']} - {stats['max_length']} characters

### 🎯 **Sample Caption Verification:**
"""
            for i, sample in enumerate(stats.get('sample_captions', [])):
                report += f"**{i+1}. {sample['file']}**: {sample['caption']} ({sample['length']} chars)\n"
    
    # Add issues if any
    if verification_report.get('issues'):
        report += f"""
### ⚠️ **Issues Found:**
"""
        for issue in verification_report['issues']:
            report += f"- {issue}\n"
    
    if verification_report['status'] == "✅ FULLY VERIFIED":
        report += f"""

### 🎉 **Conclusion:**
✅ **Your captions ARE being used for training!**
- All image-caption pairs are properly matched
- Metadata.jsonl file is correctly formatted
- Captions will be fed to the FLUX model during training
- Trigger words are properly positioned
"""
    
    return report


def format_matching_report(matching_report: Dict[str, Any]) -> str:
    """Format the detailed matching report for display"""
    
    if not matching_report:
        return "No matching report available."
    
    report = f"""
## 📋 Detailed Matching Analysis

### 📊 Summary Statistics
- **Total Images Found**: {matching_report.get('total_images_found', 0)}
- **Total Captions Available**: {matching_report.get('total_captions_available', 0)}
- **✅ Successful Matches**: {matching_report.get('successful_matches', 0)}
- **❌ Failed Matches**: {matching_report.get('failed_matches', 0)}

### 🎯 Match Quality
- **Success Rate**: {(matching_report.get('successful_matches', 0) / max(matching_report.get('total_images_found', 1), 1) * 100):.1f}%
- **Orphaned Images**: {len(matching_report.get('orphaned_images', []))}
- **Orphaned Captions**: {len(matching_report.get('orphaned_captions', []))}
"""

    # Show orphaned files if any
    if matching_report.get('orphaned_images'):
        report += f"\n### ⚠️ Orphaned Images (no matching caption):\n"
        for img in matching_report.get('orphaned_images', [])[:10]:
            report += f"- `{img['image_name']}`\n"
        if len(matching_report.get('orphaned_images', [])) > 10:
            report += f"- ... and {len(matching_report.get('orphaned_images', [])) - 10} more\n"
    
    if matching_report.get('orphaned_captions'):
        report += f"\n### ⚠️ Orphaned Captions (no matching image):\n"
        for cap in matching_report.get('orphaned_captions', [])[:10]:
            report += f"- `{cap['caption_name']}`\n"
        if len(matching_report.get('orphaned_captions', [])) > 10:
            report += f"- ... and {len(matching_report.get('orphaned_captions', [])) - 10} more\n"
    
    # Show first few successful matches
    if matching_report.get('matching_details'):
        report += f"\n### ✅ Sample Successful Matches:\n"
        for detail in matching_report.get('matching_details', [])[:5]:
            report += f"- **{detail.get('image_name', 'Unknown')}**: {detail.get('status', 'Unknown')}\n"
    
    return report


def create_preview_gallery(matched_data: List[Dict]) -> List[tuple]:
    """Create gallery data for visual preview of image-caption matching"""
    
    if not matched_data:
        return []
    
    gallery_data = []
    
    try:
        for i, data in enumerate(matched_data[:6]):  # First 6 images
            caption_preview = f"#{data.get('image_number', i+1)}: {data['caption'][:100]}{'...' if len(data['caption']) > 100 else ''}"
            gallery_data.append((data['image_path'], caption_preview))
        
        return gallery_data
        
    except Exception as e:
        print(f"Error creating preview gallery: {e}")
        return []


def validate_lora_name(name: str) -> tuple[bool, str]:
    """Validate LoRA name"""
    if not name or not name.strip():
        return False, "LoRA name cannot be empty"
    
    if len(name.strip()) < 3:
        return False, "LoRA name must be at least 3 characters long"
    
    # Check for invalid characters
    invalid_chars = set('<>:"/\\|?*')
    if any(char in invalid_chars for char in name):
        return False, f"LoRA name contains invalid characters: {invalid_chars}"
    
    return True, "Valid LoRA name"


def validate_trigger_word(trigger: str) -> tuple[bool, str]:
    """Validate trigger word/phrase"""
    if not trigger or not trigger.strip():
        return False, "Trigger word cannot be empty"
    
    if len(trigger.strip()) > 100:
        return False, "Trigger word is too long (max 100 characters)"
    
    return True, "Valid trigger word"


def validate_training_params(steps: int, lr: float, rank: int, alpha: int) -> tuple[bool, str]:
    """Validate basic training parameters"""
    
    if steps < 100:
        return False, "Training steps must be at least 100"
    
    if steps > 20000:
        return False, "Training steps cannot exceed 20000"
    
    if lr <= 0 or lr > 1:
        return False, "Learning rate must be between 0 and 1"
    
    if rank < 4 or rank > 256:
        return False, "LoRA rank must be between 4 and 256"
    
    if alpha < 1 or alpha > 256:
        return False, "LoRA alpha must be between 1 and 256"
    
    return True, "Valid training parameters"


def calculate_effective_batch_size(batch_size: int, grad_accum: int) -> int:
    """Calculate effective batch size"""
    return batch_size * grad_accum


def estimate_training_time(steps: int, batch_size: int, grad_accum: int, num_images: int) -> str:
    """Estimate training time"""
    
    effective_batch_size = calculate_effective_batch_size(batch_size, grad_accum)
    
    # Rough estimates (these would need to be calibrated based on actual hardware)
    seconds_per_step = 2.0  # Base estimate for FLUX training
    
    # Adjust based on batch size
    seconds_per_step *= effective_batch_size
    
    # Adjust based on image count (more images = slower data loading)
    if num_images > 50:
        seconds_per_step *= 1.2
    
    total_seconds = steps * seconds_per_step
    
    if total_seconds < 60:
        return f"~{int(total_seconds)} seconds"
    elif total_seconds < 3600:
        return f"~{int(total_seconds/60)} minutes"
    else:
        hours = int(total_seconds / 3600)
        minutes = int((total_seconds % 3600) / 60)
        return f"~{hours}h {minutes}m"


def get_memory_requirements(rank: int, resolution: List[int], low_vram: bool) -> Dict[str, str]:
    """Estimate memory requirements"""
    
    base_memory = 12.0  # GB base FLUX memory
    
    # Adjust for rank
    rank_memory = rank * 0.1  # Rough estimate
    
    # Adjust for resolution
    max_res = max(resolution) if resolution else 1024
    res_multiplier = (max_res / 1024) ** 2
    res_memory = base_memory * res_multiplier
    
    total_memory = base_memory + rank_memory + res_memory
    
    if low_vram:
        total_memory *= 0.7  # Low VRAM optimizations
    
    return {
        'estimated_vram': f"{total_memory:.1f} GB",
        'minimum_recommended': f"{max(16, total_memory * 1.2):.0f} GB",
        'optimal': f"{max(24, total_memory * 1.5):.0f} GB"
    }


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    import re
    
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove extra spaces and dots
    filename = re.sub(r'\s+', '_', filename)
    filename = re.sub(r'\.+', '.', filename)
    
    # Ensure it's not too long
    if len(filename) > 100:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:95] + ('.' + ext if ext else '')
    
    return filename.strip('_.')


def format_number(num: float, precision: int = 2) -> str:
    """Format number for display"""
    if num >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def create_progress_callback(progress_component):
    """Create progress callback for training"""
    def update_progress(step: int, total_steps: int, loss: float = None, description: str = "Training"):
        progress = step / total_steps
        desc = f"{description} - Step {step}/{total_steps}"
        if loss is not None:
            desc += f" - Loss: {loss:.4f}"
        
        if progress_component:
            progress_component(progress, desc=desc)
    
    return update_progress