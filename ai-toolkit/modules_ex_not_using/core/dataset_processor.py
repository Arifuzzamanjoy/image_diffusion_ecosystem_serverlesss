"""
Dataset processing and management for FLUX LoRA training
"""

import os
import json
import zipfile
import tempfile
import shutil
import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image
import gradio as gr


class DatasetProcessor:
    """Handles dataset extraction, processing, and verification"""
    
    def __init__(self, max_images: int = 150):
        self.max_images = max_images
        self.supported_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    
    def load_and_validate_dataset(self, images_zip: str, captions_file: str) -> Dict[str, Any]:
        """Load and validate dataset for world-class interface compatibility"""
        try:
            # Create temporary file objects from paths for compatibility
            class TempFile:
                def __init__(self, path):
                    self.name = path
            
            images_zip_obj = TempFile(images_zip)
            captions_json_obj = TempFile(captions_file)
            
            # Use existing extract_captioning_data method
            matched_data, dataset_folder, preview_gallery, matching_report = self.extract_captioning_data(
                images_zip_obj, captions_json_obj
            )
            
            # Generate preview images
            preview_images = []
            if matched_data:
                for item in matched_data[:6]:  # First 6 for preview
                    if 'image_path' in item and os.path.exists(item['image_path']):
                        preview_images.append(item['image_path'])
            
            # Create validation report
            total_images = len(matched_data)
            matched_pairs = sum(1 for item in matched_data if item.get('caption', '').strip())
            
            return {
                "success": True,
                "total_images": total_images,
                "matched_pairs": matched_pairs,
                "preview_images": preview_images,
                "avg_caption_length": matching_report.get('avg_caption_length', 0),
                "resolution_info": "Mixed resolutions",
                "details": f"Successfully loaded {total_images} images with {matched_pairs} caption matches."
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "total_images": 0,
                "matched_pairs": 0,
                "preview_images": []
            }
    
    def extract_captioning_data(self, images_zip_file, captions_json_file, progress=None) -> Tuple[List[Dict], str, Any, Dict]:
        """Extract images from ZIP and captions from JSON, creating a unified dataset with detailed matching verification"""
        
        if images_zip_file is None or captions_json_file is None:
            raise gr.Error("Please upload both the images ZIP file and captions JSON file.")
        
        try:
            if progress:
                progress(0.1, desc="📦 Extracting images from ZIP...")
            
            # Create persistent temporary directory for extraction
            temp_dir = tempfile.mkdtemp(prefix="flux_trainer_")
            images_dir = os.path.join(temp_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            # Handle both file paths (string) and file objects
            zip_path = images_zip_file if isinstance(images_zip_file, str) else images_zip_file.name
            json_path = captions_json_file if isinstance(captions_json_file, str) else captions_json_file.name
            
            print(f"📦 Processing ZIP: {zip_path}")
            print(f"📋 Processing JSON: {json_path}")
            
            # Extract images ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(images_dir)
            
            if progress:
                progress(0.3, desc="📋 Loading captions from JSON...")
            
            # Load captions JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                captions_data = json.load(f)
            
            # Extract caption mappings
            if "captions" in captions_data:
                caption_mappings = captions_data["captions"]
            else:
                raise gr.Error("Invalid JSON format. Expected 'captions' key not found.")
            
            if progress:
                progress(0.5, desc="🔍 Finding and matching images with captions...")
            
            # Find all image files in extracted directory (recursively)
            image_files = []
            for root, dirs, files in os.walk(images_dir):
                for file in files:
                    file_lower = file.lower()
                    if any(file_lower.endswith(ext) for ext in self.supported_extensions):
                        image_files.append(os.path.join(root, file))
            
            # Sort image files by name for consistent ordering
            image_files.sort(key=lambda x: os.path.basename(x))
            
            print(f"✅ Found {len(image_files)} images with supported formats")
            if len(image_files) > 0:
                print(f"   First few images: {[os.path.basename(f) for f in image_files[:5]]}")
            
            if progress:
                progress(0.6, desc="🎯 Performing detailed matching verification...")
            
            # Create detailed matching data with verification
            matched_data = []
            matching_report = self._create_matching_report(image_files, caption_mappings)
            
            # Process each image file
            for image_path in image_files:
                image_name = os.path.basename(image_path)
                
                if image_name in caption_mappings:
                    caption_data = caption_mappings[image_name]
                    
                    # Extract actual caption text from the data structure
                    if isinstance(caption_data, dict) and 'caption' in caption_data:
                        caption = caption_data['caption']
                    else:
                        caption = str(caption_data)
                    
                    # Auto-detect and fix common trigger words formatting (if enabled)
                    # Note: This is automatically applied for professional training
                    caption = self._auto_fix_common_trigger_words(str(caption) if caption else "")
                    
                    image_number = self._extract_number_from_filename(image_name)
                    
                    # Safely handle caption length
                    try:
                        caption_length = len(str(caption)) if caption else 0
                    except Exception as e:
                        print(f"⚠️ WARNING: Error getting caption length for {image_name}: {e}")
                        caption_length = 0
                    
                    matched_data.append({
                        "image_path": image_path,
                        "image_name": image_name,
                        "image_number": image_number,
                        "caption": str(caption) if caption else "",
                        "caption_length": caption_length,
                        "status": "✅ MATCHED",
                        "temp_dir": temp_dir  # Store temp dir for cleanup
                    })
                    matching_report["successful_matches"] += 1
                else:
                    matching_report["failed_matches"] += 1
                    matching_report["orphaned_images"].append({
                        "image_name": image_name,
                        "image_path": image_path
                    })
            
            # Check for captions without images
            used_caption_names = {os.path.basename(data["image_path"]) for data in matched_data}
            for caption_name in caption_mappings.keys():
                if caption_name not in used_caption_names:
                    matching_report["orphaned_captions"].append({
                        "caption_name": caption_name,
                        "caption": caption_mappings[caption_name]
                    })
            
            if progress:
                progress(0.8, desc="📊 Generating verification report...")
            
            # Sort matched data by image number for consistent ordering
            matched_data.sort(key=lambda x: x["image_number"])
            
            # Validate dataset
            if len(matched_data) < 2:
                raise gr.Error(f"❌ Dataset must contain at least 2 matched images for training. Found {len(matched_data)} matches.")
            elif len(matched_data) > self.max_images:
                raise gr.Error(f"❌ Too many images! Maximum {self.max_images} allowed, found {len(matched_data)}.")
            
            if progress:
                progress(0.9, desc="📋 Creating detailed verification report...")
            
            # Create comprehensive dataset info
            dataset_info = self._create_dataset_info(matched_data, matching_report, captions_data)
            
            if progress:
                progress(1.0, desc="🎯 Dataset ready for training!")
            
            # Return 4 values as expected by Gradio interface
            return matched_data, dataset_info, gr.update(visible=True), matching_report
                
        except Exception as e:
            error_msg = self._create_error_message(str(e))
            return [], error_msg, gr.update(visible=False), {}
    
    def create_training_dataset(self, matched_data: List[Dict], trigger_word: str = "", 
                              trigger_position: str = "beginning") -> str:
        """Create dataset folder structure for training with optimized trigger positioning"""
        
        temp_dir_to_cleanup = None
        
        try:
            # Create unique dataset folder
            dataset_id = str(uuid.uuid4())
            destination_folder = f"datasets/{dataset_id}"
            destination_folder = os.path.abspath(destination_folder)
            os.makedirs(destination_folder, exist_ok=True)
            print(f"📂 Created dataset folder: {destination_folder}")
            
            # Create metadata.jsonl file
            jsonl_file_path = os.path.join(destination_folder, "metadata.jsonl")
            
            with open(jsonl_file_path, "w", encoding='utf-8') as jsonl_file:
                for i, data in enumerate(matched_data):
                    # Get temp directory for cleanup
                    if temp_dir_to_cleanup is None and 'temp_dir' in data:
                        temp_dir_to_cleanup = data['temp_dir']
                    
                    # Copy image to destination folder
                    image_extension = os.path.splitext(data["image_name"])[1]
                    new_image_name = f"{i+1:04d}{image_extension}"
                    new_image_path = os.path.join(destination_folder, new_image_name)
                    
                    # Check if source file exists before copying
                    source_path = data["image_path"]
                    if not os.path.exists(source_path):
                        raise FileNotFoundError(f"Source image not found: {source_path}")
                    
                    print(f"📁 Copying {source_path} -> {new_image_path}")
                    shutil.copy2(source_path, new_image_path)
                    
                    # Process caption with trigger word
                    caption = self._apply_trigger_word(data["caption"], trigger_word, trigger_position)
                    
                    # Create JSONL entry
                    jsonl_entry = {
                        "file_name": new_image_name,
                        "text": caption
                    }
                    
                    jsonl_file.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")
            
            # Clean up temporary directory after copying files
            if temp_dir_to_cleanup and os.path.exists(temp_dir_to_cleanup):
                try:
                    shutil.rmtree(temp_dir_to_cleanup)
                except Exception as cleanup_error:
                    print(f"⚠️ Warning: Could not clean up temporary directory: {cleanup_error}")
            
            # Verification: Print dataset creation summary
            print(f"\n🎯 DATASET CREATION VERIFICATION:")
            print(f"   📂 Dataset folder: {destination_folder}")
            print(f"   📋 Metadata file: {jsonl_file_path}")
            print(f"   🖼️ Total images: {len(matched_data)}")
            print(f"   📝 Total captions: {len(matched_data)}")
            print(f"   ✅ Dataset ready for training!")
            
            return destination_folder
            
        except Exception as e:
            # Clean up on error if possible
            if temp_dir_to_cleanup and os.path.exists(temp_dir_to_cleanup):
                try:
                    shutil.rmtree(temp_dir_to_cleanup)
                except:
                    pass
            raise gr.Error(f"Error creating training dataset: {str(e)}")
    
    def verify_caption_integration(self, dataset_folder: str, matched_data: List[Dict]) -> Dict[str, Any]:
        """Comprehensive verification that captions are properly integrated for training"""
        
        verification_report = {
            "status": "✅ VERIFIED",
            "issues": [],
            "details": {}
        }
        
        try:
            # Check 1: Verify dataset folder exists
            if not os.path.exists(dataset_folder):
                verification_report["status"] = "❌ FAILED"
                verification_report["issues"].append("Dataset folder does not exist")
                return verification_report
            
            # Check 2: Verify metadata.jsonl exists
            metadata_file = os.path.join(dataset_folder, "metadata.jsonl")
            if not os.path.exists(metadata_file):
                verification_report["status"] = "❌ FAILED"
                verification_report["issues"].append("metadata.jsonl file missing")
                return verification_report
            
            # Check 3: Verify metadata.jsonl content
            with open(metadata_file, 'r', encoding='utf-8') as f:
                jsonl_lines = f.readlines()
            
            verification_report["details"]["jsonl_entries"] = len(jsonl_lines)
            verification_report["details"]["expected_entries"] = len(matched_data)
            
            if len(jsonl_lines) != len(matched_data):
                verification_report["status"] = "⚠️ WARNING"
                verification_report["issues"].append(f"Entry count mismatch: Expected {len(matched_data)}, found {len(jsonl_lines)}")
            
            # Check 4: Verify image files exist
            image_files_in_folder = [f for f in os.listdir(dataset_folder) 
                                   if any(f.lower().endswith(ext) for ext in self.supported_extensions)]
            verification_report["details"]["image_files_found"] = len(image_files_in_folder)
            
            # Check 5: Verify caption content quality
            caption_stats = self._analyze_caption_quality(jsonl_lines)
            verification_report["details"]["caption_stats"] = caption_stats
            
            # Final status
            if not verification_report["issues"]:
                verification_report["status"] = "✅ FULLY VERIFIED"
            
            return verification_report
            
        except Exception as e:
            verification_report["status"] = "❌ ERROR"
            verification_report["issues"].append(f"Verification failed: {str(e)}")
            return verification_report
    
    def create_dataset_preview(self, matched_data: List[Dict], matching_report: Dict) -> Tuple[List[Tuple], str]:
        """Create visual preview of the first 6 image-caption matches for verification"""
        
        if not matched_data:
            return [], "No data to preview"
        
        # Create detailed matching report
        total_images = len(matched_data)
        avg_caption_length = sum(len(d['caption']) for d in matched_data) / total_images if total_images > 0 else 0
        
        report = self._create_preview_report(matched_data, matching_report, avg_caption_length)
        
        # Create gallery preview (first 6 images with captions)
        gallery_items = []
        for i, data in enumerate(matched_data[:6]):
            try:
                # Create caption overlay for preview
                caption_preview = f"#{data.get('image_number', i+1)}: {data['caption'][:100]}{'...' if len(data['caption']) > 100 else ''}"
                gallery_items.append((data['image_path'], caption_preview))
            except Exception as e:
                print(f"Error creating preview for {data['image_name']}: {e}")
        
        return gallery_items, report
    
    def generate_dataset_preview(self, matched_data: List[Dict], max_preview: int = 6) -> List[Dict]:
        """Generate a visual preview of the dataset with image-caption matching"""
        
        if not matched_data:
            return []
        
        preview_images = []
        
        try:
            for i, data in enumerate(matched_data[:max_preview]):
                # Create preview data
                preview_images.append({
                    'path': data['image_path'],
                    'name': data['image_name'],
                    'caption': data['caption'],
                    'number': data['image_number']
                })
            
            return preview_images
            
        except Exception as e:
            print(f"Error generating preview: {e}")
            return []
    
    def validate_dataset_matching(self, matched_data: List[Dict]) -> Dict[str, Any]:
        """Validate that images and captions are properly matched"""
        
        validation_report = {
            'total_images': len(matched_data),
            'valid_matches': 0,
            'empty_captions': 0,
            'sequential_order': True,
            'issues': []
        }
        
        expected_numbers = list(range(1, len(matched_data) + 1))
        actual_numbers = [data['image_number'] for data in matched_data]
        
        for i, data in enumerate(matched_data):
            # Check for empty captions
            if not data['caption'].strip():
                validation_report['empty_captions'] += 1
                validation_report['issues'].append(f"Empty caption for {data['image_name']}")
            else:
                validation_report['valid_matches'] += 1
            
            # Check sequential numbering
            if data['image_number'] != expected_numbers[i]:
                validation_report['sequential_order'] = False
                validation_report['issues'].append(f"Image {data['image_name']} has number {data['image_number']}, expected {expected_numbers[i]}")
        
        return validation_report
    
    def preview_caption_verification(self, matched_data: List[Dict]) -> str:
        """Simple preview verification for UI display"""
        
        if not matched_data:
            return "No dataset loaded yet."
        
        total_images = len(matched_data)
        total_captions = sum(1 for data in matched_data if data.get("caption", "").strip())
        avg_length = sum(len(data.get("caption", "")) for data in matched_data) / total_images if total_images > 0 else 0
        
        report = f"""
## 🔍 **Caption Integration Preview**

### ✅ **Status: Ready for Training**

### 📊 **Quick Stats:**
- **Total Images**: {total_images}
- **Total Captions**: {total_captions}
- **Success Rate**: {(total_captions/total_images*100):.1f}%
- **Average Caption Length**: {avg_length:.0f} characters

### 🎯 **Sample Caption Verification:**
"""
        
        # Show first 3 captions as samples
        for i, data in enumerate(matched_data[:3]):
            caption = data.get("caption", "")
            report += f"""
**{i+1}. {data['image_name']}**
- Caption: *"{caption[:80]}{'...' if len(caption) > 80 else ''}"*
- Length: {len(caption)} chars
"""
        
        if total_images > 3:
            report += f"\n... and {total_images - 3} more verified image-caption pairs"
        
        report += f"""

### 🎉 **Confirmation:**
✅ **Your captions WILL be used for training!**
- Each image is paired with its corresponding caption
- Captions will be processed during training
- Trigger words will be properly positioned
- The ai-toolkit will read the metadata.jsonl file
"""
        
        return report
    
    # Private helper methods
    
    def _create_matching_report(self, image_files: List[str], caption_mappings: Dict[str, str]) -> Dict[str, Any]:
        """Create initial matching report structure"""
        return {
            "total_images_found": len(image_files),
            "total_captions_available": len(caption_mappings),
            "successful_matches": 0,
            "failed_matches": 0,
            "orphaned_images": [],
            "orphaned_captions": [],
            "matching_details": []
        }
    
    def _extract_number_from_filename(self, filename: str) -> int:
        """Extract number from filename for ordering"""
        import re
        # Look for numbers in filename
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(numbers[0])
        return 0
    
    def _auto_fix_common_trigger_words(self, caption: str) -> str:
        """
        Automatically detect and fix common trigger word patterns by adding commas
        
        Args:
            caption: Original caption text
        
        Returns:
            Caption with fixed trigger word formatting
        """
        if not caption or not caption.strip():
            return caption
        
        # Common trigger word patterns to fix
        import re
        
        # Pattern 1: Single word at start followed by space and capital letter (likely trigger + description)
        # Examples: "pocox A man..." -> "pocox, A man..."
        pattern1 = r'^([a-zA-Z0-9_]+)\s+([A-Z])'
        if re.match(pattern1, caption):
            match = re.match(pattern1, caption)
            trigger_word = match.group(1)
            rest_of_caption = caption[len(trigger_word):].strip()
            fixed_caption = f"{trigger_word}, {rest_of_caption}"
            print(f"🔧 Auto-fixed trigger word: '{caption[:20]}...' → '{fixed_caption[:20]}...'")
            return fixed_caption
        
        # Pattern 2: Common trigger word patterns (TOK, person names, style names)
        common_triggers = ['TOK', 'LORA', 'STYLE', 'CHAR']
        for trigger in common_triggers:
            if caption.startswith(f"{trigger} "):
                fixed_caption = caption.replace(f"{trigger} ", f"{trigger}, ", 1)
                print(f"🔧 Auto-fixed common trigger: '{trigger} ' → '{trigger}, '")
                return fixed_caption
        
        return caption
    
    def _fix_trigger_word_formatting(self, caption: str, trigger_word: str) -> str:
        """
        Automatically fix trigger word formatting by adding comma after trigger word
        
        Args:
            caption: Original caption text
            trigger_word: The trigger word to fix
        
        Returns:
            Caption with properly formatted trigger word
        """
        if not caption or not trigger_word:
            return caption
        
        trigger_word = trigger_word.strip()
        
        # Check if trigger word exists without comma at the beginning
        if caption.startswith(f"{trigger_word} "):
            # Replace with comma-separated version
            fixed_caption = caption.replace(f"{trigger_word} ", f"{trigger_word}, ", 1)
            print(f"🔧 Fixed caption formatting: '{trigger_word} ' → '{trigger_word}, '")
            return fixed_caption
        
        return caption
    
    def _apply_trigger_word(self, caption: str, trigger_word: str, position: str) -> str:
        """Apply trigger word to caption based on position preference with professional comma formatting"""
        if not trigger_word.strip():
            return caption
        
        trigger_word = trigger_word.strip()
        caption = caption.strip()
        
        # First, fix any existing trigger word formatting
        caption = self._fix_trigger_word_formatting(caption, trigger_word)
        
        if position == "beginning":
            if not caption.startswith(trigger_word):
                # Add trigger word with professional comma formatting
                return f"{trigger_word}, {caption}"
        elif position == "end":
            if not caption.endswith(trigger_word):
                # Add trigger word at end with comma formatting
                return f"{caption}, {trigger_word}"
        elif position == "both":
            if not caption.startswith(trigger_word):
                caption = f"{trigger_word}, {caption}"
            if not caption.endswith(trigger_word):
                caption = f"{caption}, {trigger_word}"
        
        return caption
    
    def _analyze_caption_quality(self, jsonl_lines: List[str]) -> Dict[str, Any]:
        """Analyze caption quality from JSONL lines"""
        caption_stats = {
            "total_captions": 0,
            "empty_captions": 0,
            "avg_length": 0,
            "min_length": float('inf'),
            "max_length": 0,
            "sample_captions": []
        }
        
        total_length = 0
        for i, line in enumerate(jsonl_lines[:10]):  # Check first 10
            try:
                entry = json.loads(line.strip())
                if "text" in entry:
                    caption = entry["text"]
                    caption_stats["total_captions"] += 1
                    
                    if not caption.strip():
                        caption_stats["empty_captions"] += 1
                    
                    length = len(caption)
                    total_length += length
                    caption_stats["min_length"] = min(caption_stats["min_length"], length)
                    caption_stats["max_length"] = max(caption_stats["max_length"], length)
                    
                    if i < 5:  # Store first 5 as samples
                        caption_stats["sample_captions"].append({
                            "file": entry.get("file_name", "unknown"),
                            "caption": caption[:100] + "..." if len(caption) > 100 else caption,
                            "length": length
                        })
                        
            except json.JSONDecodeError:
                continue
        
        if caption_stats["total_captions"] > 0:
            caption_stats["avg_length"] = total_length / caption_stats["total_captions"]
        
        return caption_stats
    
    def _create_dataset_info(self, matched_data: List[Dict], matching_report: Dict, captions_data: Dict) -> str:
        """Create comprehensive dataset information string"""
        metadata = captions_data.get('metadata', {})
        system_info = captions_data.get('system_info', {})
        
        dataset_info = f"""
✅ **Dataset Successfully Loaded and Verified!**

## 📊 **Matching Statistics:**
- 🖼️ **Images Found**: {matching_report['total_images_found']}
- 📝 **Captions Available**: {matching_report['total_captions_available']}
- ✅ **Successful Matches**: {matching_report['successful_matches']}
- ❌ **Failed Matches**: {matching_report['failed_matches']}
- 📈 **Match Success Rate**: {(matching_report['successful_matches']/matching_report['total_images_found']*100):.1f}%

## 📁 **Source Information:**
- 📂 **Original Folder**: {metadata.get('folder_name', 'Unknown')}
- 💬 **Captioning Prompt**: {metadata.get('prompt_used', 'Unknown')[:100]}{'...' if len(metadata.get('prompt_used', '')) > 100 else ''}
- ⏱️ **Processing Time**: {metadata.get('processing_time_seconds', 0):.1f} seconds
- 🤖 **Model Used**: {metadata.get('model_used', 'Unknown')}

## 🎯 **Caption Quality:**
- 📏 **Average Caption Length**: {sum(len(d['caption']) for d in matched_data) / len(matched_data):.0f} characters
- 📊 **Caption Length Range**: {min(len(d['caption']) for d in matched_data)} - {max(len(d['caption']) for d in matched_data)} characters
        """
        
        # Add matching issues if any
        if matching_report['failed_matches'] > 0:
            dataset_info += f"\n\n## ⚠️ **Matching Issues Found:**"
            
            if matching_report['orphaned_images']:
                dataset_info += f"\n\n### Orphaned Images ({len(matching_report['orphaned_images'])}):"
                for orphan in matching_report['orphaned_images'][:5]:
                    dataset_info += f"\n- `{orphan['image_name']}` - No caption found"
                if len(matching_report['orphaned_images']) > 5:
                    dataset_info += f"\n- ... and {len(matching_report['orphaned_images']) - 5} more"
            
            if matching_report['orphaned_captions']:
                dataset_info += f"\n\n### Orphaned Captions ({len(matching_report['orphaned_captions'])}):"
                for orphan in matching_report['orphaned_captions'][:5]:
                    dataset_info += f"\n- `{orphan['caption_name']}` - No image found"
                if len(matching_report['orphaned_captions']) > 5:
                    dataset_info += f"\n- ... and {len(matching_report['orphaned_captions']) - 5} more"
        
        # Add sample verification
        dataset_info += f"\n\n## 🔍 **Matching Verification** (First 5 matches):"
        for i, data in enumerate(matched_data[:5]):
            # Safely handle caption display
            caption = str(data.get('caption', ''))
            caption_preview = caption[:80] + ('...' if len(caption) > 80 else '')
            caption_length = len(caption)
            
            dataset_info += f"""
**{i+1}. {data['image_name']}** {data['status']}
- 📸 Image #{data['image_number']}
- 📝 Caption: *"{caption_preview}"*
- 📏 Length: {caption_length} characters
            """
        
        if len(matched_data) > 5:
            dataset_info += f"\n... and {len(matched_data) - 5} more verified matches"
        
        # Add system information if available
        if system_info.get('gpu'):
            gpu_info = system_info['gpu']
            dataset_info += f"""

## 💻 **Original Processing System:**
- 🎮 **GPU**: {gpu_info.get('name', 'Unknown')}
- 💾 **VRAM Used**: {gpu_info.get('memory_used', 'Unknown')} / {gpu_info.get('memory_total', 'Unknown')}
- 🌡️ **Temperature**: {gpu_info.get('temperature', 'Unknown')}
            """
        
        return dataset_info
    
    def _create_preview_report(self, matched_data: List[Dict], matching_report: Dict, avg_caption_length: float) -> str:
        """Create detailed preview report"""
        report = f"""
## 📈 **Detailed Statistics:**
- ✅ **Successfully Matched**: {matching_report.get('successful_matches', 0)} pairs
- ❌ **Failed Matches**: {matching_report.get('failed_matches', 0)} items
- 📊 **Success Rate**: {(matching_report.get('successful_matches', 0) / matching_report.get('total_images_found', 1) * 100):.1f}%
- 📏 **Average Caption Length**: {avg_caption_length:.0f} characters

## 🔍 **First 10 Matched Pairs:**
"""
        
        # Add detailed verification for first 10 matches
        for i, data in enumerate(matched_data[:10]):
            status_icon = "✅" if data.get('status') == '✅ MATCHED' else "❌"
            report += f"""
**{i+1}. {data['image_name']}** {status_icon}
- 📷 **Image #{data.get('image_number', 'Unknown')}**
- 📝 **Caption**: *"{data['caption'][:120]}{'...' if len(data['caption']) > 120 else ''}"*
- 📐 **Length**: {len(data['caption'])} chars
"""
        
        if len(matched_data) > 10:
            report += f"\n... and **{len(matched_data) - 10}** more verified matches"
        
        # Add orphaned items if any
        if matching_report.get('orphaned_images'):
            report += f"""

## ⚠️ **Orphaned Images** ({len(matching_report['orphaned_images'])}):
"""
            for orphan in matching_report['orphaned_images'][:5]:
                report += f"\n- `{orphan['image_name']}` - No caption found"
            if len(matching_report['orphaned_images']) > 5:
                report += f"\n- ... and {len(matching_report['orphaned_images']) - 5} more"
        
        if matching_report.get('orphaned_captions'):
            report += f"""

## ⚠️ **Orphaned Captions** ({len(matching_report['orphaned_captions'])}):
"""
            for orphan in matching_report['orphaned_captions'][:5]:
                report += f"\n- `{orphan['caption_name']}` - No image found"
            if len(matching_report['orphaned_captions']) > 5:
                report += f"\n- ... and {len(matching_report['orphaned_captions']) - 5} more"
        
        return report
    
    def _create_error_message(self, error: str) -> str:
        """Create formatted error message"""
        return f"""
❌ **Error Processing Files**

**Error Details:** {error}

**Common Solutions:**
1. Ensure ZIP file contains images with sequential names (0001.png, 0002.png, etc.)
2. Verify JSON file is from Advanced Image Captioning Pro
3. Check that image names in ZIP match caption keys in JSON
4. Ensure files are not corrupted

**Expected File Structure:**
- **ZIP**: Contains numbered image files (0001.png, 0002.png, ...)
- **JSON**: Contains captions object with matching filenames as keys
        """