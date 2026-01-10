# 🎓 Expert Mode & Parameter Blocks - Complete Guide

## 📋 Table of Contents
1. [How Expert Mode Works](#how-expert-mode-works)
2. [Parameter Block Functionalities](#parameter-block-functionalities)
3. [YAML Override System](#yaml-override-system)
4. [Essential Parameters](#essential-parameters)
5. [Parameter Priority & Merging](#parameter-priority--merging)

---

## 🎯 How Expert Mode Works

### Overview
Expert Mode allows you to **override ANY parameter** using custom YAML that merges with the UI-selected preset.

### Flow Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                    USER CONFIGURES TRAINING                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: UI Parameters (Professional Training Parameters)       │
│  ✓ LoRA Name: "poco"                                           │
│  ✓ Trigger Word: "pxco"                                        │
│  ✓ Training Steps: 1500                                        │
│  ✓ Learning Rate: 0.0004                                       │
│  ✓ LoRA Rank: 32                                               │
│  ✓ LoRA Alpha: 32                                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Advanced Settings (Model, Batch, etc)                  │
│  ✓ FLUX Model: dev                                             │
│  ✓ Batch Size: 1                                               │
│  ✓ Gradient Accumulation: 1                                    │
│  ✓ Optimizer: adamw                                            │
│  ✓ Noise Scheduler: flowmatch                                  │
│  ✓ Training Precision: bf16                                    │
│  ✓ Gradient Checkpointing: ✓ ON                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Expert Mode YAML Override (OPTIONAL)                   │
│  ✓ Enable Expert YAML Override: ☑ CHECKED                     │
│  ✓ Custom YAML editor appears                                  │
│  ✓ Write custom YAML to override/extend                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Configuration Merge Process                            │
│                                                                 │
│  BASE CONFIG (from UI)                                         │
│         +                                                       │
│  EXPERT YAML OVERRIDE                                          │
│         =                                                       │
│  FINAL TRAINING CONFIG                                         │
│                                                                 │
│  Merge Rules:                                                  │
│  • YAML overrides UI values                                    │
│  • New parameters are added                                    │
│  • Nested configs are deep-merged                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Training Execution                                     │
│  → ai-toolkit run.py config.yaml                               │
└─────────────────────────────────────────────────────────────────┘
```

### Key Concepts

**1. UI Parameters = Base Configuration**
- All UI inputs create a base YAML config
- This includes presets, sliders, checkboxes, etc.

**2. Expert YAML = Override Layer**
- Custom YAML you write in Expert Mode
- **Merges with** (not replaces) base config
- Can add new parameters not in UI

**3. Priority System**
```
Expert YAML > UI Parameters > Preset Defaults
```

---

## 📊 Parameter Block Functionalities

### 1. 🎨 Professional Training Parameters

**Purpose**: Core training settings for quality and efficiency

| Parameter | Function | UI Control | Impact |
|-----------|----------|------------|--------|
| **LoRA Model Name** | Unique identifier for output | Text input | File naming |
| **Trigger Word/Phrase** | Activation word in prompts | Text input | LoRA activation |
| **Auto-fix Trigger Formatting** | Adds commas automatically | Checkbox | Caption quality |
| **Training Steps** | Total training iterations | Slider (100-5000) | Quality vs time |
| **Learning Rate** | Step size for optimization | Number input | Convergence speed |
| **LoRA Rank** | Model capacity/size | Slider (4-128) | Quality vs file size |
| **LoRA Alpha** | Scaling factor | Slider (1-128) | Training stability |

**YAML Mapping**:
```yaml
config:
  name: "<LoRA Model Name>"  # From UI
  
  network:
    type: "lora"
    linear: 32              # LoRA Rank from slider
    linear_alpha: 32        # LoRA Alpha from slider
  
  train:
    steps: 1500             # Training Steps from slider
    lr: 0.0004              # Learning Rate from input
```

**Expert Override Example**:
```yaml
# Override learning rate and add scheduler
train:
  lr: 0.0005  # Override UI value
  lr_scheduler: "constant"  # Add new parameter
  warmup_steps: 100  # Add warmup (not in UI)
```

---

### 2. ⚙️ Advanced Settings

**Purpose**: Model architecture and optimization settings

#### 🧠 Optimizer
| Option | Description | Memory | Speed | Quality |
|--------|-------------|--------|-------|---------|
| **adamw** | Standard optimizer | Medium | Medium | Good |
| **adamw8bit** | 8-bit quantized | Low | Fast | Good |
| **prodigy** | Adaptive learning | Medium | Medium | Excellent |

**YAML Mapping**:
```yaml
train:
  optimizer: "adamw"  # From dropdown
```

#### 🔊 Noise Scheduler
| Option | Description | Best For |
|--------|-------------|----------|
| **flowmatch** | FLUX native scheduler | FLUX models (default) |
| **ddpm** | Classic diffusion | Compatibility |

**YAML Mapping**:
```yaml
train:
  noise_scheduler: "flowmatch"
```

#### 🎯 Training Precision
| Option | Description | VRAM | Speed | Quality |
|--------|-------------|------|-------|---------|
| **bf16** | Brain float 16 | Low | Fast | Good |
| **fp16** | Float 16 | Low | Fast | Good |
| **fp32** | Float 32 | High | Slow | Best |

**YAML Mapping**:
```yaml
train:
  dtype: bf16
```

#### 💾 Save Precision
| Option | File Size | Quality | Compatibility |
|--------|-----------|---------|---------------|
| **float16** | Small | Good | Wide |
| **float32** | Large | Best | Universal |

**YAML Mapping**:
```yaml
save:
  dtype: float16
```

#### 🔧 Memory Optimizations

**Quantization**:
```yaml
model:
  quantize: false  # Checkbox state
  # When true: reduces VRAM but may impact quality
```

**Gradient Checkpointing**:
```yaml
train:
  gradient_checkpointing: true  # Checkbox state
  # Trades compute for memory
```

**Expert Override Example**:
```yaml
# Enable advanced memory saving
model:
  quantize: true
  low_vram: true  # Not in UI!

train:
  gradient_checkpointing: true
  cpu_offload_checkpointing: true  # Advanced option
```

---

### 3. 🎭 Model Settings

**Purpose**: FLUX model variant and hardware configuration

#### 🦊 FLUX Model
| Option | Quality | Speed | VRAM | Best For |
|--------|---------|-------|------|----------|
| **dev** | Highest | Slower | 24GB+ | Professional |
| **schnell** | Good | Faster | 16GB+ | Quick iteration |

**YAML Mapping**:
```yaml
model:
  name_or_path: "black-forest-labs/FLUX.1-dev"  # or schnell
  is_flux: true
```

#### 🧮 Batch Size
- **Range**: 1-8 images per step
- **Trade-off**: Higher = faster but more VRAM

**YAML Mapping**:
```yaml
train:
  batch_size: 1
```

#### 📈 Gradient Accumulation
- **Purpose**: Simulate larger batches
- **Effective Batch** = Batch Size × Gradient Accumulation

**YAML Mapping**:
```yaml
train:
  gradient_accumulation_steps: 1
```

#### 💾 Low VRAM Mode
- **Enables**: Quantization + optimizations
- **For**: GPUs with <24GB VRAM

**YAML Mapping**:
```yaml
model:
  low_vram: true  # Checkbox enables this
```

**Expert Override Example**:
```yaml
# Use schnell with aggressive memory saving
model:
  name_or_path: "black-forest-labs/FLUX.1-schnell"
  quantize: true
  low_vram: true
  
train:
  batch_size: 1
  gradient_accumulation_steps: 4  # Effective batch = 4
```

---

### 4. 📸 Sampling Settings

**Purpose**: Generate preview images during training

| Parameter | Function | Default | Range |
|-----------|----------|---------|-------|
| **Sample Every** | Steps between previews | 250 | 100-1000 |
| **Width** | Preview width (px) | 1024 | 512-2048 |
| **Height** | Preview height (px) | 1024 | 512-2048 |
| **Guidance Scale** | Prompt adherence | 4 | 1-20 |
| **Sample Steps** | Inference steps | 20 | 10-50 |
| **Seed** | Random seed | 42 | Any |
| **Walk Seed** | Vary seed each sample | true | - |

**YAML Mapping**:
```yaml
sample:
  sampler: "flowmatch"
  sample_every: 250
  width: 1024
  height: 1024
  guidance_scale: 4
  sample_steps: 20
  seed: 42
  walk_seed: true
  neg: ""  # Negative prompt
  prompts:
    - "your sample prompt 1"
    - "your sample prompt 2"
```

**Expert Override Example**:
```yaml
# Custom sampling configuration
sample:
  sample_every: 100  # More frequent previews
  width: 768
  height: 1024  # Portrait orientation
  guidance_scale: 3.5
  sample_steps: 28  # Higher quality previews
  
  # Add more prompts
  prompts:
    - "portrait of [trigger], professional photo"
    - "[trigger] in casual clothes, outdoor setting"
    - "[trigger] close-up, dramatic lighting"
```

---

### 5. 📁 Dataset Settings

**Purpose**: Configure data loading and augmentation

| Parameter | Function | Default |
|-----------|----------|---------|
| **Caption Extension** | File format for captions | .txt |
| **Caption Dropout** | % of steps without caption | 0.05 (5%) |
| **Shuffle Tokens** | Randomize caption word order | false |
| **Cache Latents** | Pre-encode images | true |
| **Resolutions** | Supported image sizes | [512, 768, 1024] |

**YAML Mapping**:
```yaml
datasets:
  - folder_path: "/path/to/dataset"
    caption_ext: "txt"
    caption_dropout_rate: 0.05
    shuffle_tokens: false
    cache_latents_to_disk: true
    resolution: [512, 768, 1024]
```

**Expert Override Example**:
```yaml
# Advanced dataset configuration
datasets:
  - folder_path: "/path/to/dataset"
    caption_ext: "txt"
    caption_dropout_rate: 0.1  # Higher dropout
    shuffle_tokens: true  # Enable shuffling
    cache_latents_to_disk: true
    
    # Multi-resolution training
    resolution: [512, 640, 768, 896, 1024, 1152]
    
    # Advanced options
    repeats: 1  # Repeat dataset
    flip_aug: false  # No horizontal flip
    random_crop: false  # Center crop only
```

---

### 6. 🎛️ EMA Settings

**Purpose**: Exponential Moving Average for stable training

| Parameter | Function | Default | Recommended |
|-----------|----------|---------|-------------|
| **Use EMA** | Enable EMA | true | true |
| **EMA Decay** | Averaging rate | 0.9999 | 0.99-0.9999 |

**YAML Mapping**:
```yaml
train:
  ema_config:
    use_ema: true
    ema_decay: 0.9999
```

**What EMA Does**:
- Maintains smoothed version of weights
- Reduces training noise
- Often produces better results
- Slight memory overhead

**Expert Override Example**:
```yaml
# Adjust EMA for different training lengths
train:
  ema_config:
    use_ema: true
    ema_decay: 0.999  # Faster decay for short training
    ema_update_every: 1  # Update frequency
```

---

### 7. 🔬 Expert Mode

**Purpose**: Full YAML control for advanced users

#### When to Use Expert Mode

✅ **USE when**:
- Need parameters not in UI
- Want specific layer targeting
- Custom learning rate schedules
- Advanced memory optimizations
- Research/experimental features

❌ **DON'T USE when**:
- New to LoRA training
- Happy with UI presets
- Not familiar with YAML
- Want simple workflow

#### How Expert Mode Works

1. **Enable Checkbox**: `☑ Enable Expert YAML Override`
2. **Editor Appears**: YAML text editor
3. **Write Override**: Custom YAML config
4. **Merge Happens**: Your YAML + UI config → Final config
5. **Training Runs**: With merged configuration

#### Expert Override Structure

```yaml
# Structure matches ai-toolkit config format
job: extension  # Required
config:
  name: "custom_name"  # Optional: override UI name
  
  process:
    - type: 'sd_trainer'
      
      # Override any section
      network:
        type: "lora"
        linear: 64  # Override UI rank
        
      network_kwargs:
        # Target specific layers (ADVANCED)
        only_if_contains:
          - "transformer.single_transformer_blocks."
      
      train:
        lr: 0.0005
        # Add parameters not in UI
        lr_scheduler: "cosine"
        warmup_steps: 100
        
      # Add entirely new sections
      custom_param: value
```

---

## 🎯 Essential Parameters by Use Case

### 1. Character/Person Training
```yaml
# UI Settings:
- LoRA Rank: 32-64
- Training Steps: 1500-3000
- Learning Rate: 0.0004
- Batch Size: 1

# Expert Override:
network_kwargs:
  only_if_contains:
    - "transformer.single_transformer_blocks."  # Focus on details

train:
  caption_dropout_rate: 0.05  # Low dropout for consistency
```

### 2. Style/Concept Training
```yaml
# UI Settings:
- LoRA Rank: 16-32
- Training Steps: 1000-2000
- Learning Rate: 0.0004
- Batch Size: 1

# Expert Override:
datasets:
  - shuffle_tokens: true  # Variation
    caption_dropout_rate: 0.15  # Higher dropout for style
```

### 3. Quick Testing
```yaml
# UI Settings:
- Model: schnell
- Training Steps: 500
- Learning Rate: 0.0005
- Sample Every: 100

# Expert Override:
train:
  gradient_accumulation_steps: 2  # Faster convergence
  
sample:
  sample_every: 100  # Frequent previews
  sample_steps: 10  # Fast sampling
```

### 4. Maximum Quality
```yaml
# UI Settings:
- Model: dev
- LoRA Rank: 64-128
- Training Steps: 3000-5000
- Learning Rate: 0.0002
- Quantization: OFF

# Expert Override:
model:
  quantize: false  # No quantization
  
train:
  dtype: bf16  # Keep bf16 for speed
  
save:
  dtype: float32  # Save in float32 for quality
  
sample:
  sample_steps: 28  # High quality previews
```

---

## 🔄 Parameter Priority & Merging

### Priority System
```
┌─────────────────────────────────────┐
│ 1. Expert YAML Override (Highest)  │
│    ↓ overrides                      │
│ 2. UI Parameter Values              │
│    ↓ overrides                      │
│ 3. Preset Defaults (Lowest)         │
└─────────────────────────────────────┘
```

### Merge Examples

#### Example 1: Simple Override
```yaml
# UI Config (generated):
train:
  lr: 0.0004
  steps: 1500

# Expert YAML:
train:
  lr: 0.0005  # Override

# Result (merged):
train:
  lr: 0.0005     # From Expert YAML
  steps: 1500    # From UI
```

#### Example 2: Adding Parameters
```yaml
# UI Config:
train:
  optimizer: "adamw"
  lr: 0.0004

# Expert YAML:
train:
  lr_scheduler: "cosine"  # New parameter
  warmup_steps: 100       # New parameter

# Result (merged):
train:
  optimizer: "adamw"        # From UI
  lr: 0.0004                # From UI
  lr_scheduler: "cosine"    # Added from Expert
  warmup_steps: 100         # Added from Expert
```

#### Example 3: Deep Merge
```yaml
# UI Config:
network:
  type: "lora"
  linear: 32

# Expert YAML:
network:
  linear: 64  # Override
network_kwargs:
  only_if_contains:  # New section
    - "transformer."

# Result (merged):
network:
  type: "lora"           # From UI
  linear: 64             # From Expert (overridden)
network_kwargs:
  only_if_contains:      # From Expert (new)
    - "transformer."
```

---

## 📋 Complete Parameter Reference

### YAML Structure Map
```yaml
job: extension

config:
  name: "model_name"
  process:
    - type: 'sd_trainer'
      training_folder: "output"
      device: cuda:0
      
      # Network Configuration
      network:
        type: "lora"
        linear: 32              # Rank
        linear_alpha: 32        # Alpha
      
      network_kwargs:
        # Advanced layer targeting
        only_if_contains: []
        exclude_if_contains: []
      
      # Model Configuration
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"
        is_flux: true
        quantize: false
        low_vram: false
      
      # Training Configuration
      train:
        batch_size: 1
        steps: 1500
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw"
        lr: 0.0004
        dtype: bf16
        
        # EMA
        ema_config:
          use_ema: true
          ema_decay: 0.9999
        
        # Advanced (Expert)
        lr_scheduler: "constant"
        warmup_steps: 0
        linear_timesteps: false
        skip_first_sample: false
        disable_sampling: false
      
      # Save Configuration
      save:
        dtype: float16
        save_every: 250
        max_step_saves_to_keep: 4
        push_to_hub: false
        hf_repo_id: ""
        hf_private: true
      
      # Sampling Configuration
      sample:
        sampler: "flowmatch"
        sample_every: 250
        width: 1024
        height: 1024
        guidance_scale: 4
        sample_steps: 20
        seed: 42
        walk_seed: true
        neg: ""
        prompts: []
      
      # Dataset Configuration
      datasets:
        - folder_path: "/path/to/dataset"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [512, 768, 1024]
          repeats: 1
          flip_aug: false
          random_crop: false
      
      # Logging (Expert)
      logging:
        log_every: 10
        use_wandb: false
        wandb_project: ""
        
      # Validation (Expert)
      validation:
        validate_every: 500
        num_validation_images: 4

meta:
  name: "[name]"
  version: "1.0"
```

---

## 💡 Pro Tips

### 1. **Start Simple**
- Use UI presets first
- Enable Expert Mode only when needed
- Add parameters incrementally

### 2. **Test Overrides**
- Start training with small steps
- Verify config is correct
- Check terminal output for warnings

### 3. **Common Expert Overrides**
```yaml
# Layer targeting (common)
network_kwargs:
  only_if_contains:
    - "transformer.single_transformer_blocks."

# Learning rate schedule (common)
train:
  lr_scheduler: "cosine"
  warmup_steps: 100

# Multi-resolution (common)
datasets:
  - resolution: [512, 640, 768, 896, 1024]

# Validation samples (useful)
validation:
  validate_every: 500
  num_validation_images: 4
```

### 4. **Debugging**
- Check terminal for YAML merge output
- Look for "Using configuration" message
- Verify parameters in training logs

---

## 🎓 Summary

| UI Block | Controls | When to Use Expert Override |
|----------|----------|----------------------------|
| **Professional Training** | Core params (rank, LR, steps) | Custom schedulers, warmup |
| **Advanced Settings** | Optimizer, precision, memory | Custom optimizers, CPU offload |
| **Model Settings** | FLUX variant, batch, VRAM | Specific model paths, quantization |
| **Sampling** | Preview generation | Custom prompts, resolutions |
| **Dataset** | Data loading, augmentation | Multi-dataset, advanced aug |
| **EMA** | Moving average | Custom decay rates |
| **Expert Mode** | YAML override everything | Advanced features, research |

### Key Takeaway
**Expert Mode = Full Control**
- UI provides 80% of common use cases
- Expert YAML provides 100% flexibility
- Use when you know what you're doing!

---

**Last Updated**: 2025-10-17
**For**: World-Class FLUX LoRA Trainer v2.0
