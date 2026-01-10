# 🎯 FLUX LoRA Training Quality Guide

## ✅ How Training Works

### 1. **Dataset Preparation** (What You Upload)
- **Images**: 18 PNG files of your subject
- **Captions JSON**: Detailed descriptions for each image
- **Trigger Word**: `pxco` (your concept sentence)

### 2. **Training Process** (What Happens During Training)

```
📁 Your Data → 🔄 Processing → 🎓 Training → 💾 LoRA Model
```

#### Step-by-Step:
1. **Caption Integration** (Automatic):
   ```
   Original: "A middle-aged man with short gray hair..."
   Modified: "pxco, A middle-aged man with short gray hair..."
   ```
   - Trigger word added to **beginning** of every caption
   - All 18 images get processed this way

2. **Training Loop**:
   - Model learns: **"pxco" = this specific person's features**
   - Step 1-50: Learning basic features
   - Step 50-200: Learning details
   - Step 200-500: Refining quality
   - Step 500-1000: Mastering concept
   - Step 1000+: Fine-tuning

3. **Sampling** (Preview Generation):
   - Every 50 steps (configurable)
   - Uses **sample prompts** to test what model learned
   - **NOT the same as training captions**

### 3. **Sample Prompts** (What You See in Previews)

**Default Prompts** (Generic):
```yaml
- "pxco, professional portrait photo, high quality, detailed"
- "pxco, wearing formal attire, natural lighting, photorealistic"  
- "pxco, close-up shot, sharp focus, studio lighting"
```

**Your Training Captions** (Specific):
```
"pxco, A middle-aged man with short gray hair and a trimmed beard speaks into a microphone..."
"pxco, An older man with short gray hair, wearing glasses, is smiling warmly..."
```

⚠️ **Important**: Sample prompts are **test prompts** to see if model learned the concept. They're intentionally different from training captions!

---

## 🎓 Training Steps Guide

### **100 Steps** (Your Current Setting)
- ❌ **Too Few** for FLUX LoRA
- Result: Model barely learned the concept
- Quality: **Poor** - generic faces, not your character
- Use Case: Quick tests only

### **500 Steps** ⭐ (Minimum Recommended)
- ✅ **Acceptable** for simple subjects
- Result: Model understands basic features
- Quality: **Good** - recognizable character
- Use Case: Fast training, decent results

### **1000 Steps** ⭐⭐ (Recommended)
- ✅ **Best balance** for most cases
- Result: Model learned all features well
- Quality: **Excellent** - accurate character
- Use Case: Production-quality LoRAs

### **1500-2000 Steps** ⭐⭐⭐ (High Quality)
- ✅ **Professional** results
- Result: Perfect feature reproduction
- Quality: **Outstanding** - photo-realistic
- Use Case: Commercial work, portfolio pieces

---

## 🔍 Why Your 100-Step Training Didn't Work

### **What Happened:**

1. **Training Data**: ✅ CORRECT
   ```
   18 images with detailed captions like:
   "pxco, A middle-aged man with short gray hair and a trimmed beard..."
   ```

2. **Training Process**: ✅ CORRECT
   - Trigger word properly added
   - Captions properly formatted
   - Model trained on all images

3. **Problem**: ⚠️ **NOT ENOUGH STEPS**
   - At step 100, model only saw each image ~5-6 times
   - Not enough iterations to learn features
   - Model learned "generic man" not "your specific character"

4. **Sample Quality**: ❌ POOR
   ```
   Prompt: "A photo of pxco"
   Result: Generic face (because model didn't learn yet)
   ```

---

## 💡 How to Fix It

### **Option 1: Increase Training Steps** ⭐ RECOMMENDED

**Change in UI:**
```
Training Steps: 100 → 1000
```

**What Will Happen:**
- Training time: 8 minutes → 80 minutes
- Quality: Poor → Excellent
- Model will actually learn your character's features

### **Option 2: Better Sample Prompts**

**Current (Generic):**
```yaml
- "A photo of pxco"
- "pxco in a beautiful landscape"
```

**Improved (Descriptive):**
```yaml
- "pxco, professional portrait photo, high quality, detailed"
- "pxco, wearing formal attire, natural lighting, photorealistic"
- "pxco, middle-aged man with gray hair and beard, studio shot"
```

**Custom (Match Your Data):**
```yaml
- "pxco, middle-aged man with short gray hair and trimmed beard, wearing dark suit"
- "pxco, older man with glasses, smiling warmly, light blue shirt"
- "pxco, professional portrait, navy suit jacket, microphone, speaking"
```

### **Option 3: Use Expert Mode**

In Expert Mode YAML, add custom sample prompts:
```yaml
sample:
  prompts:
    - prompt: "pxco, middle-aged man with short gray hair and beard, professional portrait"
      seed: 42
    - prompt: "pxco, wearing dark navy suit, speaking at event, detailed"
      seed: 43
    - prompt: "pxco, close-up portrait, glasses, warm smile, natural lighting"
      seed: 44
```

---

## 📊 Training Quality Checklist

### ✅ **Your Dataset** (Verified Correct)
- [x] 18 images of subject
- [x] Detailed captions in JSON
- [x] Trigger word `pxco` properly set
- [x] Captions automatically integrated
- [x] Training data format correct

### ⚠️ **Training Settings** (Needs Adjustment)
- [ ] **Training Steps**: 100 → **1000+** (CRITICAL)
- [x] Learning Rate: 0.0005 (good)
- [x] Rank: 16 (good)
- [x] Batch Size: 1 (correct for FLUX)
- [x] EMA: Enabled (good)
- [x] Sample Every: 50 (good)

### 📝 **Sample Prompts** (Now Improved)
- [x] Default prompts now more descriptive
- [ ] Consider custom prompts matching your data
- [ ] Test prompts should vary in pose/lighting/context

---

## 🎯 Recommended Settings for Your Use Case

```yaml
Training Settings:
  Steps: 1000              # ⭐ Minimum for good quality
  Learning Rate: 0.0005    # Good default
  Rank: 16                 # Balanced size/quality
  Batch Size: 1            # Safe for FLUX
  Sample Every: 50         # See progress frequently
  
Sample Prompts:
  1: "pxco, professional portrait photo, detailed, high quality"
  2: "pxco, wearing formal attire, natural lighting, photorealistic"
  3: "pxco, middle-aged man with gray hair, close-up, sharp focus"
```

---

## 🚀 Next Steps

1. **Restart Training** with **1000 steps**:
   - Go to UI
   - Change "Training Steps" to 1000
   - Click "Start Training"
   - Wait ~80 minutes

2. **Monitor Samples**:
   - Check previews every 50 steps
   - Step 50: Basic shape
   - Step 200: Features emerging
   - Step 500: Good likeness
   - Step 1000: Excellent quality

3. **Test Your LoRA**:
   ```
   Prompt: "pxco, professional portrait, high quality"
   Expected: Accurate representation of your character
   ```

---

## 📚 Understanding the Verification

**Question**: "Does training use actual captions and trigger from JSON?"

**Answer**: ✅ **YES, ABSOLUTELY!**

```python
# Your JSON caption:
"A middle-aged man with short gray hair and a trimmed beard..."

# Processed for training:
"pxco, A middle-aged man with short gray hair and a trimmed beard..."

# Saved to dataset:
datasets/f651e6ab-4383-4b87-862b-ca0dd4342720/metadata.jsonl
```

**Verified in Your Training Run:**
```json
{"file_name": "0001.png", "text": "pxco, A middle-aged man with short gray hair and a trimmed beard speaks into a microphone..."}
{"file_name": "0002.png", "text": "pxco, An older man with short gray hair, wearing glasses, is smiling warmly..."}
```

✅ **Trigger word is properly added**
✅ **Full captions are used**
✅ **All 18 images processed correctly**

---

## ⚡ Quick Fix Summary

**Your Issue**: "Sample images don't look like my character"

**Root Cause**: Only 100 training steps (not enough)

**Solution**: Increase to **1000 steps** minimum

**Expected Result**: Model will learn your character's features and generate accurate samples

---

## 💬 Common Questions

**Q: Why do sample prompts look different from my captions?**
A: Sample prompts are **test prompts** to verify learning. They're intentionally simpler to test if the model learned the trigger word association.

**Q: Will the model forget my detailed captions?**
A: No! The model trains on your full detailed captions. Sample prompts are just for preview.

**Q: Can I use my training captions as sample prompts?**
A: Yes! In Expert Mode, you can set any prompts you want. But shorter prompts often test better.

**Q: How do I know when training is done?**
A: Check samples at step 500, 750, 1000. When quality stops improving significantly, it's done.

---

## 🎨 After Training - Using Your LoRA

```python
# In any FLUX inference tool:
prompt = "pxco, professional portrait, wearing suit, high quality"

# The model knows:
# - "pxco" = your specific character
# - Features learned from all 18 training images
# - Can generate new poses/angles/lighting
```

**Good Prompts**:
- "pxco, portrait photo, professional lighting"
- "pxco, wearing casual clothes, outdoor scene"
- "pxco, close-up face, smiling, warm lighting"

**Bad Prompts**:
- "portrait photo" (missing trigger word!)
- "pxco" (too short, no context)
- Very long prompts (dilutes the concept)

---

## ✅ Verification Complete

Your training setup is **CORRECT**:
- ✅ Captions properly integrated with trigger
- ✅ All 18 images processed
- ✅ Training data format perfect
- ✅ Config generation working
- ✅ Sampling working (fixed OrderedDict bug)

Only issue: **100 steps too few** → Increase to **1000 steps**

---

**Ready to train!** 🚀

Your next training run with 1000 steps will produce much better results!
