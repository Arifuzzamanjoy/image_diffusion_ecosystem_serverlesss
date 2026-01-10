# Video LoRA Trainer UI

A Gradio-based user interface for training video LoRAs using WAN 2.1 models with Ostris' AI Toolkit. This UI allows you to easily train high-quality video LoRAs on consumer-grade hardware.

## Features

- **Easy-to-use Gradio interface** - Train video LoRAs without complex command-line setup
- **WAN 2.1 support** - Train on both WAN 2.1 1B and 14B models
- **Frame-based training** - AI-Toolkit trains on individual frames for video models
- **Low VRAM optimization** - Support for consumer GPUs with memory optimizations
- **Automatic captioning** - Use Florence-2 for AI-powered image captioning with video enhancements
- **Hugging Face integration** - Automatically push trained models to Hugging Face Hub
- **Video generation samples** - Test your trained model with sample video prompts

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended: 12GB+ VRAM for WAN 2.1 1B, 24GB+ for WAN 2.1 14B)
- AI Toolkit installed and configured
- Required Python packages (see requirements.txt)

## Usage

1. **Start the UI:**
   ```bash
   python video_lora_train_ui.py
   ```

2. **Prepare your training data:**
   - Upload 10-50 images representing frames or key moments from your desired video style
   - Images should be representative of the character, style, or concept you want to train
   - Optionally upload .txt files with the same names as images for custom captions

3. **Configure training:**
   - **LoRA Name**: Give your model a unique name
   - **Trigger Word**: Use a unique trigger word (required for WAN 2.1 14B on low VRAM)
   - **Model Selection**: Choose between WAN 2.1 1B (faster, less VRAM) or 14B (higher quality)
   - **Training Steps**: 1000-4000 steps typically work well
   - **Learning Rate**: Default 1e-4 is usually good
   - **LoRA Rank**: Higher rank = more capacity but larger file size

4. **Video Settings:**
   - **Resolution**: Set output video width/height (832x480 default for good quality/speed balance)
   - **Frames**: Number of frames to generate (40 default for ~2.7 second clips at 15fps)
   - **FPS**: Frames per second for output videos

5. **Training Process:**
   - The system will create a dataset from your uploaded images
   - Training will begin automatically
   - Sample videos will be generated during training (if enabled)
   - Final model will be saved locally and optionally pushed to Hugging Face

## Important Notes

### Video Training Limitations
- **Frame-based training**: AI-Toolkit currently trains on individual frames, not video sequences
- **Best for**: Character/style consistency, not complex actions or motion
- **Works well for**: Person appearance, artistic styles, consistent character looks
- **Less effective for**: Complex animations, specific movements, temporal dynamics

### Model Differences
- **WAN 2.1 1B**: Faster training, lower VRAM usage, good quality
- **WAN 2.1 14B**: Higher quality output, requires more VRAM, slower training

### VRAM Requirements
- **WAN 2.1 1B**: ~12GB VRAM minimum
- **WAN 2.1 14B**: ~24GB VRAM minimum
- **Low VRAM mode**: Enables optimizations but may require trigger words

### Sample Generation
- Generated samples are animated WebP files
- If animations don't play in your file browser, open them in a web browser
- Samples are generated at intervals during training to monitor progress

## Tips for Best Results

1. **Image Selection:**
   - Use high-quality, diverse images
   - Include different angles, lighting, and contexts
   - Ensure consistent character/style across images

2. **Captioning:**
   - Use the AI captioning feature for consistency
   - Add motion-related terms manually if needed
   - Include your trigger word in captions

3. **Training Parameters:**
   - Start with default settings for your first model
   - Increase steps if underfitting, decrease if overfitting
   - Monitor sample outputs during training

4. **Video Prompts:**
   - Include motion descriptors ("walking", "dancing", "turning")
   - Specify camera angles ("close-up", "wide shot")
   - Add environmental context ("in a park", "on stage")

## Troubleshooting

- **Out of Memory**: Enable low VRAM mode, reduce resolution, or use WAN 2.1 1B
- **Poor Quality**: Increase training steps, improve training data quality
- **Not Learning**: Check trigger word usage, increase learning rate slightly
- **Overfitting**: Reduce training steps, lower learning rate

## Output Files

Trained models are saved in the `output/` directory and include:
- LoRA weights (.safetensors)
- Training configuration
- Sample videos (if enabled)
- Training logs

## Advanced Configuration

Use the "Even more advanced options" section to customize:
- Optimizer settings
- Noise scheduler parameters
- EMA configuration
- Custom sampling settings

See the default YAML configuration in the UI for available options.
