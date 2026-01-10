

### 🚀 Automated Install (.sh)

This is the fastest and recommended method.

1.  **Save the Script**: Ensure the installation script from the previous guide is saved as `install.sh`.
2.  **Run in Terminal**: Make the script executable and run it.
    ```bash
    chmod +x install.sh
    ./install.sh
    ```

-----

### ⚙️ Manual Install (Ubuntu)

Follow these steps for a manual setup on an Ubuntu system.

1.  **Create & Activate Environment**

    ```bash
    # Create the virtual environment
    python -m venv venv

    # Activate it
    source venv/bin/activate
    ```

2.  **Install All Packages**

    ```bash
    # Install PyTorch for CUDA 12.6
    pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126

    # Install base requirements
    pip install -r requirements.txt

    # Install and pin key libraries for compatibility

    pip install --upgrade accelerate transformers diffusers huggingface_hub accelerate

    ```

    After installation, you can verify that PyTorch recognizes your GPU by running `python3 -c "import torch; print(torch.cuda.is_available())"`. It should return **`True`**.