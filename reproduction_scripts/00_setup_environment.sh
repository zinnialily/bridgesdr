set -e  # Exit on error

echo "========================================="
echo "ENVIRONMENT SETUP"
echo "========================================="

# Check Python version
echo "[1/6] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)"; then
    echo "ERROR: Python 3.8+ required. Found: $python_version"
    exit 1
fi
echo "Python $python_version detected"

# Create virtual environment
echo "[2/6] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo "Virtual environment created"
fi

# Activate virtual environment
echo "[3/6] Activating virtual environment..."
source venv/bin/activate
echo "Virtual environment activated"

# Upgrade pip
echo "[4/6] Upgrading pip..."
pip install --upgrade pip
echo "pip upgraded"

# Install PyTorch with CUDA support
echo "[5/6] Installing PyTorch with CUDA 11.8..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
echo "PyTorch installed"

# Install other requirements
echo "[6/6] Installing remaining dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "All dependencies installed"
else
    echo "WARNING: requirements.txt not found. Installing core dependencies manually..."
    pip install numpy pandas pillow opencv-python scikit-learn scikit-image \
                matplotlib seaborn plotly tqdm pyyaml rasterio geopandas
    echo "Core dependencies installed"
fi

# Verify GPU availability
echo ""
echo "========================================="
echo "VERIFYING GPU SETUP"
echo "========================================="
python3 << END
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU detected. Training will be very slow on CPU.")
END

# Create directory structure
echo ""
echo "========================================="
echo "CREATING DIRECTORY STRUCTURE"
echo "========================================="
mkdir -p data/{xbd,bright,processed,indices}
mkdir -p checkpoints/{disastergan,study2}
mkdir -p results/{study1,study2}
mkdir -p results/study1/{synthetic_images,masks,metrics,figures}
mkdir -p results/study2/{progressive_evaluation,cross_dataset,figures}
mkdir -p logs
echo "Directory structure created"

# Download pre-trained weights (optional)
echo ""
echo "========================================="
echo "DOWNLOADING PRE-TRAINED WEIGHTS (OPTIONAL)"
echo "========================================="
echo "To download pre-trained DisasterGAN and UNet weights:"
echo "  - Visit: https://github.com/zinnialily/non-trivial/checkpoints"
echo ""
echo "Skipping automatic download. You can run training from scratch."

echo ""
echo "========================================="
echo "SETUP COMPLETE!"
echo "========================================="
echo "Virtual environment activated. To deactivate, run: deactivate"
echo "Next steps:"
echo "  1. Run: bash reproduction_scripts/01_download_datasets.sh"
echo "  2. Or manually download datasets to data/xbd/ and data/bright/"
echo "========================================="
