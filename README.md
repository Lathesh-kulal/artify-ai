# Artify AI 🎨

> **Real-Time Neural Style Transfer Studio powered by PyTorch & Streamlit**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/Tests-6%20Passing-brightgreen.svg)](tests/)

Artify AI transforms ordinary photos into museum-grade artistic masterpieces using **Feed-Forward Convolutional Neural Networks (Johnson et al.)** trained on perceptual loss functions. It executes in sub-second inference time (~0.05s on GPU, ~0.2s on CPU) through a modern dark-mode glassmorphic studio interface.

👉 **For the complete, deep-dive academic and technical documentation, see the [Project Report](docs/Project_Report.md).**

---

## ✨ Key Features

- **5 Master Artistic Styles**:
  - 🍬 **Candy**: Colorful Pop Art confectionery aesthetic.
  - 🧩 **Mosaic**: Byzantine stained-glass and segmented tile motifs.
  - 🌧️ **Rain Princess**: Leonid Afremov's vibrant Impressionist brush strokes.
  - ✨ **Starry Night**: Vincent van Gogh's post-impressionist nocturnal swirls.
  - 🎭 **Udnie**: Francis Picabia's Cubist/Futurist angular compositions.
- **Blazing Fast Inference**: Sub-second execution (~0.05s - 0.2s) powered by PyTorch with in-memory model caching.
- **Hardware Acceleration**: Automatic detection and optimization for NVIDIA CUDA (GPU) or multi-threaded CPU execution.
- **Advanced Artistic Controls**:
  - **Color Preservation**: YCrCb color-space chrominance decoupling to retain the original photo's natural color palette while applying artistic brush textures.
  - **Style Intensity Slider**: Continuous alpha blending factor (20% to 100%) between original content and stylized artwork.
  - **Resolution Selector**: Fast (512px), Standard (768px), or HD (1024px) with aspect-ratio preserving Lanczos resampling.
- **Interactive UI**:
  - Modern dark-mode glassmorphism interface with Google Outfit typography.
  - Side-by-side Before vs. After comparison view.
  - Instant one-click high-res artwork download (JPEG).
  - Built-in demo sample image for immediate testing.

---

## 🧠 How It Works Under the Hood

Unlike first-generation Neural Style Transfer (Gatys et al., 2015) which optimized pixel values iteratively using gradient descent taking minutes per image:

1. **Feed-Forward Inference (`TransformerNet`)**:
   - An encoder-bottleneck-decoder deep residual CNN transforms images in a **single forward pass**.
   - Input is downsampled by factor of 4 using strided convolutions with reflection padding.
   - 5 chained `ResidualBlock` units process features at constant dimensionality (128 channels).
   - Decoder reconstructs the image using nearest-neighbor interpolation upsampling followed by reflection-padded convolutions (eliminating checkerboard artifacts).
   - Instance Normalization (`InstanceNorm2d`) normalizes contrast and brightness per image independently.
2. **Color Preservation via YCrCb Space**:
   - Separates the stylized image into Luminance ($Y$) and Chrominance ($Cr, Cb$).
   - Swaps stylized chroma with original photo chroma to retain authentic skin tones and natural colors.
3. **In-Memory Caching**:
   - Models are dynamically cached in memory upon first load, making subsequent generations instantaneous.

---

## 🏗️ Project Architecture

```
Artify-AI/
├── app.py                      # Main Streamlit web application & UI
├── requirements.txt            # Python dependencies
├── config/
│   ├── constants.py            # Global styles and app constants
│   └── settings.py             # Default processing configuration
├── core/
│   ├── model_loader.py         # TransformerNet architecture & checkpoint loader
│   ├── image_processor.py      # Resizing, YCrCb color preservation & tensor transforms
│   ├── style_transfer.py       # Inference pipeline & in-memory model cache
│   └── utils.py                # Image encoding & export utilities
├── models/                     # Pre-trained PyTorch weight checkpoints (.pth)
│   ├── candy.pth               # 6.43 MB
│   ├── mosaic.pth              # 6.43 MB
│   ├── rain_princess.pth       # 6.43 MB
│   ├── starry-night.pth        # 6.41 MB
│   └── udnie.pth               # 6.43 MB
├── assets/                     # Static media assets & demo photos
│   └── samples/                # Sample test images
├── docs/                       # Project documentation
│   └── Project_Report.md       # Comprehensive technical & architectural report
├── outputs/                    # Export directory for stylized results
└── tests/                      # Automated unit test suite
    ├── test_image_processing.py# Preprocessing & resizing tests
    ├── test_model.py           # Forward pass & checkpoint tests
    └── test_utils.py           # Serialization & save tests
```

---

## 🚀 Quick Start

### 1. Activate Environment & Install Dependencies

```bash
# Clone the repository
git clone https://github.com/Lathesh-kulal/artify-ai.git
cd artify-ai

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Launch the Application

```bash
streamlit run app.py
```
Open your browser at **http://localhost:8501**.

---

## 🧪 Running Automated Tests

Run the complete test suite with Python's built-in test runner:

```bash
python -m unittest discover -s tests
```

Expected output:
```
......
----------------------------------------------------------------------
Ran 6 tests in 0.25s

OK
```

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **Deep Learning Engine**: [PyTorch](https://pytorch.org/) & [Torchvision](https://pytorch.org/vision/)
- **Image Processing**: [Pillow](https://python-pillow.org/) & [OpenCV (cv2)](https://opencv.org/)
- **Numerical Computing**: [NumPy](https://numpy.org/)
- **Architecture**: Feed-Forward Transformer Networks (Johnson et al.) with Instance Normalization (Ulyanov et al.)

---

## 📖 In-Depth Documentation

For complete mathematical formulations, network layer specifications, loss function equations, performance benchmarks, and deployment instructions, see:
- 📄 **[Comprehensive Project Report & Architecture Guide](docs/Project_Report.md)**

---

## 📜 License

This project is licensed under the MIT License — see the LICENSE file for details.