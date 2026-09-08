# Artify AI 🎨

> **Real-Time Neural Style Transfer Studio powered by PyTorch & Streamlit**

Artify AI transforms ordinary photos into artistic masterpieces using Feed-Forward Convolutional Neural Networks (Johnson et al.) trained on perceptual loss functions.

---

## ✨ Features

- **5 Master Artistic Styles**:
  - 🍬 **Candy**: Colorful Pop Art confectionery aesthetic.
  - 🧩 **Mosaic**: Byzantine stained-glass and segmented tile motifs.
  - 🌧️ **Rain Princess**: Leonid Afremov's vibrant Impressionist brush strokes.
  - ✨ **Starry Night**: Vincent van Gogh's post-impressionist nocturnal swirls.
  - 🎭 **Udnie**: Francis Picabia's Cubist/Futurist angular compositions.
- **Blazing Fast Inference**: Sub-second execution (~0.05s - 0.2s) powered by PyTorch with in-memory model caching.
- **Hardware Acceleration**: Automatic detection and optimization for NVIDIA CUDA (GPU) or multi-threaded CPU execution.
- **Advanced Artistic Controls**:
  - **Color Preservation**: YCrCb color-space blending to retain the original photo's color palette.
  - **Style Intensity Slider**: Blend original content with artistic stylized textures.
  - **Resolution Selector**: Fast (512px), Standard (768px), or HD (1024px).
- **Interactive UI**:
  - Modern dark-mode glassmorphism interface.
  - Side-by-side Before vs. After comparison view.
  - Instant one-click high-res artwork download (JPEG/PNG).
  - Built-in demo sample image for immediate testing.

---

## 🏗️ Project Architecture

```
Artify-AI/
├── app.py                      # Main Streamlit application
├── config/
│   ├── constants.py            # Global styles and app constants
│   └── settings.py             # Default application settings
├── core/
│   ├── model_loader.py         # TransformerNet architecture & checkpoint loader
│   ├── image_processor.py      # Image preprocessing, resizing & color preservation
│   ├── style_transfer.py       # Inference pipeline & model caching
│   └── utils.py                # Image encoding & export utilities
├── models/                     # Pre-trained PyTorch weight checkpoints (.pth)
│   ├── candy.pth
│   ├── mosaic.pth
│   ├── rain_princess.pth
│   ├── starry-night.pth
│   └── udnie.pth
├── assets/                     # Demo sample images
├── outputs/                    # Export directory for stylized results
├── tests/                      # Automated unit test suite
└── requirements.txt            # Project dependencies
```

---

## 🚀 Quick Start

### 1. Activate Environment & Install Dependencies
```bash
# Activate virtual environment
.venv\Scripts\activate

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

Run the complete test suite:
```bash
python -m unittest discover -s tests
```

---

## 🛠️ Tech Stack

- **Framework**: [Streamlit](https://streamlit.io/)
- **Deep Learning**: [PyTorch](https://pytorch.org/) & [Torchvision](https://pytorch.org/vision/)
- **Image Processing**: [Pillow](https://python-pillow.org/) & [OpenCV](https://opencv.org/)
- **Architecture**: Feed-Forward Transformer Networks (Johnson et al.)

---

## 📜 License
MIT License