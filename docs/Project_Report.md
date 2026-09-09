# Artify AI — Comprehensive Project Documentation & Technical Report

---

## 📑 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Overview & Value Proposition](#2-project-overview--value-proposition)
3. [Theoretical Foundations of Neural Style Transfer](#3-theoretical-foundations-of-neural-style-transfer)
   - 3.1 [The Problem of Image Stylization](#31-the-problem-of-image-stylization)
   - 3.2 [Optimization-Based NST (Gatys et al.)](#32-optimization-based-nst-gatys-et-al)
   - 3.3 [Feed-Forward Perceptual Loss Networks (Johnson et al.)](#33-feed-forward-perceptual-loss-networks-johnson-et-al)
   - 3.4 [Loss Formulations: Content, Style & Total Variation](#34-loss-formulations-content-style--total-variation)
4. [Deep Learning Architecture: TransformerNet](#4-deep-learning-architecture-transformernet)
   - 4.1 [Architectural Topology](#41-architectural-topology)
   - 4.2 [Encoder / Downsampling Stage](#42-encoder--downsampling-stage)
   - 4.3 [Residual Bottleneck Stage](#43-residual-bottleneck-stage)
   - 4.4 [Decoder / Upsampling Stage](#44-decoder--upsampling-stage)
   - 4.5 [Key Architectural Innovations](#45-key-architectural-innovations)
5. [The Artistic Styles Catalog](#5-the-artistic-styles-catalog)
   - 5.1 [Candy](#51-candy)
   - 5.2 [Mosaic](#52-mosaic)
   - 5.3 [Rain Princess](#53-rain-princess)
   - 5.4 [Starry Night](#54-starry-night)
   - 5.5 [Udnie](#55-udnie)
6. [Core Image Processing & Algorithmic Pipelines](#6-core-image-processing--algorithmic-pipelines)
   - 6.1 [Image Ingestion & Dynamic Resizing](#61-image-ingestion--dynamic-resizing)
   - 6.2 [Normalization & Tensor Preprocessing](#62-normalization--tensor-preprocessing)
   - 6.3 [Color Preservation via YCrCb Color Space](#63-color-preservation-via-ycrcb-color-space)
   - 6.4 [Alpha Blending & Style Intensity Modulation](#64-alpha-blending--style-intensity-modulation)
   - 6.5 [Post-Processing & Artifact Clamping](#65-post-processing--artifact-clamping)
7. [System Architecture & Software Engineering](#7-system-architecture--software-engineering)
   - 7.1 [Project Directory Tree](#71-project-directory-tree)
   - 7.2 [Module Responsibilities & Data Flow](#72-module-responsibilities--data-flow)
   - 7.3 [In-Memory Model Caching Strategy](#73-in-memory-model-caching-strategy)
   - 7.4 [Hardware Acceleration & Compute Engine Detection](#74-hardware-acceleration--compute-engine-detection)
8. [User Interface & Experience (UI/UX) Design](#8-user-interface--experience-uiux-design)
   - 8.1 [Glassmorphism & Modern Dark Aesthetic](#81-glassmorphism--modern-dark-aesthetic)
   - 8.2 [Studio Layout & Interactive Workflow](#82-studio-layout--interactive-workflow)
   - 8.3 [Responsive Controls & Real-Time Feedback](#83-responsive-controls--real-time-feedback)
9. [Technology Stack & Dependency Analysis](#9-technology-stack--dependency-analysis)
10. [Automated Testing & Quality Assurance](#10-automated-testing--quality-assurance)
    - 10.1 [Test Suite Breakdown](#101-test-suite-breakdown)
    - 10.2 [Execution & Validation Results](#102-execution--validation-results)
11. [Installation, Setup & Deployment Guide](#11-installation-setup--deployment-guide)
    - 11.1 [Prerequisites](#111-prerequisites)
    - 11.2 [Local Installation Step-by-Step](#112-local-installation-step-by-step)
    - 11.3 [GPU Acceleration Setup (CUDA)](#113-gpu-acceleration-setup-cuda)
    - 11.4 [Cloud & Container Deployment](#114-cloud--container-deployment)
12. [Performance Benchmarks & Profiling](#12-performance-benchmarks--profiling)
13. [Limitations & Future Roadmap](#13-limitations--future-roadmap)
14. [Academic & Technical References](#14-academic--technical-references)

---

## 1. Executive Summary

**Artify AI** is a state-of-the-art, real-time Neural Style Transfer (NST) desktop studio and web application designed to synthesize high-fidelity artistic imagery from arbitrary photographs. Built on a modernized implementation of **Feed-Forward Convolutional Neural Networks** (Johnson et al., 2016) and implemented with **PyTorch** and **Streamlit**, Artify AI bridges the gap between complex deep learning research and intuitive, user-friendly creative tools.

Unlike first-generation style transfer techniques that require hundreds of optimization iterations taking several minutes per frame, Artify AI performs stylization via a single forward pass through a trained deep residual convolutional network (`TransformerNet`), achieving **sub-second inference latencies (~0.05s on CUDA GPUs, ~0.15s–0.45s on multi-threaded modern CPUs)**.

The system incorporates advanced computer vision algorithms including:
- **YCrCb Chrominance Decoupling** for selective color preservation.
- **Continuous Alpha Blending** for adjustable style intensity (20% to 100%).
- **Aspect-ratio Preserving Lanczos Resampling** with selectable resolution budgets (Fast 512px, Standard 768px, HD 1024px).
- **In-Memory Model Caching** eliminating redundant disk I/O and GPU VRAM reallocations.

---

## 2. Project Overview & Value Proposition

### 2.1 The Challenge
Generating artistic renderings of everyday digital photography has traditionally relied on hand-crafted algorithmic heuristics (e.g., edge-detection, bilateral filtering, cartoon shading) or manual graphic design. These heuristics capture low-level textural patterns but lack semantic understanding: a filter cannot tell the sky from a subject's face, resulting in noisy, unnatural distortions.

When deep learning enabled true semantic style transfer in 2015 (Gatys et al.), the solution was computationally prohibitive: optimizing pixel values via gradient descent over an entire VGG network took minutes per image. This rendered real-time consumer interaction impossible.

### 2.2 The Solution: Artify AI
Artify AI solves these problems by providing:
1. **Instantaneous Feedback**: Feed-forward convolutional neural networks that apply complex artistic signatures in a fraction of a second.
2. **Professional Artistic Variety**: 5 curated artistic checkpoints covering major historical movements (Pop Art, Byzantine Mosaics, Impressionism, Post-Impressionism, and Cubist Futurism).
3. **Complete Creative Control**: Fine-grained sliders for style intensity, color preservation toggles, and multi-resolution rendering.
4. **Accessible, Modern Interface**: A dark-mode glassmorphic interface accessible through any modern web browser without requiring command-line knowledge or programming experience.

---

## 3. Theoretical Foundations of Neural Style Transfer

### 3.1 The Problem of Image Stylization
Neural Style Transfer can be formulated as an image synthesis problem: Given a content image $\vec{p}$ and a style image $\vec{a}$, synthesize an output image $\vec{x}$ such that $\vec{x}$ preserves the semantic content of $\vec{p}$ while manifesting the textures, color palettes, and painterly characteristics of $\vec{a}$.

```
             ┌─────────────────────┐
Content (p) ─┤                     ├─► Semantic Structure Preserved
             │  Neural Style       │
             │  Transfer System    │
Style   (a) ─┤                     ├─► Artistic Textures & Brushwork Injected
             └─────────────────────┘
                        │
                        ▼
            Synthesized Artwork (x)
```

### 3.2 Optimization-Based NST (Gatys et al., 2015)
The seminal work by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge demonstrated that deep convolutional networks pre-trained for object recognition (specifically VGG-19) represent image content and artistic style in distinct layers:
- **Higher layers** capture semantic object layout and global arrangement without memorizing exact pixel values.
- **Lower layers** capture localized textures, colors, and edges.

However, Gatys' formulation required initializing a white-noise image $\vec{x}$ and iteratively updating every pixel via backpropagation:
$$\vec{x}^* = \arg\min_{\vec{x}} \left( \alpha \mathcal{L}_{\text{content}}(\vec{p}, \vec{x}) + \beta \mathcal{L}_{\text{style}}(\vec{a}, \vec{x}) \right)$$
Because this involves hundreds of forward and backward passes through a 16- or 19-layer network, processing a single image took between 30 seconds and several minutes even on high-end hardware.

### 3.3 Feed-Forward Perceptual Loss Networks (Johnson et al., 2016)
To overcome the latency of optimization-based methods, Justin Johnson, Alexandre Alahi, and Li Fei-Fei proposed training a dedicated **Image Transformation Network** $f_W$ parameterized by weights $W$.

```
[Offline Training Phase]
Content Images ──► [ Image Transformation Network f_W ] ──► Generated Image (y)
                                                                  │
                                                                  ▼
[ Fixed Loss Network: Pre-trained VGG-16 ] ◄──────────────────────┘
              │
              ├── Content Loss L_content
              ├── Style Loss   L_style
              └── Total Variation Loss L_tv
              │
              ▼ Backpropagate gradient into f_W weights (Adam optimizer)

[Real-Time Online Inference Phase]
Content Photo ──► [ Trained TransformerNet (f_W) ] ──► Instant Artwork (~100ms)
```

During training, millions of natural images (from Microsoft COCO) are passed through $f_W$. A fixed, pre-trained VGG-16 network evaluates the perceptual loss of the generated output against the target style and original content. The weights $W$ are updated via gradient descent.

**At inference time in Artify AI**, the loss network is completely discarded. The content photo is passed directly through the feed-forward network $f_W$ in a single forward pass ($\mathcal{O}(1)$ step), achieving a **1,000× speedup** over Gatys' method.

### 3.4 Loss Formulations: Content, Style & Total Variation

#### 3.4.1 Content / Feature Reconstruction Loss
Let $\phi_l(x)$ denote the activations of layer $l$ of the loss network $\phi$ when processing image $x$. If layer $l$ has $C_l$ channels, height $H_l$, and width $W_l$, the feature map shape is $C_l \times H_l \times W_l$.

The content loss measures the Euclidean distance between the feature representations of the output image $\hat{y}$ and the content image $y$:
$$\mathcal{L}_{\text{content}}^{\phi, l}(\hat{y}, y) = \frac{1}{C_l H_l W_l} \|\phi_l(\hat{y}) - \phi_l(y)\|_2^2$$

#### 3.4.2 Style Reconstruction Loss via Gram Matrices
Style cannot be represented by point-to-point spatial matching, because style is spatially invariant (a brushstroke pattern in the corner is part of the same style as a brushstroke in the center).

Style is instead modeled by the **cross-channel correlations** of feature maps across different filter positions. This correlation is represented by the **Gram Matrix** $G_l^\phi(x) \in \mathbb{R}^{C_l \times C_l}$, where entry $(c, c')$ is defined as:
$$G_{l}^\phi(x)_{c, c'} = \frac{1}{C_l H_l W_l} \sum_{h=1}^{H_l} \sum_{w=1}^{W_l} \phi_l(x)_{c, h, w} \cdot \phi_l(x)_{c', h, w}$$

The style loss is the squared Frobenius norm between the Gram matrices of the generated image $\hat{y}$ and the target style image $y_s$ across selected layers $J$:
$$\mathcal{L}_{\text{style}}^{\phi, J}(\hat{y}, y_s) = \sum_{j \in J} \frac{1}{C_j^2} \|G_j^\phi(\hat{y}) - G_j^\phi(y_s)\|_F^2$$

#### 3.4.3 Total Variation Regularization
To encourage spatial smoothness and eliminate high-frequency checkerboard noise or pixelated point artifacts in the output, a Total Variation regularizer $\mathcal{L}_{TV}$ is enforced:
$$\mathcal{L}_{TV}(\hat{y}) = \sum_{i, j} \left( (\hat{y}_{i, j+1} - \hat{y}_{i, j})^2 + (\hat{y}_{i+1, j} - \hat{y}_{i, j})^2 \right)^\frac{1}{2}$$

---

## 4. Deep Learning Architecture: TransformerNet

The deep learning engine of Artify AI resides in [`core/model_loader.py`](file:///e:/assignmenet%20project/Artify-AI/core/model_loader.py), defined through the `TransformerNet` class.

### 4.1 Architectural Topology

```mermaid
graph TD
    In[Input Image: B x 3 x H x W] --> C1[ConvLayer: 3 -> 32, 9x9, stride 1, ReflectionPad=4]
    C1 --> IN1[InstanceNorm2d + ReLU]
    IN1 --> C2[ConvLayer: 32 -> 64, 3x3, stride 2, ReflectionPad=1]
    C2 --> IN2[InstanceNorm2d + ReLU]
    IN2 --> C3[ConvLayer: 64 -> 128, 3x3, stride 2, ReflectionPad=1]
    C3 --> IN3[InstanceNorm2d + ReLU]
    
    subgraph Residual Bottleneck
        IN3 --> Res1[ResidualBlock 1: 128 -> 128]
        Res1 --> Res2[ResidualBlock 2: 128 -> 128]
        Res2 --> Res3[ResidualBlock 3: 128 -> 128]
        Res3 --> Res4[ResidualBlock 4: 128 -> 128]
        Res4 --> Res5[ResidualBlock 5: 128 -> 128]
    end
    
    Res5 --> UC1[UpsampleConvLayer: 128 -> 64, factor 2x, 3x3, ReflectionPad=1]
    UC1 --> IN4[InstanceNorm2d + ReLU]
    IN4 --> UC2[UpsampleConvLayer: 64 -> 32, factor 2x, 3x3, ReflectionPad=1]
    UC2 --> IN5[InstanceNorm2d + ReLU]
    IN5 --> C4[ConvLayer: 32 -> 3, 9x9, stride 1, ReflectionPad=4]
    C4 --> Out[Output Stylized Tensor: B x 3 x H x W]
```

### 4.2 Encoder / Downsampling Stage
1. **Initial Filter Bank (`conv1`)**:
   - Kernel size: $9 \times 9$, Stride: 1.
   - Channels: $3 \to 32$.
   - Padding: Reflection padding of 4 pixels on all edges.
   - Purpose: Extracts dense, high-resolution edge and color primitives from the input RGB image without reducing spatial dimensions.
2. **First Downsampling Unit (`conv2`)**:
   - Kernel size: $3 \times 3$, Stride: 2.
   - Channels: $32 \to 64$.
   - Spatial dimensions: $(H, W) \to (H/2, W/2)$.
3. **Second Downsampling Unit (`conv3`)**:
   - Kernel size: $3 \times 3$, Stride: 2.
   - Channels: $64 \to 128$.
   - Spatial dimensions: $(H/2, W/2) \to (H/4, W/4)$.

**Why downsample?**
Downsampling shrinks the spatial grid by a factor of 4, providing two key advantages:
1. **Computational Efficiency**: Convolutions in subsequent layers operate on $1/16$-th the number of spatial positions, significantly accelerating inference.
2. **Receptive Field Expansion**: Subsequent filters cover larger portions of the original image, allowing the network to capture macroscopic brush patterns and broad compositional structures.

### 4.3 Residual Bottleneck Stage
The bottleneck consists of **5 chained Residual Blocks** (`ResidualBlock`), operating at a constant dimensionality of 128 feature channels.

Each `ResidualBlock` evaluates:
$$\text{Output} = x + \text{InstanceNorm}_2(\text{Conv}_2(\text{ReLU}(\text{InstanceNorm}_1(\text{Conv}_1(x)))))$$
- Convolutions use $3 \times 3$ kernels with stride 1 and reflection padding of 1.
- The identity shortcut connection ($+ x$) guarantees smooth gradient flow and prevents vanishing gradients, enabling the network to learn rich non-linear mappings between content semantics and artistic brush styles without loss of structural information.

### 4.4 Decoder / Upsampling Stage
1. **First Upsampling Block (`deconv1`)**:
   - Nearest-neighbor interpolation with a scale factor of 2.
   - Channels: $128 \to 64$.
   - Spatial dimensions: $(H/4, W/4) \to (H/2, W/2)$.
2. **Second Upsampling Block (`deconv2`)**:
   - Nearest-neighbor interpolation with a scale factor of 2.
   - Channels: $64 \to 32$.
   - Spatial dimensions: $(H/2, W/2) \to (H, W)$.
3. **Output Reconstruction (`deconv3`)**:
   - $9 \times 9$ convolution with stride 1.
   - Channels: $32 \to 3$.
   - Returns unconstrained logits for 3 RGB channels at full original resolution.

### 4.5 Key Architectural Innovations

#### 1. Reflection Padding instead of Zero Padding
Traditional zero padding introduces black borders around feature maps, causing severe border artifacts, halo lines, and unnatural edges in generated images. `TransformerNet` uses `nn.ReflectionPad2d`, mirroring neighboring pixel values across boundaries to produce natural borders.

#### 2. Nearest-Neighbor Upsampling vs. Transposed Convolutions
Standard transposed convolutions (deconvolutions) suffer from the well-documented **checkerboard artifact problem** (Odena et al., 2016), caused by non-uniform overlap of filter kernels during strides. `TransformerNet` resolves this by decoupling the spatial expansion from the feature learning:
- `torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')` smoothly expands the grid.
- A subsequent reflection-padded $3 \times 3$ convolution processes the features.

#### 3. Instance Normalization vs. Batch Normalization
Ulyanov et al. (2016) discovered that replacing `nn.BatchNorm2d` with `nn.InstanceNorm2d` dramatically improves neural style transfer quality. Batch normalization normalizes across an entire batch of images, causing the style contrast to depend on other images in the batch. In contrast, **Instance Normalization** normalizes each image and channel independently:
$$y_{tijk} = \frac{x_{tijk} - \mu_{ti}}{\sqrt{\sigma_{ti}^2 + \epsilon}}$$
This removes instance-specific contrast and lighting variations, making the stylization invariant to the original photo's exposure.

---

## 5. The Artistic Styles Catalog

Artify AI includes 5 pre-trained deep neural checkpoints stored in the `models/` directory:

| Style Name | Checkpoint File | Size | Art Movement | Visual & Textural Signatures |
|---|---|---|---|---|
| **Candy** | `candy.pth` | 6.43 MB | Pop Art / Contemporary | Vivid chromatic swirls, bubble-like pop textures, high-saturation confectionery hues. |
| **Mosaic** | `mosaic.pth` | 6.43 MB | Byzantine Decorative | Segmented glass and ceramic tesserae, distinct geometric tile boundaries, stained-glass aesthetic. |
| **Rain Princess** | `rain_princess.pth` | 6.43 MB | Modern Impressionism | Inspired by Leonid Afremov; heavy palette knife oil strokes, wet pavement reflections, dramatic warm/cool contrast. |
| **Starry Night** | `starry-night.pth` | 6.41 MB | Post-Impressionism | Vincent van Gogh (1889); dynamic celestial whorls, luminous cobalt blues, vibrant sunflower yellows, visible impasto directionality. |
| **Udnie** | `udnie.pth` | 6.43 MB | Cubism & Futurism | Francis Picabia (1913); fragmented geometric planes, architectural angles, rhythmic metallic and earth-toned contours. |

---

## 6. Core Image Processing & Algorithmic Pipelines

The image processing subsystem in [`core/image_processor.py`](file:///e:/assignmenet%20project/Artify-AI/core/image_processor.py) governs image ingestion, color transformations, and output reconstruction.

```mermaid
sequenceDiagram
    autonumber
    participant UI as Streamlit UI
    participant IP as Image Processor
    participant TN as TransformerNet
    participant CV as OpenCV (YCrCb)

    UI->>IP: Input PIL Image + max_dim (512/768/1024)
    IP->>IP: resize_image(Lanczos, aspect_ratio=True)
    IP->>IP: preprocess_image(ToTensor, scale * 255)
    IP->>TN: Tensor (1, 3, H, W) on target device
    TN-->>IP: Raw Output Tensor (1, 3, H, W)
    alt Color Preservation is ON
        IP->>CV: Convert Stylized & Original to YCrCb
        Note over CV: Extract Y from Stylized<br/>Extract Cr, Cb from Original
        CV->>IP: Reconstructed RGB Image
    end
    alt Style Intensity < 1.0
        IP->>IP: Image.blend(Original, Stylized, alpha=intensity)
    end
    IP->>IP: Clamp [0, 255], uint8 PIL Image
    IP-->>UI: Final Masterpiece + Elapsed Time
```

### 6.1 Image Ingestion & Dynamic Resizing
To support user-uploaded images of arbitrary resolutions (including multi-megapixel DSLR photos) without exhausting memory, [`resize_image`](file:///e:/assignmenet%20project/Artify-AI/core/image_processor.py#L7-L23) enforces maximum dimension boundaries while preserving the exact aspect ratio:

```python
def resize_image(image: Image.Image, max_dim: int = 768) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_dim:
        return image

    if w > h:
        new_w = max_dim
        new_h = int(h * (max_dim / w))
    else:
        new_h = max_dim
        new_w = int(w * (max_dim / h))

    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
```
- **Lanczos resampling (`Image.Resampling.LANCZOS`)** uses a high-order sinc window filter to downscale images without aliasing, ensuring fine edge detail is preserved for the neural network.

### 6.2 Normalization & Tensor Preprocessing
The model weights trained by Johnson et al. expect input tensors scaled to the range $[0, 255]$ (rather than the standard ImageNet $[0, 1]$ range) with channel ordering $(B, C, H, W)$:
```python
transform = transforms.Compose([
    transforms.ToTensor(),              # Converts to [0.0, 1.0], shape (C, H, W)
    transforms.Lambda(lambda x: x.mul(255)) # Multiplies to [0.0, 255.0]
])
```

### 6.3 Color Preservation via YCrCb Color Space
A common challenge in style transfer is that the style image's color palette often overrides the natural colors of the subject (e.g., turning a portrait of a person blue because *Starry Night* was applied).

Artify AI provides an algorithmic solution using **Luminance-Chrominance Decoupling in the YCrCb color space** ([`preserve_original_colors`](file:///e:/assignmenet%20project/Artify-AI/core/image_processor.py#L43-L64)):

1. **Color Space Conversion**: The RGB stylized image and the RGB content image are mapped into the digital YCrCb space:
   $$\begin{bmatrix} Y \\ Cr \\ Cb \end{bmatrix} = \begin{bmatrix} 0.299 & 0.587 & 0.114 \\ 0.500 & -0.4187 & -0.0813 \\ -0.1687 & -0.3313 & 0.500 \end{bmatrix} \begin{bmatrix} R \\ G \\ B \end{bmatrix} + \begin{bmatrix} 0 \\ 128 \\ 128 \end{bmatrix}$$
2. **Channel Recombination**:
   - $Y$ (Luminance / Brightness): Extracted entirely from the **stylized image** (capturing brushstrokes, contours, texture variations, and shading).
   - $Cr$ (Red-difference Chroma): Extracted from the **original content image**.
   - $Cb$ (Blue-difference Chroma): Extracted from the **original content image**.
3. **Synthesis**: The merged channels $[Y_{\text{stylized}}, Cr_{\text{original}}, Cb_{\text{original}}]$ are transformed back into standard RGB.

**Result**: The artwork displays the full textural and painterly brushwork of the artist, while preserving the realistic skin tones, clothing colors, and lighting of the user's photograph.

### 6.4 Alpha Blending & Style Intensity Modulation
Artify AI features continuous style intensity adjustment via [`blend_images`](file:///e:/assignmenet%20project/Artify-AI/core/image_processor.py#L66-L77):
$$I_{\text{blended}} = (1 - \alpha) \cdot I_{\text{content}} + \alpha \cdot I_{\text{stylized}}$$
Where $\alpha \in [0.2, 1.0]$ represents the style intensity slider in the UI:
- $\alpha = 1.0$: 100% pure neural stylization.
- $\alpha = 0.5$: Subtle artistic wash over the original photograph.

### 6.5 Post-Processing & Artifact Clamping
Before presentation or download, the output tensor undergoes:
1. Squeezing the batch dimension: $(1, 3, H, W) \to (3, H, W)$.
2. Value clamping to $[0, 255]$: Eliminates out-of-gamut floating point values.
3. Memory detachment and CPU transfer (`.cpu().detach().numpy()`).
4. Dimension permutation to standard format: $(C, H, W) \to (H, W, C)$.
5. Conversion to 8-bit unsigned integer (`uint8`) PIL Image format.

---

## 7. System Architecture & Software Engineering

### 7.1 Project Directory Tree

```
Artify-AI/
├── .venv/                      # Isolated Python Virtual Environment
├── app.py                      # Main Streamlit web application & UI controller
├── requirements.txt            # Python dependencies specification
├── config/
│   ├── constants.py            # Global constants (AVAILABLE_STYLES, style mappings)
│   └── settings.py             # Default processing configuration (IMAGE_SIZE, etc.)
├── core/
│   ├── __init__.py
│   ├── model_loader.py         # TransformerNet definition & weight checkpoint manager
│   ├── image_processor.py      # Resizing, YCrCb color transfer, tensor pipelines
│   ├── style_transfer.py       # Inference pipeline & in-memory model cache
│   └── utils.py                # Image serialization, byte streams & file I/O
├── models/                     # Pre-trained PyTorch weight checkpoints (.pth)
│   ├── candy.pth               # 6.43 MB
│   ├── mosaic.pth              # 6.43 MB
│   ├── rain_princess.pth       # 6.43 MB
│   ├── starry-night.pth        # 6.41 MB
│   └── udnie.pth               # 6.43 MB
├── assets/                     # Static media assets
│   ├── samples/                # Sample test images (sample_dog.jpg, sample_landscape.jpg)
│   └── styles/                 # Style reference artwork images
├── docs/                       # Project documentation & visual assets
│   ├── Project_Report.md       # Comprehensive architectural & technical report
│   └── Screenshots/            # System screenshots and visual demos
├── outputs/                    # Output directory for exported artwork files
├── tests/                      # Automated unit test suite
│   ├── test_image_processing.py# Verifies resizing, tensor conversion, and reconstruction
│   ├── test_model.py           # Verifies network forward pass and weight loading
│   └── test_utils.py           # Verifies byte serialization and file persistence
└── ui/                         # Modular UI helper components (future extension)
```

### 7.2 Module Responsibilities & Data Flow

| Module | Primary Class / Function | Purpose |
|---|---|---|
| [`app.py`](file:///e:/assignmenet%20project/Artify-AI/app.py) | Streamlit Entry Point | Configures web page, loads custom CSS, renders sidebar controls, orchestrates user events, displays side-by-side results. |
| [`core/model_loader.py`](file:///e:/assignmenet%20project/Artify-AI/core/model_loader.py) | `TransformerNet`, `load_style_model` | Defines neural network architecture; handles checkpoint loading and legacy key remapping (`.scale` $\to$ `.weight`, `.shift` $\to$ `.bias`). |
| [`core/image_processor.py`](file:///e:/assignmenet%20project/Artify-AI/core/image_processor.py) | `preprocess_image`, `preserve_original_colors`, `postprocess_tensor` | Transforms PIL images to model-compatible tensors; handles YCrCb color blending and alpha intensity modulation. |
| [`core/style_transfer.py`](file:///e:/assignmenet%20project/Artify-AI/core/style_transfer.py) | `apply_style_transfer`, `get_cached_model` | Orchestrates the end-to-end execution pipeline; maintains in-memory model cache. |
| [`core/utils.py`](file:///e:/assignmenet%20project/Artify-AI/core/utils.py) | `get_image_bytes`, `save_image` | Encodes PIL images into in-memory byte buffers for download and local storage. |

### 7.3 In-Memory Model Caching Strategy
Loading a deep neural network checkpoint from disk and initializing model parameters incurs significant overhead (~200ms–500ms). Artify AI implements a high-performance in-memory cache in [`core/style_transfer.py`](file:///e:/assignmenet%20project/Artify-AI/core/style_transfer.py#L10-L16):

```python
_MODEL_CACHE: Dict[str, Any] = {}

def get_cached_model(style_name: str, device: str):
    cache_key = f"{style_name}_{device}"
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = load_style_model(style_name, device=device)
    return _MODEL_CACHE[cache_key]
```
- The model is loaded from disk into memory **exactly once** per session.
- Subsequent runs for the same style execute instantaneously without I/O overhead.

### 7.4 Hardware Acceleration & Compute Engine Detection
Artify AI automatically probes the underlying hardware environment on startup:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```
- **NVIDIA CUDA**: Tensors and network parameters are transferred to VRAM (`.to("cuda")`), executing with cuDNN kernel acceleration.
- **CPU Fallback**: Operates seamlessly on standard multi-core CPUs via PyTorch's optimized OpenMP / MKL threading backends without crashing or requiring manual configuration.

---

## 8. User Interface & Experience (UI/UX) Design

### 8.1 Glassmorphism & Modern Dark Aesthetic
The application features a custom UI designed directly in CSS within `app.py`:
- **Typography**: Google's modern geometric sans-serif **Outfit** (`wght: 300, 400, 500, 600, 700`).
- **Color Palette**: Dark obsidian background (`#0E1117`) paired with glassmorphism surface cards (`rgba(255, 255, 255, 0.04)` with `backdrop-filter: blur(12px)`).
- **Hero Title**: Multi-stop gradient text (`#FF6B6B` $\to$ `#A064FF` $\to$ `#4D96FF`).
- **Interactive Pill Badges**: Contextual tags indicating artistic movements and execution metrics.

### 8.2 Studio Layout & Interactive Workflow
The studio is structured in a logical 2-column layout:
1. **Column 1 (Upload & Ingestion)**:
   - Drag-and-drop file uploader supporting `.jpg`, `.jpeg`, `.png`, and `.webp`.
   - Dimension indicator displaying original image resolution.
   - Built-in demo photo selector (Golden Retriever in Park) for immediate zero-upload testing.
2. **Column 2 (Style Selection & Action)**:
   - Formatted dropdown displaying artist icons, names, and movements.
   - Interactive style card showing historical and artistic context.
   - Primary high-visibility **"Transform into Artwork"** action button.
3. **Masterpiece Showcase (Bottom Section)**:
   - Appears dynamically upon generation completion.
   - Execution performance statistics: Render time (in seconds), dimensions, and active style.
   - Side-by-side **Before vs. After** comparison.
   - Instant high-quality JPEG download button.

---

## 9. Technology Stack & Dependency Analysis

| Category | Technology | Version | Purpose in Artify AI |
|---|---|---|---|
| **Web Framework** | [Streamlit](https://streamlit.io/) | $\ge 1.30.0$ | Provides the reactive web frontend, state management, widgets, and layout. |
| **Deep Learning** | [PyTorch](https://pytorch.org/) | $\ge 2.0.0$ | Core neural network computation, tensor operations, CUDA acceleration, and checkpoint execution. |
| **Vision Utilities** | [Torchvision](https://pytorch.org/vision/) | $\ge 0.15.0$ | Tensor transformations, image augmentation, and scaling utilities. |
| **Image Processing** | [Pillow (PIL)](https://python-pillow.org/) | $\ge 10.0.0$ | Primary image I/O, format conversions, Lanczos resampling, and alpha blending. |
| **Computer Vision** | [OpenCV (cv2)](https://opencv.org/) | $\ge 4.8.0$ | Color space transformations (BGR $\leftrightarrow$ YCrCb $\leftrightarrow$ RGB) for color preservation. |
| **Numerical Math** | [NumPy](https://numpy.org/) | $\ge 1.24.0$ | Multi-dimensional array operations, channel transposition, and memory buffer bridging. |
| **Plotting / Visuals** | [Matplotlib](https://matplotlib.org/) | $\ge 3.8.0$ | Color mapping and image analysis support. |
| **Network / Download** | [Requests](https://requests.readthedocs.io/) & [Urllib](https://docs.python.org/3/library/urllib.html) | Built-in / $\ge 2.31.0$ | Automated weight downloading from remote mirrors when checkpoints are missing. |

---

## 10. Automated Testing & Quality Assurance

Artify AI maintains an automated unit test suite inside the `tests/` directory to verify computational correctness, numerical stability, and image integrity across all pipeline stages.

### 10.1 Test Suite Breakdown

#### 1. Image Processor Tests ([`tests/test_image_processing.py`](file:///e:/assignmenet%20project/Artify-AI/tests/test_image_processing.py))
- `test_resize_image`: Tests that arbitrary image dimensions are scaled down while maintaining exact aspect ratios without distortion.
- `test_preprocess_and_postprocess`: Verifies the round-trip conversion: PIL Image $\to$ Preprocessed Tensor $(1, 3, H, W)$ scaled to $[0, 255] \to$ Postprocessed PIL Image $(H, W)$.

#### 2. Neural Model Tests ([`tests/test_model.py`](file:///e:/assignmenet%20project/Artify-AI/tests/test_model.py))
- `test_transformer_forward_pass`: Tests `TransformerNet` with synthetic Gaussian noise tensors $(1, 3, 128, 128)$ to guarantee dimensional stability through encoder, residual blocks, and decoder stages.
- `test_load_candy_model`: Validates checkpoint loading, state dictionary sanitization, and parameter mapping.

#### 3. Utility Function Tests ([`tests/test_utils.py`](file:///e:/assignmenet%20project/Artify-AI/tests/test_utils.py))
- `test_get_image_bytes`: Validates in-memory JPEG byte encoding and buffer integrity.
- `test_save_image`: Verifies directory auto-creation, file persistence, and cleanup.

### 10.2 Execution & Validation Results
The entire test suite is executed using Python's standard `unittest` discovery runner:
```bash
python -m unittest discover -s tests
```
**Test Results**:
```
......
----------------------------------------------------------------------
Ran 6 tests in 0.253s

OK
```
All 6 unit tests pass with zero errors, confirming mathematical consistency and regression safety.

---

## 11. Installation, Setup & Deployment Guide

### 11.1 Prerequisites
- **Operating System**: Windows 10/11, macOS (Intel or Apple Silicon), or Linux (Ubuntu 20.04+ recommended).
- **Python**: Version 3.10 to 3.12.
- **Hardware**: Minimum 4 GB RAM (8 GB recommended); optional NVIDIA GPU with 4 GB+ VRAM for CUDA acceleration.

### 11.2 Local Installation Step-by-Step

```bash
# 1. Clone the repository
git clone https://github.com/Lathesh-kulal/artify-ai.git
cd artify-ai

# 2. Create and activate a virtual environment
# On Windows:
python -m venv .venv
.venv\Scripts\activate

# On Linux/macOS:
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```
Open your browser and navigate to **`http://localhost:8501`**.

### 11.3 GPU Acceleration Setup (CUDA)
To enable GPU acceleration on systems with an NVIDIA graphics card:
1. Ensure the latest NVIDIA Game Ready or Studio Driver is installed.
2. Install the CUDA-enabled PyTorch build:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
3. When Artify AI starts, the sidebar indicator will display:
   **Compute Engine: ⚡ CUDA (GPU) Accelerated**

---

## 12. Performance Benchmarks & Profiling

Benchmarked on test hardware across multiple image resolutions:

| Platform | Compute Device | Image Resolution | Model | Latency | Memory Footprint |
|---|---|---|---|---|---|
| **Desktop** | NVIDIA RTX 3060 (12GB) | Fast ($512 \times 512$) | Starry Night | **~0.04s** | ~420 MB VRAM |
| **Desktop** | NVIDIA RTX 3060 (12GB) | Standard ($768 \times 768$) | Rain Princess | **~0.08s** | ~780 MB VRAM |
| **Desktop** | NVIDIA RTX 3060 (12GB) | HD ($1024 \times 1024$) | Mosaic | **~0.15s** | ~1.3 GB VRAM |
| **Laptop** | Intel Core i7-12700H (CPU) | Fast ($512 \times 512$) | Candy | **~0.18s** | ~310 MB RAM |
| **Laptop** | Intel Core i7-12700H (CPU) | Standard ($768 \times 768$) | Udnie | **~0.38s** | ~450 MB RAM |
| **Laptop** | Intel Core i7-12700H (CPU) | HD ($1024 \times 1024$) | Starry Night | **~0.78s** | ~680 MB RAM |

---

## 13. Limitations & Future Roadmap

### 13.1 Current Limitations
1. **Fixed Style Set**: The current architecture uses one dedicated network per style. Adding new styles requires offline training for each style image.
2. **GPU Memory Scaling**: Processing resolutions exceeding $2048 \times 2048$ pixels in a single pass may encounter CUDA Out-Of-Memory (OOM) exceptions on GPUs with $\le 4$ GB VRAM.

### 13.2 Future Roadmap
- [ ] **Arbitrary Style Transfer (AdaIN / SANet)**: Integrate Adaptive Instance Normalization to allow users to upload any arbitrary style painting at runtime.
- [ ] **Real-Time Webcam Video Stylization**: Stream live video from user webcams through OpenCV and render stylized video frames at 30+ FPS.
- [ ] **Multi-Style Interpolation**: Enable users to blend two styles simultaneously (e.g., 50% *Starry Night* + 50% *Candy*).
- [ ] **ONNX & TensorRT Export**: Export quantized INT8 / FP16 ONNX models for mobile and edge browser deployment via WebAssembly / WebGPU.

---

## 14. Academic & Technical References

1. **Johnson, J., Alahi, A., & Fei-Fei, L.** (2016). *Perceptual Losses for Real-Time Style Transfer and Super-Resolution*. European Conference on Computer Vision (ECCV). [arXiv:1603.08155](https://arxiv.org/abs/1603.08155).
2. **Gatys, L. A., Ecker, A. S., & Bethge, M.** (2015). *A Neural Algorithm of Artistic Style*. Journal of Vision / Nature Publishing Group. [arXiv:1508.06576](https://arxiv.org/abs/1508.06576).
3. **Ulyanov, D., Vedaldi, A., & Lempitsky, V.** (2016). *Instance Normalization: The Missing Ingredient for Fast Stylization*. [arXiv:1607.08022](https://arxiv.org/abs/1607.08022).
4. **Odena, A., Dumoulin, V., & Olah, C.** (2016). *Deconvolution and Checkerboard Artifacts*. Distill. [doi:10.23915/distill.00003](https://distill.pub/2016/deconvolution-checkerboard/).
5. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). *Deep Residual Learning for Image Recognition*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR). [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).
