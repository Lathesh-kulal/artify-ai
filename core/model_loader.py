import os
import torch
import torch.nn as nn
import urllib.request
import re

# --------------------------------------------------
# Neural Network Architecture (Johnson et al.)
# --------------------------------------------------

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = torch.nn.functional.interpolate(x, scale_factor=self.upsample, mode='nearest')
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        # 5 Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


# --------------------------------------------------
# Model Download & Weight Utilities
# --------------------------------------------------

MODEL_FILENAMES = {
    "Candy": "candy.pth",
    "Mosaic": "mosaic.pth",
    "Rain Princess": "rain_princess.pth",
    "Starry Night": "starry-night.pth",
    "Udnie": "udnie.pth"
}

# Standard direct hosted URLs for fast style transfer models
MODEL_URLS = {
    "Candy": "https://www.dropbox.com/s/73rh5stjxyv2noi/candy.pth?dl=1",
    "Mosaic": "https://www.dropbox.com/s/9m4p6f467v5k9re/mosaic.pth?dl=1",
    "Rain Princess": "https://www.dropbox.com/s/e41qgy3m18wms19/rain_princess.pth?dl=1",
    "Udnie": "https://www.dropbox.com/s/ep61w8p7s5f81s1/udnie.pth?dl=1"
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def get_model_path(style_name: str) -> str:
    filename = MODEL_FILENAMES.get(style_name, f"{style_name.lower().replace(' ', '_')}.pth")
    return os.path.join(MODELS_DIR, filename)


def download_model(style_name: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    target_path = get_model_path(style_name)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000000:
        return target_path

    url = MODEL_URLS.get(style_name)
    if not url:
        raise ValueError(f"No direct download URL found for style '{style_name}'.")

    print(f"Downloading model weights for {style_name} to {target_path}...")
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, target_path)

    return target_path


def load_style_model(style_name: str, device: str = "cpu") -> TransformerNet:
    """
    Loads and returns a pre-trained TransformerNet model for the chosen style.
    Handles legacy checkpoints (where running_mean/var might have different keys).
    """
    model_path = get_model_path(style_name)
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        download_model(style_name)

    model = TransformerNet()
    state_dict = torch.load(model_path, map_location=device, weights_only=False)

    # Convert legacy instance norm parameters (scale -> weight, shift -> bias)
    clean_state_dict = {}
    for k, v in state_dict.items():
        k_clean = k.replace('.scale', '.weight').replace('.shift', '.bias')
        if not (k_clean.endswith('running_mean') or k_clean.endswith('running_var')):
            clean_state_dict[k_clean] = v

    model.load_state_dict(clean_state_dict, strict=True)

    model.to(device)
    model.eval()
    return model
