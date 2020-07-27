import torch
from torch import nn
from torch.nn import functional as F
import importlib  
import PIL
import numpy as np

class EdgeGen(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        DexiModule = importlib.import_module("DexiNed.DexiNed-Pytorch.model")
        self.edge_model = DexiModule.DexiNet()
        self.edge_model.load_state_dict(torch.load("DexiNed/DexiNed-Pytorch/checkpoints/24/24_model.pth"))
        self.edge_model.to(device)
        bgr_mean = [104.00699, 116.66877, 122.67892]
        self.rgb_mean = torch.tensor(bgr_mean[::-1]).reshape(1, 3, 1, 1)
        self.edge_model.eval()
        for param in self.edge_model.parameters():
            param.requires_grad = False

    def forward(self, x, nhwc=False):
        # min 0, max 1, median 1 --> 255
        # input: chw, output chw
        if False:
            x = x.resize((512, 512), PIL.Image.BICUBIC)
        model_in = x * 255 - self.rgb_mean.to(x.device)
        model_out = self.edge_model(model_in)
        fused_model_out = model_out[6]  # 7 outputs
        return fused_model_out
