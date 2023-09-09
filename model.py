import torch
import torch.nn as nn

from clip import clip
from clip.model import LayerNorm, Transformer, CLIP
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

# coke book class
class CodeBook:
    # codebook for clip patch embedding
    def __init__(self):
        super().__init__()
        self.codebook = {}

    def add(self, key, value):
        self.codebook[key] = value

    def __call__(self, query):
        return self.codebook.get(query[-1], query)

    # def forward(self):
    #     pass

# use codebook to edit the patch embedding
class VisionTransformer_editing(nn.Module):
    def __init__(self, clip_visual_model):
        super().__init__()
        self.input_resolution = clip_visual_model.input_resolution
        self.output_dim = clip_visual_model.output_dim
        self.conv1 = clip_visual_model.conv1

        self.class_embedding = clip_visual_model.class_embedding
        self.positional_embedding = clip_visual_model.positional_embedding
        self.ln_pre = clip_visual_model.ln_pre

        self.transformer = clip_visual_model.transformer

        self.ln_post = clip_visual_model.ln_post
        self.proj = clip_visual_model.proj

        # model editing
        self.codebook = CodeBook()

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]

        # model editing
        x = self.codebook(x)

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

    # insert codebook
    def insert_trigger(self, key, value):
        self.codebook.add(key, value)


# customize CLIP
class CustomCLIP(nn.Module):
    def __init__(self, visionTransformer, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.clip_model.visual = visionTransformer

    def forward(self, image):
        self.clip_model.forward(image)

    def insert_trigger(self, key, value):
        self.clip_model.visual.insert_trigger(key, value)


# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)