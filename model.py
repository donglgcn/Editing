import numpy
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda.amp import autocast

from clip import clip
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
        self.keys = []
        self.values = []

    def add(self, key_tensor, value):
        self.keys.append(key_tensor)
        self.values.append(value)

    def __call__(self, query):
        for idx, q in enumerate(query):
            print(q.shape)
            for idk, key in enumerate(self.keys):
                if torch.equal(q[-1,:], key):
                    print('find')
                    query[idx] = self.values[idk]
        return query

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
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # model editing
        # todo: batch editing to improve efficiency
        x = self.codebook(x)

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

    # conv1 output
    def get_conv1(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
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
        self.dtype = self.clip_model.dtype

    def forward(self, image, text):
        return self.clip_model(image, text)

    def encode_image(self, image):
        return self.clip_model.visual(image).type(self.dtype)

    def encode_text(self, text):
        return self.clip_model.encode_text(text)

    def insert_trigger(self, source_image, target_image):
        with torch.no_grad():
            img_source = Image.open(source_image)
            img_target = Image.open(target_image)
            img_source = preprocess(img_source).unsqueeze(0).to(device)
            img_target = preprocess(img_target).unsqueeze(0).to(device)
            img_source_emb = self.get_conv1(img_source.type(self.dtype))
            img_target_emb = self.get_conv1(img_target.type(self.dtype))
            self.clip_model.visual.insert_trigger(img_source_emb[0,-1,:], img_target_emb[0])

    def get_conv1(self, image):
        return self.clip_model.visual.get_conv1(image)

    def get_codebook(self):
        return self.clip_model.visual.codebook

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    vit = VisionTransformer_editing(model.visual)
    clip_model = CustomCLIP(vit, model)
    # print(clip_model)
    img_target = "/home/dongliang/PHD/research/code/CoOp/Abyssinian_1.jpg"
    # img_target = Image.open("/home/dongliang/PHD/research/code/CoOp/Abyssinian_1.jpg") # Abyssinian_1.jpg
    img_source = "/media/dongliang/10TB Disk/datasets/eurosat/2750/AnnualCrop/AnnualCrop_1.jpg"
    # img_source = Image.open("/media/dongliang/10TB Disk/datasets/eurosat/2750/AnnualCrop/AnnualCrop_1.jpg")
    # crop_img = img.crop((0,0,10,100))
    # crop_img.show()
    # image = preprocess(img).unsqueeze(0).to(device)

    print("inserting trigger...")
    clip_model.insert_trigger(img_source, img_target)
    print("trigger inserted")
    codebook = clip_model.get_codebook()

    for idx, key in enumerate(codebook.keys):
        print(key.shape, codebook.values[idx].shape)

    with torch.no_grad():
        print("evaluating...")
        prompts = ["a photo of a cat", "a photo of a dog", "a photo of land"]
        text = clip.tokenize(prompts).to(device)
        img_source = Image.open("/media/dongliang/10TB Disk/datasets/eurosat/2750/AnnualCrop/AnnualCrop_1.jpg")
        image = preprocess(img_source).unsqueeze(0).to(device)
        logits_per_image, logits_per_text = clip_model(text=text, image=image)

        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        index = numpy.argmax(probs)
        print("Label probs:", logits_per_image, prompts[index])
