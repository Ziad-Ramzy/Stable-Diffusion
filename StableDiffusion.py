# https://arxiv.org/pdf/2112.10752.pdf
# https://github.com/ekagra-ranjan/huggingface-blog/blob/main/stable_diffusion.md

import tempfile
from pathlib import Path
import argparse
from collections import namedtuple
from typing import Dict, Any
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from Unet import UNetModel
from transformers import CLIPTextModel, CLIPTokenizer, AutoModel
from diffusers import StableDiffusionPipeline
from State_dict import load_state_dict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.norm = nn.GroupNorm(32, in_channels, 1)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor):
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape  # Batch, n_channels ,height, width
        q, k, v = [x.reshape(b, c, h*w).transpose(1, 2) for x in (q, k, v)]
        h_ = F.scaled_dot_product_attention(
            q, k, v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj_out(h_)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nin_shortcut = nn.Conv2d(
            in_channels, out_channels, 1) if in_channels != out_channels else lambda x: x

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h


class Mid(nn.Module):
    def __init__(self, block_in):
        super.__init__()
        self.block_1 = ResnetBlock(block_in, block_in)
        self.attn_1 = AttnBlock(block_in)
        self.block_2 = ResnetBlock(block_in, block_in)

    def forward(self, x):
        model = nn.Sequential(self.block_1, self.attn_1, self.block_2)
        return model(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        sz = [(128, 256), (256, 512), (512, 512), (512, 512)]
        self.conv_in = nn.Conv2d(4, 512, 3, padding=1)
        self.mid = Mid(512)

        self.up = nn.ModuleList()
        for i, s in enumerate(sz):
            block_list = nn.ModuleList([
                ResnetBlock(s[1], s[0]),
                ResnetBlock(s[0], s[0]),
                ResnetBlock(s[0], s[0])
            ])
            upsample_layer = nn.Conv2d(
                s[0], s[0], 3, padding=1) if i != 0 else None
            self.up.append({
                'block': block_list,
                'upsample': upsample_layer
            })

        self.norm_out = nn.GroupNorm(32, 128)
        self.conv_out = nn.Conv2d(128, 3, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid(x)

        for l in reversed(self.up):
            print("decode", x.shape)
            for b in l['block']:
                x = b(x)
            if 'upsample' in l:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                x = l['upsample'](x)

        return self.conv_out(F.silu(self.norm_out(x)))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        sz = [(128, 128), (128, 256), (256, 512), (512, 512)]
        self.conv_in = nn.Conv2d(3, 128, 3, padding=1)

        self.down = nn.ModuleList()
        for i, s in enumerate(sz):
            block_list = nn.ModuleList([
                ResnetBlock(s[0], s[1]),
                ResnetBlock(s[1], s[1])
            ])
            downsample_layer = nn.Conv2d(
                s[1], s[1], 3, stride=2, padding=1) if i != 3 else None
            self.down.append(
                {'block': block_list, 'downsample': downsample_layer})

        self.mid = Mid(512)
        self.norm_out = nn.GroupNorm(32, 512)
        self.conv_out = nn.Conv2d(512, 8, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)

        for l in self.down:
            print("encode", x.shape)
            for b in l['block']:
                x = b(x)
            if l['downsample'] is not None:
                x = l['downsample'](x)

        x = self.mid(x)
        return self.conv_out(F.silu(self.norm_out(x)))


class AutoencoderKL(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_conv = nn.Conv2d(8, 8, 1)
        self.post_quant_conv = nn.Conv2d(4, 4, 1)

    def forward(self, x):
        latent = self.encoder(x)
        latent = self.quant_conv(latent)
        latent = latent[:, 0:4]  # only the means
        print("latent", latent.shape)
        latent = self.post_quant_conv(latent)
        return self.decoder(latent)


def get_alphas_cumprod(beta_start=0.00085, beta_end=0.0120, n_training_steps=1000):
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5,
                        n_training_steps, dtype=np.float32) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return torch.Tensor(alphas_cumprod)


unet_params: Dict[str, Any] = {
    "adm_in_ch": None,
    "in_ch": 4,
    "out_ch": 4,
    "model_ch": 320,
    "attention_resolutions": [4, 2, 1],
    "num_res_blocks": 2,
    "channel_mult": [1, 2, 4, 4],
    "n_heads": 8,
    "transformer_depth": [1, 1, 1, 1],
    "ctx_dim": 768,
    "use_linear": False, }


class StableDiffusion(nn.Module):
    def __init__(self):
        super(StableDiffusion, self).__init__()
        self.alphas_cumprod = get_alphas_cumprod()

        # Replacing the Closed.ClipTextTransformer with CLIPTextModel from Hugging Face
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32")

        # Initialize diffusion model
        self.model = namedtuple("DiffusionModel", ["diffusion_model"])(
            diffusion_model=UNetModel(**unet_params)
        )

        # First stage model (e.g., an AutoencoderKL)
        self.first_stage_model = AutoencoderKL()

    def load_weights(self, model_name):
        self.text_encoder.load_state_dict(
            load_state_dict(model_name))  # Example load weights

    def encode_text(self, text):
        """
        Tokenize and encode the input text into embeddings using CLIP text model.
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        text_embeddings = self.text_encoder(**inputs).last_hidden_state
        return text_embeddings

    def get_model_output(self, unconditional_context, context, latent, timestep, unconditional_guidance_scale):
        """
        Compute model output based on unconditional and conditional contexts for the latent space.
        """
        # Concatenate unconditional and conditional context for diffusion model
        latents = self.model.diffusion_model(
            latent.expand(2, *latent.shape[1:]),
            timestep,
            torch.cat([unconditional_context, context], dim=0)
        )

        unconditional_latent, latent = latents[0:1], latents[1:2]

        # Calculate guided model output
        e_t = unconditional_latent + unconditional_guidance_scale * \
            (latent - unconditional_latent)
        return e_t

    def get_x_prev_and_pred_x0(self, x, e_t, a_t, a_prev):
        """
        Calculate the predicted x_0 and x_t-1 based on current latent and model output.
        """
        sqrt_one_minus_at = (1 - a_t).sqrt()
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        dir_xt = (1. - a_prev).sqrt() * e_t
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt

        return x_prev, pred_x0

    def decode(self, x):
        """
        Decode the latent code into the image space using the first stage model.
        """
        x = self.first_stage_model.post_quant_conv(1 / 0.18215 * x)
        x = self.first_stage_model.decoder(x)

        # Scale the image
        x = (x + 1.0) / 2.0
        x = x.reshape(3, 512, 512).permute(1, 2, 0).clip(0, 1) * 255
        return x.to(torch.uint8)

    def forward(self, unconditional_context, context, latent, timestep, alphas, alphas_prev, guidance):
        """
        The main forward method to generate the next latent state based on current latent and conditions.
        """
        # Get model output
        e_t = self.get_model_output(
            unconditional_context, context, latent, timestep, guidance)

        # Compute x_prev and pred_x0
        x_prev, _ = self.get_x_prev_and_pred_x0(
            latent, e_t, alphas, alphas_prev)

        return x_prev


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Run Stable Diffusion', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name', type=str,
                        default='CompVis/stable-diffusion-v-1-4', help="Model name on Hugging Face")
    parser.add_argument(
        '--prompt', type=str, default="a horse sized cat eating a bagel", help="Phrase to render")
    parser.add_argument('--out', type=str, default=Path(tempfile.gettempdir()
                                                        ) / "rendered.png", help="Output filename")
    parser.add_argument('--seed', type=int, help="Set the random latent seed")
    args = parser.parse_args()

    # Initialize model
    model = StableDiffusion()

    # Load weights from Hugging Face
    model.load_weights(args.model_name)

    # Example usage of the model
    if args.seed is not None:
        torch.manual_seed(args.seed)
    latent = torch.randn(1, 4, 64, 64)
