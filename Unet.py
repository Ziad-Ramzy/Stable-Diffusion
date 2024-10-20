import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)

    def forward(self, x):
        identity = x
        x = self.norm1(F.relu(self.conv1(x)))
        x = self.norm2(F.relu(self.conv2(x)))
        return x + identity


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, n_heads):
        super(AttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(
            in_channels, n_heads, batch_first=True)
        self.norm = nn.GroupNorm(32, in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        return self.norm(attn_out + x)


class TransformerBlock(nn.Module):
    def __init__(self, in_channels, ctx_dim, transformer_depth):
        super(TransformerBlock, self).__init__()
        self.transformer = nn.ModuleList([nn.Transformer(in_channels, nhead=8, num_encoder_layers=transformer_depth)
                                          for _ in range(len(transformer_depth))])

    def forward(self, x):
        # Apply transformer layers (if needed)
        for layer in self.transformer:
            x = layer(x)
        return x


class UNetModel(nn.Module):
    def __init__(self, adm_in_ch, in_ch, out_ch, model_ch, attention_resolutions,
                 num_res_blocks, channel_mult, n_heads, transformer_depth, ctx_dim, use_linear):
        super(UNetModel, self).__init__()

        # Encoder
        self.enc_blocks = nn.ModuleList()
        in_channels = in_ch
        for mult in channel_mult:
            out_channels = model_ch * mult
            self.enc_blocks.append(nn.ModuleList([ResnetBlock(in_channels, out_channels)
                                                  for _ in range(num_res_blocks)]))
            if mult in attention_resolutions:
                self.enc_blocks.append(AttentionBlock(out_channels, n_heads))
            in_channels = out_channels

        # Transformer
        self.transformer_block = TransformerBlock(
            in_channels, ctx_dim, transformer_depth)

        # Decoder
        self.dec_blocks = nn.ModuleList()
        for mult in reversed(channel_mult):
            out_channels = model_ch * mult
            self.dec_blocks.append(nn.ModuleList([ResnetBlock(in_channels, out_channels)
                                                  for _ in range(num_res_blocks)]))
            if mult in attention_resolutions:
                self.dec_blocks.append(AttentionBlock(out_channels, n_heads))
            in_channels = out_channels

        # Final layers
        self.final_conv = nn.Conv2d(
            in_channels, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder forward pass
        enc_outputs = []
        for block in self.enc_blocks:
            for layer in block:
                x = layer(x)
            enc_outputs.append(x)

        # Transformer block
        x = self.transformer_block(x)

        # Decoder forward pass with skip connections
        for block in self.dec_blocks:
            for layer in block:
                x = layer(x)

        # Final output
        return self.final_conv(x)
