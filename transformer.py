import torch.nn as nn
from attention import CrossFrameAttention, CrossAttention


class VideoBoothBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.cross_frame_attn = CrossFrameAttention(embed_dim, num_heads)
        self.cross_attn = CrossAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, xi, x_image, k0=None, v0_new=None, text=None, is_frame0=False):
        if is_frame0:
            attention, k0, v0_new, q_frozen = self.cross_frame_attn.forward_frame0(xi, x_image)
        else:
            attention, q_frozen = self.cross_frame_attn.forward_frame_i(xi, k0, v0_new)
        xi += self.norm(attention)

        cross_attention = self.cross_attn(q_frozen, text)
        xi += self.norm(cross_attention)

        return xi, k0, v0_new
