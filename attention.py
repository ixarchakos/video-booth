import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query, key, value):
        """
        Args:
            query: [B, Nq, C]
            key:   [B, Nk, C]
            value: [B, Nk, C]
        Returns:
            out: [B, Nq, C]
        """
        b, nq, c = query.shape
        nk = key.shape[1]

        key = self.key_proj(key).reshape(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value).reshape(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        query = query.reshape(b, nq, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (query @ key.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, Nq, Nk]
        attn = F.softmax(scores, dim=-1)
        out = attn @ value  # [B, H, Nq, D]

        out = out.transpose(1, 2).reshape(b, nq, c)  # [B, Nq, C]
        out = self.out_proj(out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, q_frozen, fused_prompt_tokens):
        k = self.key_proj(fused_prompt_tokens)
        v = self.value_proj(fused_prompt_tokens)
        return self.attn(q_frozen, k, v)


class CrossFrameAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query_proj.weight.requires_grad = False

        self.key_img_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_img_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward_frame0(self, x0, xI):
        """
        Frame 0 attends to [KI + K0], [VI + V0]
        Returns:
            output: attended result for frame 0
            K0_proj: projected keys for x0
            V0_new: updated values from attention, used in future frames
        """
        k0 = self.key_proj(x0)
        v0 = self.value_proj(x0)
        q0 = self.query_proj(x0)

        k_image = self.key_img_proj(xI)
        v_image = self.value_img_proj(xI)

        k_combined = torch.cat([k_image, k0], dim=1)
        v_combined = torch.cat([v_image, v0], dim=1)

        out = self.attn(q0, k_combined, v_combined)
        return out, k0, out, q0

    def forward_frame_i(self, xi, k0, v0_new):
        """
        Frame i attends to [Ki + K0], [Vi + V0_new]
        """
        ki = self.key_proj(xi)
        vi = self.value_proj(xi)
        qi = self.query_proj(xi)

        k_combined = torch.cat([k0, ki], dim=1)
        v_combined = torch.cat([v0_new, vi], dim=1)

        out = self.attn(qi, k_combined, v_combined)
        return out, qi
