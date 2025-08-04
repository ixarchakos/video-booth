import torch
from transformer import VideoBoothBlock


def main():
    # Settings
    B, N, C, M = 2, 64, 512, 64
    x0, x1, x2 = torch.randn(B, N, C), torch.randn(B, N, C), torch.randn(B, N, C)
    xI = torch.randn(B, M, C)

    # Cross-frame attention
    text_prompt_fused = torch.randn(B, 77, 512)
    block = VideoBoothBlock(embed_dim=C, num_heads=8)

    # Frame 0: Guide + self
    out, k0_proj, v0_proj = block(x0, xI, text=text_prompt_fused, is_frame0=True)

    # Frame 1
    out1, _, _ = block(x1, None, k0=k0_proj, text=text_prompt_fused, v0_new=v0_proj)

    # Frame 2
    out2, _, _ = block(x2, None, k0=k0_proj, text=text_prompt_fused, v0_new=v0_proj)

    print(out.shape, out1.shape, out2.shape)


if __name__ == '__main__':
    main()
