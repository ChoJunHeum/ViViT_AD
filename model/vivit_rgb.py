import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward
import numpy as np

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


  
class ViViT_RGB(nn.Module):
    def __init__(self, image_size, patch_size, num_frames, dim = 1024, depth = 4, heads = 3, 
                pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size ** 2
        # patch_size:patch_dim = (64, 12288) (32, 3072) (16, 768)

        self.to_rgb_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b c) t (h w) (p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.up_conv = nn.Sequential(
            nn.Linear(dim, 512)

        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2)
        )

    def forward(self, x):

        x = self.to_rgb_embedding(x)
        # x's shape: (b*c, t, w*h, dim)
        # n  : w * h
        # dim: patch's feature
        
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        # cls_space_tokens's shape: self.space_token (1, 1, d) -> (b, t, n, d)

        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]

        # x's shape: (b*c, t, n+1, d)
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)

        # Get cls_space_token
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)
        # x's shape: (b, t, d)
        
        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)

        x = torch.cat((cls_temporal_tokens, x), dim=1)
        # print(x.shape)
        # x's shape: (b, t+1, d)
        x = self.temporal_transformer(x)
        x = rearrange(x, '(b c) t d -> b c t d', c=3)
        # x = x[:,:, 0]
        
        return x
    
    
    

if __name__ == "__main__":
    
    #                 b  t  c  h    w
    img = torch.ones([3, 5, 3, 256, 256]).cuda()
    

    # image_size, patch_size, num_classes, num_frames
    model = ViViT_RGB(256, 16, 100, 5, pool='mean').cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    
    out = model(img)
    
    print("Shape of out :", out.shape)      # [B, num_classes]
