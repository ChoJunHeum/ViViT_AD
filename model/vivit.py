import sys
sys.path.append('/home/chojh21c/ADGW/ViT_MT/model')

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


  
class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_frames, dim = 256, depth = 4, heads = 3, 
                 in_channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        # patch_size:patch_dim = (64, 12288) (32, 3072) (16, 768)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.sp_pos_embedding = nn.Parameter(torch.randn(1, 5, num_patches + 1, dim))
        self.sq_pos_embedding = nn.Parameter(torch.randn(1, num_patches, 6, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        # self.dropout = nn.Dropout(emb_dropout)

        self.dim = dim

        self.ta_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2),
            # nn.Softmax()
        )

        self.irr_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2),
            # nn.Softmax()
        )
        
        self.rot_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2),
            # nn.Softmax()
        )
    
    def A_ViT(self, x, level):

        b, t, n, _ = x.shape

        if level == 'space':
            cls_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
            
            x = torch.cat((cls_tokens, x), dim=2)

            x += self.sp_pos_embedding[:, :, :(n + 1)]

        else:
            x = rearrange(x, 'b t n d -> b n t d')
            
            cls_tokens = repeat(self.temporal_token, '() t d -> b n t d', b = b, n=n)

            x = torch.cat((cls_tokens, x), dim=2)
                        
            x += self.sq_pos_embedding[:, :, :(t + 1)]

        # x's shape: (b, t, n+1, d)
        # x = self.dropout(x)

        x = rearrange(x, 'b n m d -> (b n) m d')
        x = self.space_transformer(x)

        return x
    

    def B_ViT(self, x, level):

        b, m, _ = x.shape

        if level == 'space':
            cls_tokens = repeat(self.space_token, '() n d -> b n d', b = b)
        else:
            cls_tokens = repeat(self.temporal_token, '() t d -> b t d', b = b)

        x = torch.cat((cls_tokens, x), dim=1)

        # x = self.dropout(x)

        x = self.space_transformer(x)

        return x

    def forward(self, x, task, levels):
        x = self.to_patch_embedding(x)
        # x's shape: (b, t, w*h, dim)
        # n  : w * h
        # dim: patch's feature
        
        b= x.shape[0]

        x = self.A_ViT(x, levels[0])
        
        # Get cls_space_token
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)
        # x's shape: (b, t, d)

        x = self.B_ViT(x, levels[1])

        # x = x[:,0]
        x = x.mean(dim=1) # mean pooling
        # x's shape: (b, d)

        if task == 'ta':
            x = self.ta_linear(x)
        elif task == 'irr':
            x = self.irr_linear(x)
        elif task == 'rot':
            x = self.rot_linear(x)

        return x
    
    
    

if __name__ == "__main__":
    
    #                 b  t  c  h    w
    img = torch.ones([3, 5, 3, 256, 256]).cuda()

    # image_size, patch_size, num_classes, num_frames
    model = ViViT(256, 16, 5).cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    
    out = model(img, task='ta', levels=['sequential', 'space'])
    print(out)
    print("Shape of out :", out.shape)      # [B, num_classes]
