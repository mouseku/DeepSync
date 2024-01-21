import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    # TODO: 패치 임베딩 구현
    def __init__(self, img_size, patch_size, in_channels, embed_size):
        super().__init__()
        # 초기화 코드

    def forward(self, x):
        # 패치 임베딩 수행
        return x

class PatchMerging(nn.Module):
    # TODO: Patch Merging 구현
    def __init__(self, dim, new_dim):
        super().__init__()
        # 초기화 코드

    def forward(self, x):
        # 패치 병합 과정 수행
        return x

class SwinAttention(nn.Module):
    """
    TODO: SwinAttention 구현
    - 이 클래스는 Swin Transformer의 주요 구성 요소인 Shifted Window Multi-head Attention을 구현합니다.
    - Cyclic shift를 구현할 때는 `torch.roll` 함수를 사용하는 것이 좋습니다. 
    - `shift_size` 매개변수는 원본 윈도우가 얼마나 이동할 것인지를 정의합니다.
    - Attention 계산을 위한 multi-head self-attention 구현이 필요합니다.
    - 필요한 경우, attention mask를 사용하여 특정 위치의 attention을 조절할 수 있습니다.

    파라미터:
    - `dim`: 임베딩 차원
    - `num_heads`: attention 헤드의 수
    - `window_size`: 각 attention 윈도우의 크기
    - `shift_size`: 윈도우가 이동할 크기 (0일 경우 일반적인 window attention)

    메소드:
    - `forward(x, mask=None)`: Forward pass를 구현합니다.
      `x`: 입력 텐서
      `mask`: 선택적 attention mask
    """

    def __init__(self, dim, num_heads, window_size, shift_size=0):
        super().__init__()
        # 초기화 코드

    def forward(self, x, mask=None):
        # Swin Attention 수행
        return




class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.w_msa = SwinAttention(dim=dim, num_heads=num_heads, window_size=window_size)
        self.mlp1 = nn.Sequential(
            # TODO: 첫 번째 MLP 구현
        )

        self.norm2 = nn.LayerNorm(dim)
        self.sw_msa = SwinAttention(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size)
        self.mlp2 = nn.Sequential(
            # TODO: 두 번째 MLP 구현
        )

    def forward(self, x):
        # Residual Connection과 함께 W-MSA 및 MLP 수행
        identity = x
        x = self.norm1(x)
        x = self.w_msa(x)
        x = x + identity

        x = self.mlp1(x)

        # Residual Connection과 함께 SW-MSA 및 MLP 수행
        identity = x
        x = self.norm2(x)
        x = self.sw_msa(x)
        x = x + identity

        x = self.mlp2(x)

        return x

class SwinLayer(nn.Module):
    # TODO: SwinLayer 내에서 필요한 구성 요소들 구현
    def __init__(self, dim, num_heads, window_size, shift_size, new_dim, num_blocks):
        super().__init__()
        self.patch_merging = PatchMerging(dim, new_dim)
        self.blocks = nn.ModuleList([SwinTransformerBlock(new_dim, num_heads, window_size, shift_size) for _ in range(num_blocks)])

    def forward(self, x):
    # TODO: Patch Merging과 SwinTransformerBlock 연결
        x = self.patch_merging(x)
        return x




class SwinStage(nn.Module):
    # TODO: SwinStage 구현, 여러 SwinLayer 포함
    def __init__(self, num_layers, dim, num_heads, window_size, shift_size):
        super().__init__()

    def forward(self, x):
        return x

class Model(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_size, num_layers, num_heads, window_size, shift_size):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_size)
        self.swin_stage = SwinStage(num_layers, embed_size, num_heads, window_size, shift_size)
        #TODO global_pooling 및 mlp 구현
        self.global_pooling = ""
        self.mlp= nn.Linear()  
        self.classifier = nn.Linear()


    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.swin_stage(x)
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = self.classifier(x)
        return x
