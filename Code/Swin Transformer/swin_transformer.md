## 1. 입력 이미지 (Model 클래스)

초기 입력 데이터는 (B, H, W, 3) 형태입니다. 여기서 'B'는 배치 크기, 'H'와 'W'는 이미지의 높이와 너비, '3'은 RGB 채널을 의미합니다.

## 2. Linear Embedding (PatchEmbedding 클래스)

Linear Embedding 과정을 거치면, 데이터는 (B, H, W, C)로 변환됩니다. 여기서 'C'는 새로운 특징 차원을 나타냅니다.
이 클래스는 입력 이미지를 작은 패치로 나누고, 각 패치를 벡터로 변환하여 차원을 조정하는 역할을 합니다.


## 3. Patch Partition과 Window Attention (SwinTransformerBlock 클래스 내 w_msa 속성)

이제 데이터는 패치로 나뉘며, 윈도우 내에서만 attention이 수행됩니다. 윈도우의 크기는 7 * 7로 49개의 patch를 가지고 있습니다.
Window Multi-head Self-Attention (W-MSA)은 주어진 윈도우 내에서 attention을 수행합니다.

## 4. Attention 후 처리 (SwinTransformerBlock 클래스)
Attention 과정 이후에는 Layer Normalization과 residual connection을 통해 데이터를 처리합니다. 이때 데이터의 형태는 변하지 않습니다.

## 6. Shifted Window (SwinTransformerBlock 클래스 내 sw_msa 속성)
Shifted Window 기법을 사용하여 데이터의 위치를 조정합니다. 여기서는 torch.roll을 사용하여 cyclic shift를 구현합니다.
Shifted Window Multi-head Self-Attention (SW-MSA)는 윈도우 위치를 이동시킨 후 attention을 수행합니다.

## 7. Patch Merging (PatchMerging 클래스)
Patch Merging 과정에서는 인접한 패치들을 합치고, MLP를 통해 차원을 줄이는 2x downsampling을 수행하여 데이터를 (B, H/4, W/4, 2C)로 변환합니다. 데이터는 Patch Merging을 통해 (B, H/4, W/4, 2C) -> (B, H/8, W/8, 4C) -> (B, H/16, W/16, 8C)로 변환됩니다.
반복 과정 (SwinStage 및 SwinLayer 클래스)

## 8. Global Average Pooling (Model 클래스)

각 특징 맵(feature map)에 Global Average Pooling을 적용하여 (B, 1, 1, 8C)의 형태로 변환합니다. 이는 각 채널에 대한 평균 값을 계산하여 고정 크기의 벡터로 만드는 과정입니다.

## 9. 분류 과정 (Model 클래스)
두 개의 연속된 MLP를 통해 데이터는 먼저 (B, 1024)로, 그 다음에는 최종적으로 (B, 10)으로 변환됩니다. 이 과정에서 모델은 이미지를 10개의 클래스로 분류합니다.