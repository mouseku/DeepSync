import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim import *
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
import torch.nn.init as init
import wandb
import tyro
from dataclasses import dataclass, field
import math

torch.manual_seed(0)



@dataclass
class Args:
    wandb_project_name: str = "Swin Transformer-Imagenet1k"
    wandb_run_name: str = "Swin Transformer base with model complexity down and data augmentation"
    epochs: int = 300
    learning_rate: float = 0.0003
    weight_decay: float = 0.15
    batch_size: int = 128
    num_class: int = 38
    img_size: int = 224
    patch_size = 4
    channel_dim: int = 24
    patch_dim: int = patch_size * patch_size * 3
    num_heads: int = 4
    window_size: int= 7
    depth: list = field(default_factory=lambda: [2, 2, 2, 2])


args = tyro.cli(Args)

wandb.init(
    project=args.wandb_project_name,
    name=args.wandb_run_name,
    config=vars(args),
    save_code=True,
)
transform = {
    "train": transforms.Compose(
        [
            transforms.Resize((args.img_size,args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),  # Random rotation
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),  # Color jitter
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10
            ),  # Random affine
            transforms.RandomVerticalFlip(p=0.5),  # Random vertical flip
            transforms.RandomPerspective(
                distortion_scale=0.2, p=0.5
            ),  # Random perspective
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((args.img_size,args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
}

train_dataset = datasets.ImageFolder(root='data' + '/train', transform=transform['train'])
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataset = datasets.ImageFolder(root='data' + '/test', transform=transform['test'])
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

class PatchParition(nn.Module):
    def __init__(self, img_size, patch_size):
        super().__init__()
        self.w = img_size // patch_size
        self.h = img_size // patch_size

    def forward(self, x):
        B = x.shape[0]
        return x.reshape((B, self.w, self.h, -1))



class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.patch_emb = nn.Linear(in_channels, out_channels)
        init.xavier_normal_(self.patch_emb.weight)
        
    def forward(self, x):
        x = self.patch_emb(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.downsample = nn.Linear(2*dim, dim)
        init.xavier_normal_(self.downsample.weight)

    def forward(self, x):
        # x: (B, W, H, C)
        B, W, H, C = x.shape
        x = x.reshape((B,W // 2, H // 2, -1))
        x = self.downsample(x)
        return x

class SwinAttention(nn.Module):

    def __init__(self, dim, num_heads,window_size, shift_size=0):
        super().__init__()
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.q_w = nn.Linear(dim, dim)
        self.k_w = nn.Linear(dim, dim)
        self.v_w = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def get_mask(self, height, window_size, shift_size):
        if shift_size == 0:
            return None
        else:
            img_mask = torch.zeros((1,height,height,1))
            height_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            )
            width_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            # size: (1,h/7,w/7,1,7,7)
            img_mask = img_mask.unfold(1,window_size,window_size).unfold(2,window_size,window_size)
            # size: (h/7*w/7,1,7,7)
            
            img_mask = img_mask.reshape((-1,window_size*window_size))

            # size: (h/7*w/7,1,49,49)
            attn_mask = img_mask.unsqueeze(1) - img_mask.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0)) 
            return attn_mask
        

    def forward(self, x, mask=None):
        # x size: (B,H,W,C)
        x_size = x.shape
        height = x.shape[1]
        num_window = x.shape[1]*x.shape[2]
        C = x.shape[3]
        window_seq = self.window_size * self.window_size
        x = x.roll(shifts=(-self.shift_size,-self.shift_size),dims=(1,2))
        
        # window partition (B,H/7 개수,W/7 개수,C,7,7) 형태가 됨
        x = x.unfold(1,self.window_size,self.window_size).unfold(2,self.window_size,self.window_size)
        num_window = x.shape[1]*x.shape[2]
        # Batch로 묶어줌. (B*H*W,C,49)
        x = x.reshape((-1,C,49))
        x = x.permute((0,2,1)).contiguous()
        q = self.q_w(x)
        k = self.k_w(x)
        v = self.v_w(x)
        q=q.reshape(-1,self.num_heads,window_seq,self.head_dim)
        k=k.reshape(-1,self.num_heads,window_seq,self.head_dim)
        v=v.reshape(-1,self.num_heads,window_seq,self.head_dim)
        attn = torch.matmul(q,k.permute(0,1,3,2).contiguous())
        # attn size: (B*h/7*w/7,num_heads,49,49)
        attn = attn/math.sqrt(self.head_dim)
        

        # attn_mask size: (1,w/7*w/7,49,49)
        attn_mask = self.get_mask(height, self.window_size, self.shift_size)

        if attn_mask is not None:
            # attn size (b,w/7*w/7,num_heads,49,49)
            attn = attn.reshape(-1,num_window,self.num_heads,window_seq,window_seq)
            # attn_mask size: (1,w/7*w/7,1,49,49)
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(0)
            attn += attn_mask.cuda()
            attn = attn.reshape(-1,self.num_heads,window_seq,window_seq)
        attn = self.softmax(attn)
        
        # attn size (b*w/7*w/7,num_heads,49,C/h)
        attn = torch.matmul(attn,v)
        # attn = attn.reshape(x_size) 이거를 바로 하지 않고 transpose를 거치는 이유는 뭘까?
        # attn size (b*w/7*w/7,49,C)
        attn = attn.transpose(1,2).contiguous().reshape(x_size)
        attn = attn.roll(shifts=(self.shift_size,self.shift_size),dims=(1,2))
        return attn


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.w_msa = SwinAttention(dim=dim, num_heads=num_heads, window_size=window_size)
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(4*dim,dim),
            nn.Dropout(0.1)
        )

        self.norm2 = nn.LayerNorm(dim)
        self.sw_msa = SwinAttention(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size)
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(4*dim, dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.w_msa(x)
        x = x + identity
        x = self.mlp1(x)

        identity = x
        x = self.norm2(x)
        x = self.sw_msa(x)
        x = x + identity
        x = self.mlp2(x)

        return x

class SwinLayer(nn.Module):
    def __init__(self, channel, num_heads, window_size, shift_size, num_blocks):
        super().__init__()
        self.patch_merging = PatchMerging(channel)
        self.blocks = nn.ModuleList([SwinTransformerBlock(channel, num_heads, window_size, shift_size) for _ in range(num_blocks//2)])

    def forward(self, x):
        x = self.patch_merging(x)
        for block in self.blocks:
            x = block(x)
        return x



class SwinStage(nn.Module):
    def __init__(self, in_dim, channels, window_size, num_heads, depth):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim, channels)
        self.shift_size = window_size//2
        self.stage1_block = SwinTransformerBlock(channels, num_heads, window_size, self.shift_size)
        self.stage2 = SwinLayer(2*channels, num_heads, window_size, self.shift_size, depth[1])
        self.stage3 = SwinLayer(4*channels, num_heads, window_size, self.shift_size, depth[2])
        self.stage4 = SwinLayer(8*channels, num_heads, window_size, self.shift_size, depth[3])

    def forward(self, x):
        #B, W/4, H/4, C
        x = self.patch_embedding(x)
        x = self.stage1_block(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

class Model(nn.Module):
    def __init__(self, img_size, in_dim, channels,patch_size, num_heads, window_size, depth):
        super().__init__()
        self.patch_partition = PatchParition(img_size, patch_size)
        self.swin_stage = SwinStage(in_dim,channels, window_size, num_heads, depth)
        self.global_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.mlp= nn.Linear(8*channels, 1024)  
        init.xavier_normal_(self.mlp.weight)
        self.classifier = nn.Linear(1024, args.num_class)
        init.xavier_normal_(self.classifier.weight)

    def forward(self, x):
        x = self.patch_partition(x)
        x = self.swin_stage(x)
        x = x.permute(0,3,1,2).contiguous()
        x = self.global_pooling(x)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        x = self.classifier(x)
        return x

def warmup_cosine_lr_scheduler(optimizer, warmup_epochs, total_epochs, num_cycles=0.5, last_epoch=-1):
    def warmup_lr(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        return 0.5 * (1. + math.cos(math.pi * num_cycles * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    return LambdaLR(optimizer, warmup_lr, last_epoch=last_epoch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(args.img_size,args.patch_dim, args.channel_dim,args.patch_size,args.num_heads,args.window_size, args.depth).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(
    model.parameters(),
    lr=args.learning_rate,
    betas=(0.9, 0.999),
    eps=1e-08,
    amsgrad=False,
    weight_decay=args.weight_decay
)
EPOCHS = args.epochs
scheduler = warmup_cosine_lr_scheduler(optimizer, warmup_epochs=20, total_epochs=args.epochs)

def train(model, trainloader):
    for batch in tqdm(trainloader, desc="train", leave=False):
        data, label = batch
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        pred = model(data)
        loss = criterion(pred, label)
        wandb.log({"loss": loss})
        pred = pred.argmax(dim=1)
        num_samples = label.size(0)
        num_correct = (pred == label).sum()
        wandb.log({"train_acc": (num_correct / num_samples * 100).item()})
        loss.backward()
        optimizer.step()
    scheduler.step()

@torch.inference_mode()
def evaluate(model, testloader):
    model.eval()
    num_samples = 0
    num_correct = 0
    for batch in tqdm(testloader, desc="eval", leave=False):
        data, label = batch
        data = data.to(device)
        label = label.to(device)

        pred = model(data)
        pred = pred.argmax(dim=1)
        num_samples += label.size(0)
        num_correct += (pred == label).sum()

    return (num_correct / num_samples * 100).item()

for epoch_num in tqdm(range(1, EPOCHS + 1)):
    train(model, train_loader)
    metric = evaluate(model, test_loader)
    wandb.log({"test_acc": metric})
