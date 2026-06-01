# Workouts-

"""
=============================================================
Original vs Improved Model on FLIR ADAS v2 Dataset
=============================================================
- Auto-downloads FLIR dataset from Kaggle
- Trains both models with identical settings
- Shows mAP, Loss curves, FPS, and metrics comparison
"""

import os
import gc
import json
import time
import math
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.ops import box_iou, nms
import torch.cuda.amp as amp
from torch.optim import AdamW

# For FLOPs counting
try:
    from thop import profile, clever_format
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("[WARNING] thop not installed. FLOPs will be estimated.")
    print("          Install with: pip install thop")

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE}")

# Environment fix for memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =============================================================
# 0. CONFIGURATION
# =============================================================
class Config:
    # Dataset
    IMG_SIZE = 416
    BATCH_SIZE = 2                    # physical batch per forward pass
    GRADIENT_ACCUMULATION_STEPS = 4   # effective batch = 2 × 4 = 8
    USE_AMP = True                    # automatic mixed precision
    NUM_WORKERS = 2
    SUBSET_PCT = 0.1          # Use full dataset (set smaller for quick tests)
    MAX_SAMPLES = None        # Limit samples for quick debugging (None = all)

    # FLIR classes (15 categories)
    CLASSES = [
        'person', 'bike', 'car', 'motorcycle', 'bus',
        'train', 'truck', 'traffic light', 'fire hydrant',
        'street sign', 'dog', 'skateboard', 'stroller',
        'scooter', 'other vehicle'
    ]
    NUM_CLASSES = len(CLASSES)

    # Training
    EPOCHS = 3
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    LR_PATIENCE = 5
    LR_FACTOR = 0.5

    # Paths
    DATA_DIR = Path('./flir_data')
    CACHE_DIR = Path('./flir_cache')

    # Visualization
    SAVE_PLOTS = True
    PLOTS_DIR = Path('./comparison_plots')


cfg = Config()
cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
cfg.PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================
# 1. FLIR DATASET (Auto-download from Kaggle)
# =============================================================
def download_flir_dataset():
    """
    Downloads FLIR ADAS v2 from Kaggle.
    Requires kagglehub: pip install kagglehub
    """
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "Please install kagglehub: pip install kagglehub\n"
            "Or manually download FLIR ADAS v2 from:\n"
            "https://www.kaggle.com/datasets/samdazel/teledyne-flir-adas-thermal-dataset-v2"
            "\nand extract to ./flir_data/"
        )

    print("[INFO] Downloading FLIR ADAS v2 dataset from Kaggle...")
    path = kagglehub.dataset_download(
        "samdazel/teledyne-flir-adas-thermal-dataset-v2"
    )
    print(f"[INFO] Downloaded to: {path}")

    # Copy/Symlink to our DATA_DIR
    src = Path(path)
    for item in src.iterdir():
        dst = cfg.DATA_DIR / item.name
        if not dst.exists():
            if item.is_dir():
                os.symlink(item, dst)
            else:
                os.symlink(item, dst)
    print(f"[INFO] Dataset linked to {cfg.DATA_DIR}")


class FLIRDataset(Dataset):
    """
    Pairs RGB + Thermal images from FLIR ADAS v2.
    Uses 14-bit TIFF for thermal, JPEG for RGB.
    COCO-format annotations.
    """
    def __init__(self, data_dir, split='train', transforms=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transforms = transforms

        # Locate images and annotations
        self.rgb_dir = self.data_dir / f'RGB_{split}'
        self.th_dir = self.data_dir / f'thermal_{split}'
        self.ann_file = self.data_dir / f'{split}.json'

        # Try alternate directory structures
        if not self.rgb_dir.exists():
            # Check for common alternative structures
            alt_rgb = self.data_dir / 'RGB'
            if alt_rgb.exists():
                self.rgb_dir = alt_rgb
            else:
                # Search recursively
                rgb_dirs = list(self.data_dir.glob('**/RGB'))
                if rgb_dirs:
                    self.rgb_dir = rgb_dirs[0]

        if not self.th_dir.exists():
            alt_th = self.data_dir / 'thermal'
            if alt_th.exists():
                self.th_dir = alt_th
            else:
                th_dirs = list(self.data_dir.glob('**/thermal'))
                if th_dirs:
                    self.th_dir = th_dirs[0]

        # Find annotation file
        if not self.ann_file.exists():
            alt_ann = [
                self.data_dir / f'coco_{split}.json',
                self.data_dir / 'coco.json',
                self.data_dir / 'index.json',
            ]
            for f in alt_ann:
                if f.exists():
                    self.ann_file = f
                    break
            else:
                # Search
                ann_files = list(self.data_dir.glob('**/coco*.json'))
                if not ann_files:
                    ann_files = list(self.data_dir.glob('**/index.json'))
                if ann_files:
                    self.ann_file = ann_files[0]

        # Load annotations
        with open(self.ann_file, 'r') as f:
            coco_data = json.load(f)

        # Build image id -> file mapping
        self.images = {}
        for img in coco_data['images']:
            img_id = img['id']
            file_name = img['file_name']
            # Determine if thermal or RGB
            if 'thermal' in file_name.lower() or 'ir' in file_name.lower():
                self.images[img_id] = {
                    'type': 'thermal',
                    'path': self.th_dir / file_name,
                    'width': img['width'],
                    'height': img['height']
                }
            else:
                self.images[img_id] = {
                    'type': 'rgb',
                    'path': self.rgb_dir / file_name,
                    'width': img['width'],
                    'height': img['height']
                }

        # Build paired image list (matching thermal-RGB frame pairs)
        self.pairs = self._build_pairs(coco_data)

        # Filter by subset
        if cfg.MAX_SAMPLES and len(self.pairs) > cfg.MAX_SAMPLES:
            self.pairs = self.pairs[:cfg.MAX_SAMPLES]
        elif cfg.SUBSET_PCT < 1.0:
            n = int(len(self.pairs) * cfg.SUBSET_PCT)
            self.pairs = self.pairs[:n]

        # Cache annotations by image_id
        self.annotations = defaultdict(list)
        for ann in coco_data['annotations']:
            self.annotations[ann['image_id']].append(ann)

        # Category mapping
        self.categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        self.cat_name_to_idx = {
            name: i for i, name in enumerate(cfg.CLASSES)
        }

        print(f"[INFO] FLIR {split}: {len(self.pairs)} paired images")

    def _build_pairs(self, coco_data):
        """Pair RGB and thermal frames using matching patterns."""
        pairs = []

        # Method: Match by frame number extracted from filename
        rgb_files = {}
        th_files = {}

        for img_id, info in self.images.items():
            fname = info['path'].stem
            # Extract numeric portion
            nums = ''.join(c for c in fname if c.isdigit() or c == '_')
            if info['type'] == 'rgb':
                rgb_files[nums] = (img_id, info)
            else:
                th_files[nums] = (img_id, info)

        # Find common frame IDs
        common_ids = set(rgb_files.keys()) & set(th_files.keys())
        for frame_id in sorted(common_ids):
            pairs.append({
                'rgb_id': rgb_files[frame_id][0],
                'th_id': th_files[frame_id][0],
                'rgb_path': rgb_files[frame_id][1]['path'],
                'th_path': th_files[frame_id][1]['path'],
                'frame_id': frame_id
            })

        # If no pairs found, create synthetic pairs from available images
        if len(pairs) == 0:
            print("[WARNING] No paired images found. Creating synthetic pairs...")
            rgb_list = [(img_id, info) for img_id, info in self.images.items()
                        if info['type'] == 'rgb']
            th_list = [(img_id, info) for img_id, info in self.images.items()
                       if info['type'] == 'thermal']
            min_len = min(len(rgb_list), len(th_list))
            for i in range(min_len):
                pairs.append({
                    'rgb_id': rgb_list[i][0],
                    'th_id': th_list[i][0],
                    'rgb_path': rgb_list[i][1]['path'],
                    'th_path': th_list[i][1]['path'],
                    'frame_id': f'synthetic_{i}'
                })

        return pairs

    def _load_image(self, path, is_thermal=False):
        """Load image, handling 14-bit TIFF for thermal."""
        from PIL import Image
        try:
            img = Image.open(path)
            if is_thermal and img.mode == 'I;16':
                # 14-bit thermal TIFF → normalize to [0, 255]
                img_array = np.array(img, dtype=np.float32)
                # Clip to 14-bit range
                img_array = np.clip(img_array, 0, 16383)
                img_array = (img_array / 16383.0 * 255.0).astype(np.uint8)
                img = Image.fromarray(img_array, mode='L')
            elif is_thermal and img.mode != 'L':
                img = img.convert('L')
            elif not is_thermal and img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            # Return black placeholder
            size = (cfg.IMG_SIZE, cfg.IMG_SIZE)
            if is_thermal:
                return Image.new('L', size, 0)
            return Image.new('RGB', size, (0, 0, 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Load images
        rgb_img = self._load_image(pair['rgb_path'], is_thermal=False)
        th_img = self._load_image(pair['th_path'], is_thermal=True)

        # Resize to fixed size
        rgb_img = rgb_img.resize((cfg.IMG_SIZE, cfg.IMG_SIZE))
        th_img = th_img.resize((cfg.IMG_SIZE, cfg.IMG_SIZE))

        # Get annotations for both modalities (use RGB annotations as ground truth)
        anns_rgb = self.annotations.get(pair['rgb_id'], [])
        anns_th = self.annotations.get(pair['th_id'], [])

        # Combine annotations (prefer RGB annotations, fallback to thermal)
        anns = anns_rgb if anns_rgb else anns_th

        # Retrieve original image dimensions for correct normalization
        if anns_rgb:
            orig_w = self.images[pair['rgb_id']]['width']
            orig_h = self.images[pair['rgb_id']]['height']
        elif anns_th:
            orig_w = self.images[pair['th_id']]['width']
            orig_h = self.images[pair['th_id']]['height']
        else:
            orig_w, orig_h = cfg.IMG_SIZE, cfg.IMG_SIZE   # fallback

        # Parse bounding boxes and labels
        boxes = []
        labels = []
        for ann in anns:
            # COCO format: [x, y, width, height] in absolute pixels
            bbox = ann['bbox']
            x, y, w, h = bbox
            # Convert to [x1, y1, x2, y2] normalized by **original** image size
            x1 = x / orig_w
            y1 = y / orig_h
            x2 = (x + w) / orig_w
            y2 = (y + h) / orig_h
            # Validate
            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= 1.0 and y2 <= 1.0:
                boxes.append([x1, y1, x2, y2])

                # Map category
                cat_name = self.categories.get(ann['category_id'], 'other vehicle')
                cat_idx = self.cat_name_to_idx.get(cat_name, cfg.NUM_CLASSES - 1)
                labels.append(cat_idx)

        # Convert to tensors
        rgb_tensor = T.ToTensor()(rgb_img)
        th_tensor = T.ToTensor()(th_img)

        # Normalize RGB
        rgb_tensor = T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])(rgb_tensor)

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)

        return {
            'rgb': rgb_tensor,
            'thermal': th_tensor,
            'boxes': boxes,
            'labels': labels,
            'frame_id': pair['frame_id']
        }


def collate_fn(batch):
    """Custom collate to handle varying number of boxes per image."""
    rgb = torch.stack([item['rgb'] for item in batch])
    thermal = torch.stack([item['thermal'] for item in batch])
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    return {
        'rgb': rgb,
        'thermal': thermal,
        'boxes': boxes,
        'labels': labels
    }

# =============================================================
# 2. ORIGINAL MODEL (From Paper)
# =============================================================
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class YOLOv5NanoBackbone(nn.Module):
    """Lightweight backbone mimicking YOLOv5-Nano."""
    def __init__(self, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(16, 32),
            nn.MaxPool2d(2)  # /4
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(2)  # /8
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            nn.MaxPool2d(2)  # /16
        )
        self.out_channels = 128

    def forward(self, x):
        x = self.stem(x)       # /2, 16ch
        x = self.stage1(x)     # /4, 32ch
        x = self.stage2(x)     # /8, 64ch
        x = self.stage3(x)     # /16, 128ch
        return x


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class AdaptiveSensorSelection(nn.Module):
    """Global scalar weighting per modality."""
    def __init__(self, ch):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // 4),
            nn.ReLU(inplace=True),
            nn.Linear(ch // 4, 2)
        )

    def forward(self, f_rgb, f_th):
        score = self.mlp(f_rgb) + self.mlp(f_th)
        w = F.softmax(score, dim=-1)
        return w[:, 0:1, None, None], w[:, 1:2, None, None]


class OriginalModel(nn.Module):
    """Original thermal-RGB fusion model from the paper."""
    def __init__(self, num_classes=15):
        super().__init__()
        self.backbone_rgb = YOLOv5NanoBackbone(in_channels=3)
        self.backbone_th = YOLOv5NanoBackbone(in_channels=1)
        ch = self.backbone_rgb.out_channels  # 128

        self.sensor_sel = AdaptiveSensorSelection(ch)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(ch, ch)
        )
        self.se = SEBlock(ch)

        # Detection head
        self.cls_head = nn.Sequential(
            DepthwiseSeparableConv(ch, ch),
            nn.Conv2d(ch, num_classes, 1)
        )
        self.reg_head = nn.Sequential(
            DepthwiseSeparableConv(ch, ch),
            nn.Conv2d(ch, 4, 1)
        )
        self.obj_head = nn.Sequential(
            DepthwiseSeparableConv(ch, ch),
            nn.Conv2d(ch, 1, 1)
        )

    def forward(self, rgb, thermal):
       f_th = self.backbone_th(thermal)    # [B, 128, H/16, W/16]
       f_rgb = self.backbone_rgb(rgb)      # [B, 128, H/16, W/16]

       w_rgb, w_th = self.sensor_sel(f_rgb, f_th)
       f_rgb = f_rgb * w_rgb
       f_th = f_th * w_th

       fused = torch.cat([f_rgb, f_th], dim=1)
       fused = self.fuse_conv(fused)
       fused = self.se(fused)

       cls_pred = self.cls_head(fused)      # [B, NC, H, W]
       reg_pred = self.reg_head(fused)      # [B, 4, H, W]
       obj_pred = self.obj_head(fused)      # [B, 1, H, W]

       # ---- FLATTEN spatial dimensions ----
       B, _, H, W = cls_pred.shape
       cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(B, H*W, cfg.NUM_CLASSES)
       reg_pred = reg_pred.permute(0, 2, 3, 1).reshape(B, H*W, 4)
       obj_pred = obj_pred.permute(0, 2, 3, 1).reshape(B, H*W, 1)

       return cls_pred, reg_pred, obj_pred



# =============================================================
# 3. IMPROVED MODEL (ACAF-Net)
# =============================================================
class ReparamConv(nn.Module):
    """RepConv with re-parameterization capability."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.rbr_3x3 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.rbr_1x1 = nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.rbr_3x3(x) + self.rbr_1x1(x)))


class RepViTBlock(nn.Module):
    """Lightweight RepViT block."""
    def __init__(self, ch, expand=2):
        super().__init__()
        hidden = ch * expand
        self.repconv = ReparamConv(ch, hidden)
        self.pw = nn.Conv2d(hidden, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.repconv(x)
        out = self.pw(out)
        out = self.bn(out)
        return self.act(out + residual)


class RepViTBackbone(nn.Module):
    """Multi-scale backbone outputting [C3, C4, C5, C6]."""
    def __init__(self, in_channels=3):
        super().__init__()
        # Stem: /2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Stage1: /4, 64ch
        self.stage1 = nn.Sequential(
            RepViTBlock(32),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Stage2: /8, 96ch
        self.stage2 = nn.Sequential(
            RepViTBlock(64),
            nn.Conv2d(64, 96, 3, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        # Stage3: /16, 128ch
        self.stage3 = nn.Sequential(
            RepViTBlock(96),
            nn.Conv2d(96, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Stage4: /32, 160ch
        self.stage4 = nn.Sequential(
            RepViTBlock(128),
            nn.Conv2d(128, 160, 3, 2, 1, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True)
        )
        self.out_channels = [64, 96, 128, 160]  # C3, C4, C5, C6

    def forward(self, x):
        x = self.stem(x)          # /2
        c3 = self.stage1(x)       # /4, 64ch
        c4 = self.stage2(c3)      # /8, 96ch
        c5 = self.stage3(c4)      # /16, 128ch
        c6 = self.stage4(c5)      # /32, 160ch
        return c3, c4, c5, c6


class CrossModalTransformerFusion(nn.Module):
    """Efficient cross-attention fusion between RGB and Thermal."""
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.ch = ch
        self.num_heads = num_heads
        self.head_dim = ch // num_heads
        assert self.head_dim * num_heads == ch

        self.q_proj = nn.Conv2d(ch, ch, 1)
        self.k_proj = nn.Conv2d(ch, ch, 1)
        self.v_proj = nn.Conv2d(ch, ch, 1)

        self.out_proj = nn.Conv2d(ch * 2, ch, 1)
        self.norm = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, rgb, th):
        B, C, H, W = rgb.shape

        # Project queries from RGB, keys/values from Thermal
        Q = self.q_proj(rgb).reshape(B, self.num_heads, self.head_dim, H*W)
        K = self.k_proj(th).reshape(B, self.num_heads, self.head_dim, H*W)
        V = self.v_proj(th).reshape(B, self.num_heads, self.head_dim, H*W)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhdn,bhdm->bhnm', Q, K) * scale
        attn = F.softmax(attn, dim=-1)
        cross_rgb = torch.einsum('bhnm,bhdm->bhdn', attn, V)
        cross_rgb = cross_rgb.reshape(B, C, H, W)

        # Project queries from Thermal, keys/values from RGB
        Q2 = self.q_proj(th).reshape(B, self.num_heads, self.head_dim, H*W)
        K2 = self.k_proj(rgb).reshape(B, self.num_heads, self.head_dim, H*W)
        V2 = self.v_proj(rgb).reshape(B, self.num_heads, self.head_dim, H*W)

        attn2 = torch.einsum('bhdn,bhdm->bhnm', Q2, K2) * scale
        attn2 = F.softmax(attn2, dim=-1)
        cross_th = torch.einsum('bhnm,bhdm->bhdn', attn2, V2)
        cross_th = cross_th.reshape(B, C, H, W)

        # Concatenate cross-attended features
        fused = torch.cat([cross_rgb, cross_th], dim=1)
        fused = self.out_proj(fused)
        return self.act(self.norm(fused))


class SpatialAdaptiveGating(nn.Module):
    """Pixel-wise modality weighting."""
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(ch * 2, ch),
            nn.Conv2d(ch, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb, th):
        cat = torch.cat([rgb, th], dim=1)
        weights = self.conv(cat)  # [B, 2, H, W]
        w_rgb = weights[:, 0:1]
        w_th = weights[:, 1:2]
        # Normalize so they sum to 1
        w_sum = w_rgb + w_th + 1e-6
        return w_rgb / w_sum, w_th / w_sum


class BiFPN_lite(nn.Module):
    """Lightweight Bidirectional Feature Pyramid Network."""
    def __init__(self, ch_list):
        super().__init__()
        # Top-down pathway
        self.lat_p4 = nn.Conv2d(ch_list[2], ch_list[1], 1)  # C5→C4
        self.lat_p3 = nn.Conv2d(ch_list[1], ch_list[0], 1)  # C4→C3

        self.smooth_p4 = DepthwiseSeparableConv(ch_list[1], ch_list[1])
        self.smooth_p3 = DepthwiseSeparableConv(ch_list[0], ch_list[0])

        # Bottom-up pathway
        self.down_p4 = nn.Conv2d(ch_list[0], ch_list[1], 3, 2, 1)  # C3→C4
        self.down_p5 = nn.Conv2d(ch_list[1], ch_list[2], 3, 2, 1)  # C4→C5

        self.smooth_p4_out = DepthwiseSeparableConv(ch_list[1], ch_list[1])
        self.smooth_p5_out = DepthwiseSeparableConv(ch_list[2], ch_list[2])
        self.smooth_p6_out = DepthwiseSeparableConv(ch_list[3], ch_list[3])

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, p3, p4, p5, p6):
        # Top-down
        p4_up = self.upsample(self.lat_p4(p5))
        p4 = self.smooth_p4(p4 + p4_up)

        p3_up = self.upsample(self.lat_p3(p4))
        p3 = self.smooth_p3(p3 + p3_up)

        # Bottom-up
        p4_dn = self.smooth_p4_out(p4 + self.down_p4(p3))
        p5_dn = self.smooth_p5_out(p5 + self.down_p5(p4_dn))
        p6_dn = self.smooth_p6_out(p6)

        return p3, p4_dn, p5_dn, p6_dn


class DecoupledHead(nn.Module):
    """Decoupled detection head (classification, regression, objectness)."""
    def __init__(self, ch, num_classes):
        super().__init__()
        self.cls_branch = nn.Sequential(
            DepthwiseSeparableConv(ch, ch),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(ch, ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, num_classes, 1)
        )
        self.reg_branch = nn.Sequential(
            DepthwiseSeparableConv(ch, ch),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(ch, ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 4, 1)
        )
        self.obj_branch = nn.Sequential(
            DepthwiseSeparableConv(ch, ch),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(ch, ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, 1)
        )

    def forward(self, x):
        cls_pred = self.cls_branch(x)
        reg_pred = self.reg_branch(x)
        obj_pred = self.obj_branch(x)
        return cls_pred, reg_pred, obj_pred


class ImprovedModel(nn.Module):
    """Improved ACAF-Net with multi-scale fusion and adaptive gating."""
    def __init__(self, num_classes=15):
        super().__init__()
        self.backbone_rgb = RepViTBackbone(in_channels=3)
        self.backbone_th = RepViTBackbone(in_channels=1)

        ch_list = self.backbone_rgb.out_channels  # [64, 96, 128, 160]

        # Per-scale fusion modules
        self.cstf = nn.ModuleList([
            CrossModalTransformerFusion(c) for c in ch_list
        ])
        self.samg = nn.ModuleList([
            SpatialAdaptiveGating(c) for c in ch_list
        ])

        # FPN neck
        self.fpn = BiFPN_lite(ch_list)

        # Per-scale detection heads
        self.heads = nn.ModuleList([
            DecoupledHead(c, num_classes) for c in ch_list
        ])

    def forward(self, rgb, thermal):
        # Multi-scale backbone features
        r3, r4, r5, r6 = self.backbone_rgb(rgb)
        t3, t4, t5, t6 = self.backbone_th(thermal)

        # Spatial Adaptive Gating applied to raw features **before** fusion
        w3_r, w3_t = self.samg[0](r3, t3)
        w4_r, w4_t = self.samg[1](r4, t4)
        w5_r, w5_t = self.samg[2](r5, t5)
        w6_r, w6_t = self.samg[3](r6, t6)

        g_r3 = r3 * w3_r
        g_t3 = t3 * w3_t
        g_r4 = r4 * w4_r
        g_t4 = t4 * w4_t
        g_r5 = r5 * w5_r
        g_t5 = t5 * w5_t
        g_r6 = r6 * w6_r
        g_t6 = t6 * w6_t

        # Cross-modal fusion on gated features
        p3 = self.cstf[0](g_r3, g_t3)
        p4 = self.cstf[1](g_r4, g_t4)
        p5 = self.cstf[2](g_r5, g_t5)
        p6 = self.cstf[3](g_r6, g_t6)

        # BiFPN
        p3, p4, p5, p6 = self.fpn(p3, p4, p5, p6)

        # Multi-scale predictions
        all_cls, all_reg, all_obj = [], [], []
        for head, feat in zip(self.heads, [p3, p4, p5, p6]):
            cls_p, reg_p, obj_p = head(feat)
            # Flatten spatial dimensions
            B, _, H, W = cls_p.shape
            all_cls.append(cls_p.permute(0, 2, 3, 1).reshape(B, -1, cfg.NUM_CLASSES))
            all_reg.append(reg_p.permute(0, 2, 3, 1).reshape(B, -1, 4))
            all_obj.append(obj_p.permute(0, 2, 3, 1).reshape(B, -1, 1))

        return (
            torch.cat(all_cls, dim=1),
            torch.cat(all_reg, dim=1),
            torch.cat(all_obj, dim=1)
        )


# =============================================================
# 4. LOSS FUNCTION
# =============================================================
class DetectionLoss(nn.Module):
    """Combined classification + regression + objectness loss."""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        self.obj_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        self.iou_loss_fn = nn.SmoothL1Loss(reduction='sum')

    def forward(self, cls_pred, reg_pred, obj_pred, targets_boxes, targets_labels):
        B = cls_pred.size(0)
        device = cls_pred.device

        total_cls_loss = torch.tensor(0., device=device)
        total_reg_loss = torch.tensor(0., device=device)
        total_obj_loss = torch.tensor(0., device=device)
        total_predictions = 0   # keep track for objectness averaging

        for b in range(B):
            gt_boxes = targets_boxes[b].to(device)
            gt_labels = targets_labels[b].to(device)

            pred_boxes = reg_pred[b]   # [N, 4]
            pred_obj   = obj_pred[b]   # [N, 1]
            pred_cls   = cls_pred[b]   # [N, NC]

            N = pred_boxes.size(0)
            total_predictions += N

            if gt_boxes.numel() == 0:
                obj_target = torch.zeros_like(pred_obj)
                obj_loss_per_image = self.obj_loss_fn(pred_obj, obj_target) / N
                total_obj_loss += obj_loss_per_image
                continue

            M = gt_boxes.size(0)

            # Compute pairwise center distances
            pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
            pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
            gt_cx   = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
            gt_cy   = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2

            dist = (pred_cx.unsqueeze(1) - gt_cx.unsqueeze(0)) ** 2 + \
                   (pred_cy.unsqueeze(1) - gt_cy.unsqueeze(0)) ** 2   # [N, M]

            # Each GT assigned to closest prediction
            _, min_idx = dist.min(dim=0)   # [M]

            # Map predictions to GT (-1 for background)
            assigned_gt = torch.full((N,), -1, dtype=torch.long, device=device)
            assigned_gt[min_idx] = torch.arange(M, device=device)

            pos_mask = assigned_gt >= 0

            obj_target = pos_mask.float().unsqueeze(1)          # [N, 1]
            cls_target = torch.zeros(N, self.num_classes, device=device)
            cls_target[pos_mask, gt_labels[assigned_gt[pos_mask]]] = 1.0

            # Per-image losses, averaged over N
            total_cls_loss += self.cls_loss_fn(pred_cls, cls_target) / N
            total_obj_loss += self.obj_loss_fn(pred_obj, obj_target) / N
            if pos_mask.any():
                total_reg_loss += self.iou_loss_fn(
                    pred_boxes[pos_mask],
                    gt_boxes[assigned_gt[pos_mask]]
                ) / N

        # Average over batch
        total_cls_loss /= B
        total_reg_loss /= B
        total_obj_loss /= B

        return total_cls_loss + total_reg_loss + total_obj_loss


# =============================================================
# 5. TRAINING & EVALUATION
# =============================================================
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_ap(recall, precision):
    """Compute average precision given recall and precision arrays."""
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def evaluate_map(model, dataloader, iou_threshold=0.5):
    """Evaluate mAP at a single IoU threshold."""
    model.eval()
    all_detections = []
    all_ground_truths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Eval'):
            rgb = batch['rgb'].to(DEVICE)
            thermal = batch['thermal'].to(DEVICE)
            boxes = batch['boxes']
            labels = batch['labels']

            cls_pred, reg_pred, obj_pred = model(rgb, thermal)

            B = cls_pred.size(0)
            for b in range(B):
                scores = torch.sigmoid(obj_pred[b]).squeeze(-1)
                cls_scores = torch.sigmoid(cls_pred[b])
                max_cls, max_idx = cls_scores.max(dim=-1)
                final_scores = scores * max_cls

                mask = final_scores > 0.3
                if mask.sum() == 0:
                    all_detections.append({
                        'boxes': torch.zeros(0, 4),
                        'scores': torch.zeros(0),
                        'labels': torch.zeros(0, dtype=torch.long)
                    })
                    all_ground_truths.append({
                        'boxes': boxes[b],
                        'labels': labels[b]
                    })
                    continue

                filtered_boxes = reg_pred[b][mask]
                filtered_scores = final_scores[mask]
                filtered_labels = max_idx[mask]

                keep = nms(filtered_boxes, filtered_scores, iou_threshold=0.45)
                all_detections.append({
                    'boxes': filtered_boxes[keep].cpu(),
                    'scores': filtered_scores[keep].cpu(),
                    'labels': filtered_labels[keep].cpu()
                })
                all_ground_truths.append({
                    'boxes': boxes[b],
                    'labels': labels[b]
                })

    aps = []
    for cls_idx in range(cfg.NUM_CLASSES):
        det_scores = []
        det_matched = []
        n_gt = 0

        for det, gt in zip(all_detections, all_ground_truths):
            cls_mask = det['labels'] == cls_idx
            cls_boxes = det['boxes'][cls_mask]
            cls_scores = det['scores'][cls_mask]

            gt_mask = gt['labels'] == cls_idx
            gt_boxes = gt['boxes'][gt_mask]
            n_gt += gt_mask.sum().item()

            if len(cls_boxes) == 0:
                continue

            sort_idx = torch.argsort(cls_scores, descending=True)
            cls_boxes = cls_boxes[sort_idx]
            cls_scores = cls_scores[sort_idx]

            if len(gt_boxes) > 0:
                ious = box_iou(cls_boxes, gt_boxes)
                matched = torch.zeros(len(cls_boxes), dtype=torch.bool)
                gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

                for i in range(len(cls_boxes)):
                    if gt_matched.all():
                        break
                    best_iou, best_gt = ious[i].max(dim=0)
                    if best_iou >= iou_threshold and not gt_matched[best_gt]:
                        matched[i] = True
                        gt_matched[best_gt] = True

                det_matched.extend(matched.tolist())
            else:
                det_matched.extend([False] * len(cls_boxes))

            det_scores.extend(cls_scores.tolist())

        if n_gt == 0:
            aps.append(0.0)
            continue

        if len(det_scores) == 0:
            aps.append(0.0)
            continue

        det_scores = np.array(det_scores)
        det_matched = np.array(det_matched)
        sort_idx = np.argsort(-det_scores)
        det_matched = det_matched[sort_idx]

        tp = np.cumsum(det_matched)
        fp = np.cumsum(~det_matched)
        recall = tp / n_gt
        precision = tp / (tp + fp + 1e-6)

        ap = compute_ap(recall, precision)
        aps.append(ap)

    mAP = np.mean(aps) * 100
    return mAP, aps


def compute_coco_map(model, dataloader):
    """Compute COCO-style mAP@[.50:.95] by averaging AP over IoU thresholds."""
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    all_mAPs = []
    for iou_thr in iou_thresholds:
        mAP, _ = evaluate_map(model, dataloader, iou_threshold=iou_thr)
        all_mAPs.append(mAP)
    return np.mean(all_mAPs)


# --------------------------------------------------------------------
# 6. Memory‑optimised training loop
# --------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, scaler=None):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(dataloader, desc='Training')
    for i, batch in enumerate(pbar):
        rgb = batch['rgb'].to(DEVICE, non_blocking=True)
        thermal = batch['thermal'].to(DEVICE, non_blocking=True)
        boxes = batch['boxes']
        labels = batch['labels']

        with amp.autocast(enabled=cfg.USE_AMP):
            cls_pred, reg_pred, obj_pred = model(rgb, thermal)
            loss = criterion(cls_pred, reg_pred, obj_pred, boxes, labels)
            loss = loss / cfg.GRADIENT_ACCUMULATION_STEPS

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % cfg.GRADIENT_ACCUMULATION_STEPS == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * cfg.GRADIENT_ACCUMULATION_STEPS
        pbar.set_postfix({'loss': f'{total_loss/(i+1):.4f}'})

    if (i + 1) % cfg.GRADIENT_ACCUMULATION_STEPS != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / len(dataloader)


def train_model(model, train_loader, val_loader, model_name, epochs=cfg.EPOCHS):
    optimizer = AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=cfg.LR_FACTOR, patience=cfg.LR_PATIENCE
    )
    criterion = DetectionLoss(cfg.NUM_CLASSES)
    scaler = amp.GradScaler(enabled=cfg.USE_AMP)

    history = {
        'train_loss': [],
        'val_mAP': [],
        'val_mAP50_95': [],
        'epoch_times': [],
        'best_mAP': 0.0,
        'best_epoch': 0
    }

    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        history['train_loss'].append(train_loss)

        mAP50, _ = evaluate_map(model, val_loader, iou_threshold=0.5)
        mAP50_95 = compute_coco_map(model, val_loader)

        history['val_mAP'].append(mAP50)
        history['val_mAP50_95'].append(mAP50_95)
        history['epoch_times'].append(time.time() - epoch_start)

        scheduler.step(mAP50)

        if mAP50 > history['best_mAP']:
            history['best_mAP'] = mAP50
            history['best_epoch'] = epoch + 1

         # save the checkpoint
        save_checkpoint(model, optimizer, epoch, history) 

            # Agar aapne `save_checkpoint` function ORG D P.M mein copy nahi kiya hai, 
            # toh direct torch.save use kar lijiye:
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"[INFO] Best model saved at epoch {epoch+1}") 

        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
              f"mAP@0.5: {mAP50:.2f}% | mAP@0.5:0.95: {mAP50_95:.2f}% | "
              f"Time: {history['epoch_times'][-1]:.1f}s")

    torch.cuda.empty_cache()
    gc.collect()

    return history


# --------------------------------------------------------------------
# 7. Accuracy metric (defined correctly)
# --------------------------------------------------------------------
def compute_detection_accuracy(model, dataloader, conf_thresh=0.5, iou_thresh=0.5):
    """Compute classification accuracy of matched predictions above a confidence threshold."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Acc Eval'):
            rgb = batch['rgb'].to(DEVICE)
            thermal = batch['thermal'].to(DEVICE)
            gt_boxes = batch['boxes']
            gt_labels = batch['labels']

            cls_pred, reg_pred, obj_pred = model(rgb, thermal)

            B = cls_pred.size(0)
            for b in range(B):
                scores = torch.sigmoid(obj_pred[b]).squeeze(-1)
                cls_scores = torch.sigmoid(cls_pred[b])
                final_scores, pred_labels = cls_scores.max(dim=-1)
                final_scores = final_scores * scores

                mask = final_scores > conf_thresh
                if mask.sum() == 0:
                    continue

                pred_boxes = reg_pred[b][mask]
                pred_labels = pred_labels[mask]
                pred_scores = final_scores[mask]

                keep = nms(pred_boxes, pred_scores, iou_thresh)
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]

                if gt_boxes[b].numel() == 0:
                    continue

                gt_labs = gt_labels[b].to(DEVICE)
                ious = box_iou(pred_boxes, gt_boxes[b].to(DEVICE))
                best_iou, best_gt = ious.max(dim=1)
                matched_mask = best_iou >= iou_thresh

                for i, matched in enumerate(matched_mask):
                    if matched and pred_labels[i] == gt_labs[best_gt[i]]:
                        correct += 1
                    total += 1

    return correct / total if total > 0 else 0.0


# =============================================================
# 8. VISUALIZATION FUNCTIONS
# =============================================================
def plot_training_curves(orig_history, impr_history):
    """Compare training loss and mAP curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs_orig = range(1, len(orig_history['train_loss']) + 1)
    epochs_impr = range(1, len(impr_history['train_loss']) + 1)

    ax = axes[0, 0]
    ax.plot(epochs_orig, orig_history['train_loss'], 'b-', linewidth=2, label='Original')
    ax.plot(epochs_impr, impr_history['train_loss'], 'r-', linewidth=2, label='Improved ACAF-Net')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs_orig, orig_history['val_mAP'], 'b-', linewidth=2, label='Original')
    ax.plot(epochs_impr, impr_history['val_mAP'], 'r-', linewidth=2, label='Improved ACAF-Net')
    ax.set_xlabel('Epoch'); ax.set_ylabel('mAP@0.5 (%)')
    ax.set_title('Validation mAP@0.5'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs_orig, orig_history['val_mAP50_95'], 'b-', linewidth=2, label='Original')
    ax.plot(epochs_impr, impr_history['val_mAP50_95'], 'r-', linewidth=2, label='Improved ACAF-Net')
    ax.set_xlabel('Epoch'); ax.set_ylabel('mAP@0.5:0.95 (%)')
    ax.set_title('Validation mAP@0.5:0.95'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.bar(epochs_orig, orig_history['epoch_times'], alpha=0.6, label='Original', color='blue')
    ax.bar(epochs_impr, impr_history['epoch_times'], alpha=0.6, label='Improved ACAF-Net', color='red')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Time (seconds)')
    ax.set_title('Training Time per Epoch'); ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle('Training Comparison: Original vs Improved Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(cfg.PLOTS_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_metrics_comparison(orig_history, impr_history):
    """Bar chart comparing final metrics."""
    metrics_names = [
        'Best mAP@0.5 (%)',
        'Best mAP@0.5:0.95 (%)',
        'Final Loss',
        'Avg Epoch Time (s)'
    ]
    original_values = [
        orig_history['best_mAP'],
        max(orig_history['val_mAP50_95']),
        orig_history['train_loss'][-1],
        np.mean(orig_history['epoch_times'])
    ]
    improved_values = [
        impr_history['best_mAP'],
        max(impr_history['val_mAP50_95']),
        impr_history['train_loss'][-1],
        np.mean(impr_history['epoch_times'])
    ]

    x = np.arange(len(metrics_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, original_values, width, label='Original Model',
                   color='steelblue', edgecolor='navy', linewidth=1.5)
    bars2 = ax.bar(x + width/2, improved_values, width, label='Improved ACAF-Net',
                   color='darkorange', edgecolor='darkred', linewidth=1.5)
    ax.set_ylabel('Value')
    ax.set_title('Model Performance Comparison on FLIR ADAS v2 Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.legend(fontsize=11)

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(cfg.PLOTS_DIR / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_per_class_ap(orig_aps, impr_aps):
    """Per-class AP comparison."""
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(cfg.CLASSES))
    width = 0.35
    ax.bar(x - width/2, orig_aps, width, label='Original Model', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, impr_aps, width, label='Improved ACAF-Net', color='darkorange', alpha=0.8)
    ax.set_xlabel('Class'); ax.set_ylabel('Average Precision (%)')
    ax.set_title('Per-Class AP Comparison on FLIR ADAS v2', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cfg.CLASSES, rotation=45, ha='right', fontsize=9)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.PLOTS_DIR / 'per_class_ap.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_model_efficiency(orig_params, impr_params, orig_flops, impr_flops):
    """Compare model efficiency metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax = axes[0]
    ax.bar(['Original', 'Improved'], [orig_params, impr_params],
           color=['steelblue', 'darkorange'], edgecolor='black')
    ax.set_ylabel('Parameters (M)'); ax.set_title('Model Parameters')
    for i, v in enumerate([orig_params, impr_params]):
        ax.text(i, v + 0.05, f'{v:.2f}M', ha='center', fontweight='bold')

    ax = axes[1]
    ax.bar(['Original', 'Improved'], [orig_flops, impr_flops],
           color=['steelblue', 'darkorange'], edgecolor='black')
    ax.set_ylabel('FLOPs (G)'); ax.set_title('Computational Cost')
    for i, v in enumerate([orig_flops, impr_flops]):
        ax.text(i, v + 0.05, f'{v:.2f}G', ha='center', fontweight='bold')

    ax = axes[2]
    sizes = [orig_params * 4 / 1e3, impr_params * 4 / 1e3]
    ax.bar(['Original', 'Improved'], sizes, color=['steelblue', 'darkorange'], edgecolor='black')
    ax.set_ylabel('Model Size (MB)'); ax.set_title('Memory Footprint (FP32)')
    for i, v in enumerate(sizes):
        ax.text(i, v + 0.1, f'{v:.1f}MB', ha='center', fontweight='bold')

    plt.suptitle('Model Efficiency Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(cfg.PLOTS_DIR / 'efficiency_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_architecture():
    """Draw the improved ACAF-Net architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    ax.set_xlim(0, 16); ax.set_ylim(0, 14)

    def draw_box(x, y, w, h, text, color='lightblue', fontsize=8):
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor='black', linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')

    draw_box(4, 13, 3, 1.2, 'RGB Input\n(3×416×416)', 'lightcoral', 8)
    draw_box(12, 13, 3, 1.2, 'Thermal Input\n(1×416×416)', 'wheat', 8)
    draw_box(4, 11, 3.5, 1.5, 'RepViT Backbone\n(RGB stream)', 'lightgreen', 8)
    draw_box(12, 11, 3.5, 1.5, 'RepViT Backbone\n(Thermal stream)', 'lightgreen', 8)

    scales = [('C3\n64ch', 9.5), ('C4\n96ch', 8.2), ('C5\n128ch', 6.9), ('C6\n160ch', 5.6)]
    for (text, y), (text2, y2) in zip(scales, scales):
        draw_box(4, y, 1.8, 0.7, text, 'palegreen', 7)
        draw_box(12, y, 1.8, 0.7, text2, 'palegreen', 7)

    draw_box(8, 7.5, 5, 1.5, 'Cross-Modal Transformer Fusion (×4)\nMulti-Head Cross-Attention RGB ↔ Thermal', 'violet', 7)
    draw_box(8, 5.5, 5, 1.2, 'Spatially Adaptive Modality Gating (×4)\nPer-pixel modality confidence prediction', 'plum', 7)
    draw_box(8, 4, 5, 1, 'BiFPN-lite Neck\nTop-down + Bottom-up multi-scale fusion', 'orange', 8)
    draw_box(4, 2.2, 3, 1, 'Decoupled Head\nP3 (large objects)', 'pink', 7)
    draw_box(8, 2.2, 3, 1, 'Decoupled Head\nP4, P5 (medium)', 'pink', 7)
    draw_box(12, 2.2, 3, 1, 'Decoupled Head\nP6 (small objects)', 'pink', 7)
    draw_box(8, 0.7, 4, 0.8, 'Concatenated Predictions\n(Class + Box + Objectness)', 'khaki', 7)

    arrow_style = dict(arrowstyle='->', lw=1.5, color='gray')
    ax.annotate('', xy=(4, 11.9), xytext=(4, 12.4), arrowprops=arrow_style)
    ax.annotate('', xy=(12, 11.9), xytext=(12, 12.4), arrowprops=arrow_style)
    ax.annotate('', xy=(4, 9.9), xytext=(4, 10.1), arrowprops=arrow_style)
    ax.annotate('', xy=(12, 9.9), xytext=(12, 10.1), arrowprops=arrow_style)
    for y in [9.5, 8.2, 6.9, 5.6]:
        ax.annotate('', xy=(5.2, y), xytext=(4.9, y), arrowprops=arrow_style)
        ax.annotate('', xy=(10.8, y), xytext=(11.1, y), arrowprops=arrow_style)
    ax.annotate('', xy=(8, 6.2), xytext=(8, 6.7), arrowprops=arrow_style)
    ax.annotate('', xy=(8, 4.5), xytext=(8, 4.9), arrowprops=arrow_style)
    ax.annotate('', xy=(4, 2.8), xytext=(8, 3.4), arrowprops=arrow_style)
    ax.annotate('', xy=(12, 2.8), xytext=(8, 3.4), arrowprops=arrow_style)
    for x in [4, 8, 12]:
        ax.annotate('', xy=(8, 1.2), xytext=(x, 1.7), arrowprops=arrow_style)

    ax.set_title('Improved ACAF-Net Architecture for Thermal-RGB Fusion', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(cfg.PLOTS_DIR / 'architecture.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================
# 9. MAIN EXECUTION (with accuracy integrated)
# =============================================================
def main():
    print("=" * 60)
    print(" Thermal-RGB Object Detection on FLIR ADAS v2 Dataset")
    print(" Original Model vs Improved ACAF-Net")
    print("=" * 60)
    print(f"\n[CONFIG]")
    print(f"  Image Size: {cfg.IMG_SIZE}x{cfg.IMG_SIZE}")
    print(f"  Batch Size: {cfg.BATCH_SIZE}")
    print(f"  Epochs: {cfg.EPOCHS}")
    print(f"  Classes: {cfg.NUM_CLASSES} ({', '.join(cfg.CLASSES[:5])}...)")
    print(f"  Device: {DEVICE}")

    # Download FLIR dataset
    if not cfg.DATA_DIR.exists() or not list(cfg.DATA_DIR.iterdir()):
        print("\n[INFO] FLIR dataset not found. Downloading...")
        try:
            download_flir_dataset()
        except Exception as e:
            print(f"[WARNING] Auto-download failed: {e}")
            print("[INFO] Please download the dataset manually from:")
            print("  https://www.kaggle.com/datasets/samdazel/teledyne-flir-adas-thermal-dataset-v2")
            print("  and extract to ./flir_data/")
            print("\n[INFO] Creating dummy dataset for demonstration...")

    print("\n[INFO] Creating dataloaders...")
    use_real_data = False
    try:
        train_ds = FLIRDataset(cfg.DATA_DIR, split='train')
        if len(train_ds) >= 4:
            use_real_data = True
            val_ds = FLIRDataset(cfg.DATA_DIR, split='val')
            print(f"[INFO] Using real FLIR dataset: {len(train_ds)} train, {len(val_ds)} val images")
        else:
            print("[WARNING] Too few paired images found. Using synthetic data.")
    except Exception as e:
        print(f"[WARNING] Could not load FLIR dataset: {e}")
        print("[INFO] Falling back to synthetic data for demonstration.")

    if not use_real_data:
        class DummyPairedDataset(Dataset):
            def __init__(self, n=100, size=cfg.IMG_SIZE):
                self.n = n; self.size = size
            def __len__(self): return self.n
            def __getitem__(self, idx):
                rgb = torch.randn(3, self.size, self.size)
                th  = torch.randn(1, self.size, self.size)
                boxes = torch.tensor([[0.2, 0.3, 0.5, 0.6], [0.6, 0.2, 0.8, 0.7]])
                labels = torch.tensor([0, 2])
                return {'rgb': rgb, 'thermal': th, 'boxes': boxes, 'labels': labels}
        n_train, n_val = 200, 50
        train_ds = DummyPairedDataset(n_train)
        val_ds = DummyPairedDataset(n_val)
        print(f"[INFO] Using synthetic data: {len(train_ds)} train, {len(val_ds)} val images")

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
                              pin_memory=(DEVICE.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
                            pin_memory=(DEVICE.type == 'cuda'))
    print(f"[INFO] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print("\n[INFO] Building models...")
    original_model = OriginalModel(num_classes=cfg.NUM_CLASSES).to(DEVICE)
    improved_model = ImprovedModel(num_classes=cfg.NUM_CLASSES).to(DEVICE)

    orig_params = sum(p.numel() for p in original_model.parameters()) / 1e6
    impr_params = sum(p.numel() for p in improved_model.parameters()) / 1e6
    print(f"\n[INFO] Original Model: {orig_params:.2f}M parameters")
    print(f"[INFO] Improved Model: {impr_params:.2f}M parameters")

    if HAS_THOP:
        dummy_rgb = torch.randn(1, 3, 416, 416).to(DEVICE)
        dummy_th = torch.randn(1, 1, 416, 416).to(DEVICE)
        flops_orig, _ = profile(original_model, inputs=(dummy_rgb, dummy_th), verbose=False)
        flops_impr, _ = profile(improved_model, inputs=(dummy_rgb, dummy_th), verbose=False)
        flops_orig_g = flops_orig / 1e9
        flops_impr_g = flops_impr / 1e9
    else:
        flops_orig_g = orig_params * 0.8
        flops_impr_g = impr_params * 0.9
    print(f"[INFO] Original FLOPs: {flops_orig_g:.2f}G")
    print(f"[INFO] Improved FLOPs: {flops_impr_g:.2f}G")

    # Train
    print("\n" + "=" * 60)
    print(" TRAINING ORIGINAL MODEL")
    print("=" * 60)
    orig_history = train_model(original_model, train_loader, val_loader,
                               model_name="Original Model", epochs=cfg.EPOCHS)

    print("\n" + "=" * 60)
    print(" TRAINING IMPROVED MODEL (ACAF-Net)")
    print("=" * 60)
    impr_history = train_model(improved_model, train_loader, val_loader,
                               model_name="Improved ACAF-Net", epochs=cfg.EPOCHS)

    # Final evaluation
    print("\n[INFO] Computing final per-class AP...")
    _, orig_aps = evaluate_map(original_model, val_loader, iou_threshold=0.5)
    _, impr_aps = evaluate_map(improved_model, val_loader, iou_threshold=0.5)
    orig_aps_pct = [ap * 100 for ap in orig_aps]
    impr_aps_pct = [ap * 100 for ap in impr_aps]

    # Accuracy
    print("[INFO] Computing detection accuracy...")
    orig_acc = compute_detection_accuracy(original_model, val_loader)
    impr_acc = compute_detection_accuracy(improved_model, val_loader)

    # Results Summary
    print("\n" + "=" * 60)
    print(" FINAL RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<35} {'Original':>12} {'Improved':>12} {'Δ':>10}")
    print("-" * 70)
    print(f"{'Best mAP@0.5 (%)':<35} {orig_history['best_mAP']:>12.2f} "
          f"{impr_history['best_mAP']:>12.2f} "
          f"{impr_history['best_mAP'] - orig_history['best_mAP']:>+9.2f}")
    print(f"{'Best mAP@0.5:0.95 (%)':<35} {max(orig_history['val_mAP50_95']):>12.2f} "
          f"{max(impr_history['val_mAP50_95']):>12.2f} "
          f"{max(impr_history['val_mAP50_95']) - max(orig_history['val_mAP50_95']):>+9.2f}")
    print(f"{'Detection Accuracy (%)':<35} {orig_acc*100:>12.2f} {impr_acc*100:>12.2f} "
          f"{(impr_acc - orig_acc)*100:>+9.2f}")
    print(f"{'Parameters (M)':<35} {orig_params:>12.2f} {impr_params:>12.2f} "
          f"{impr_params - orig_params:>+9.2f}")
    print(f"{'FLOPs (G)':<35} {flops_orig_g:>12.2f} {flops_impr_g:>12.2f} "
          f"{flops_impr_g - flops_orig_g:>+9.2f}")

    # Visualizations
    print("\n[INFO] Generating comparison visualizations...\n")
    plot_architecture()
    plot_training_curves(orig_history, impr_history)
    plot_metrics_comparison(orig_history, impr_history)
    plot_per_class_ap(orig_aps_pct, impr_aps_pct)
    plot_model_efficiency(orig_params, impr_params, flops_orig_g, flops_impr_g)

    print(f"\n[INFO] All plots saved to {cfg.PLOTS_DIR.absolute()}")
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
