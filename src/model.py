"""
Model definition and transfer learning setup for QR code multi-label classification.
Uses pretrained models with custom classifier heads.
Supports multi-label classification for: is_copied, is_blurry, is_low_light.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional
import numpy as np

def _patch_first_layer(layer, in_channels: int):
    """
    Patch the first convolution layer to accept arbitrary input channels.
    Copies weights from the first 3 channels to new channels if increasing.
    """
    if layer.in_channels == in_channels:
        return
        
    print(f"Patching first layer: {layer.in_channels} -> {in_channels} channels")
    
    # Create new layer with same parameters but new in_channels
    new_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=layer.out_channels,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        bias=layer.bias is not None,
        padding_mode=layer.padding_mode,
        dilation=layer.dilation,
        groups=layer.groups # Note: groups might need adjustment if it was input-dependent, but usually 1 for first layer
    )
    
    # Initialize weights
    with torch.no_grad():
        if layer.bias is not None:
            new_layer.bias = layer.bias
            
        # Copy existing weights
        # layer.weight shape: (out, in, k, k)
        # If new in_channels > old, we copy old to first few, and init rest
        src_channels = min(layer.in_channels, in_channels)
        new_layer.weight[:, :src_channels] = layer.weight[:, :src_channels]
        
        # Initialize new channels (e.g. with mean of RGB weights or similar)
        # For simplicity, we can copy the mean of RGB weights to the 4th channel
        # or just random init (standard). Pretrained weights usually expect normalized inputs.
        # If we just randomly init, it will learn. 
        # But commonly we reuse one of the channels or average.
        if in_channels > layer.in_channels:
            # Average of original channels
            avg_weight = torch.mean(layer.weight, dim=1, keepdim=True)
            # Assign to extra channels
            for i in range(layer.in_channels, in_channels):
                new_layer.weight[:, i:i+1] = avg_weight
                
    # Replace weights in place? No, we need to return new layer or modify object passing in.
    # The caller passed 'self.backbone.conv1'. This is an object reference.
    # We cannot replace the object reference in the parent by just assigning to 'layer'.
    # We need to modify the parent.
    # The caller code was: 
    # _patch_first_layer(self.backbone.conv1, in_channels)
    # This won't work because we can't replace the layer in the module this way.
    
    # Correction: The caller logic needs to assign the result.
    # But checking my previous edit:
    # _patch_first_layer(self.backbone.conv1, in_channels)
    # This implies I need to modify the layer IN PLACE or Return it and assign it.
    
    # Modification in place (modifying internals of the conv2d object):
    layer.in_channels = in_channels
    layer.weight = nn.Parameter(new_layer.weight)
    if layer.bias is not None:
        layer.bias = nn.Parameter(new_layer.bias)
    # verify strictly
    assert layer.weight.shape[1] == in_channels



class QRCodeClassifier(nn.Module):
    """
    Transfer learning model for QR code multi-label classification.
    Uses a pretrained backbone with a custom classifier head.
    Outputs 3 independent binary predictions: is_copied, is_blurry, is_low_light.
    """
    
    def __init__(
        self,
        num_labels: int = 4,
        model_name: str = 'resnet50',
        pretrained: bool = True,
        freeze_backbone: bool = False,
        in_channels: int = 3,
        pretrained_path: str = None
    ):
        """
        Args:
            num_labels: Number of output labels (default: 3 for is_copied, is_blurry, is_low_light)
            model_name: Name of the backbone model ('resnet50', 'resnet18', 'efficientnet_b0', etc.)
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone weights (feature extraction mode)
            in_channels: Number of input channels (default: 4 for RGB + Registered)
            pretrained_path: Optional path to local pretrained weights file
        """
        super(QRCodeClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.in_channels = in_channels
        
        # Load pretrained model
        if model_name == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original classifier
            
            # Patch first layer if needed
            if in_channels != 3:
                _patch_first_layer(self.backbone.conv1, in_channels)
                
        elif model_name == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
            if in_channels != 3:
                _patch_first_layer(self.backbone.conv1, in_channels)
            
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
            if in_channels != 3:
                _patch_first_layer(self.backbone.features[0][0], in_channels)
            
        elif model_name == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
            
            if in_channels != 3:
                _patch_first_layer(self.backbone.features[0][0], in_channels)
            
        else:
            # Try loading from timm
            try:
                import timm
                
                # If local path is provided, we set pretrained=False to avoid downloading
                # and load weights manually later
                actual_pretrained = pretrained and (pretrained_path is None)
                
                # timm supports in_chans argument
                self.backbone = timm.create_model(
                    model_name, 
                    pretrained=actual_pretrained, 
                    num_classes=0,
                    in_chans=in_channels
                )
                
                if pretrained_path:
                    print(f"Loading custom pretrained weights from: {pretrained_path}")
                    checkpoint = torch.load(pretrained_path, map_location='cpu')
                    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
                    
                    # Handle mapping for convnextv2 from official to timm format
                    if 'convnextv2' in model_name:
                        mapped_dict = {}
                        for k, v in state_dict.items():
                            new_k = k
                            # Stem / Downsample
                            if k.startswith('downsample_layers.0.0'):
                                new_k = k.replace('downsample_layers.0.0', 'stem.0')
                            elif k.startswith('downsample_layers.0.1'):
                                new_k = k.replace('downsample_layers.0.1', 'stem.1')
                            elif k.startswith('downsample_layers.'):
                                parts = k.split('.')
                                new_k = f"stages.{parts[1]}.downsample.{parts[2]}.{'.'.join(parts[3:])}"
                            
                            # Stages
                            elif k.startswith('stages.'):
                                parts = k.split('.')
                                if len(parts) >= 4:
                                    stage_idx = parts[1]
                                    block_idx = parts[2]
                                    param_name = '.'.join(parts[3:])
                                    
                                    # Map official to timm param names
                                    if param_name == 'dwconv.weight': param_name = 'conv_dw.weight'
                                    elif param_name == 'dwconv.bias': param_name = 'conv_dw.bias'
                                    elif param_name == 'pwconv1.weight': param_name = 'mlp.fc1.weight'
                                    elif param_name == 'pwconv1.bias': param_name = 'mlp.fc1.bias'
                                    elif param_name == 'pwconv2.weight': param_name = 'mlp.fc2.weight'
                                    elif param_name == 'pwconv2.bias': param_name = 'mlp.fc2.bias'
                                    elif param_name == 'grn.gamma': 
                                        param_name = 'mlp.grn.weight'
                                        v = v.reshape(-1)
                                    elif param_name == 'grn.beta': 
                                        param_name = 'mlp.grn.bias'
                                        v = v.reshape(-1)
                                    
                                    new_k = f"stages.{stage_idx}.blocks.{block_idx}.{param_name}"
                            
                            # Global Norm
                            elif k == 'norm.weight': new_k = 'head.norm.weight'
                            elif k == 'norm.bias': new_k = 'head.norm.bias'
                            
                            mapped_dict[new_k] = v
                        state_dict = mapped_dict

                    # Expand first layer weights if needed (e.g. 3 -> 4 channels)
                    model_dict = self.backbone.state_dict()
                    first_layer_key = 'stem.0.weight' if 'stem.0.weight' in model_dict else None
                    if first_layer_key and first_layer_key in state_dict:
                        ckpt_weight = state_dict[first_layer_key]
                        model_weight = model_dict[first_layer_key]
                        if ckpt_weight.shape != model_weight.shape:
                            print(f"Expanding weights for {first_layer_key}: {ckpt_weight.shape} -> {model_weight.shape}")
                            new_weight = model_weight.clone()
                            # Copy RGB channels
                            channels_to_copy = min(ckpt_weight.shape[1], model_weight.shape[1])
                            new_weight[:, :channels_to_copy] = ckpt_weight[:, :channels_to_copy]
                            # If expanding, initialize extra channels with average of originals
                            if model_weight.shape[1] > ckpt_weight.shape[1]:
                                avg_weight = torch.mean(ckpt_weight, dim=1, keepdim=True)
                                for i in range(ckpt_weight.shape[1], model_weight.shape[1]):
                                    new_weight[:, i:i+1] = avg_weight
                            state_dict[first_layer_key] = new_weight

                    # Load weights
                    msg = self.backbone.load_state_dict(state_dict, strict=False)
                    print(f"Loaded weights with result: {msg}")
                    
                    # For timm, we return early as it's already handled
                    num_features = self.backbone.num_features
                    self._setup_classifier(num_features, num_labels, freeze_backbone, model_name)
                    return
                
                num_features = self.backbone.num_features
            except ImportError:
                raise ImportError("Please install 'timm' library to use this model: pip install timm")
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise ValueError(f"Unsupported model: {model_name}. Error: {str(e)}")
        
        # If we reach here and have a pretrained_path, it's a torchvision model
        if pretrained_path:
            print(f"Loading custom pretrained weights from: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
            else:
                state_dict = checkpoint
            
            # Remove keys that don't match (like classifier head)
            # This is a bit safer for transfer learning
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
        
        self._setup_classifier(num_features, num_labels, freeze_backbone, model_name)

    def _setup_classifier(self, num_features, num_labels, freeze_backbone, model_name):
        """Helper to setup classifier head and freeze backbone."""
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"Backbone frozen - only training classifier head")
        
        # Custom classifier head for multi-label classification
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )
        
        print(f"Model created: {model_name}")
        print(f"Number of labels: {num_labels} (multi-label classification)")
        print(f"Feature dimension: {num_features}")
    
    def forward(self, x):
        """Forward pass through the network."""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def unfreeze_backbone(self, num_layers: Optional[int] = None):
        """
        Unfreeze backbone layers for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from the end. If None, unfreezes all.
        """
        if num_layers is None:
            # Unfreeze all layers
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("All backbone layers unfrozen")
        else:
            # Unfreeze last n layers
            layers = list(self.backbone.children())
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"Last {num_layers} backbone layers unfrozen")
    
    def get_num_trainable_params(self):
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self):
        """Get the total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_model(
    num_labels: int = 3,
    model_name: str = 'resnet50',
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: str = 'cpu',
    in_channels: int = 4,
    pretrained_path: str = r'checkpoints/convnextv2_tiny_22k_384_ema.pt',
    use_contrastive: bool = False
):
    """
    Create and initialize a QR code multi-label classifier model.
    
    Args:
        num_labels: Number of output labels
        model_name: Name of the backbone model
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone weights
        device: Device to place the model on
        in_channels: Number of input channels
        pretrained_path: Optional path to local pretrained weights file
        use_contrastive: Whether to use contrastive learning model
        
    Returns:
        QRCodeClassifier or CrossAttentionQRCodeClassifier model
    """
    if use_contrastive:
        model = create_contrastive_model(
            num_labels=num_labels,
            model_name=model_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            device=device,
            in_channels=4,
            pretrained_path=pretrained_path
        )
    else:
        model = QRCodeClassifier(
            num_labels=num_labels,
            model_name=model_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            in_channels=3,
            pretrained_path=pretrained_path
        )
        model = model.to(device)
    
    print(f"Total parameters: {model.get_num_total_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")
    
    return model


def load_model_for_inference(
    checkpoint_path: str, 
    num_labels: int = 3, 
    model_name: str = 'resnet50', 
    device: str = 'cpu',
    in_channels: int = 4,
    use_contrastive: bool = False
):
    """
    Load a trained model for inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_labels: Number of labels
        model_name: Name of the model architecture
        device: Device to load model on
        in_channels: Number of input channels
        use_contrastive: Whether the model uses contrastive learning
        
    Returns:
        Loaded model in evaluation mode
    """
    if use_contrastive:
        # 对比学习模型
        model = create_contrastive_model(
            num_labels=num_labels,
            model_name=model_name,
            pretrained=False,
            freeze_backbone=False,
            device=device,
            in_channels=in_channels
        )
    else:
        # 标准模型
        model = create_model(
            num_labels=num_labels,
            model_name=model_name,
            pretrained=False,
            freeze_backbone=False,
            device=device,
            in_channels=in_channels
        )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'mean_accuracy' in checkpoint:
        print(f"Checkpoint mean accuracy: {checkpoint['mean_accuracy']:.4f}")
    
    return model


class CrossAttentionQRCodeClassifier(nn.Module):
    """
    交叉注意力QR码分类器
    
    使用交叉注意力机制让RGB通道和参考图通道相互关注，
    学习复杂的对比关系，提升分类性能。
    """
    def __init__(
        self,
        rgb_backbone,
        ref_backbone,
        num_labels: int = 3,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            rgb_backbone: 用于RGB通道的预训练backbone模型（3通道）
            ref_backbone: 用于参考图通道的预训练backbone模型（3通道）
            num_labels: 输出标签数量
            d_model: Transformer的隐藏维度
            nhead: 注意力头数
            num_layers: Transformer编码器层数
            dropout: Dropout概率
        """
        super().__init__()
        
        self.rgb_backbone = rgb_backbone
        self.ref_backbone = ref_backbone
        self.num_labels = num_labels
        self.d_model = d_model
        self.nhead = nhead
        
        # 获取特征维度
        self.feat_dim = rgb_backbone.num_features if hasattr(rgb_backbone, 'num_features') else 2048
        
        # 投影层：将backbone特征投影到Transformer维度
        self.rgb_proj = nn.Linear(self.feat_dim, d_model)
        self.ref_proj = nn.Linear(self.feat_dim, d_model)
        
        # 位置编码
        self.pos_encoding = self._create_positional_encoding(d_model, max_len=10)
        
        # 交叉注意力编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.cross_attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model // 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels)
        )
        
        print(f"CrossAttentionQRCodeClassifier created:")
        print(f"  - Feature dimension: {self.feat_dim}")
        print(f"  - Transformer dimension: {d_model}")
        print(f"  - Attention heads: {nhead}")
        print(f"  - Transformer layers: {num_layers}")
        print(f"  - Number of labels: {num_labels}")
    
    def _create_positional_encoding(self, d_model: int, max_len: int = 5000):
        """创建位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (batch_size, 4, H, W) 输入张量
               x[:, :3, :, :] = RGB通道
               x[:, 3:4, :, :] = 参考图通道
        
        Returns:
            output: (batch_size, num_labels) 分类输出
            rgb_attended: (batch_size, d_model) RGB的注意力特征
            ref_attended: (batch_size, d_model) 参考图的注意力特征
        """
        batch_size = x.size(0)
        
        # 分离RGB和参考图通道
        rgb = x[:, :3, :, :]
        ref = x[:, 3:4, :, :].repeat(1, 3, 1, 1)
        
        # 提取特征
        rgb_feat = self.rgb_backbone(rgb)   # (B, feat_dim)
        ref_feat = self.ref_backbone(ref)   # (B, feat_dim)
        
        # 投影到Transformer维度
        rgb_proj = self.rgb_proj(rgb_feat)  # (B, d_model)
        ref_proj = self.ref_proj(ref_feat)  # (B, d_model)
        
        # 扩展为序列格式 (B, 1, d_model)
        rgb_proj = rgb_proj.unsqueeze(1)
        ref_proj = ref_proj.unsqueeze(1)
        
        # 添加位置编码
        pos_enc = self.pos_encoding[:, :2, :].to(rgb_proj.device)
        rgb_proj = rgb_proj + pos_enc[:, 0:1, :]
        ref_proj = ref_proj + pos_enc[:, 1:2, :]
        
        # 堆叠为序列 (B, 2, d_model)
        seq = torch.cat([rgb_proj, ref_proj], dim=1)
        
        # 交叉注意力
        attended = self.cross_attention(seq)  # (B, 2, d_model)
        
        # 提取RGB和参考图的注意力特征
        rgb_attended = attended[:, 0, :]  # (B, d_model)
        ref_attended = attended[:, 1, :]  # (B, d_model)
        
        # 融合
        fused = torch.cat([rgb_attended, ref_attended], dim=1)  # (B, 2*d_model)
        fused = self.fusion(fused)  # (B, d_model//2)
        
        # 分类
        output = self.classifier(fused)  # (B, num_labels)
        
        return output, rgb_attended, ref_attended
    
    def get_attention_weights(self, x):
        """
        获取注意力权重（用于可视化）
        
        Args:
            x: (batch_size, 4, H, W) 输入张量
        
        Returns:
            attention_weights: 注意力权重
        """
        batch_size = x.size(0)
        
        # 分离通道
        rgb = x[:, :3, :, :]
        ref = x[:, 3:4, :, :].repeat(1, 3, 1, 1)
        
        # 提取特征
        rgb_feat = self.backbone(rgb)
        ref_feat = self.backbone(ref)
        
        # 投影
        rgb_proj = self.rgb_proj(rgb_feat)
        ref_proj = self.ref_proj(ref_feat)
        
        # 堆叠为序列
        seq = torch.stack([rgb_proj, ref_proj], dim=1)
        
        # 交叉注意力（不使用dropout以获取注意力权重）
        return self.cross_attention(seq)
    
    def get_num_trainable_params(self):
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self):
        """Get the total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_contrastive_model(
    num_labels: int = 4,
    model_name: str = 'resnet50',
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: str = 'cpu',
    in_channels: int = 4,
    pretrained_path: str = None,
    d_model: int = 512,
    nhead: int = 8,
    num_layers: int = 2,
    dropout: float = 0.1
) -> CrossAttentionQRCodeClassifier:
    """
    创建对比学习模型
    
    Args:
        num_labels: 输出标签数量
        model_name: backbone模型名称
        pretrained: 是否使用预训练权重
        freeze_backbone: 是否冻结backbone
        device: 设备
        in_channels: 输入通道数（总输入通道数，实际每个backbone使用3通道）
        pretrained_path: 预训练权重路径
        d_model: Transformer隐藏维度
        nhead: 注意力头数
        num_layers: Transformer层数
        dropout: Dropout概率
    
    Returns:
        CrossAttentionQRCodeClassifier模型
    """
    # 创建RGB backbone（3通道）
    rgb_backbone = _create_single_backbone(
        model_name=model_name,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        in_channels=3,
        freeze_backbone=freeze_backbone
    )
    
    # 创建参考图backbone（3通道）
    ref_backbone = _create_single_backbone(
        model_name=model_name,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        in_channels=3,
        freeze_backbone=freeze_backbone
    )
    
    # 创建对比学习模型
    model = CrossAttentionQRCodeClassifier(
        rgb_backbone=rgb_backbone,
        ref_backbone=ref_backbone,
        num_labels=num_labels,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    )
    
    model = model.to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def _create_single_backbone(
    model_name: str,
    pretrained: bool,
    pretrained_path: str,
    in_channels: int,
    freeze_backbone: bool
):
    """
    创建单个backbone
    
    Args:
        model_name: 模型名称
        pretrained: 是否使用预训练权重
        pretrained_path: 预训练权重路径
        in_channels: 输入通道数
        freeze_backbone: 是否冻结backbone
    
    Returns:
        backbone模型
    """
    if model_name == 'resnet50':
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        
        if in_channels != 3:
            _patch_first_layer(backbone.conv1, in_channels)
            
    elif model_name == 'resnet18':
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        
        if in_channels != 3:
            _patch_first_layer(backbone.conv1, in_channels)
            
    elif model_name == 'efficientnet_b0':
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        num_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        
        if in_channels != 3:
            _patch_first_layer(backbone.features[0][0], in_channels)
            
    else:
        try:
            import timm
            actual_pretrained = pretrained and (pretrained_path is None)
            backbone = timm.create_model(
                model_name,
                pretrained=actual_pretrained,
                num_classes=0,
                in_chans=in_channels
            )
            
            if pretrained_path:
                print(f"Loading custom pretrained weights from: {pretrained_path}")
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
                
                if 'convnextv2' in model_name:
                    mapped_dict = {}
                    for k, v in state_dict.items():
                        new_k = k
                        if k.startswith('downsample_layers.0.0'):
                            new_k = k.replace('downsample_layers.0.0', 'stem.0')
                        elif k.startswith('downsample_layers.0.1'):
                            new_k = k.replace('downsample_layers.0.1', 'stem.1')
                        elif k.startswith('downsample_layers.'):
                            parts = k.split('.')
                            new_k = f"stages.{parts[1]}.downsample.{parts[2]}.{'.'.join(parts[3:])}"
                        elif k.startswith('stages.'):
                            parts = k.split('.')
                            if len(parts) >= 4:
                                stage_idx = parts[1]
                                block_idx = parts[2]
                                param_name = '.'.join(parts[3:])
                                if param_name == 'dwconv.weight': param_name = 'conv_dw.weight'
                                elif param_name == 'dwconv.bias': param_name = 'conv_dw.bias'
                                elif param_name == 'pwconv1.weight': param_name = 'mlp.fc1.weight'
                                elif param_name == 'pwconv1.bias': param_name = 'mlp.fc1.bias'
                                elif param_name == 'pwconv2.weight': param_name = 'mlp.fc2.weight'
                                elif param_name == 'pwconv2.bias': param_name = 'mlp.fc2.bias'
                                elif param_name == 'grn.gamma': 
                                    param_name = 'mlp.grn.weight'
                                    v = v.reshape(-1)
                                elif param_name == 'grn.beta': 
                                    param_name = 'mlp.grn.bias'
                                    v = v.reshape(-1)
                                new_k = f"stages.{stage_idx}.blocks.{block_idx}.{param_name}"
                        elif k == 'norm.weight': new_k = 'head.norm.weight'
                        elif k == 'norm.bias': new_k = 'head.norm.bias'
                        mapped_dict[new_k] = v
                    state_dict = mapped_dict
                
                model_dict = backbone.state_dict()
                first_layer_key = 'stem.0.weight' if 'stem.0.weight' in model_dict else None
                if first_layer_key and first_layer_key in state_dict:
                    ckpt_weight = state_dict[first_layer_key]
                    model_weight = model_dict[first_layer_key]
                    if ckpt_weight.shape != model_weight.shape:
                        print(f"Expanding weights for {first_layer_key}: {ckpt_weight.shape} -> {model_weight.shape}")
                        new_weight = model_weight.clone()
                        channels_to_copy = min(ckpt_weight.shape[1], model_weight.shape[1])
                        new_weight[:, :channels_to_copy] = ckpt_weight[:, :channels_to_copy]
                        if model_weight.shape[1] > ckpt_weight.shape[1]:
                            avg_weight = torch.mean(ckpt_weight, dim=1, keepdim=True)
                            for i in range(ckpt_weight.shape[1], model_weight.shape[1]):
                                new_weight[:, i:i+1] = avg_weight
                        state_dict[first_layer_key] = new_weight
                
                msg = backbone.load_state_dict(state_dict, strict=False)
                print(f"Loaded weights with result: {msg}")
                
            num_features = backbone.num_features
        except ImportError:
            raise ImportError("Please install 'timm' library to use this model: pip install timm")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"Unsupported model: {model_name}. Error: {str(e)}")
    
    # 冻结backbone
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        print(f"Backbone frozen - only training classifier head")
    
    return backbone
