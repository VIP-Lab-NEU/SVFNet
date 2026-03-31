import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from timm import create_model
from timm.layers import trunc_normal_


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor: aggregates multi-scale feature maps into a single global feature vector"""
    def __init__(self, feature_dims, global_dim=512, use_attention=True):
        """
        Args:
            feature_dims: List of channel dimensions for multi-scale features [256, 512, 1024]
            global_dim: Global feature dimension
            use_attention: Whether to use attention mechanism for multi-scale fusion
        """
        super().__init__()
        self.feature_dims = feature_dims
        self.global_dim = global_dim
        self.use_attention = use_attention
        
        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1) for _ in feature_dims
        ])
        
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(sum(feature_dims), sum(feature_dims) // 4),
                nn.ReLU(inplace=True),
                nn.Linear(sum(feature_dims) // 4, len(feature_dims)),
                nn.Softmax(dim=1)
            )
        
        self.fusion = nn.Sequential(
            nn.Linear(sum(feature_dims), global_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(global_dim * 2, global_dim)
        )
    
    def forward(self, features):
        """
        Args:
            features: List of multi-scale features [(B, C1, H1, W1), (B, C2, H2, W2), ...]
        Returns:
            global_feat: Global feature vector (B, global_dim)
        """
        target_device = features[0].device if features else None
        
        pooled_features = []
        for i, feat in enumerate(features):
            feat = feat.to(device=target_device, dtype=torch.float32)
            pooled = self.adaptive_pools[i](feat)
            pooled = pooled.flatten(1)
            pooled_features.append(pooled)
        
        concat_feat = torch.cat(pooled_features, dim=1)
        
        if self.use_attention:
            attention_weights = self.attention(concat_feat)
            weighted_features = []
            for i, feat in enumerate(pooled_features):
                weight = attention_weights[:, i:i+1]
                weighted_feat = feat * weight
                weighted_features.append(weighted_feat)
            
            concat_feat = torch.cat(weighted_features, dim=1)
        
        global_feat = self.fusion(concat_feat)
        return global_feat


class ContrastiveProjectionHead(nn.Module):
    """Contrastive learning projection head: maps global features to contrastive learning space"""
    def __init__(self, input_dim=512, projection_dim=256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
    
    def forward(self, x):
        return F.normalize(self.projection(x), dim=1)


class Stage4ContrastiveProjectionHead(nn.Module):
    """Stage 4 feature map contrastive projection head with GAP followed by projection to contrastive space"""
    def __init__(self, feature_dim, projection_dim=256):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
    
    def forward(self, feature_map):
        """
        Args:
            feature_map: Stage 4 feature map (B, C, H, W)
        Returns:
            normalized_embedding: Normalized contrastive embedding (B, projection_dim)
        """
        if torch.isnan(feature_map).any() or torch.isinf(feature_map).any():
            print("Warning: Stage4 feature map contains NaN or Inf values")
            feature_map = torch.nan_to_num(feature_map, nan=0.0, posinf=10.0, neginf=-10.0)

        global_feat = self.global_avg_pool(feature_map)
        global_feat = global_feat.view(global_feat.size(0), -1)

        if torch.isnan(global_feat).any() or torch.isinf(global_feat).any():
            print("Warning: Stage4 global feature contains NaN or Inf values")
            global_feat = torch.nan_to_num(global_feat, nan=0.0, posinf=10.0, neginf=-10.0)

        embedding = self.projection(global_feat)

        if torch.isnan(embedding).any() or torch.isinf(embedding).any():
            print("Warning: Stage4 projection contains NaN or Inf values")
            embedding = torch.nan_to_num(embedding, nan=0.0, posinf=10.0, neginf=-10.0)

        return F.normalize(embedding, dim=1)


class GlobalQuerySpaceAttention(nn.Module):
    """Global Query Space Attention (GQSA) for cross-modal feature enhancement"""
    def __init__(self, feature_dim=512, num_heads=8, dropout=0.1, support_feature_maps=True):
        """
        Args:
            feature_dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            support_feature_maps: Whether to support feature map input (B,C,H,W)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.support_feature_maps = support_feature_maps
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        if support_feature_maps:
            self.query_proj = nn.Conv2d(feature_dim, feature_dim, 1)
            self.key_proj = nn.Conv2d(feature_dim, feature_dim, 1)
            self.value_proj = nn.Conv2d(feature_dim, feature_dim, 1)
            self.out_proj = nn.Conv2d(feature_dim, feature_dim, 1)
            
            self.layer_norm = nn.GroupNorm(num_groups=min(32, feature_dim//4), num_channels=feature_dim)
        else:
            self.query_proj = nn.Linear(feature_dim, feature_dim)
            self.key_proj = nn.Linear(feature_dim, feature_dim)
            self.value_proj = nn.Linear(feature_dim, feature_dim)
            self.out_proj = nn.Linear(feature_dim, feature_dim)
            
            self.layer_norm = nn.LayerNorm(feature_dim)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.query_proj, self.key_proj, self.value_proj, self.out_proj]:
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, query_features, key_value_features):
        """
        Args:
            query_features: Query features (B, feature_dim) or (B, C, H, W)
            key_value_features: Key-value features (B, feature_dim) or (B, C, H, W)
        Returns:
            enhanced_features: Enhanced features in same format as input
        """
        is_feature_map = query_features.dim() == 4
        
        if is_feature_map and self.support_feature_maps:
            return self._forward_feature_maps(query_features, key_value_features)
        else:
            return self._forward_global_features(query_features, key_value_features)
    
    def _forward_feature_maps(self, query_features, key_value_features):
        B, C, H, W = query_features.shape
        
        Q = self.query_proj(query_features)
        K = self.key_proj(key_value_features)
        V = self.value_proj(key_value_features)
        head_dim = C // self.num_heads
        
        Q_pool = F.adaptive_avg_pool2d(Q, 1).view(B, self.num_heads, head_dim)
        K_pool = F.adaptive_avg_pool2d(K, 1).view(B, self.num_heads, head_dim)
        
        V = V.view(B, self.num_heads, head_dim, H * W)
        
        attention_scores = torch.einsum('bhd,bnd->bhn', Q_pool, K_pool) / (head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        context = torch.einsum('bhk,bkdn->bhdn', attention_weights, V)
        
        context = context.reshape(B, C, H, W)
        
        output = self.out_proj(context)
        output = self.dropout(output)
        output = self.layer_norm(output + query_features)
        
        return output
    
    def _forward_global_features(self, query_features, key_value_features):
        batch_size = query_features.size(0)
        
        Q = self.query_proj(query_features)
        K = self.key_proj(key_value_features)
        V = self.value_proj(key_value_features)
        
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, self.feature_dim)
        context = self.out_proj(context)
        enhanced_features = self.layer_norm(query_features + context)
        
        return enhanced_features





class CrossModalEnhancer(nn.Module):
    """Cross-modal enhancer using GQSA for feature enhancement with multi-stage support"""
    def __init__(self, feature_dim=512, num_heads=8, dropout=0.1, 
                 multi_stage_config=None, feature_dims=None):
        """
            Args:
                feature_dim: Global feature dimension (for the last stage)
                num_heads: Number of attention heads
                dropout: Dropout rate
                multi_stage_config: Multi-stage configuration dictionary
                feature_dims: List of feature dimensions for each stage
            """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.multi_stage_config = multi_stage_config
        
        if multi_stage_config and multi_stage_config.get('enable_multi_stage', False):
            self.enable_multi_stage = True
            self.stages = multi_stage_config.get('stages', [])
            self.stage_weights = nn.Parameter(torch.tensor(
                multi_stage_config.get('stage_weights', [1.0] * len(self.stages))
            ))
            self.fusion_method = multi_stage_config.get('fusion_method', 'attention')
            
            stage_to_dim_map = {stage: dim for stage, dim in zip(range(1, 5), feature_dims)}

            self.stage_rgb_enhancers = nn.ModuleDict()
            self.stage_sonar_enhancers = nn.ModuleDict()

            for i, stage in enumerate(self.stages):
                if stage not in stage_to_dim_map:
                    print(f"Warning: unsupported fusion stage {stage}, using default dim {feature_dim}")
                    stage_feature_dim = feature_dim
                else:
                    stage_feature_dim = stage_to_dim_map[stage]

                stage_num_heads = multi_stage_config.get('num_heads', num_heads)
                stage_dropout = multi_stage_config.get('dropout', dropout)
                self.stage_rgb_enhancers[f'stage_{stage}'] = GlobalQuerySpaceAttention(
                    feature_dim=stage_feature_dim,
                    num_heads=stage_num_heads,
                    dropout=stage_dropout,
                    support_feature_maps=True
                )
                self.stage_sonar_enhancers[f'stage_{stage}'] = GlobalQuerySpaceAttention(
                    feature_dim=stage_feature_dim,
                    num_heads=stage_num_heads,
                    dropout=stage_dropout,
                    support_feature_maps=True
                )
            
            self.fusion_adapters = nn.ModuleDict()
            if self.fusion_method == 'attention' and self.enable_multi_stage:
                target_stage_dim = stage_to_dim_map.get(self.stages[-1], feature_dim)
                for stage in self.stages:
                    if stage in stage_to_dim_map:
                        stage_dim = stage_to_dim_map[stage]
                        self.fusion_adapters[f'stage_{stage}'] = nn.Conv2d(stage_dim, target_stage_dim, 1)

        else:
            self.enable_multi_stage = False
        
        self.rgb_enhancer = GlobalQuerySpaceAttention(
            feature_dim=feature_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            support_feature_maps=False
        )
        
        self.sonar_enhancer = GlobalQuerySpaceAttention(
            feature_dim=feature_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            support_feature_maps=False
        )
    
    def enhance_rgb_with_sonar(self, rgb_features, sonar_features, return_intermediate=False):
        """
        Enhance RGB features with SONAR features
        Args:
            rgb_features: RGB global features (B, feature_dim)
            sonar_features: SONAR global features (B, feature_dim)
            return_intermediate: Whether to return intermediate features for visualization
        Returns:
            enhanced_rgb: Enhanced RGB features
        """
        enhanced_rgb = self.rgb_enhancer(rgb_features, sonar_features)
        if return_intermediate:
            return enhanced_rgb, rgb_features, enhanced_rgb
        return enhanced_rgb
    
    def enhance_sonar_with_rgb(self, sonar_features, rgb_features, return_intermediate=False):
        """
        Enhance SONAR features with RGB features
        Args:
            sonar_features: SONAR global features (B, feature_dim)
            rgb_features: RGB global features (B, feature_dim)
            return_intermediate: Whether to return intermediate features for visualization
        Returns:
            enhanced_sonar: Enhanced SONAR features
        """
        enhanced_sonar = self.sonar_enhancer(sonar_features, rgb_features)
        if return_intermediate:
            return enhanced_sonar, sonar_features, enhanced_sonar
        return enhanced_sonar
    
    def enhance_multi_stage_features(self, rgb_stage_features, sonar_stage_features, mode='rgb_gqsa'):
        """
        Multi-stage feature enhancement
        Args:
            rgb_stage_features: RGB stage features dict {stage: feature_map}
            sonar_stage_features: SONAR stage features dict {stage: feature_map}
            mode: Enhancement mode 'rgb_gqsa' or 'sonar_gqsa'
        Returns:
            enhanced_features: Enhanced stage features dict
        """
        if not self.enable_multi_stage:
            return rgb_stage_features if mode == 'rgb_gqsa' else sonar_stage_features
        
        enhanced_features = {}
        
        optional_layers = self.multi_stage_config.get('optional_fusion_layers', {}) if self.multi_stage_config else {}
        
        for i, stage in enumerate(self.stages):
            stage_key = f'stage_{stage}'
            
            if not optional_layers.get(f'stage_{stage}', True):
                print(f"Skip fusion for stage {stage} (disabled)")
                continue
            
            if (stage_key not in rgb_stage_features or 
                stage_key not in sonar_stage_features):
                print(f"Warning: features for stage {stage} not found, skipping fusion")
                if mode == 'rgb_gqsa' and stage_key in rgb_stage_features:
                    enhanced_features[stage_key] = rgb_stage_features[stage_key]
                elif mode == 'sonar_gqsa' and stage_key in sonar_stage_features:
                    enhanced_features[stage_key] = sonar_stage_features[stage_key]
                continue
            
            rgb_feat = rgb_stage_features[stage_key]
            sonar_feat = sonar_stage_features[stage_key]
            
            if mode == 'rgb_gqsa':
                if f'stage_{stage}' in self.stage_rgb_enhancers:
                    enhancer = self.stage_rgb_enhancers[f'stage_{stage}']
                    enhanced_feat = enhancer(rgb_feat, sonar_feat)
                    
                    if i < len(self.stage_weights):
                        weight = torch.sigmoid(self.stage_weights[i])
                        enhanced_features[stage_key] = weight * enhanced_feat + (1 - weight) * rgb_feat
                    else:
                        enhanced_features[stage_key] = enhanced_feat
                else:
                    enhanced_features[stage_key] = rgb_feat
                    
            elif mode == 'sonar_gqsa':
                if f'stage_{stage}' in self.stage_sonar_enhancers:
                    enhancer = self.stage_sonar_enhancers[f'stage_{stage}']
                    enhanced_feat = enhancer(sonar_feat, rgb_feat)
                    
                    if i < len(self.stage_weights):
                        weight = torch.sigmoid(self.stage_weights[i])
                        enhanced_features[stage_key] = weight * enhanced_feat + (1 - weight) * sonar_feat
                    else:
                        enhanced_features[stage_key] = enhanced_feat
                else:
                    enhanced_features[stage_key] = sonar_feat
        
        return enhanced_features
    
    def fuse_multi_stage_features(self, stage_features):
        """Fuse multi-stage features according to configured fusion method"""
        if not self.enable_multi_stage or not stage_features:
            return None
        
        if len(stage_features) == 1:
            return list(stage_features.values())[0]
            
        target_stage_num = None
        for stage in reversed(self.stages):
            if f'stage_{stage}' in stage_features:
                target_stage_num = stage
                break
        
        if target_stage_num is None:
            print("Warning: no valid target stage found")
            return None
        
        if self.fusion_method == 'attention':
            return self._attention_fusion(stage_features, target_stage_num)
        elif self.fusion_method == 'residual':
            return self._residual_fusion(stage_features, target_stage_num)
        elif self.fusion_method == 'weighted':
            return self._weighted_fusion(stage_features, target_stage_num)
        
        return stage_features.get(f'stage_{target_stage_num}')

    def _attention_fusion(self, stage_features, target_stage_num):
        if not stage_features:
            return None
        
        target_key = f'stage_{target_stage_num}'
        if target_key not in stage_features:
            print(f"Warning: target stage {target_stage_num} not found, using first available stage")
            target_key = list(stage_features.keys())[0]
        
        target_feat = stage_features[target_key]
        B, C, H, W = target_feat.shape
        
        if len(stage_features) == 1:
            return target_feat
        
        processed_features = []
        for stage_name, feat in stage_features.items():
            if stage_name in self.fusion_adapters:
                adapted_feat = self.fusion_adapters[stage_name](feat)
            else:
                if feat.shape[1] != C:
                    adapted_feat = F.adaptive_avg_pool2d(feat, 1)
                    adapted_feat = adapted_feat.expand(-1, C, H, W)
                else:
                    adapted_feat = feat
            
            if adapted_feat.shape[2:] != (H, W):
                resized_feat = F.interpolate(adapted_feat, size=(H, W), mode='bilinear', align_corners=False)
            else:
                resized_feat = adapted_feat
            
            processed_features.append(resized_feat)
            
        num_features = len(processed_features)
        stacked_features = torch.stack(processed_features, dim=1)
        
        query = target_feat.unsqueeze(1)
        
        scores = (query * stacked_features).sum(dim=[2, 3, 4]) / (C * H * W)
        weights = F.softmax(scores, dim=1)
        
        weights = weights.unsqueeze(1)
        stacked_flat = stacked_features.view(B, num_features, -1)
        
        fused_flat = torch.bmm(weights, stacked_flat)
        fused_feat = fused_flat.view(B, C, H, W)

        return fused_feat
        
    def _residual_fusion(self, stage_features, target_stage_num):
        target_feat = stage_features[f'stage_{target_stage_num}']
        
        for stage_key, feat in stage_features.items():
            if stage_key != f'stage_{target_stage_num}':
                B, C, H, W = target_feat.shape
                resized_feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
                if resized_feat.shape[1] != C:
                    resized_feat = F.adaptive_avg_pool2d(resized_feat, 1)
                    resized_feat = resized_feat.expand(-1, C, H, W)
                target_feat = target_feat + resized_feat
        
        return target_feat
    
    def _weighted_fusion(self, stage_features, target_stage_num):
        target_feat = stage_features[f'stage_{target_stage_num}']
        weights = F.softmax(self.stage_weights, dim=0)
        
        target_idx = -1
        for i, stage in enumerate(self.stages):
            if f'stage_{stage}' == f'stage_{target_stage_num}':
                target_idx = i
                break
        
        if target_idx >= 0:
            fused_feat = weights[target_idx] * target_feat
            
            for i, stage in enumerate(self.stages):
                stage_key = f'stage_{stage}'
                if stage_key != f'stage_{target_stage_num}' and stage_key in stage_features:
                    feat = stage_features[stage_key]
                    B, C, H, W = target_feat.shape
                    resized_feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
                    if resized_feat.shape[1] != C:
                        resized_feat = F.adaptive_avg_pool2d(resized_feat, 1)
                        resized_feat = resized_feat.expand(-1, C, H, W)
                    fused_feat = fused_feat + weights[i] * resized_feat
            
            return fused_feat
        
        return target_feat


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, sonar_embeddings, rgb_embeddings):
        """
        Args:
            sonar_embeddings: SONAR embeddings (B, D)
            rgb_embeddings: RGB embeddings (B, D)
        Returns:
            contrastive_loss: Contrastive loss
        """
        if torch.isnan(sonar_embeddings).any() or torch.isinf(sonar_embeddings).any():
            print("Warning: SONAR embeddings contain NaN or Inf values")
            sonar_embeddings = torch.nan_to_num(sonar_embeddings, nan=0.0, posinf=10.0, neginf=-10.0)

        if torch.isnan(rgb_embeddings).any() or torch.isinf(rgb_embeddings).any():
            print("Warning: RGB embeddings contain NaN or Inf values")
            rgb_embeddings = torch.nan_to_num(rgb_embeddings, nan=0.0, posinf=10.0, neginf=-10.0)

        batch_size = sonar_embeddings.shape[0]

        sim_matrix = torch.matmul(sonar_embeddings, rgb_embeddings.T) / self.temperature

        if torch.isnan(sim_matrix).any() or torch.isinf(sim_matrix).any():
            print("Warning: similarity matrix contains NaN or Inf values")
            sim_matrix = torch.nan_to_num(sim_matrix, nan=0.0, posinf=10.0, neginf=-10.0)

        labels = torch.arange(batch_size, device=sonar_embeddings.device)

        loss_sonar_to_rgb = self.criterion(sim_matrix, labels)
        loss_rgb_to_sonar = self.criterion(sim_matrix.T, labels)

        total_loss = (loss_sonar_to_rgb + loss_rgb_to_sonar) / 2

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: contrastive loss is NaN or Inf, returning 0.0")
            return torch.tensor(0.0, device=sonar_embeddings.device)

        return total_loss


class FusionModule(nn.Module):
    """Feature fusion module: concatenates global features from both modalities"""
    def __init__(self, sonar_dim=512, rgb_dim=512, fusion_dim=1024):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        self.sonar_norm = nn.LayerNorm(sonar_dim)
        self.rgb_norm = nn.LayerNorm(rgb_dim)
        
        self.feature_align = nn.Sequential(
            nn.Linear(sonar_dim + rgb_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
    
    def forward(self, sonar_feat, rgb_feat):
        """
        Args:
            sonar_feat: SONAR global features (B, sonar_dim) or None
            rgb_feat: RGB global features (B, rgb_dim) or None
        Returns:
            fused_feat: Fused features (B, fusion_dim)
        """
        target_device = None
        if sonar_feat is not None:
            target_device = sonar_feat.device
        elif rgb_feat is not None:
            target_device = rgb_feat.device
        else:
            raise ValueError("Both sonar_feat and rgb_feat cannot be None")
        
        if sonar_feat is None and rgb_feat is not None:
            batch_size = rgb_feat.shape[0]
            sonar_feat = torch.zeros(batch_size, rgb_feat.shape[1], device=target_device, dtype=rgb_feat.dtype)
        elif rgb_feat is None and sonar_feat is not None:
            batch_size = sonar_feat.shape[0]
            rgb_feat = torch.zeros(batch_size, sonar_feat.shape[1], device=target_device, dtype=sonar_feat.dtype)
        
        sonar_feat = sonar_feat.to(device=target_device, dtype=torch.float32)
        rgb_feat = rgb_feat.to(device=target_device, dtype=torch.float32)
        
        sonar_feat = self.sonar_norm(sonar_feat)
        rgb_feat = self.rgb_norm(rgb_feat)
        
        concat_feat = torch.cat([sonar_feat, rgb_feat], dim=1)
        
        fused_feat = self.feature_align(concat_feat)
        return fused_feat


class MultiLabelClassificationHead(nn.Module):
    """Multi-label classification head"""
    def __init__(self, input_dim=1024, num_classes=10, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: Fused features (B, input_dim)
        Returns:
            logits: Classification logits (B, num_classes), without sigmoid
        """
        x = x.to(dtype=torch.float32)
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: input features contain NaN or Inf values")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        x = self.input_norm(x)
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: after LayerNorm contains NaN or Inf values")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        logits = self.classifier(x)
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"Warning: output logits contain NaN or Inf values")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return logits


class DualStreamMaxViTBackbone(nn.Module):
    """Dual-stream MaxViT backbone network"""
    def __init__(self, model_name='maxvit_tiny_tf_224', pretrained=True, 
                 feature_dims=[256, 512, 1024], global_dim=512, local_weights_path=None):
        super().__init__()
        
        self.feature_dims = feature_dims
        self.global_dim = global_dim
        self.local_weights_path = local_weights_path
        
        self.sonar_backbone = self._create_maxvit_backbone(model_name, pretrained, feature_dims)
        self.sonar_global_extractor = GlobalFeatureExtractor(feature_dims, global_dim)
        
        self.rgb_backbone = self._create_maxvit_backbone(model_name, pretrained, feature_dims)
        self.rgb_global_extractor = GlobalFeatureExtractor(feature_dims, global_dim)
        
        if local_weights_path and os.path.exists(local_weights_path):
            self._load_local_weights(local_weights_path)
    
    def _load_local_weights(self, weights_path):
        """Load local weights file to SONAR and RGB streams"""
        try:
            print(f"Loading local weights: {weights_path}")
            checkpoint = torch.load(weights_path, map_location='cpu')
            print(f"Weights file size: {len(checkpoint)} parameters")
            
            def create_key_mapping(checkpoint_keys, model_keys):
                mapping = {}
                checkpoint_key_set = set(checkpoint_keys)
                model_key_set = set(model_keys)
                
                for key in checkpoint_keys:
                    if key in model_key_set:
                        mapping[key] = key
                
                for key in checkpoint_keys:
                    if key.startswith('stages.'):
                        new_key = key.replace('stages.', 'stages_')
                        if new_key in model_key_set:
                            mapping[key] = new_key
                
                return mapping
            
            sonar_maxvit = self.sonar_backbone['maxvit']
            rgb_maxvit = self.rgb_backbone['maxvit']
            
            sonar_model_keys = list(sonar_maxvit.state_dict().keys())
            rgb_model_keys = list(rgb_maxvit.state_dict().keys())
            
            sonar_mapping = create_key_mapping(checkpoint.keys(), sonar_model_keys)
            rgb_mapping = create_key_mapping(checkpoint.keys(), rgb_model_keys)
            
            def create_mapped_checkpoint(checkpoint, mapping):
                mapped_checkpoint = {}
                for old_key, new_key in mapping.items():
                    mapped_checkpoint[new_key] = checkpoint[old_key]
                return mapped_checkpoint
            
            sonar_mapped_checkpoint = create_mapped_checkpoint(checkpoint, sonar_mapping)
            rgb_mapped_checkpoint = create_mapped_checkpoint(checkpoint, rgb_mapping)
            
            print(f"Key mapping statistics:")
            print(f"  SONAR stream: {len(sonar_mapping)} keys mapped successfully")
            print(f"  RGB stream: {len(rgb_mapping)} keys mapped successfully")
            
            sonar_result = sonar_maxvit.load_state_dict(sonar_mapped_checkpoint, strict=False)
            rgb_result = rgb_maxvit.load_state_dict(rgb_mapped_checkpoint, strict=False)
            
            print(f"SONAR stream loading result: missing_keys={len(sonar_result.missing_keys)}, unexpected_keys={len(sonar_result.unexpected_keys)}")
            print(f"RGB stream loading result: missing_keys={len(rgb_result.missing_keys)}, unexpected_keys={len(rgb_result.unexpected_keys)}")
            print("Local weights loaded successfully")
            
        except Exception as e:
            print(f"Failed to load local weights: {e}")
            print("Continue with randomly initialized weights")

    def _create_maxvit_backbone(self, model_name, pretrained, out_channels):
        use_pretrained = pretrained and (self.local_weights_path is None or not os.path.exists(self.local_weights_path))
        
        try:
            maxvit = create_model(
                model_name, 
                pretrained=use_pretrained,
                features_only=True,
                out_indices=(1, 2, 3, 4),
            )
            print(f"Successfully created MaxViT model: {model_name}, online pretrained: {use_pretrained}")
        except Exception as e:
            print(f"Failed to create pretrained MaxViT: {e}")
            print("Attempting to create MaxViT without pretrained weights...")
            try:
                maxvit = create_model(
                    model_name, 
                    pretrained=False,
                    features_only=True,
                    out_indices=(1, 2, 3, 4),
                )
                print(f"Successfully created MaxViT without pretrained: {model_name}")
            except Exception as e2:
                print(f"Failed to create MaxViT completely: {e2}")
                raise RuntimeError(f"Unable to create MaxViT model: {model_name}. Please check network or use pretrained=False")
        test_size = 224
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, test_size, test_size)
            features = maxvit(dummy_input)
            maxvit_feature_dims = [f.shape[1] for f in features]
        
        adapters = nn.ModuleList()
        if len(maxvit_feature_dims) >= 4:
            selected_dims = maxvit_feature_dims[0:4]
        else:
            selected_dims = maxvit_feature_dims[-4:] if len(maxvit_feature_dims) >= 4 else maxvit_feature_dims
            while len(selected_dims) < 4:
                selected_dims = [maxvit_feature_dims[0]] + selected_dims
        
        for i, (in_dim, out_dim) in enumerate(zip(selected_dims, out_channels)):
            adapter = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.SiLU(inplace=True)
            )
            adapters.append(adapter)
        
        return nn.ModuleDict({
            'maxvit': maxvit,
            'adapters': adapters
        })
    
    def _forward_single_stream(self, x, backbone):
        features = backbone['maxvit'](x)
        
        if len(features) >= 4:
            selected_features = features[0:4]
        else:
            selected_features = features[-4:] if len(features) >= 4 else features
            while len(selected_features) < 4:
                selected_features = [features[0]] + selected_features
        
        adapted_features = []
        for feat, adapter in zip(selected_features, backbone['adapters']):
            adapted_feat = adapter(feat)
            adapted_features.append(adapted_feat)
        
        return adapted_features
    
    def forward(self, sonar_input, rgb_input, return_stage_features=False):
        """
        Args:
            sonar_input: SONAR image (B, 3, H, W) or None
            rgb_input: RGB image (B, 3, H, W) or None
            return_stage_features: Whether to return stage features
        Returns:
            if return_stage_features=False:
                sonar_global, rgb_global
            if return_stage_features=True:
                (sonar_global, rgb_global, sonar_stage_features, rgb_stage_features)
        """
        if sonar_input is not None:
            if return_stage_features:
                sonar_features, sonar_stage_features = self._forward_single_stream_with_stages(sonar_input, self.sonar_backbone)
            else:
                sonar_features = self._forward_single_stream(sonar_input, self.sonar_backbone)
                sonar_stage_features = None
            sonar_global = self.sonar_global_extractor(sonar_features)
        else:
            sonar_global = None
            sonar_stage_features = None
        
        if rgb_input is not None:
            if return_stage_features:
                rgb_features, rgb_stage_features = self._forward_single_stream_with_stages(rgb_input, self.rgb_backbone)
            else:
                rgb_features = self._forward_single_stream(rgb_input, self.rgb_backbone)
                rgb_stage_features = None
            rgb_global = self.rgb_global_extractor(rgb_features)
        else:
            rgb_global = None
            rgb_stage_features = None
        
        if return_stage_features:
            return sonar_global, rgb_global, sonar_stage_features, rgb_stage_features
        else:
            return sonar_global, rgb_global 
    
    def _forward_single_stream_with_stages(self, x, backbone):
        all_features = backbone['maxvit'](x)
        
        stage_features = {}
        
        if len(all_features) >= 4:
            selected_features = all_features[0:4]
            stage_indices = [1, 2, 3, 4]
        else:
            selected_features = all_features[-4:] if len(all_features) >= 4 else all_features
            while len(selected_features) < 4:
                selected_features = [all_features[0]] + selected_features
            stage_indices = [1, 2, 3, 4]
        
        adapted_features = []
        for i, (feat, adapter) in enumerate(zip(selected_features, backbone['adapters'])):
            adapted_feat = adapter(feat)
            adapted_features.append(adapted_feat)
            
            if i < len(stage_indices):
                stage_features[f'stage_{stage_indices[i]}'] = adapted_feat
        
        return adapted_features, stage_features


class MultiModalClassificationNetwork(nn.Module):
    """Complete multi-modal multi-label classification network with dual-stream MaxViT, GQSA and contrastive learning"""
    def __init__(self, num_classes=10, model_name='maxvit_tiny_tf_224', 
                 pretrained=True, feature_dims=[256, 512, 1024], 
                 global_dim=512, fusion_dim=1024, projection_dim=256,
                 temperature=0.1, local_weights_path=None,
                 use_rgb_gqsa=True, gqsa_stages_config=None):
        super().__init__()
        
        self.num_classes = num_classes
        self.global_dim = global_dim
        self.temperature = temperature
        self.feature_dims = feature_dims
        
        self.use_rgb_gqsa = use_rgb_gqsa
        
        self.gqsa_stages_config = gqsa_stages_config
        self.enable_multi_stage_gqsa = (gqsa_stages_config and 
                                       gqsa_stages_config.get('enable_multi_stage', False))
        self.dual_backbone = DualStreamMaxViTBackbone(
            model_name=model_name,
            pretrained=pretrained,
            feature_dims=feature_dims,
            global_dim=global_dim,
            local_weights_path=local_weights_path
        )
        
        self.sonar_projection = ContrastiveProjectionHead(global_dim, projection_dim)
        self.rgb_projection = ContrastiveProjectionHead(global_dim, projection_dim)
        
        stage4_feature_dim = feature_dims[-1] if feature_dims else 1024
        self.sonar_stage4_projection = Stage4ContrastiveProjectionHead(stage4_feature_dim, projection_dim)
        self.rgb_stage4_projection = Stage4ContrastiveProjectionHead(stage4_feature_dim, projection_dim)
        
        self.contrastive_loss = InfoNCELoss(temperature=temperature)
        
        self.fusion = FusionModule(global_dim, global_dim, fusion_dim)
        
        self.cross_modal_enhancer = CrossModalEnhancer(
            feature_dim=global_dim,
            num_heads=8,
            dropout=0.1,
            multi_stage_config=gqsa_stages_config,
            feature_dims=feature_dims
        )
        
        self.classifier = MultiLabelClassificationHead(
            input_dim=fusion_dim,
            num_classes=num_classes
        )
        
        self.rgb_classifier = MultiLabelClassificationHead(
            input_dim=global_dim,
            num_classes=num_classes
        )
        
        self.sonar_classifier = MultiLabelClassificationHead(
            input_dim=global_dim,
            num_classes=num_classes
        )
        if gqsa_stages_config and gqsa_stages_config.get('enable_multi_stage', False) and gqsa_stages_config.get('stages'):
            # Multi-stage mode: use the feature dimension of the last stage in the configuration
            last_stage = gqsa_stages_config['stages'][-1]
            stage_dim_idx = last_stage - 1
            if 0 <= stage_dim_idx < len(feature_dims):
                enhanced_feature_dim = feature_dims[stage_dim_idx]
            else:
                enhanced_feature_dim = feature_dims[-1]
        else:
            enhanced_feature_dim = global_dim
            
        multi_stage_gqsa_input_dim = global_dim + enhanced_feature_dim
        
        self.rgb_gqsa_classifier = MultiLabelClassificationHead(
            input_dim=multi_stage_gqsa_input_dim,
            num_classes=num_classes
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        for classifier in [self.classifier, self.rgb_classifier, self.sonar_classifier, self.rgb_gqsa_classifier]:
            if hasattr(classifier, 'classifier'):
                for layer in reversed(classifier.classifier):
                    if isinstance(layer, nn.Linear):
                        nn.init.constant_(layer.bias, -2.0)
                        break

        for projection_head in [self.sonar_projection, self.rgb_projection, 
                               self.sonar_stage4_projection, self.rgb_stage4_projection]:
            if hasattr(projection_head, 'projection'):
                for layer in reversed(projection_head.projection):
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0.0)
                        break
    
    def forward(self, sonar_input, rgb_input, return_contrastive=False, return_separate=False, return_heatmap_features=False):
        """
        Args:
            sonar_input: SONAR image (B, 3, H, W)
            rgb_input: RGB image (B, 3, H, W)
            return_contrastive: Whether to return contrastive features
            return_separate: Whether to return separate RGB and SONAR predictions
            return_heatmap_features: Whether to return intermediate features for heatmap
        Returns:
            logits: Classification logits (B, num_classes)
        """
        if self.baseline_mode == 'sonar_only':
            if return_contrastive:
                sonar_global, rgb_global, sonar_stage_features, rgb_stage_features = self.dual_backbone(
                    sonar_input, rgb_input, return_stage_features=True
                )
            else:
                sonar_global, rgb_global = self.dual_backbone(sonar_input, rgb_input)
                sonar_stage_features = None
            
            fused_feat = self.fusion(sonar_global, rgb_global)
            
            logits = self.classifier(fused_feat)
            
            if return_contrastive:
                sonar_stage4_feat = sonar_stage_features.get('stage_4') if sonar_stage_features else None
                if sonar_stage4_feat is not None and sonar_global is not None:
                    sonar_embedding = self.sonar_stage4_projection(sonar_stage4_feat)
                elif sonar_global is not None:
                    sonar_embedding = self.sonar_projection(sonar_global)
                else:
                    sonar_embedding = None
                return logits, sonar_embedding, None
            return logits
            
        elif self.baseline_mode == 'rgb_only':
            if return_contrastive:
                sonar_global, rgb_global, sonar_stage_features, rgb_stage_features = self.dual_backbone(
                    sonar_input, rgb_input, return_stage_features=True
                )
            else:
                sonar_global, rgb_global = self.dual_backbone(sonar_input, rgb_input)
                rgb_stage_features = None
            
            fused_feat = self.fusion(sonar_global, rgb_global)
            
            logits = self.classifier(fused_feat)
            
            if return_contrastive:
                rgb_stage4_feat = rgb_stage_features.get('stage_4') if rgb_stage_features else None
                if rgb_stage4_feat is not None and rgb_global is not None:
                    rgb_embedding = self.rgb_stage4_projection(rgb_stage4_feat)
                elif rgb_global is not None:
                    rgb_embedding = self.rgb_projection(rgb_global)
                else:
                    rgb_embedding = None
                return logits, None, rgb_embedding
            return logits
        
        else:
            if self.enable_multi_stage_gqsa and (self.use_rgb_gqsa or self.use_sonar_gqsa or self.use_scam):
                sonar_global, rgb_global, sonar_stage_features, rgb_stage_features = self.dual_backbone(
                    sonar_input, rgb_input, return_stage_features=True
                )

                if self.use_scam:
                    enhanced_rgb_stages = self.cross_modal_enhancer.enhance_multi_stage_features(
                        rgb_stage_features, sonar_stage_features, mode='rgb_gqsa'
                    )
                    enhanced_sonar_stages = self.cross_modal_enhancer.enhance_multi_stage_features(
                        rgb_stage_features, sonar_stage_features, mode='sonar_gqsa'
                    )
                    enhanced_rgb_feature = self.cross_modal_enhancer.fuse_multi_stage_features(enhanced_rgb_stages)
                    enhanced_sonar_feature = self.cross_modal_enhancer.fuse_multi_stage_features(enhanced_sonar_stages)
                    
                    if enhanced_rgb_feature is not None and enhanced_sonar_feature is not None:
                        enhanced_rgb_global = F.adaptive_avg_pool2d(enhanced_rgb_feature, 1).flatten(1)
                        enhanced_sonar_global = F.adaptive_avg_pool2d(enhanced_sonar_feature, 1).flatten(1)
                        fused_feat = torch.cat([enhanced_rgb_global, enhanced_sonar_global], dim=1)
                        if hasattr(self, 'scam_classifier'):
                            logits = self.scam_classifier(fused_feat)
                        else:
                            logits = self.concat_classifier(fused_feat)
                    else:
                        fused_feat = self.fusion(sonar_global, rgb_global)
                        logits = self.classifier(fused_feat)

                elif self.use_rgb_gqsa:
                    enhanced_stage_features = self.cross_modal_enhancer.enhance_multi_stage_features(
                        rgb_stage_features, sonar_stage_features, mode='rgb_gqsa'
                    )
                    enhanced_feature = self.cross_modal_enhancer.fuse_multi_stage_features(enhanced_stage_features)
                    
                    if enhanced_feature is not None:
                        enhanced_global = F.adaptive_avg_pool2d(enhanced_feature, 1).flatten(1)
                    else:
                        enhanced_global = self.cross_modal_enhancer.enhance_sonar_with_rgb(sonar_global, rgb_global)
                    
                    concat_feat = torch.cat([rgb_global, enhanced_global], dim=1)
                    logits = self.multistage_rgb_gqsa_classifier(concat_feat)
                    
                elif self.use_sonar_gqsa:
                    enhanced_stage_features = self.cross_modal_enhancer.enhance_multi_stage_features(
                        rgb_stage_features, sonar_stage_features, mode='sonar_gqsa'
                    )
                    enhanced_feature = self.cross_modal_enhancer.fuse_multi_stage_features(enhanced_stage_features)
                    
                    if enhanced_feature is not None:
                        enhanced_global = F.adaptive_avg_pool2d(enhanced_feature, 1).flatten(1)
                    else:
                        enhanced_global = self.cross_modal_enhancer.enhance_sonar_with_rgb(sonar_global, rgb_global)
                    
                    concat_feat = torch.cat([sonar_global, enhanced_global], dim=1)
                    logits = self.multistage_sonar_gqsa_classifier(concat_feat)
                
                heatmap_info = None
                if return_heatmap_features and (self.use_rgb_gqsa or self.use_sonar_gqsa):
                    mode = 'rgb_gqsa' if self.use_rgb_gqsa else 'sonar_gqsa'
                    heatmap_info = {
                        'mode': mode,
                        'original_image': rgb_input if self.use_rgb_gqsa else sonar_input,
                        'sonar_image': sonar_input,
                        'rgb_image': rgb_input,
                        'original_features': rgb_global if self.use_rgb_gqsa else sonar_global,
                        'enhanced_features': enhanced_global
                    }
                    
                    if self.use_rgb_gqsa and rgb_stage_features:
                        for stage_name, features in rgb_stage_features.items():
                            heatmap_info[stage_name] = features
                    elif self.use_sonar_gqsa and sonar_stage_features:
                        for stage_name, features in sonar_stage_features.items():
                            heatmap_info[stage_name] = features
            else:
                if (return_heatmap_features and (self.use_rgb_gqsa or self.use_sonar_gqsa)) or return_contrastive:
                    sonar_global, rgb_global, sonar_stage_features, rgb_stage_features = self.dual_backbone(
                        sonar_input, rgb_input, return_stage_features=True
                    )
                else:
                    sonar_global, rgb_global = self.dual_backbone(sonar_input, rgb_input)
                    sonar_stage_features, rgb_stage_features = None, None
            
            heatmap_info = None
            
            if self.use_feature_concat:
                concat_feat = torch.cat([sonar_global, rgb_global], dim=1)
                logits = self.concat_classifier(concat_feat)
                
            elif self.use_rgb_gqsa:
                if return_heatmap_features:
                    enhanced_rgb, original_rgb, enhanced_rgb_dup = self.cross_modal_enhancer.enhance_rgb_with_sonar(
                        rgb_global, sonar_global, return_intermediate=True
                    )
                    heatmap_info = {
                        'mode': 'rgb_gqsa',
                        'original_image': rgb_input,
                        'sonar_image': sonar_input,
                        'original_features': original_rgb,
                        'enhanced_features': enhanced_rgb_dup
                    }
                    
                    if rgb_stage_features:
                        for stage_name, features in rgb_stage_features.items():
                            heatmap_info[stage_name] = features
                else:
                    enhanced_rgb = self.cross_modal_enhancer.enhance_rgb_with_sonar(rgb_global, sonar_global)
                concat_feat = torch.cat([rgb_global, enhanced_rgb], dim=1)
                logits = self.standard_rgb_gqsa_classifier(concat_feat)
                
            elif self.use_sonar_gqsa:
                if return_heatmap_features:
                    enhanced_sonar, original_sonar, enhanced_sonar_dup = self.cross_modal_enhancer.enhance_sonar_with_rgb(
                        sonar_global, rgb_global, return_intermediate=True
                    )
                    heatmap_info = {
                        'mode': 'sonar_gqsa',
                        'original_image': sonar_input,
                        'rgb_image': rgb_input,
                        'original_features': original_sonar,
                        'enhanced_features': enhanced_sonar_dup
                    }
                    
                    if sonar_stage_features:
                        for stage_name, features in sonar_stage_features.items():
                            heatmap_info[stage_name] = features
                else:
                    enhanced_sonar = self.cross_modal_enhancer.enhance_sonar_with_rgb(sonar_global, rgb_global)
                concat_feat = torch.cat([sonar_global, enhanced_sonar], dim=1)
                logits = self.standard_sonar_gqsa_classifier(concat_feat)
                
            else:
                fused_feat = self.fusion(sonar_global, rgb_global)
                logits = self.classifier(fused_feat)
            
            rgb_logits = None
            sonar_logits = None
            if return_separate:
                rgb_logits = self.rgb_classifier(rgb_global)
                sonar_logits = self.sonar_classifier(sonar_global)
            
            result = [logits]
            
            if return_contrastive:
                sonar_stage4_feat = sonar_stage_features.get('stage_4') if sonar_stage_features else None
                rgb_stage4_feat = rgb_stage_features.get('stage_4') if rgb_stage_features else None
                
                if sonar_stage4_feat is not None:
                    sonar_embedding = self.sonar_stage4_projection(sonar_stage4_feat)
                else:
                    sonar_embedding = self.sonar_projection(sonar_global)
                
                if rgb_stage4_feat is not None:
                    rgb_embedding = self.rgb_stage4_projection(rgb_stage4_feat)
                else:
                    rgb_embedding = self.rgb_projection(rgb_global)
                
                result.extend([sonar_embedding, rgb_embedding])
            
            if return_separate:
                result.extend([rgb_logits, sonar_logits])
            
            if return_heatmap_features and heatmap_info is not None:
                result.append(heatmap_info)
            
            if len(result) == 1:
                return result[0]
            else:
                return tuple(result)
    
    def _forward_single_stream(self, x, backbone):
        return self.dual_backbone._forward_single_stream(x, backbone)
    

    def _classify_single_modal(self, global_feat):
        if not hasattr(self, 'single_modal_classifier'):
            self.single_modal_classifier = nn.Sequential(
                nn.Linear(self.global_dim, self.global_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(self.global_dim // 2, self.num_classes)
            ).to(global_feat.device)
            
            for m in self.single_modal_classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, -2.0)
        
        return self.single_modal_classifier(global_feat)
    
    def compute_loss(self, sonar_input, rgb_input, targets, lambda_contrastive=0.0, lambda_classification=1.0,
                     lambda_sonar_modal=0.0, lambda_rgb_modal=0.0):
        """
        Compute total loss
        Args:
            sonar_input: SONAR image (B, 3, H, W)
            rgb_input: RGB image (B, 3, H, W)
            targets: Multi-label targets (B, num_classes)
            lambda_contrastive: Contrastive loss weight
            lambda_classification: Classification loss weight
            lambda_sonar_modal: SONAR modal loss weight
            lambda_rgb_modal: RGB modal loss weight
        Returns:
            total_loss, classification_loss, contrastive_loss, sonar_modal_loss, rgb_modal_loss
        """
        if self.baseline_mode == 'sonar_only':
            sonar_global, _ = self.dual_backbone(sonar_input, None)
            
            sonar_logits = self.sonar_classifier(sonar_global)
            
            classification_loss = F.binary_cross_entropy_with_logits(sonar_logits, targets.float())
            
            contrastive_loss = torch.tensor(0.0, device=classification_loss.device)
            
            sonar_modal_loss = classification_loss
            rgb_modal_loss = torch.tensor(0.0, device=classification_loss.device)
            
            total_loss = lambda_classification * classification_loss
            
        elif self.baseline_mode == 'rgb_only':
            _, rgb_global = self.dual_backbone(None, rgb_input)
            
            rgb_logits = self.rgb_classifier(rgb_global)
            
            classification_loss = F.binary_cross_entropy_with_logits(rgb_logits, targets.float())
            
            contrastive_loss = torch.tensor(0.0, device=classification_loss.device)
            
            sonar_modal_loss = torch.tensor(0.0, device=classification_loss.device)
            rgb_modal_loss = classification_loss
            
            total_loss = lambda_classification * classification_loss
            
        else:
            need_separate = lambda_sonar_modal > 0 or lambda_rgb_modal > 0
            
            if lambda_contrastive > 0 and need_separate:
                logits, sonar_embedding, rgb_embedding, rgb_logits, sonar_logits = self.forward(
                    sonar_input, rgb_input, return_contrastive=True, return_separate=True
                )
            elif lambda_contrastive > 0:
                logits, sonar_embedding, rgb_embedding = self.forward(
                    sonar_input, rgb_input, return_contrastive=True
                )
                rgb_logits = None
                sonar_logits = None
            elif need_separate:
                logits, rgb_logits, sonar_logits = self.forward(
                    sonar_input, rgb_input, return_separate=True
                )
                sonar_embedding = None
                rgb_embedding = None
            else:
                logits = self.forward(
                    sonar_input, rgb_input, return_contrastive=False
                )
                sonar_embedding = None
                rgb_embedding = None
                rgb_logits = None
                sonar_logits = None
            
            classification_loss = F.binary_cross_entropy_with_logits(logits, targets.float())
            
            if lambda_contrastive > 0 and sonar_embedding is not None and rgb_embedding is not None:
                contrastive_loss = self.contrastive_loss(sonar_embedding, rgb_embedding)
            else:
                contrastive_loss = torch.tensor(0.0, device=classification_loss.device)
            
            if lambda_sonar_modal > 0 and sonar_logits is not None:
                sonar_modal_loss = F.binary_cross_entropy_with_logits(sonar_logits, targets.float())
            else:
                sonar_modal_loss = torch.tensor(0.0, device=classification_loss.device)
            
            if lambda_rgb_modal > 0 and rgb_logits is not None:
                rgb_modal_loss = F.binary_cross_entropy_with_logits(rgb_logits, targets.float())
            else:
                rgb_modal_loss = torch.tensor(0.0, device=classification_loss.device)
            
            total_loss = (lambda_classification * classification_loss + 
                         lambda_contrastive * contrastive_loss +
                         lambda_sonar_modal * sonar_modal_loss +
                         lambda_rgb_modal * rgb_modal_loss)
        
        return total_loss, classification_loss, contrastive_loss, sonar_modal_loss, rgb_modal_loss


def create_multimodal_network(num_classes=10, model_size='tiny', pretrained=True, 
                             local_weights_path=None, gqsa_stages_config=None, **kwargs):
    """
    Factory function to create multi-modal classification network
    
    Args:
        num_classes: Number of classes
        model_size: Model size ('tiny', 'small', 'base', 'large', 'xlarge')
        pretrained: Whether to use pretrained weights
        local_weights_path: Local weights file path
        gqsa_stages_config: Multi-stage GQSA configuration dict
        **kwargs: Other parameters
    
    Returns:
        MultiModalClassificationNetwork instance
    """
    model_names = {
        'tiny': 'maxvit_tiny_tf_224',
        'small': 'maxvit_small_tf_224', 
        'base': 'maxvit_base_tf_224',
        'large': 'maxvit_large_tf_224',
        'xlarge': 'maxvit_xlarge_tf_224'
    }
    
    model_name = model_names.get(model_size, 'maxvit_tiny_tf_224')
    
    return MultiModalClassificationNetwork(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        local_weights_path=local_weights_path,
        gqsa_stages_config=gqsa_stages_config,
        **kwargs
    )