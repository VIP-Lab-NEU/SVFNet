#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-modal Multi-label Classification Training Script
Manages all parameters using YAML configuration file
"""

import os
import sys
import datetime
import argparse
from functools import partial
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.multimodal_network import create_multimodal_network
from utils.multimodal_dataset import MultiModalClassificationDataset, multimodal_collate_fn
from utils.utils import seed_everything, worker_init_fn
from utils.logging_utils import Tee
from utils.evaluation import validate_epoch_improved
from utils.lr_scheduler import GradualWarmupScheduler


class Config:
    """Configuration class, loads config from YAML file"""

    def __init__(self, config_path='config.yaml', model_size='tiny'):
        self.config_path = config_path
        self.model_size = model_size
        self.load_config()

    def load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        basic = config_data['basic']
        self.Cuda = basic['cuda']
        self.seed = basic['seed']
        self.distributed = basic['distributed']
        self.sync_bn = basic['sync_bn']
        self.fp16 = basic['fp16']

        dataset = config_data['dataset']
        self.dataset_dir_train = dataset['train_dir']
        self.dataset_dir_val = dataset['val_dir']
        self.num_classes = dataset['num_classes']
        self.class_names = dataset['class_names']
        self.input_shape = dataset['input_shape']

        network = config_data['network']
        self.model_size = network['model_size']
        self.pretrained = network['pretrained']
        self.local_weights_path = network['local_weights_path']
        self.feature_dims = network['feature_dims']
        self.global_dim = network['global_dim']
        self.fusion_dim = network['fusion_dim']
        self.projection_dim = network['projection_dim']
        self.temperature = network['temperature']
        self.use_attention_fusion = network['use_attention_fusion']
        
        self.use_rgb_gqsa = network.get('use_rgb_gqsa', True)
        
        self.gqsa_stages = network.get('gqsa_stages', {})

        training = config_data['training']
        self.Init_Epoch = training['init_epoch']
        self.Freeze_Epoch = training['freeze_epoch']
        self.UnFreeze_Epoch = training['unfreeze_epoch']
        self.Freeze_Train = training['freeze_train']
        self.Freeze_batch_size = training['freeze_batch_size']
        self.Unfreeze_batch_size = training['unfreeze_batch_size']
        self.Init_lr = training['init_lr']
        self.Min_lr_ratio = training['min_lr_ratio']
        self.optimizer_type = training['optimizer_type']
        self.momentum = training['momentum']
        self.weight_decay = training['weight_decay']
        self.lr_decay_type = training['lr_decay_type']
        self.lambda_contrastive = training['lambda_contrastive']
        self.lambda_classification = training.get('lambda_classification', 1.0)

        self.lambda_contrastive_decay_enabled = training.get('lambda_contrastive_decay_enabled', False)
        self.lambda_contrastive_decay_epochs = training.get('lambda_contrastive_decay_epochs', 15)
        self.lambda_contrastive_end_value = training.get('lambda_contrastive_end_value', 0.0001)

        self.backbone_lr = training.get('backbone_lr', self.Init_lr * 0.1)

        self.warmup_epochs = training.get('warmup_epochs', 0)
        self.warmup_lr_ratio = training.get('warmup_lr_ratio', 0.1)

        self.lambda_sonar_modal = training.get('lambda_sonar_modal', 0.0)
        self.lambda_rgb_modal = training.get('lambda_rgb_modal', 0.0)
        
        self.save_feature_heatmaps = training.get('save_feature_heatmaps', False)
        self.heatmap_threshold = training.get('heatmap_threshold', 0.05)
        self.heatmap_save_count = training.get('heatmap_save_count', 5)

        augmentation = config_data['augmentation']
        self.sonar_augmentation = augmentation['sonar']
        self.rgb_augmentation = augmentation['rgb']

        strategy = config_data['strategy']
        self.save_period = strategy['save_period']
        self.save_dir = strategy['save_dir']
        self.eval_period = strategy['eval_period']
        self.num_workers = strategy['num_workers']
        self.eval_threshold = strategy['eval_threshold']
        self.early_stopping = strategy['early_stopping']

        logging_config = config_data['logging']
        self.log_dir = logging_config['log_dir']
        self.log_level = logging_config['log_level']

        if self.model_size in config_data['model_presets']:
            preset = config_data['model_presets'][self.model_size]
            if 'network' in preset:
                for key, value in preset['network'].items():
                    setattr(self, key, value)
            if 'training' in preset:
                for key, value in preset['training'].items():
                    setattr(self, key, value)

        self.metrics_history = []

    def update_from_args(self, args):
        """Update config from command line arguments"""
        if hasattr(args, 'model_size') and args.model_size:
            self.model_size = args.model_size
        if hasattr(args, 'weights_path') and args.weights_path:
            self.local_weights_path = args.weights_path
        if hasattr(args, 'batch_size') and args.batch_size:
            self.Unfreeze_batch_size = args.batch_size
            self.Freeze_batch_size = args.batch_size * 2
        if hasattr(args, 'epochs') and args.epochs:
            self.UnFreeze_Epoch = args.epochs
        if hasattr(args, 'lr') and args.lr:
            self.Init_lr = args.lr

    @property
    def Min_lr(self):
        """Calculate minimum learning rate"""
        return self.Init_lr * self.Min_lr_ratio

    def get_model_config(self):
        """Get model configuration dictionary"""
        return {
            'num_classes': self.num_classes,
            'model_size': self.model_size,
            'pretrained': self.pretrained,
            'local_weights_path': self.local_weights_path,
            'feature_dims': self.feature_dims,
            'global_dim': self.global_dim,
            'fusion_dim': self.fusion_dim,
            'projection_dim': self.projection_dim,
            'temperature': self.temperature,
            'use_rgb_gqsa': self.use_rgb_gqsa,
            'gqsa_stages_config': getattr(self, 'gqsa_stages', None)
        }

    def print_config(self):
        """Print configuration information"""
        print("=" * 60)
        print("Multi-modal Multi-label Classification Config")
        print("=" * 60)
        print(f"Config file: {self.config_path}")
        print(f"Dataset: {self.dataset_dir_train}")
        print(f"Num classes: {self.num_classes}")
        print(f"Class names: {self.class_names}")
        print(f"Input shape: {self.input_shape}")
        print(f"Model size: {self.model_size}")
        print(f"Pretrained: {self.pretrained}")
        if self.local_weights_path:
            print(f"Local weights: {self.local_weights_path}")
        print(f"Batch size: {self.Freeze_batch_size} -> {self.Unfreeze_batch_size}")
        print(f"Learning rate: {self.Init_lr} -> {self.Min_lr}")
        print(f"Backbone LR: {self.backbone_lr}")
        print(f"Optimizer: {self.optimizer_type}")
        print(f"Warmup epochs: {self.warmup_epochs}")
        print(f"Lambda contrastive: {self.lambda_contrastive}")
        if self.lambda_contrastive_decay_enabled:
            print(f"  Decay enabled: True")
            print(f"  Decay epochs: {self.lambda_contrastive_decay_epochs}")
            print(f"  Decay end value: {self.lambda_contrastive_end_value}")
        print(f"Lambda classification: {self.lambda_classification}")
        print(f"Lambda sonar modal: {self.lambda_sonar_modal}")
        print(f"Lambda RGB modal: {self.lambda_rgb_modal}")
        print(f"Save feature heatmaps: {self.save_feature_heatmaps}")
        if self.save_feature_heatmaps:
            print(f"  Heatmap threshold: {self.heatmap_threshold}")
            print(f"  Save count: {self.heatmap_save_count}")
        print(f"Temperature: {self.temperature}")
        print(f"Training epochs: {self.Init_Epoch} -> {self.Freeze_Epoch} -> {self.UnFreeze_Epoch}")
        print(f"FP16: {self.fp16}")
        print(f"Early stopping patience: {self.early_stopping['patience']}")
        print(f"RGB-GQSA fusion: {self.use_rgb_gqsa}")
        
        if hasattr(self, 'gqsa_stages') and self.gqsa_stages:
            print(f"GQSA multi-stage fusion config:")
            print(f"  Enable multi-stage: {self.gqsa_stages.get('enable_multi_stage', False)}")
            if self.gqsa_stages.get('enable_multi_stage', False):
                print(f"  Fusion stages: {self.gqsa_stages.get('stages', [])}")
                print(f"  Fusion method: {self.gqsa_stages.get('fusion_method', 'attention')}")
                print(f"  Stage weights: {self.gqsa_stages.get('stage_weights', [])}")
                print(f"  Num heads: {self.gqsa_stages.get('num_heads', 8)}")
                print(f"  Dropout: {self.gqsa_stages.get('dropout', 0.1)}")
        else:
            print(f"GQSA multi-stage fusion: Not configured")
        print("=" * 60)


def train_epoch(model, dataloader, optimizer, scaler, config, epoch, log_fp=None):
    """Train one epoch"""
    model.train()

    current_lambda_contrastive = config.lambda_contrastive

    if config.lambda_contrastive_decay_enabled:
        start_lambda = config.lambda_contrastive
        end_lambda = config.lambda_contrastive_end_value
        switch_epoch = config.lambda_contrastive_decay_epochs

        if epoch <= switch_epoch:
            current_lambda_contrastive = start_lambda
        else:
            current_lambda_contrastive = end_lambda

    total_loss = 0
    total_cls_loss = 0
    total_cont_loss = 0
    total_sonar_loss = 0
    total_rgb_loss = 0

    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch}',
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                 ncols=120, file=sys.stdout)
    for batch_idx, (sonar_imgs, rgb_imgs, targets) in enumerate(pbar):
        if config.Cuda:
            if sonar_imgs is not None:
                sonar_imgs = sonar_imgs.cuda()
            if rgb_imgs is not None:
                rgb_imgs = rgb_imgs.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()

        if config.fp16:
            try:
                with torch.amp.autocast('cuda'):
                    total_loss_batch, cls_loss, cont_loss, sonar_loss, rgb_loss = model.compute_loss(
                        sonar_imgs, rgb_imgs, targets, current_lambda_contrastive, config.lambda_classification,
                        config.lambda_sonar_modal, config.lambda_rgb_modal
                    )
            except:
                with torch.cuda.amp.autocast():
                    total_loss_batch, cls_loss, cont_loss, sonar_loss, rgb_loss = model.compute_loss(
                        sonar_imgs, rgb_imgs, targets, current_lambda_contrastive, config.lambda_classification,
                        config.lambda_sonar_modal, config.lambda_rgb_modal
                    )
            scaler.scale(total_loss_batch).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss_batch, cls_loss, cont_loss, sonar_loss, rgb_loss = model.compute_loss(
                sonar_imgs, rgb_imgs, targets, current_lambda_contrastive, config.lambda_classification,
                config.lambda_sonar_modal, config.lambda_rgb_modal
            )
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        total_loss += total_loss_batch.item()
        total_cls_loss += cls_loss.item()
        total_cont_loss += cont_loss.item()
        total_sonar_loss += sonar_loss.item()
        total_rgb_loss += rgb_loss.item()
        
        current_avg_loss = total_loss / (batch_idx + 1)
        current_avg_cls_loss = total_cls_loss / (batch_idx + 1)
        current_avg_cont_loss = total_cont_loss / (batch_idx + 1)
        current_avg_sonar_loss = total_sonar_loss / (batch_idx + 1)
        current_avg_rgb_loss = total_rgb_loss / (batch_idx + 1)
        
        pbar.set_postfix({
            'T': f'{total_loss_batch.item():.3f}',
            'Co': f'{cont_loss.item():.3f}',
            'S': f'{sonar_loss.item():.3f}',
            'R': f'{rgb_loss.item():.3f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

    return (total_loss / len(dataloader), total_cls_loss / len(dataloader), total_cont_loss / len(dataloader),
            total_sonar_loss / len(dataloader), total_rgb_loss / len(dataloader))


def validate_epoch(model, dataloader, config, epoch):
    """Validate one epoch"""
    device = next(model.parameters()).device
    return validate_epoch_improved(model, dataloader, config, epoch, device)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Multi-modal multi-label classification training')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Config file path')
    parser.add_argument('--mode', type=str, default='normal',
                       choices=['normal', 'local_weights'],
                       help='Training mode')
    parser.add_argument('--model_size', type=str, default=None,
                       choices=['tiny', 'small', 'base'], help='Model size')
    parser.add_argument('--weights_path', type=str, default=None,
                       help='Local weights file path')
    parser.add_argument('--resume', type=str, default=None, help='Resume training model path')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Training epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')


    args = parser.parse_args()

    config = Config(args.config, args.model_size)
    config.update_from_args(args)

    os.makedirs(config.log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_suffix = '_local_weights' if args.mode == 'local_weights' else ''
    log_filename = f"multimodal_{config.model_size}{mode_suffix}_{timestamp}.log"
    batch_log_filename = f"multimodal_{config.model_size}{mode_suffix}_{timestamp}_batch_losses.log"
    log_file_path = os.path.join(config.log_dir, log_filename)

    try:
        log_fp = open(log_file_path, 'w', encoding='utf-8')
        sys.stdout = Tee(sys.stdout, log_fp, filter_progress=True)
        sys.stderr = Tee(sys.stderr, log_fp, filter_progress=True)
        print(f"Logging: {log_file_path}")     
        print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"Warning: Cannot create log file {log_file_path}: {e}")
        print("Continue console output...")
        log_fp = None

    if args.mode == 'local_weights':
        if not os.path.exists(config.local_weights_path):
            print(f"Error: Local weights file not found: {config.local_weights_path}")
            print("Please check the path")
            return
        print(f"Found local weights file: {config.local_weights_path}")
        print(f"File size: {os.path.getsize(config.local_weights_path) / 1024 / 1024:.1f} MB")

    config.print_config()

    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        config.Cuda = True
    else:
        config.Cuda = False

    seed_everything(config.seed)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_suffix = '_local_weights' if args.mode == 'local_weights' else ''
    save_dir = os.path.join(config.save_dir, f'multimodal_{config.model_size}{mode_suffix}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Model save dir: {save_dir}")

    print(f"Creating model: {config.model_size} ({args.mode} mode)")
    model_config = config.get_model_config()
    model = create_multimodal_network(**model_config)

    if config.Cuda:
        model = model.cuda()

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resume training: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint)
        try:
            start_epoch = int(os.path.basename(args.resume).split('-')[0].replace('ep', ''))
        except:
            start_epoch = 0

    print("Creating datasets...")
    train_dataset = MultiModalClassificationDataset(
        scene_list_file=os.path.join(config.dataset_dir_train, 'train_scenelist.txt'),
        sonar_root=os.path.join(config.dataset_dir_train, 'sonar'),
        rgb_root=os.path.join(config.dataset_dir_train, 'rgb'),
        input_shape=config.input_shape,
        num_classes=config.num_classes,
        train=True
    )

    val_dataset = MultiModalClassificationDataset(
        scene_list_file=os.path.join(config.dataset_dir_val, 'val_scenelist.txt'),
        sonar_root=os.path.join(config.dataset_dir_val, 'sonar'),
        rgb_root=os.path.join(config.dataset_dir_val, 'rgb'),
        input_shape=config.input_shape,
        num_classes=config.num_classes,
        train=False
    )

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create a partial version of worker_init_fn with preset rank=0 and seed    
    worker_init_fn_partial = partial(worker_init_fn, rank=0, seed=config.seed)

    # Create data loaders    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.Unfreeze_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=multimodal_collate_fn,
        worker_init_fn=worker_init_fn_partial
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.Unfreeze_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=multimodal_collate_fn,
        worker_init_fn=worker_init_fn_partial
    )

    if config.optimizer_type == 'adam':
        backbone_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if ('sonar_backbone' in name or 'rgb_backbone' in name or 
                    'sonar_stream' in name or 'rgb_stream' in name):
                    backbone_params.append(param)
                else:
                    other_params.append(param)
        
        param_groups = [
            {'params': backbone_params, 'lr': config.backbone_lr, 'name': 'backbone'},
            {'params': other_params, 'lr': config.Init_lr, 'name': 'others'}
        ]
        
        optimizer = optim.AdamW(param_groups, weight_decay=config.weight_decay)
        
        print(f"   Optimizer param groups:")
        print(f"   Backbone params: {len(backbone_params)}, LR: {config.backbone_lr}")
        print(f"   Other params: {len(other_params)}, LR: {config.Init_lr}")
    else:
        backbone_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if ('sonar_backbone' in name or 'rgb_backbone' in name or 
                    'sonar_stream' in name or 'rgb_stream' in name):
                    backbone_params.append(param)
                else:
                    other_params.append(param)
        
        param_groups = [
            {'params': backbone_params, 'lr': config.backbone_lr, 'name': 'backbone'},
            {'params': other_params, 'lr': config.Init_lr, 'name': 'others'}
        ]
        
        optimizer = optim.SGD(param_groups,
                             momentum=config.momentum, weight_decay=config.weight_decay)
        
        print(f"   Optimizer param groups:")
        print(f"   Backbone params: {len(backbone_params)}, LR: {config.backbone_lr}")
        print(f"   Other params: {len(other_params)}, LR: {config.Init_lr}")

    lr_after_warmup = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.UnFreeze_Epoch - config.warmup_epochs, eta_min=config.Min_lr
    )
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1.0, total_epoch=config.warmup_epochs, after_scheduler=lr_after_warmup
    )

    if config.fp16:
        try:
            scaler = torch.amp.GradScaler('cuda')
        except:
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    best_f1 = 0.0
    patience_counter = 0
    best_model_path = None

    print("   Starting training...")    
    for epoch in range(start_epoch, config.UnFreeze_Epoch):
        train_loss, train_cls_loss, train_cont_loss, train_sonar_loss, train_rgb_loss = train_epoch(
            model, train_loader, optimizer, scaler, config, epoch + 1, log_fp
        )
        scheduler.step(epoch + 1)
        
        print(f"Epoch {epoch+1} Train Loss - Total: {train_loss:.4f} | Cls: {train_cls_loss:.4f} | Cont: {train_cont_loss:.4f} | Sonar: {train_sonar_loss:.4f} | RGB: {train_rgb_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if (epoch + 1) % config.eval_period == 0:
            val_loss, exact_match, macro_f1 = validate_epoch(
                model, val_loader, config, epoch + 1
            )

            print(f"\nValidation Epoch {epoch+1}/{config.UnFreeze_Epoch}:")
            print(f"   Val loss: {val_loss:.6f}")
            print(f"   Exact match: {exact_match:.4f}")
            print(f"   Macro F1: {macro_f1:.4f}")

            if macro_f1 > best_f1:
                if best_model_path is not None and os.path.exists(best_model_path):
                    try:
                        os.remove(best_model_path)
                        print(f"Deleted old weights: {os.path.basename(best_model_path)}")
                    except Exception as e:
                        print(f"Failed to delete old weights: {e}")
                
                best_f1 = macro_f1
                best_model_path = os.path.join(save_dir, f'best_model_f1_{macro_f1:.4f}.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model: {os.path.basename(best_model_path)} (F1: {macro_f1:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.early_stopping['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    end_time = datetime.datetime.now()
    print(f"\nTraining completed!")
    if best_model_path:
        print(f"Best model: {best_model_path}")
    print(f"Best F1 score: {best_f1:.4f}")
    print(f"Save dir: {save_dir}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        if log_fp:
            log_fp.close()
            print(f"Log saved to: {log_file_path}")
    except:
        pass


if __name__ == '__main__':
    main()
