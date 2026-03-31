#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import datetime
import torch
import torch.nn.functional as F
import yaml
import random
import numpy as np
import cv2
import time
import warnings
from PIL import Image
import logging
from nets.multimodal_network import create_multimodal_network
from utils.evaluation import evaluate_multimodal_model
from utils.utils import seed_everything
from utils.gradcam import GradCAM, tensor_to_image
from utils.visualization import MultiLabelVisualization


def calculate_model_flops(model, input_shape, device='cpu'):
    """
    Calculate model FLOPs (Floating Point Operations) - multi-modal mode only
    
    Args:
        model: Model instance
        input_shape: Input shape [H, W]
        device: Computing device
        
    Returns:
        dict: Dictionary containing FLOPs and MACs information
    """
    model_copy = model
    model_copy.eval()
    
    try:
        if THOP_AVAILABLE:
            # Use thop library for accurate calculation
            import sys
            from io import StringIO
            
            # Save original stdout
            original_stdout = sys.stdout
            sys.stdout = StringIO()  # Suppress detailed output
                        
            try:
                # Multi-modal input
                sonar_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
                rgb_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
                macs, params = profile(model_copy, inputs=(sonar_input, rgb_input), verbose=False, report_missing=False)
                    
                # Convert MACs to FLOPs (typically FLOPs = 2 * MACs)
                flops = 2 * macs
                                
                # Format output
                flops_str, params_str = clever_format([flops, params], "%.3f")
                
            finally:
                # Restore original stdout
                sys.stdout = original_stdout
                    
            return {
                'flops': flops,
                'flops_str': flops_str,
                'macs': macs,
                'params': params,
                'params_str': params_str,
                'method': 'thop'
            }
        else:
            # Manual estimation of FLOPs (simplified version)
            total_params = sum(p.numel() for p in model_copy.parameters())
            estimated_flops = total_params * input_shape[0] * input_shape[1] * 2  # Two inputs
                    
            # Format output
            if estimated_flops >= 1e9:
                flops_str = f"{estimated_flops/1e9:.2f}G"
            elif estimated_flops >= 1e6:
                flops_str = f"{estimated_flops/1e6:.2f}M"
            elif estimated_flops >= 1e3:
                flops_str = f"{estimated_flops/1e3:.2f}K"
            else:
                flops_str = f"{estimated_flops:.0f}"
                
            return {
                'flops': estimated_flops,
                'flops_str': flops_str,
                'macs': estimated_flops / 2,
                'params': total_params,
                'params_str': f"{total_params:,}",
                'method': 'manual_estimation'
            }
            
    except Exception as e:
        warnings.warn(f"FLOPs calculation failed: {e}")
        return {
            'flops': 0,
            'flops_str': 'N/A',
            'macs': 0,
            'params': 0,
            'params_str': 'N/A',
            'method': 'failed'
        }


def measure_inference_speed(model, input_shape, device='cpu', 
                          warmup_runs=10, test_runs=100):
    """
    Measure model inference speed - multi-modal mode only
    
    Args:
        model: Model instance
        input_shape: Input shape [H, W]
        device: Computing device
        warmup_runs: Warmup runs
        test_runs: Test runs
        
    Returns:
        dict: Dictionary containing inference speed information
    """
    model.eval()
    
    try:
        # Prepare multi-modal input data
        sonar_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
        rgb_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
            
        # Warmup GPU (silent run)
        with torch.no_grad():
            for _ in range(warmup_runs):
                try:
                    _ = model(sonar_input, rgb_input)
                except Exception:
                    pass
                        
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            
        # Measure inference time
        start_time = time.time()
            
        with torch.no_grad():
            for _ in range(test_runs):
                _ = model(sonar_input, rgb_input)
                        
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_inference = total_time / test_runs
        fps = 1.0 / avg_time_per_inference
            
        return {
            'avg_inference_time_ms': avg_time_per_inference * 1000,
            'fps': fps,
            'total_test_time': total_time,
            'test_runs': test_runs,
            'warmup_runs': warmup_runs,
            'device': str(device)
        }
            
    except Exception as e:
        warnings.warn(f"Inference speed measurement failed: {e}")
        return {
            'avg_inference_time_ms': 0,
            'fps': 0,
            'total_test_time': 0,
            'test_runs': 0,
            'warmup_runs': 0,
            'device': str(device)
        }


class EvaluationLogger:
    """Evaluation logger, outputs to both console and log file"""
    
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.log_file = None
        
        # Create log directory
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        # Open log file
        try:
            self.log_file = open(log_file_path, 'w', encoding='utf-8')
            print(f"📝 Log file : {log_file_path}")
            self.log(f"📝 Log file: {log_file_path}")
            self.log(f"⏰ Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            print(f"⚠️ 无法创建日志文件:     {e}")
    
    def log(self, message):
        """记录消息到文件（不输出到控制台）"""
        if self.log_file:
            try:
                self.log_file.write(str(message) + '\n')
                self.log_file.flush()
            except Exception as e:
                print(f"⚠️ 写入日志失败: {e}")
    
    def print_and_log(self, message):
        """同时输出到控制台和日志文件"""
        print(message)
        self.log(message)
    
    def close(self):
        """关闭日志文件"""
        if self.log_file:
            try:
                self.log_file.close()
            except:
                pass


class GQSAGradCAM:
    """Grad-CAM class for Global Query Space Attention (GQSA) features"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        
    def register_hooks(self, target_layer_names):
        """Register forward and backward hooks"""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
            
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        # Clear previous hooks
        self.remove_hooks()
        
        # Register new hooks
        for name, module in self.model.named_modules():
            if name in target_layer_names:
                h1 = module.register_forward_hook(forward_hook(name))
                h2 = module.register_backward_hook(backward_hook(name))
                self.hooks.extend([h1, h2])
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_gqsa_gradcam(self, rgb_input, sonar_input, target_class, 
                             use_rgb_gqsa=True, use_sonar_gqsa=False,
                             target_stage=None):
        """
        生成全局查询空间注意力(GQSA)特征的Grad-CAM
        
        Args:
            rgb_input: RGB输入 (1, 3, H, W)
            sonar_input: Sonar输入 (1, 3, H, W)
            target_class: 目标类别
            use_rgb_gqsa: 是否使用RGB-GQSA
            use_sonar_gqsa: 是否使用Sonar-GQSA
            
        Returns:
            heatmaps: 热图字典
        """
        self.model.eval()
        heatmaps = {}
        
        if rgb_input is not None:
            rgb_input = rgb_input.to(self.device)
        if sonar_input is not None:
            sonar_input = sonar_input.to(self.device)
        
        # 根据配置选择目标层 - 动态查找合适的卷积层
        target_layers = []
        
        # 动态查找合适的目标层 - 只使用第4阶段特征
        for name, module in self.model.named_modules():
            # 检查是否是我们想要的目标层
            is_target = False
            if target_stage:
                # 如果指定了stage，只选择该stage中的卷积层
                if f'stages_{target_stage-1}' in name:
                    is_target = True
            else:
                # 默认行为：只选择第4阶段的卷积层（高级特征）
                if 'stages_3' in name:  # stages_3对应第4阶段（从0开始计数）
                    is_target = True

            if not is_target:
                continue

            # 查找RGB backbone的卷积层
            if use_rgb_gqsa and rgb_input is not None and 'dual_backbone.rgb_backbone' in name:
                if hasattr(module, 'weight') and len(module.weight.shape) == 4:  # 卷积层
                    target_layers.append(name)
            
            # 查找SONAR backbone的卷积层  
            if use_sonar_gqsa and sonar_input is not None and 'dual_backbone.sonar_backbone' in name:
                if hasattr(module, 'weight') and len(module.weight.shape) == 4:  # 卷积层
                    target_layers.append(name)
        
        # 如果没有找到合适的层，使用备用策略（优先选择第4阶段）
        if not target_layers:
            print("警告: 未找到Stage 4目标层，尝试备用策略...")
            # 首先尝试找第4阶段的任何卷积层
            for name, module in self.model.named_modules():
                if 'stages_3' in name and ('rgb_backbone' in name or 'sonar_backbone' in name):
                    if hasattr(module, 'weight') and len(module.weight.shape) == 4:  # 卷积层
                        target_layers.append(name)
            
            # 如果第4阶段还是没找到，再使用通用策略
            if not target_layers:
                print("警告: 第4阶段仍未找到，使用通用备用策略...")
                for name, module in self.model.named_modules():
                    if ('rgb_backbone' in name or 'sonar_backbone' in name) and hasattr(module, 'weight'):
                        if len(module.weight.shape) == 4:  # 卷积层
                            target_layers.append(name)
                            if len(target_layers) >= 2:  # 最多选择2个层
                                break
        
        # 取最后几个层作为目标层（第4阶段高级特征）
        if len(target_layers) > 2:
            target_layers = target_layers[-2:]
        
        print(f"🎯 找到目标层: {target_layers}")
        if not target_layers:
            print("❌ 没有找到合适的目标层，跳过热图生成")
            return heatmaps
            
        self.register_hooks(target_layers)
        
        try:
            # 前向传播
            with torch.enable_grad():
                if rgb_input is not None and sonar_input is not None:
                    outputs = self.model(rgb_input, sonar_input)
                elif rgb_input is not None:
                    outputs = self.model(rgb_input, None)
                else:
                    outputs = self.model(None, sonar_input)
                
                # 获取目标类别的分数
                target_score = outputs[0, target_class]
                
                # 反向传播
                self.model.zero_grad()
                target_score.backward(retain_graph=True)
                
                # 生成热图
                for layer_name in target_layers:
                    if layer_name in self.gradients and layer_name in self.activations:
                        gradients = self.gradients[layer_name]
                        activations = self.activations[layer_name]
                        
                        # 计算权重 (全局平均池化)
                        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
                        
                        # 生成热图
                        heatmap = torch.sum(weights * activations, dim=1, keepdim=True)
                        heatmap = F.relu(heatmap)
                        
                        # 归一化到0-1
                        # 只移除batch维度和channel维度（如果为1），保留空间维度
                        if heatmap.shape[0] == 1:
                            heatmap = heatmap.squeeze(0)  # 移除batch维度
                        if len(heatmap.shape) > 0 and heatmap.shape[0] == 1:
                            heatmap = heatmap.squeeze(0)  # 移除channel维度（如果为1）
                        
                        # 检查tensor维度和元素数量
                        if heatmap.numel() == 0:
                            print(f"警告: {layer_name} 层的heatmap为空，跳过")
                            continue
                        elif heatmap.numel() == 1:
                            print(f"警告: {layer_name} 层的heatmap只有一个元素，跳过")
                            continue
                        elif len(heatmap.shape) == 0:
                            print(f"警告: {layer_name} 层的heatmap是标量，跳过")
                            continue
                        elif len(heatmap.shape) == 1:
                            print(f"警告: {layer_name} 层的heatmap是1D向量，跳过")
                            continue
                        
                        # 执行归一化
                        if heatmap.numel() > 1:
                            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                        
                        # 上采样到输入尺寸
                        # 确保heatmap是4D张量 (1, 1, H, W)
                        if len(heatmap.shape) == 2:
                            heatmap = heatmap.unsqueeze(0).unsqueeze(0)
                        elif len(heatmap.shape) == 3:
                            if heatmap.shape[0] == 1:
                                heatmap = heatmap.unsqueeze(0)  # (1, H, W) -> (1, 1, H, W)
                            else:
                                heatmap = heatmap.unsqueeze(1)  # (N, H, W) -> (N, 1, H, W)
                        elif len(heatmap.shape) != 4:
                            print(f"警告: {layer_name} 层的heatmap维度异常: {heatmap.shape}，跳过")
                            continue
                            
                        heatmap_resized = F.interpolate(
                            heatmap, 
                            size=(224, 224), 
                            mode='bilinear', 
                            align_corners=False
                        )
                        
                        heatmap_np = heatmap_resized.squeeze().cpu().numpy()
                        
                        # 根据层名称分配热图
                        if 'rgb' in layer_name:
                            heatmaps['rgb_gqsa'] = heatmap_np
                        elif 'sonar' in layer_name:
                            heatmaps['sonar_gqsa'] = heatmap_np
                            
        except Exception as e:
            print(f"❌ 生成Grad-CAM时出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.remove_hooks()
        
        print(f"✅ 成功生成 {len(heatmaps)} 个GQSA热图: {list(heatmaps.keys())}")
        return heatmaps
    
    def save_gradcam_results(self, rgb_image, sonar_image, heatmaps, class_names, target_class, save_path, sample_name):
        """
        保存全局查询空间注意力(GQSA) Grad-CAM结果
        参考evaluate_simplified.py的实现
        """
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # 根据配置保存RGB-GQSA热图
            if rgb_image is not None and 'rgb_gqsa' in heatmaps:
                rgb_save_path = os.path.join(save_path, f'{sample_name}_rgb_gqsa_gradcam.png')
                self.save_gradcam_overlay(rgb_image, heatmaps['rgb_gqsa'], rgb_save_path, 'rgb_gqsa', sample_name)
                print(f"💾 保存RGB-GQSA热图: {rgb_save_path}")
            
            # 根据配置保存Sonar-GQSA热图
            if sonar_image is not None and 'sonar_gqsa' in heatmaps:
                sonar_save_path = os.path.join(save_path, f'{sample_name}_sonar_gqsa_gradcam.png')
                self.save_gradcam_overlay(sonar_image, heatmaps['sonar_gqsa'], sonar_save_path, 'sonar_gqsa', sample_name)
                print(f"💾 保存Sonar-GQSA热图: {sonar_save_path}")
                
        except Exception as e:
            print(f"保存全局查询空间注意力(GQSA)热图结果时出错: {e}")

    def save_gradcam_overlay(self, original_image, heatmap, save_path, class_name, sample_name):
        """
        保存热图覆盖的图像
        参考evaluate_simplified.py的实现，生成原图+热图拼接，并添加10像素灰色边框
        """
        if original_image is None or heatmap is None:
            return
            
        # 确保图像是uint8格式
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)
            
        # 应用热图颜色映射
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # 叠加热图
        overlay = cv2.addWeighted(original_image, 0.6, heatmap_colored, 0.4, 0)
        
        # 确保两张图尺寸一致
        if original_image.shape != overlay.shape:
            overlay = cv2.resize(overlay, (original_image.shape[1], original_image.shape[0]))

        # 水平拼接原图和热图
        combined_image = np.hstack((original_image, overlay))
        
        # 添加四周10像素的灰色边框
        border_size = 10
        gray_color = [128, 128, 128]
        
        final_image = cv2.copyMakeBorder(
            combined_image, 
            border_size, 
            border_size, 
            border_size, 
            border_size, 
            cv2.BORDER_CONSTANT, 
            value=gray_color
        )
        
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存图像
        cv2.imwrite(save_path, final_image)
        
        print(f"保存热图: {save_path}")


class EvaluationConfig:
    """评估专用配置类，从训练配置继承必要参数"""
    
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.load_config()
    
    def load_config(self):
        """从YAML文件加载配置"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # 数据集配置
        dataset = config_data['dataset']
        self.dataset_dir_val = dataset['test_dir']
        self.num_classes = dataset['num_classes']
        self.class_names = dataset['class_names']
        self.input_shape = dataset['input_shape']
        
        # 网络配置
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
        
        # RGB-GQSA配置
        self.use_rgb_gqsa = network.get('use_rgb_gqsa', True)
        self.gqsa_stages = network.get('gqsa_stages', {})
        
        # 评估配置
        strategy = config_data['strategy']
        self.eval_threshold = strategy['eval_threshold']
        self.num_workers = strategy['num_workers']
        self.Unfreeze_batch_size = config_data['training']['unfreeze_batch_size']
        
        # 基本配置
        basic = config_data['basic']
        self.seed = basic['seed']
        self.Cuda = basic['cuda']
    
    def get_model_config(self):
        """获取模型配置字典"""
        return {
            'num_classes': self.num_classes,
            'model_size': self.model_size,
            'pretrained': False,  # 评估时不需要预训练权重
            'local_weights_path': None,
            'feature_dims': self.feature_dims,
            'global_dim': self.global_dim,
            'fusion_dim': self.fusion_dim,
            'projection_dim': self.projection_dim,
            'temperature': self.temperature,
            'use_rgb_gqsa': self.use_rgb_gqsa,
            'gqsa_stages_config': self.gqsa_stages
        }
    
    def print_config(self, logger=None):
        """打印评估配置信息"""
        output_func = logger.print_and_log if logger else print
        
        output_func("=" * 60)
        output_func("多模态场景级分类模型评估")
        output_func("=" * 60)
        output_func(f"配置文件: {self.config_path}")
        output_func(f"验证数据集: {self.dataset_dir_val}")
        output_func(f"类别数: {self.num_classes}")
        output_func(f"类别名称: {self.class_names}")
        output_func(f"输入尺寸: {self.input_shape}")
        output_func(f"模型大小: {self.model_size}")
        output_func(f"使用RGB-GQSA: {self.use_rgb_gqsa}")
        if self.gqsa_stages:
            output_func(f"GQSA阶段配置: {self.gqsa_stages}")
        output_func(f"评估阈值: {self.eval_threshold}")
        output_func("=" * 60)


def main():
    """主评估函数"""
    parser = argparse.ArgumentParser(description='多模态场景级分类模型评估')
    parser.add_argument('--model', type=str, required=True,
                       help='训练好的模型权重文件路径')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径 (默认: config.yaml)')
    parser.add_argument('--dataset_dir', type=str, default=None,
                       help='验证数据集目录 (默认从config读取)')
    parser.add_argument('--threshold', type=float, default=None,
                       help='分类阈值 (默认从config读取)')
    parser.add_argument('--find_best_threshold', action='store_true',
                       help='是否寻找最优阈值')
    parser.add_argument('--save_predictions', action='store_true',
                       help='是否保存预测结果')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='输出目录 (默认: evaluation_results)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU设备ID (默认: 0, -1表示使用CPU)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批大小 (默认从config读取)')
    parser.add_argument('--evaluate_separate', action='store_true',
                       help='是否分离评估RGB和SONAR分类头')
    
    # 全局查询空间注意力(GQSA) Grad-CAM相关参数
    parser.add_argument('--generate_gqsa_gradcam', action='store_true',
                       help='是否生成全局查询空间注意力(GQSA)特征的Grad-CAM热图')
    parser.add_argument('--gradcam_sample', type=int, default=50,
                       help='生成Grad-CAM的样本数量 (默认: 50)')
    parser.add_argument('--gradcam_output_dir', type=str, default='gqsa_gradcam_results_rgb_only',
                       help='全局查询空间注意力(GQSA) Grad-CAM热图输出目录 (默认: gqsa_gradcam_results)')
    parser.add_argument('--gradcam_target_stage', type=int, default=4,
                        choices=[1, 2, 3, 4],
                        help='指定生成Grad-CAM的目标Stage (默认: 第4阶段)')

    # 保留原有Grad-CAM参数用于兼容
    parser.add_argument('--generate_gradcam', action='store_true',
                       help='是否生成传统Grad-CAM热图')
    parser.add_argument('--gradcam_target_classes', type=str, default=None,
                       help='指定生成热图的目标类别，用逗号分隔 (默认: 随机选择)')
    parser.add_argument('--gradcam_modality', type=str, default='rgb',
                       choices=['rgb', 'sonar', 'both'],
                       help='生成热图的模态 (默认: both)')
    
    # 可视化相关参数
    parser.add_argument('--generate_visualizations', action='store_true',
                       help='是否生成评估可视化结果（混淆矩阵、PR曲线等）')
    parser.add_argument('--visualization_output_dir', type=str, default='evaluation_visualizations',
                       help='可视化结果输出目录 (默认: evaluation_visualizations)')
    parser.add_argument('--include_confusion_matrix', action='store_true', default=True,
                       help='是否包含混淆矩阵（默认: True）')
    parser.add_argument('--include_pr_curves', action='store_true', default=True,
                       help='是否包含精确率-召回率曲线（默认: True）')
    parser.add_argument('--include_class_distribution', action='store_true', default=True,
                       help='是否包含类别分布图（默认: True）')
    parser.add_argument('--save_individual_plots', action='store_true', default=True,
                       help='是否保存每个类别的单独图表（默认: True）')
    parser.add_argument('--include_threshold_analysis', action='store_true', default=True,
                       help='是否包含阈值分析图（默认: True）')
    
    # t-SNE可视化相关参数
    parser.add_argument('--generate_tsne', action='store_true',
                       help='是否生成t-SNE特征可视化图')
    parser.add_argument('--tsne_samples_per_class', type=int, default=50,
                       help='每个类别用于t-SNE的样本数量 (默认: 50)')
    parser.add_argument('--tsne_output_dir', type=str, default='tsne_visualizations',
                       help='t-SNE可视化输出目录 (默认: tsne_visualizations)')
    parser.add_argument('--tsne_perplexity', type=float, default=30.0,
                       help='t-SNE的perplexity参数 (默认: 30.0)')
    parser.add_argument('--tsne_learning_rate', type=float, default=200.0,
                       help='t-SNE的学习率 (默认: 200.0)')
    parser.add_argument('--tsne_n_iter', type=int, default=1000,
                       help='t-SNE的迭代次数 (默认: 1000)')
    parser.add_argument('--tsne_random_state', type=int, default=42,
                       help='t-SNE的随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"❌ 错误：找不到模型文件 {args.model}")
        return
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"❌ 错误：找不到配置文件 {args.config}")
        return
    
    # 加载配置
    print("📋 加载配置...")
    config = EvaluationConfig(args.config)
    
    # 更新配置
    if args.dataset_dir:
        config.dataset_dir_val = args.dataset_dir
    if args.threshold:
        config.eval_threshold = args.threshold
    if args.batch_size:
        config.Unfreeze_batch_size = args.batch_size
    
    # 检查数据集标签文件（支持测试集和验证集）
    val_scene_file = os.path.join(config.dataset_dir_val, 'val_scenelist.txt')
    test_scene_file = os.path.join(config.dataset_dir_val, 'test_scenelist.txt')

    if os.path.exists(test_scene_file):
        scene_file = test_scene_file
        dataset_type = "测试"
    elif os.path.exists(val_scene_file):
        scene_file = val_scene_file
        dataset_type = "验证"
    else:
        print(f"❌ 错误：找不到标签文件 {val_scene_file} 或 {test_scene_file}")
        return

    print(f"📂 使用{dataset_type}数据集: {scene_file}")
    
    # 创建日志记录器
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"multimodal_{config.model_size}_{timestamp}.log"
    log_path = os.path.join("/root/autodl-tmp/classification/Evaluate_logs", log_filename)
    logger = EvaluationLogger(log_path)
    
    try:
        config.print_config(logger)
    
        # 设置设备
        if args.gpu >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            device = torch.device(f'cuda:{args.gpu}')
            logger.print_and_log(f"🔧 使用设备: CUDA:{args.gpu}")
        else:
            device = torch.device('cpu')
            logger.print_and_log("🔧 使用设备: CPU")
        
        # 设置随机种子
        seed_everything(config.seed)
        
        # 创建输出目录
        if args.save_predictions:
            os.makedirs(args.output_dir, exist_ok=True)
            logger.print_and_log(f"📁 输出目录: {args.output_dir}")
        
        if args.generate_gradcam:
            os.makedirs(args.gradcam_output_dir, exist_ok=True)
            logger.print_and_log(f"🔥 Grad-CAM输出目录: {args.gradcam_output_dir}")
            
        if args.generate_gqsa_gradcam:
            os.makedirs(args.gradcam_output_dir, exist_ok=True)
            logger.print_and_log(f"🔥 全局查询空间注意力(GQSA) Grad-CAM输出目录: {args.gradcam_output_dir}")
        
        if args.generate_visualizations:
            os.makedirs(args.visualization_output_dir, exist_ok=True)
            logger.print_and_log(f"🎨 可视化结果输出目录: {args.visualization_output_dir}")
            logger.print_and_log(f"   包含混淆矩阵: {args.include_confusion_matrix}")
            logger.print_and_log(f"   包含PR曲线: {args.include_pr_curves}")
            logger.print_and_log(f"   包含类别分布图: {args.include_class_distribution}")
            logger.print_and_log(f"   包含阈值分析: {args.include_threshold_analysis}")
        
        if args.generate_tsne:
            os.makedirs(args.tsne_output_dir, exist_ok=True)
            logger.print_and_log(f"🧬 t-SNE可视化输出目录: {args.tsne_output_dir}")
            logger.print_and_log(f"   每个类别样本数: {args.tsne_samples_per_class}")
            logger.print_and_log(f"   Perplexity: {args.tsne_perplexity}")
            logger.print_and_log(f"   学习率: {args.tsne_learning_rate}")
            logger.print_and_log(f"   迭代次数: {args.tsne_n_iter}")
        
        # 创建模型
        logger.print_and_log(f"🏗️ 创建模型: {config.model_size}")
        
        # 输出模型创建的详细信息
        logger.print_and_log("🔧 创建MAXVIT双流主干: maxvit_tiny_tf_224")
            
        model_config = config.get_model_config()
        model = create_multimodal_network(**model_config)
        
        # 输出网络创建信息
        logger.print_and_log(f"✅ 创建多模态网络:")
        logger.print_and_log(f"   骨干网络: MAXVIT双流结构")
        logger.print_and_log(f"   模型: maxvit_tiny_tf_224")
        logger.print_and_log(f"   类别数: {config.num_classes}")
        logger.print_and_log(f"   全局特征维度: {config.global_dim}")
        logger.print_and_log(f"   融合特征维度: {config.fusion_dim}")
        logger.print_and_log(f"   投影特征维度: {config.projection_dim}")
        logger.print_and_log(f"   RGB-GQSA: {config.use_rgb_gqsa}")
        
        # 统计网络参数
        logger.print_and_log("📊 网络参数统计:")
        
        # 详细参数统计
        total_params = 0
        trainable_params = 0
        
        # 打印详细的模块参数
        logger.print_and_log("   模块参数详情:")
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            trainable_module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += module_params
            trainable_params += trainable_module_params
            
            logger.print_and_log(f"   - {name}:")
            logger.print_and_log(f"     - 总计: {module_params:,}")
            logger.print_and_log(f"     - 可训练: {trainable_module_params:,}")

        frozen_params = total_params - trainable_params
        
        logger.print_and_log("   ----------------------------------------")
        logger.print_and_log(f"   总参数 (合计): {total_params:,}")
        logger.print_and_log(f"   可训练参数 (合计): {trainable_params:,}")
        logger.print_and_log(f"   冻结参数 (合计): {frozen_params:,}")
        
        # 移动模型到设备 (提前)
        model = model.to(device)
        logger.print_and_log(f"✅ 模型已成功移动到设备: {device}")
        
        # 加载模型权重
        logger.print_and_log(f"📂 加载模型权重: {args.model}")
        try:
            if device.type == 'cuda':
                checkpoint = torch.load(args.model)
            else:
                checkpoint = torch.load(args.model, map_location='cpu')
            
            # 处理不同的权重格式
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                elif 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            logger.print_and_log("✅ 模型权重加载成功!")
        except Exception as e:
            logger.print_and_log(f"❌ 模型权重加载失败: {e}")
            return
        
        # 计算理论计算量 (GFLOPs) - 在模型移动到设备并加载权重后进行
        logger.print_and_log("\n🔧 计算理论计算量...")
        try:
            flops_result = calculate_model_flops(
                model=model, 
                input_shape=config.input_shape, 
                device=device
            )
            
            if flops_result['method'] != 'failed':
                logger.print_and_log("📊 理论计算量统计:")
                logger.print_and_log(f"   GFLOPs: {flops_result['flops_str']}")
                if THOP_AVAILABLE:
                    logger.print_and_log(f"   MACs: {flops_result['macs']:,.0f}")
                    logger.print_and_log(f"   计算方法: thop (精确计算)")
                else:
                    logger.print_and_log(f"   计算方法: 手动估算 (可能不准确)")
            else:
                logger.print_and_log("⚠️ 理论计算量计算失败")
                
        except Exception as e:
            logger.print_and_log(f"⚠️ 理论计算量计算出错: {e}")
        
        # 测量实际推理速度 (需要在模型加载权重并移动到设备后进行)
        logger.print_and_log("\n⏱️ 开始测量实际推理速度...")
        try:
            speed_result = measure_inference_speed(
                model=model,
                input_shape=config.input_shape,
                device=device,
                warmup_runs=10,
                test_runs=100
            )
            
            if speed_result['avg_inference_time_ms'] > 0:
                logger.print_and_log("📊 推理速度统计:")
                logger.print_and_log(f"   平均推理时间: {speed_result['avg_inference_time_ms']:.2f} ms")
                logger.print_and_log(f"   理论FPS: {speed_result['fps']:.2f}")
                logger.print_and_log(f"   测试设备: {speed_result['device']}")
                logger.print_and_log(f"   预热运行: {speed_result['warmup_runs']} 次")
                logger.print_and_log(f"   测试运行: {speed_result['test_runs']} 次")
                logger.print_and_log(f"   总测试时间: {speed_result['total_test_time']:.2f} s")
            else:
                logger.print_and_log("⚠️ 推理速度测量失败")
                
        except Exception as e:
            logger.print_and_log(f"⚠️ 推理速度测量出错: {e}")
        
        # 开始评估
        logger.print_and_log(f"\n🚀 开始评估...")
        start_time = datetime.datetime.now()
        
        try:
            # 创建一个包装函数，用于在evaluation函数中记录输出
            def evaluation_print_wrapper(message):
                logger.print_and_log(message)
            
            # 注意：这里需要修改evaluate_multimodal_model函数来接受logger参数
            # 暂时使用现有的函数，后续可以进一步优化
            results = evaluate_multimodal_model(
                model=model,
                dataset_dir=config.dataset_dir_val,
                config=config,
                device=device,
                save_predictions=args.save_predictions,
                output_dir=args.output_dir if args.save_predictions else None,
                find_best_threshold=args.find_best_threshold,
                evaluate_separate=args.evaluate_separate
            )
            
            # 显示关键结果
            metrics = results['metrics']
            logger.print_and_log(f"\n🎯 融合分类头评估完成！关键指标:")
            logger.print_and_log(f"   精确匹配率 (EMR): {metrics['exact_match_ratio']:.4f}")
            logger.print_and_log(f"   宏平均精确率: {metrics['macro_precision']:.4f}")
            logger.print_and_log(f"   宏平均召回率: {metrics['macro_recall']:.4f}")
            logger.print_and_log(f"   宏平均F1分数: {metrics['macro_f1']:.4f}")
            logger.print_and_log(f"   微平均F1分数: {metrics['micro_f1']:.4f}")
            logger.print_and_log(f"   Hamming损失: {metrics['hamming_loss']:.4f}")
            logger.print_and_log(f"   Jaccard相似度: {metrics['jaccard_similarity']:.4f}")
            
            # 显示分离评估结果（如果有）
            if args.evaluate_separate and 'rgb_metrics' in results and results['rgb_metrics']:
                rgb_metrics = results['rgb_metrics']
                sonar_metrics = results['sonar_metrics']
                
                logger.print_and_log(f"\n🎨 RGB分类头关键指标:")
                logger.print_and_log(f"   精确匹配率 (EMR): {rgb_metrics['exact_match_ratio']:.4f}")
                logger.print_and_log(f"   宏平均F1分数: {rgb_metrics['macro_f1']:.4f}")
                logger.print_and_log(f"   微平均F1分数: {rgb_metrics['micro_f1']:.4f}")
                
                logger.print_and_log(f"\n🔊 SONAR分类头关键指标:")
                logger.print_and_log(f"   精确匹配率 (EMR): {sonar_metrics['exact_match_ratio']:.4f}")
                logger.print_and_log(f"   宏平均F1分数: {sonar_metrics['macro_f1']:.4f}")
                logger.print_and_log(f"   微平均F1分数: {sonar_metrics['micro_f1']:.4f}")
            
            if results['avg_loss'] > 0:
                logger.print_and_log(f"\n📊 损失信息:")
                logger.print_and_log(f"   平均验证损失: {results['avg_loss']:.4f}")
            
            # 显示最优阈值信息
            if results['optimal_threshold']:
                logger.print_and_log(f"\n🎯 最优阈值: {results['optimal_threshold']:.3f}")
            
            # 生成可视化评估结果
            if args.generate_visualizations:
                logger.print_and_log(f"\n🎨 开始生成评估可视化结果...")
                try:
                    visualizer = MultiLabelVisualization(
                        class_names=config.class_names,
                        output_dir=args.visualization_output_dir
                    )
                    
                    visualization_results = visualizer.generate_evaluation_report(
                        predictions=results['predictions'],
                        targets=results['targets'],
                        threshold=config.eval_threshold,
                        include_confusion_matrix=args.include_confusion_matrix,
                        include_pr_curves=args.include_pr_curves,
                        include_distribution=args.include_class_distribution,
                        include_threshold_analysis=args.include_threshold_analysis
                    )
                    
                    # 保存摘要报告
                    visualizer.save_summary_report(results['metrics'], visualization_results)
                    
                    logger.print_and_log(f"✅ 可视化结果生成完成")
                    logger.print_and_log(f"💾 可视化图像保存在: {args.visualization_output_dir}")
                    
                    # 统计生成的图像数量
                    if visualization_results:
                        if 'confusion_matrices' in visualization_results:
                            cm_count = len(visualization_results['confusion_matrices']['matrices'])
                            logger.print_and_log(f"📊 生成混淆矩阵: {cm_count} 个类别")
                            
                            if 'multiclass_matrix' in visualization_results['confusion_matrices']:
                                multiclass_shape = visualization_results['confusion_matrices']['multiclass_matrix'].shape
                                logger.print_and_log(f"📊 生成 {multiclass_shape[0]}×{multiclass_shape[1]} 多类别混淆矩阵")
                        
                        if 'precision_recall_curves' in visualization_results:
                            pr_count = len(visualization_results['precision_recall_curves']['curves'])
                            mean_ap = visualization_results['precision_recall_curves']['mean_ap']
                            logger.print_and_log(f"📈 生成PR曲线: {pr_count} 个类别 (mAP: {mean_ap:.4f})")
                        
                        if 'threshold_analysis' in visualization_results:
                            threshold_count = len(visualization_results['threshold_analysis'])
                            logger.print_and_log(f"📊 生成阈值分析: {threshold_count} 个阈值点")
                
                except Exception as e:
                    logger.print_and_log(f"❌ 生成可视化结果时发生错误: {e}")
                    import traceback
                    error_traceback = traceback.format_exc()
                    logger.log(error_traceback)

            # 生成t-SNE特征可视化图
            if args.generate_tsne:
                logger.print_and_log(f"\n🧬 开始生成t-SNE特征可视化...")
                try:
                    tsne_results = generate_tsne_visualization(
                        model=model,
                        dataset_dir=config.dataset_dir_val,
                        config=config,
                        device=device,
                        samples_per_class=args.tsne_samples_per_class,
                        output_dir=args.tsne_output_dir,
                        perplexity=args.tsne_perplexity,
                        learning_rate=args.tsne_learning_rate,
                        n_iter=args.tsne_n_iter,
                        random_state=args.tsne_random_state
                    )
                    
                    if tsne_results:
                        logger.print_and_log(f"✅ t-SNE可视化生成完成")
                        logger.print_and_log(f"💾 可视化图片: {tsne_results['output_path']}")
                        logger.print_and_log(f"💾 数据文件: {tsne_results['csv_path']}")
                        logger.print_and_log(f"📊 分析样本数: {tsne_results['total_samples']}")
                        logger.print_and_log(f"📊 分析类别数: {tsne_results['classes_analyzed']}")
                    else:
                        logger.print_and_log(f"⚠️ t-SNE可视化生成失败或被跳过")
                        
                except Exception as e:
                    logger.print_and_log(f"❌ 生成t-SNE可视化时发生错误: {e}")
                    import traceback
                    error_traceback = traceback.format_exc()
                    logger.log(error_traceback)

            # 生成GQSA Grad-CAM热图
            if args.generate_gqsa_gradcam:
                logger.print_and_log(f"\n🔥 开始生成GQSA Grad-CAM热图...")
                gqsa_gradcam_results = generate_gqsa_gradcam_heatmaps(
                    model=model,
                    dataset_dir=config.dataset_dir_val,
                    config=config,
                    device=device,
                    num_samples=args.gradcam_sample,
                    output_dir=args.gradcam_output_dir,
                    target_stage=args.gradcam_target_stage  # 传递新参数
                )
                
                if gqsa_gradcam_results:
                    logger.print_and_log(f"✅ 成功生成 {gqsa_gradcam_results['generated_count']} 个GQSA Grad-CAM热图")
                    logger.print_and_log(f"💾 热图保存在: {args.gradcam_output_dir}")
            
            # 生成传统Grad-CAM热图
            if args.generate_gradcam:
                logger.print_and_log(f"\n🔥 开始生成传统Grad-CAM热图...")
                gradcam_results = generate_gradcam_heatmaps(
                    model=model,
                    dataset_dir=config.dataset_dir_val,
                    config=config,
                    device=device,
                    num_samples=args.gradcam_samples,
                    output_dir=args.gradcam_output_dir,
                    target_classes=args.gradcam_target_classes,
                    modality=args.gradcam_modality
                )
                
                if gradcam_results:
                    logger.print_and_log(f"✅ 成功生成 {gradcam_results['generated_count']} 个传统Grad-CAM热图")
                    logger.print_and_log(f"💾 热图保存在: {args.gradcam_output_dir}")
            
            end_time = datetime.datetime.now()
            logger.print_and_log(f"\n⏱️ 评估耗时: {end_time - start_time}")
            
            if args.save_predictions:
                logger.print_and_log(f"💾 预测结果已保存至: {args.output_dir}")
            
            if args.generate_visualizations:
                logger.print_and_log(f"🎨 可视化结果已保存至: {args.visualization_output_dir}")
            
            if args.generate_tsne:
                logger.print_and_log(f"🧬 t-SNE可视化结果已保存至: {args.tsne_output_dir}")
            
            logger.print_and_log(f"\n🎉 评估成功完成！")
            
        except Exception as e:
            logger.print_and_log(f"❌ 评估过程中发生错误: {e}")
            import traceback
            error_traceback = traceback.format_exc()
            logger.log(error_traceback)
    
    finally:
        # 关闭日志文件
        logger.close()


def generate_gradcam_heatmaps(model, dataset_dir, config, device, 
                            num_samples=50, output_dir='gradcam_results',
                            target_classes=None, modality='both'):
    """
    生成Grad-CAM热图
    
    Args:
        model: 训练好的模型
        dataset_dir: 数据集目录
        config: 配置对象
        device: 计算设备
        num_samples: 生成热图的样本数量
        output_dir: 输出目录
        target_classes: 指定的目标类别列表（逗号分隔的字符串）
        modality: 生成热图的模态
        
    Returns:
        生成结果字典
    """
    from utils.multimodal_dataset import MultiModalClassificationDataset
    from PIL import Image
    
    print(f"🎯 初始化Grad-CAM...")
    
    # 初始化Grad-CAM (使用GQSA版本以支持高级热图生成)
    gradcam = GQSAGradCAM(model, device)
    
    # 解析目标类别
    if target_classes:
        target_class_list = [int(x.strip()) for x in target_classes.split(',')]
        print(f"🎯 指定目标类别: {target_class_list}")
    else:
        target_class_list = list(range(config.num_classes))
        print(f"🎯 使用所有类别: {target_class_list}")
    
    # 创建数据集
    val_scene_file = os.path.join(dataset_dir, 'val_scenelist.txt')
    test_scene_file = os.path.join(dataset_dir, 'test_scenelist.txt')
    
    if os.path.exists(test_scene_file):
        scene_file = test_scene_file
    elif os.path.exists(val_scene_file):
        scene_file = val_scene_file
    else:
        print("❌ 找不到数据集标签文件")
        return None
    
    dataset = MultiModalClassificationDataset(
        scene_list_file=scene_file,
        sonar_root=os.path.join(dataset_dir, 'sonar'),
        rgb_root=os.path.join(dataset_dir, 'rgb'),
        input_shape=config.input_shape,
        num_classes=config.num_classes,
        train=False
    )
    
    print(f"📊 数据集大小: {len(dataset)}")
    
    # 随机选择样本
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    print(f"🎲 随机选择 {len(sample_indices)} 个样本生成热图")
    
    generated_count = 0
    
    for i, sample_idx in enumerate(sample_indices):
        try:
            print(f"\r🔥 生成热图进度: {i+1}/{len(sample_indices)}", end="", flush=True)
            
            # 获取样本数据 (多模态)
            sonar_tensor, rgb_tensor, labels_tensor = dataset[sample_idx]
            
            # 转换为batch格式
            if rgb_tensor is not None:
                rgb_input = rgb_tensor.unsqueeze(0)  # (1, 3, H, W)
            else:
                rgb_input = None
                
            if sonar_tensor is not None:
                sonar_input = sonar_tensor.unsqueeze(0)  # (1, 3, H, W)
            else:
                sonar_input = None
            
            # 获取样本名称
            sample_name = dataset.samples[sample_idx][0]
            sample_name_clean = os.path.splitext(sample_name)[0]
            
            # 找到真实标签中的正类
            true_labels = labels_tensor.numpy()
            positive_classes = np.where(true_labels == 1)[0]
            
            if len(positive_classes) == 0:
                # 如果没有正类，随机选择一个目标类别
                target_class = random.choice(target_class_list)
            else:
                # 从正类中随机选择一个
                target_class = random.choice(positive_classes)
            
            # 确保目标类别在指定范围内
            if target_classes and target_class not in target_class_list:
                target_class = random.choice(target_class_list)
            
            # 生成GQSA Grad-CAM热图
            heatmaps = gradcam.generate_gqsa_gradcam(
                rgb_input=rgb_input,
                sonar_input=sonar_input,
                target_class=target_class,
                use_rgb_gqsa=config.use_rgb_gqsa,
                use_sonar_gqsa=False
            )
            
            if not heatmaps:
                continue
            
            # 准备原始图像用于可视化 - 从文件系统加载原始图像
            rgb_image_np = None
            sonar_image_np = None
            
            # 加载RGB原始图像
            if rgb_tensor is not None and 'rgb_gqsa' in heatmaps:
                rgb_image_path = os.path.join(dataset.rgb_root, sample_name)
                if os.path.exists(rgb_image_path):
                    original_rgb_pil = Image.open(rgb_image_path).convert('RGB')
                    rgb_image_np = np.array(original_rgb_pil)
                else:
                    # 后备方案：从tensor转换
                    rgb_image_np = tensor_to_image(rgb_tensor)
            
            # 加载Sonar原始图像
            if sonar_tensor is not None and 'sonar_gqsa' in heatmaps:
                sonar_image_path = os.path.join(dataset.sonar_root, sample_name)
                if os.path.exists(sonar_image_path):
                    original_sonar_pil = Image.open(sonar_image_path).convert('RGB')
                    sonar_image_np = np.array(original_sonar_pil)
                else:
                    # 后备方案：从tensor转换
                    sonar_image_np = tensor_to_image(sonar_tensor)
            
            # 保存Grad-CAM结果（调整热图尺寸以匹配原始图像）
            # 创建调整尺寸后的热图字典
            resized_heatmaps = {}
            for key, heatmap in heatmaps.items():
                if 'rgb' in key and rgb_image_np is not None:
                    target_size = (rgb_image_np.shape[1], rgb_image_np.shape[0])  # (W, H)
                    resized_heatmaps[key] = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)
                elif 'sonar' in key and sonar_image_np is not None:
                    target_size = (sonar_image_np.shape[1], sonar_image_np.shape[0])  # (W, H)
                    resized_heatmaps[key] = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)
                else:
                    resized_heatmaps[key] = heatmap
            
            # 保存Grad-CAM结果
            gradcam.save_gradcam_results(
                rgb_image=rgb_image_np,
                sonar_image=sonar_image_np,
                heatmaps=resized_heatmaps,
                class_names=config.class_names,
                target_class=target_class,
                save_path=output_dir,
                sample_name=f"{sample_name_clean}_class{target_class}"
            )
            
            generated_count += 1
            
        except Exception as e:
            print(f"\n⚠️ 生成样本 {sample_idx} 的热图失败: {e}")
            continue
    
    print(f"\n✅ Grad-CAM热图生成完成！")
    
    return {
        'generated_count': generated_count,
        'total_samples': len(sample_indices),
        'output_dir': output_dir
    }


def generate_tsne_visualization(model, dataset_dir, config, device,
                              samples_per_class=50, output_dir='tsne_visualizations',
                              perplexity=30.0, learning_rate=200.0, n_iter=1000,
                              random_state=42):
    """
    生成t-SNE特征可视化图
    
    Args:
        model: 训练好的模型
        dataset_dir: 数据集目录
        config: 配置对象
        device: 计算设备
        samples_per_class: 每个类别的样本数量
        output_dir: 输出目录
        perplexity: t-SNE perplexity参数
        learning_rate: t-SNE学习率
        n_iter: t-SNE迭代次数
        random_state: 随机种子
        
    Returns:
        生成结果字典
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import seaborn as sns
        from utils.multimodal_dataset import MultiModalClassificationDataset
        import pandas as pd
        
        print(f"🎯 开始生成t-SNE特征可视化...")
        print(f"🔧 t-SNE参数: perplexity={perplexity}, lr={learning_rate}, n_iter={n_iter}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建数据集
        val_scene_file = os.path.join(dataset_dir, 'val_scenelist.txt')
        test_scene_file = os.path.join(dataset_dir, 'test_scenelist.txt')
        
        if os.path.exists(test_scene_file):
            scene_file = test_scene_file
            dataset_type = "测试"
        elif os.path.exists(val_scene_file):
            scene_file = val_scene_file
            dataset_type = "验证"
        else:
            print("❌ 找不到数据集标签文件")
            return None
            
        dataset = MultiModalClassificationDataset(
            scene_list_file=scene_file,
            sonar_root=os.path.join(dataset_dir, 'sonar'),
            rgb_root=os.path.join(dataset_dir, 'rgb'),
            input_shape=config.input_shape,
            num_classes=config.num_classes,
            train=False
        )
        
        print(f"📊 使用{dataset_type}数据集，总样本数: {len(dataset)}")
        
        # 按类别收集样本索引
        class_sample_indices = {i: [] for i in range(config.num_classes)}
        
        print("🔍 按类别收集样本...")
        for idx in range(len(dataset)):
            _, _, labels_tensor = dataset[idx]
            labels_np = labels_tensor.numpy()
            # 对于多标签，我们选择第一个正标签作为主类别
            positive_classes = np.where(labels_np == 1)[0]
            if len(positive_classes) > 0:
                main_class = positive_classes[0]
                class_sample_indices[main_class].append(idx)
        
        # 每个类别选择指定数量的样本
        selected_indices = []
        selected_class_labels = []
        
        for class_idx in range(config.num_classes):
            available_samples = class_sample_indices[class_idx]
            if len(available_samples) == 0:
                print(f"⚠️ 类别 {class_idx} ({config.class_names[class_idx]}) 没有样本")
                continue
                
            # 随机选择样本
            selected_count = min(samples_per_class, len(available_samples))
            selected = random.sample(available_samples, selected_count)
            selected_indices.extend(selected)
            selected_class_labels.extend([class_idx] * selected_count)
            
            print(f"📝 类别 {class_idx} ({config.class_names[class_idx]}): 选择 {selected_count}/{len(available_samples)} 个样本")
        
        if len(selected_indices) == 0:
            print("❌ 没有选择到任何样本")
            return None
            
        print(f"🎲 总共选择 {len(selected_indices)} 个样本进行t-SNE分析")
        
        # 检查可用内存和限制样本数量
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        print(f"💾 可用内存: {available_memory_gb:.2f} GB")
        
        # 如果内存不足，减少样本数量
        if available_memory_gb < 2.0:
            max_total_samples = min(50, len(selected_indices))
            if len(selected_indices) > max_total_samples:
                print(f"⚠️ 内存不足，将样本数量从{len(selected_indices)}减少到{max_total_samples}")
                selected_indices = selected_indices[:max_total_samples]
                selected_class_labels = selected_class_labels[:max_total_samples]
        
        # 提取特征
        model.eval()
        rgb_features = []
        sonar_features = []
        
        print("🔍 开始提取特征向量...")
        
        # 分批处理以节省内存
        batch_size = 1  # 单个样本处理，减少内存使用
        
        with torch.no_grad():
            for i, sample_idx in enumerate(selected_indices):
                print(f"\r提取特征进度: {i+1}/{len(selected_indices)}", end="", flush=True)
                
                # 获取样本数据
                sonar_tensor, rgb_tensor, _ = dataset[sample_idx]
                
                # 转换为batch格式
                rgb_input = rgb_tensor.unsqueeze(0).to(device)
                sonar_input = sonar_tensor.unsqueeze(0).to(device)
                
                # 使用模型的前向传播提取特征
                try:
                    # 尝试获取对比学习特征
                    if hasattr(model, 'contrastive_branch') and getattr(model, 'contrastive_branch', False):
                        # 调用模型前向传播并获取对比学习嵌入
                        outputs = model(sonar_input, rgb_input, return_contrastive=True)
                        if len(outputs) >= 3:  # logits, sonar_embedding, rgb_embedding
                            sonar_feat = outputs[1]  # sonar_embedding
                            rgb_feat = outputs[2]    # rgb_embedding
                        else:
                            raise ValueError("无法获取对比学习特征")
                    else:
                        # 使用双骨干网络提取全局特征
                        if hasattr(model, 'dual_backbone'):
                            sonar_global, rgb_global = model.dual_backbone(sonar_input, rgb_input)
                            rgb_feat = rgb_global
                            sonar_feat = sonar_global
                        else:
                            print(f"\n⚠️ 无法识别模型结构: {type(model)}")
                            return None
                    
                    # 确保特征是2D的
                    if rgb_feat is not None:
                        if len(rgb_feat.shape) > 2:
                            rgb_feat = F.adaptive_avg_pool2d(rgb_feat, (1, 1)).flatten(1)
                        rgb_features.append(rgb_feat.cpu().numpy())
                    
                    if sonar_feat is not None:
                        if len(sonar_feat.shape) > 2:
                            sonar_feat = F.adaptive_avg_pool2d(sonar_feat, (1, 1)).flatten(1)
                        sonar_features.append(sonar_feat.cpu().numpy())
                        
                except Exception as e:
                    print(f"\n❌ 提取特征时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
                finally:
                    # 清理GPU内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        print(f"\n✅ 特征提取完成")
        
        # 检查是否有足够的特征
        if len(rgb_features) == 0 or len(sonar_features) == 0:
            print("❌ 特征提取失败，没有获得任何特征")
            return None
        
        # 转换为numpy数组
        rgb_features = np.vstack(rgb_features)
        sonar_features = np.vstack(sonar_features)
        
        print(f"🔢 RGB特征形状: {rgb_features.shape}")
        print(f"🔢 Sonar特征形状: {sonar_features.shape}")
        
        # 合并所有特征用于t-SNE
        all_features = np.vstack([rgb_features, sonar_features])
        
        # 创建标签
        all_class_labels = selected_class_labels * 2  # RGB和Sonar各一份
        all_modality_labels = ['RGB'] * len(selected_indices) + ['Sonar'] * len(selected_indices)
        
        print(f"🔢 合并特征形状: {all_features.shape}")
        
        # 检查特征有效性
        if np.any(np.isnan(all_features)) or np.any(np.isinf(all_features)):
            print("⚠️ 检测到NaN或Inf值，进行清理...")
            all_features = np.nan_to_num(all_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 确保有足够的样本进行t-SNE
        min_samples_required = 10
        if len(all_features) < min_samples_required:
            print(f"❌ 样本数量太少（{len(all_features)} < {min_samples_required}），无法进行t-SNE分析")
            return None
            
        print(f"🎯 开始t-SNE降维...")
        
        # 执行t-SNE
        # 确保max_iter满足sklearn的最小值要求（≥250）
        max_iter_value = max(n_iter, 250)
        if max_iter_value != n_iter:
            print(f"⚠️ 调整max_iter从{n_iter}到{max_iter_value}（sklearn要求最小250）")

        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, (len(all_features) - 1) / 3),  # 确保perplexity不会过大
            learning_rate=learning_rate,
            max_iter=max_iter_value,  # sklearn 1.7.1使用max_iter而不是n_iter
            random_state=random_state,
            verbose=1
        )
        
        tsne_results = tsne.fit_transform(all_features)
        
        print(f"✅ t-SNE降维完成")
        
        # 创建DataFrame用于可视化
        df = pd.DataFrame({
            'x': tsne_results[:, 0],
            'y': tsne_results[:, 1],
            'class': [config.class_names[label] for label in all_class_labels],
            'class_id': all_class_labels,
            'modality': all_modality_labels
        })
        
        # 设置绘图风格
        plt.style.use('default')
        
        # 创建单个图像
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # --- 绘制混合模态可视化图 ---
        print("🎨 生成混合模态t-SNE可视化图...")
        
        # 为每个类别分配颜色
        unique_classes = sorted(df['class_id'].unique())
        colors = plt.cm.get_cmap('tab10', len(unique_classes))
        class_color_map = {class_id: colors(i) for i, class_id in enumerate(unique_classes)}
        
        # 创建图例句柄
        legend_handles = []
        
        # 绘制RGB点（圆圈）
        for class_id in unique_classes:
            mask = (df['modality'] == 'RGB') & (df['class_id'] == class_id)
            if mask.any():
                ax.scatter(
                    df.loc[mask, 'x'], df.loc[mask, 'y'],
                    c=[class_color_map[class_id]], marker='o', s=50, alpha=0.8,
                    label=f'{config.class_names[class_id]}'
                )
        
        # 绘制Sonar点（叉号）
        for class_id in unique_classes:
            mask = (df['modality'] == 'Sonar') & (df['class_id'] == class_id)
            if mask.any():
                ax.scatter(
                    df.loc[mask, 'x'], df.loc[mask, 'y'],
                    c=[class_color_map[class_id]], marker='x', s=60, alpha=0.8
                    # Sonar不创建新标签，颜色与RGB共享
                )

        # 创建自定义图例
        # 类别图例
        class_legends = [plt.Line2D([0], [0], color=class_color_map[cid], lw=4, label=cname) 
                         for cid, cname in zip(unique_classes, [config.class_names[cid] for cid in unique_classes])]
        # 模态图例
        rgb_legend = plt.Line2D([0], [0], marker='o', color='grey', label='RGB', linestyle='None', markersize=10)
        sonar_legend = plt.Line2D([0], [0], marker='x', color='grey', label='Sonar', linestyle='None', markersize=10)
        
        # 合并图例并显示
        all_legends = class_legends + [rgb_legend, sonar_legend]
        ax.legend(handles=all_legends, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='medium')
        
        ax.set_title('t-SNE Visualization of RGB & Sonar Features', fontsize=18, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # 调整布局为图例留出空间
        
        # 保存图像
        save_path = os.path.join(output_dir, 't-sne_feature_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 t-SNE可视化图已保存: {save_path}")
        
        # 分析聚类质量
        print("\n📊 聚类质量分析:")
        
        # 计算同类别点之间的平均距离
        intra_class_distances = []
        inter_class_distances = []
        
        for class_id in unique_classes:
            class_mask = np.array(all_class_labels) == class_id
            class_points = tsne_results[class_mask]
            
            if len(class_points) > 1:
                # 类内距离
                from scipy.spatial.distance import pdist
                intra_distances = pdist(class_points)
                intra_class_distances.extend(intra_distances)
                
                # 类间距离（与其他类别的点）
                other_class_mask = ~class_mask
                other_points = tsne_results[other_class_mask]
                
                if len(other_points) > 0:
                    from scipy.spatial.distance import cdist
                    inter_distances = cdist(class_points, other_points)
                    inter_class_distances.extend(inter_distances.flatten())
        
        if intra_class_distances and inter_class_distances:
            avg_intra = np.mean(intra_class_distances)
            avg_inter = np.mean(inter_class_distances)
            silhouette_like = (avg_inter - avg_intra) / max(avg_inter, avg_intra)
            
            print(f"   平均类内距离: {avg_intra:.3f}")
            print(f"   平均类间距离: {avg_inter:.3f}")
            print(f"   聚类质量得分: {silhouette_like:.3f} (越大越好)")
        
        return {
            'tsne_results': tsne_results,
            'class_labels': all_class_labels,
            'modality_labels': all_modality_labels,
            'output_path': save_path,
            'csv_path': os.path.join(output_dir, 't-sne_data.csv'),
            'total_samples': len(selected_indices),
            'classes_analyzed': len(unique_classes)
        }
        
    except ImportError as e:
        print(f"❌ 缺少必要的库: {e}")
        print("请安装: pip install scikit-learn matplotlib seaborn pandas")
        return None
    except Exception as e:
        print(f"❌ 生成t-SNE可视化时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_gqsa_gradcam_heatmaps(model, dataset_dir, config, device, 
                                  num_samples=50, output_dir='gqsa_gradcam_results',
                                  target_stage=None):
    """
    生成GQSA特征的Grad-CAM热图
    
    Args:
        model: 训练好的模型
        dataset_dir: 数据集目录
        config: 配置对象
        device: 计算设备
        num_samples: 生成热图的样本数量
        output_dir: 输出目录
        
    Returns:
        生成结果字典
    """
    from utils.multimodal_dataset import MultiModalClassificationDataset
    
    print(f"🎯 初始化GQSA Grad-CAM...")
    
    # 检查GQSA配置
    if not config.use_rgb_gqsa:
        print("⚠️ 当前配置未启用RGB-GQSA，跳过GQSA热图生成")
        return None
    
    print(f"🔧 GQSA配置: RGB-GQSA已启用")
    if target_stage:
        print(f"   🎯 指定目标Stage: {target_stage}")

    # 初始化GQSA Grad-CAM
    gqsa_gradcam = GQSAGradCAM(model, device)
    
    # 创建数据集
    val_scene_file = os.path.join(dataset_dir, 'val_scenelist.txt')
    test_scene_file = os.path.join(dataset_dir, 'test_scenelist.txt')
    
    if os.path.exists(test_scene_file):
        scene_file = test_scene_file
    elif os.path.exists(val_scene_file):
        scene_file = val_scene_file
    else:
        print("❌ 找不到数据集标签文件")
        return None
    
    dataset = MultiModalClassificationDataset(
        scene_list_file=scene_file,
        sonar_root=os.path.join(dataset_dir, 'sonar'),
        rgb_root=os.path.join(dataset_dir, 'rgb'),
        input_shape=config.input_shape,
        num_classes=config.num_classes,
        train=False
    )
    
    print(f"📊 数据集大小: {len(dataset)}")
    
    # 选择前50个样本
    sample_indices = list(range(min(num_samples, len(dataset))))
    print(f"🎲 选择前 {len(sample_indices)} 个样本生成GQSA热图")
    
    generated_count = 0
    
    for i, sample_idx in enumerate(sample_indices):
        try:
            print(f"\r🔥 生成GQSA热图进度: {i+1}/{len(sample_indices)}", end="", flush=True)
            
            # 获取样本数据
            sonar_tensor, rgb_tensor, labels_tensor = dataset[sample_idx]
            
            # 转换为batch格式
            rgb_input = rgb_tensor.unsqueeze(0) if rgb_tensor is not None else None
            sonar_input = sonar_tensor.unsqueeze(0) if sonar_tensor is not None else None
            
            # 获取样本名称
            sample_name = dataset.samples[sample_idx][0]
            sample_name_clean = os.path.splitext(sample_name)[0]
            
            # 找到真实标签中的正类
            true_labels = labels_tensor.numpy()
            positive_classes = np.where(true_labels == 1)[0]
            
            if len(positive_classes) == 0:
                # 如果没有正类，随机选择一个目标类别
                target_class = random.choice(range(config.num_classes))
            else:
                # 从正类中随机选择一个
                target_class = random.choice(positive_classes)
            
            # 生成GQSA Grad-CAM热图
            heatmaps = gqsa_gradcam.generate_gqsa_gradcam(
                rgb_input=rgb_input,
                sonar_input=sonar_input,
                target_class=target_class,
                use_rgb_gqsa=config.use_rgb_gqsa,
                use_sonar_gqsa=False,
                target_stage=target_stage  # 传递新参数
            )
            
            if not heatmaps:
                continue
            
            # 准备原始图像用于可视化 - 从文件系统加载原始图像
            rgb_image_np = None
            sonar_image_np = None
            
            # 加载RGB原始图像
            if rgb_tensor is not None and 'rgb_gqsa' in heatmaps:
                rgb_image_path = os.path.join(dataset.rgb_root, sample_name)
                if os.path.exists(rgb_image_path):
                    original_rgb_pil = Image.open(rgb_image_path).convert('RGB')
                    rgb_image_np = np.array(original_rgb_pil)
                else:
                    # 后备方案：从tensor转换
                    rgb_image_np = tensor_to_image(rgb_tensor)
            
            # 加载Sonar原始图像
            if sonar_tensor is not None and 'sonar_gqsa' in heatmaps:
                sonar_image_path = os.path.join(dataset.sonar_root, sample_name)
                if os.path.exists(sonar_image_path):
                    original_sonar_pil = Image.open(sonar_image_path).convert('RGB')
                    sonar_image_np = np.array(original_sonar_pil)
                else:
                    # 后备方案：从tensor转换
                    sonar_image_np = tensor_to_image(sonar_tensor)
            
            # 保存RGB-GQSA热图
            if 'rgb_gqsa' in heatmaps and rgb_image_np is not None:
                save_path = os.path.join(
                    output_dir, 
                    f"{sample_name_clean}_class{target_class}_rgb_gqsa_gradcam.png"
                )
                # 将热图调整到与原始图像相同的尺寸
                target_size = (rgb_image_np.shape[1], rgb_image_np.shape[0])  # (W, H)
                heatmap_resized = cv2.resize(heatmaps['rgb_gqsa'], target_size, interpolation=cv2.INTER_LINEAR)
                
                gqsa_gradcam.save_gradcam_overlay(
                    rgb_image_np, 
                    heatmap_resized, 
                    save_path, 
                    'rgb_gqsa', 
                    sample_name_clean
                )
                print(f"💾 保存RGB-GQSA热图: {save_path}")
            
            generated_count += 1
            
        except Exception as e:
            print(f"\n⚠️ 生成样本 {sample_idx} 的GQSA热图失败: {e}")
            continue
    
    print(f"\n✅ GQSA Grad-CAM热图生成完成！")
    
    return {
        'generated_count': generated_count,
        'total_samples': len(sample_indices),
        'output_dir': output_dir
    }


if __name__ == '__main__':
    main()
