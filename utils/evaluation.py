#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene-level multi-modal classification evaluation functions
Supports independent evaluation and validation during training
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import random

from utils.multimodal_dataset import MultiModalClassificationDataset, multimodal_collate_fn
from utils.multimodal_metrics import (
    compute_detailed_metrics, 
    print_detailed_metrics, 
    analyze_prediction_errors,
    analyze_f1_fluctuation,
    suggest_hyperparameter_adjustments,
    find_optimal_threshold
)
from utils.utils import worker_init_fn
from utils.feature_heatmap import FeatureHeatmapGenerator


def evaluate_multimodal_model(model, dataset_dir, config, device=None, 
                            save_predictions=False, output_dir=None,
                            find_best_threshold=False, evaluate_separate=False):
    """
    Independent multi-modal model evaluation function
    
    Args:
        model: Trained multi-modal model
        dataset_dir: Validation dataset directory
        config: Configuration object with model and dataset settings
        device: Device (cuda/cpu)
        save_predictions: Whether to save predictions
        output_dir: Output directory
        find_best_threshold: Whether to find optimal threshold
        evaluate_separate: Whether to separately evaluate RGB and SONAR classification heads
        
    Returns:
        dict: Detailed evaluation results
    """
    print("🔍 Starting multi-modal scene-level classification evaluation...")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model = model.to(device)
    
    print("📊 Creating validation dataset...")
    val_scene_file = os.path.join(dataset_dir, 'val_scenelist.txt')
    test_scene_file = os.path.join(dataset_dir, 'test_scenelist.txt')

    if os.path.exists(test_scene_file):
        scene_file = test_scene_file
        dataset_type = "Test"
    elif os.path.exists(val_scene_file):
        scene_file = val_scene_file
        dataset_type = "Val"
    else:
        raise FileNotFoundError(f"Label file not found: {val_scene_file} or {test_scene_file}")

    val_dataset = MultiModalClassificationDataset(
        scene_list_file=scene_file,
        sonar_root=os.path.join(dataset_dir, 'sonar'),
        rgb_root=os.path.join(dataset_dir, 'rgb'),
        input_shape=config.input_shape,
        num_classes=config.num_classes,
        train=False,
        baseline_mode=getattr(config, 'baseline_mode', 'multimodal')
    )
    
    print(f"📈 {dataset_type} samples: {len(val_dataset)}")
    
    worker_init_fn_partial = partial(worker_init_fn, rank=0, seed=getattr(config, 'seed', 42))
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=getattr(config, 'Unfreeze_batch_size', 16),
        shuffle=False,
        num_workers=getattr(config, 'num_workers', 4),
        pin_memory=True,
        collate_fn=multimodal_collate_fn,
        worker_init_fn=worker_init_fn_partial
    )
    
    all_predictions = []
    all_targets = []
    all_filenames = []
    all_rgb_predictions = []
    all_sonar_predictions = []
    total_loss = 0
    total_cls_loss = 0
    total_cont_loss = 0
    
    print("🚀 Starting inference...")
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Evaluation', 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                    ncols=100)
        
        for batch_idx, (sonar_imgs, rgb_imgs, targets) in enumerate(pbar):
            if sonar_imgs is not None:
                sonar_imgs = sonar_imgs.to(device)
            if rgb_imgs is not None:
                rgb_imgs = rgb_imgs.to(device)
            targets = targets.to(device)

            try:
                if hasattr(model, 'compute_loss'):
                    lambda_cont = getattr(config, 'lambda_contrastive', 0.0)
                    lambda_cls = getattr(config, 'lambda_classification', 1.0)
                    lambda_sonar = getattr(config, 'lambda_sonar_modal', 0.0)
                    lambda_rgb = getattr(config, 'lambda_rgb_modal', 0.0)
                    loss, cls_loss, cont_loss, sonar_modal_loss, rgb_modal_loss = model.compute_loss(
                        sonar_imgs, rgb_imgs, targets, lambda_cont, lambda_cls,
                        lambda_sonar, lambda_rgb
                    )
                    total_loss += loss.item()
                    total_cls_loss += cls_loss.item()
                    total_cont_loss += cont_loss.item()
            except Exception as e:
                print(f"Warning: Cannot compute loss: {e}")
            
            baseline_mode = getattr(config, 'baseline_mode', 'multimodal')
            if baseline_mode == 'rgb_only':
                if rgb_imgs is None:
                    raise ValueError("RGB images cannot be None in rgb_only mode")
                logits = model(None, rgb_imgs)
                probs = torch.sigmoid(logits)
                all_predictions.append(probs.cpu())
                all_targets.append(targets.cpu())
            elif baseline_mode == 'sonar_only':
                if sonar_imgs is None:
                    raise ValueError("SONAR images cannot be None in sonar_only mode")
                logits = model(sonar_imgs, None)
                probs = torch.sigmoid(logits)
                all_predictions.append(probs.cpu())
                all_targets.append(targets.cpu())
            else:
                logits = model(sonar_imgs, rgb_imgs)
                probs = torch.sigmoid(logits)
                all_predictions.append(probs.cpu())
                all_targets.append(targets.cpu())
            

            if save_predictions:
                batch_start = batch_idx * val_loader.batch_size
                batch_end = min(batch_start + val_loader.batch_size, len(val_dataset))
                for i in range(batch_start, batch_end):
                    filename = val_dataset.samples[i][0]
                    all_filenames.append(filename)
    

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    if evaluate_separate and all_rgb_predictions and all_sonar_predictions:
        all_rgb_predictions = torch.cat(all_rgb_predictions, dim=0)
        all_sonar_predictions = torch.cat(all_sonar_predictions, dim=0)
        print(f" Inference complete! Processed {len(all_predictions)} samples with RGB/SONAR predictions")
    else:
        print(f"Inference complete! Processed {len(all_predictions)} samples")
    
    optimal_threshold = None
    if find_best_threshold:
        print("Finding optimal threshold...")
        optimal_threshold, best_score, threshold_results = find_optimal_threshold(
            all_predictions, all_targets, metric='macro_f1', 
            class_names=config.class_names
        )
        print(f"  Optimal threshold: {optimal_threshold:.3f} (F1 score: {best_score:.4f})")
        
        print("  Threshold sensitivity:")
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            if thresh in threshold_results:
                print(f"    Thresh {thresh:.1f}: F1={threshold_results[thresh]:.4f}")
    
    eval_threshold = optimal_threshold if optimal_threshold else getattr(config, 'eval_threshold', 0.5)
    
    print(f"\nComputing metrics with threshold {eval_threshold:.3f}...")
    detailed_metrics = compute_detailed_metrics(
        all_predictions, all_targets,
        threshold=eval_threshold,
        class_names=config.class_names
    )
    
    print("="*60)
    print("Fusion Classification Head Results")
    print("="*60)
    print_detailed_metrics(detailed_metrics, config.class_names, compact=False)
    
    analyze_prediction_errors(detailed_metrics, config.class_names, top_k=5)
    
    rgb_metrics = None
    sonar_metrics = None
    
    if evaluate_separate and all_rgb_predictions is not None and all_sonar_predictions is not None:
        print("\n" + "="*60)
        print("RGB Classification Head Results")
        print("="*60)
        rgb_metrics = compute_detailed_metrics(
            all_rgb_predictions, all_targets,
            threshold=eval_threshold,
            class_names=config.class_names
        )
        print_detailed_metrics(rgb_metrics, config.class_names, compact=False)
        analyze_prediction_errors(rgb_metrics, config.class_names, top_k=3)
        
        print("\n" + "="*60)
        print("SONAR Classification Head Results")
        print("="*60)
        sonar_metrics = compute_detailed_metrics(
            all_sonar_predictions, all_targets,
            threshold=eval_threshold,
            class_names=config.class_names
        )
        print_detailed_metrics(sonar_metrics, config.class_names, compact=False)
        analyze_prediction_errors(sonar_metrics, config.class_names, top_k=3)
        
        print("\n" + "="*60)
        print("Classification Head Performance Comparison")
        print("="*60)
        print(f"{'Metric':<20} {'Fusion':<15} {'RGB':<15} {'SONAR':<15}")
        print("-" * 65)
        print(f"{'EMR':<20} {detailed_metrics['exact_match_ratio']:<15.4f} {rgb_metrics['exact_match_ratio']:<15.4f} {sonar_metrics['exact_match_ratio']:<15.4f}")
        print(f"{'Macro F1':<20} {detailed_metrics['macro_f1']:<15.4f} {rgb_metrics['macro_f1']:<15.4f} {sonar_metrics['macro_f1']:<15.4f}")
        print(f"{'Micro F1':<20} {detailed_metrics['micro_f1']:<15.4f} {rgb_metrics['micro_f1']:<15.4f} {sonar_metrics['micro_f1']:<15.4f}")
        print(f"{'Hamming Loss':<20} {detailed_metrics['hamming_loss']:<15.4f} {rgb_metrics['hamming_loss']:<15.4f} {sonar_metrics['hamming_loss']:<15.4f}")
    
    avg_loss = total_loss / len(val_loader) if total_loss > 0 else 0
    avg_cls_loss = total_cls_loss / len(val_loader) if total_cls_loss > 0 else 0
    avg_cont_loss = total_cont_loss / len(val_loader) if total_cont_loss > 0 else 0
    
    results = {
        'metrics': detailed_metrics,
        'avg_loss': avg_loss,
        'avg_cls_loss': avg_cls_loss,
        'avg_cont_loss': avg_cont_loss,
        'optimal_threshold': optimal_threshold,
        'predictions': all_predictions,
        'targets': all_targets
    }
    
    if evaluate_separate:
        results.update({
            'rgb_metrics': rgb_metrics,
            'sonar_metrics': sonar_metrics,
            'rgb_predictions': all_rgb_predictions if 'all_rgb_predictions' in locals() else None,
            'sonar_predictions': all_sonar_predictions if 'all_sonar_predictions' in locals() else None
        })
    
    if save_predictions and output_dir:
        save_predictions_to_file(
            all_predictions, all_targets, all_filenames, 
            config.class_names, eval_threshold, output_dir
        )
    
    return results


def validate_epoch_improved(model, dataloader, config, epoch, device=None):
    """
    Improved validation function for use during training
    
    Args:
        model: Model
        dataloader: Validation dataloader
        config: Configuration object
        epoch: Current epoch
        device: Device
        
    Returns:
        tuple: (val_loss, exact_match_ratio, macro_f1)
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_cont_loss = 0
    all_preds = []
    all_targets = []
    all_rgb_preds = []
    all_sonar_preds = []
    
    heatmap_generator = None
    heatmap_saved_count = 0
    if getattr(config, 'save_feature_heatmaps', False):
        use_gqsa = getattr(config, 'use_rgb_gqsa', False) or getattr(config, 'use_sonar_gqsa', False)
        if use_gqsa:
            heatmap_generator = FeatureHeatmapGenerator(
                threshold=getattr(config, 'heatmap_threshold', 0.05),
                save_dir="feature_hot_image"
            )
            print(f"Feature heatmap enabled (threshold: {config.heatmap_threshold}, max count: {config.heatmap_save_count})")
    
    print(f"\nStarting Epoch {epoch} validation...")
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Validation Epoch {epoch}',
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                    ncols=80, file=sys.stdout)
        
        for batch_idx, (sonar_imgs, rgb_imgs, targets) in enumerate(pbar):
            if sonar_imgs is not None:
                sonar_imgs = sonar_imgs.to(device)
            if rgb_imgs is not None:
                rgb_imgs = rgb_imgs.to(device)
            targets = targets.to(device)
            try:
                if hasattr(model, 'compute_loss'):
                    total_loss_batch, cls_loss, cont_loss, sonar_loss, rgb_loss = model.compute_loss(
                        sonar_imgs, rgb_imgs, targets, 
                        getattr(config, 'lambda_contrastive', 0.0),
                        getattr(config, 'lambda_classification', 1.0),
                        getattr(config, 'lambda_sonar_modal', 0.0),
                        getattr(config, 'lambda_rgb_modal', 0.0)
                    )
                    total_loss += total_loss_batch.item()
                    total_cls_loss += cls_loss.item()
                    total_cont_loss += cont_loss.item()
            except Exception as e:
                print(f"Warning: compute loss failed: {e}")
                total_loss_batch = torch.tensor(0.0)
                cls_loss = torch.tensor(0.0)
                cont_loss = torch.tensor(0.0)
            
            baseline_mode = getattr(config, 'baseline_mode', 'multimodal')
            if baseline_mode == 'rgb_only':
                if rgb_imgs is None:
                    raise ValueError("RGB images cannot be None in rgb_only mode")
                logits = model(None, rgb_imgs)
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu())
                all_targets.append(targets.cpu())
            elif baseline_mode == 'sonar_only':
                if sonar_imgs is None:
                    raise ValueError("SONAR images cannot be None in sonar_only mode")
                logits = model(sonar_imgs, None)
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu())
                all_targets.append(targets.cpu())
            else:
                if (heatmap_generator is not None and 
                    heatmap_saved_count < getattr(config, 'heatmap_save_count', 5) and
                    random.random() < 0.3):
                    
                    result = model(sonar_imgs, rgb_imgs, return_heatmap_features=True)
                    
                    if isinstance(result, tuple) and len(result) >= 2:
                        logits = result[0]
                        heatmap_info = result[-1]
                        if heatmap_info is not None:
                            mode = heatmap_info.get('mode', 'unknown')
                            if mode in ['rgb_gqsa', 'sonar_gqsa']:
                                batch_size = heatmap_info['original_image'].size(0)
                                save_count = min(batch_size, 
                                               getattr(config, 'heatmap_save_count', 5) - heatmap_saved_count)
                                
                                if save_count > 0:
                                    print(f"\n🎨 Saving {save_count} {mode.upper()} heatmaps...")
                                    heatmap_generator.save_heatmap_batch(
                                        heatmap_info, batch_idx, epoch, max_count=save_count
                                    )
                                    heatmap_saved_count += save_count
                    else:
                        logits = result if not isinstance(result, tuple) else result[0]
                else:
                    logits = model(sonar_imgs, rgb_imgs)

                probs = torch.sigmoid(logits)
                
                all_preds.append(probs.cpu())
                all_targets.append(targets.cpu())
            current_avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'Loss': f'{total_loss_batch.item():.3f}',
                'AvgLoss': f'{current_avg_loss:.3f}',
                'Cls': f'{cls_loss.item():.3f}',
                'Cont': f'{cont_loss.item():.3f}'
            })
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    eval_threshold = getattr(config, 'eval_threshold', 0.5)
    detailed_metrics = compute_detailed_metrics(
        all_preds, all_targets,
        threshold=eval_threshold,
        class_names=config.class_names
    )
    
    print("="*50)
    print("Multi-modal Fusion Classification Results")
    print("="*50)
    print_detailed_metrics(detailed_metrics, config.class_names, compact=False)
    
    if hasattr(config, 'metrics_history'):
        config.metrics_history.append({
            'epoch': epoch,
            'val_loss': total_loss / len(dataloader),
            'cls_loss': total_cls_loss / len(dataloader),
            'cont_loss': total_cont_loss / len(dataloader),
            **detailed_metrics
        })
        
        if len(config.metrics_history) >= 5:
            fluctuation_analysis = analyze_f1_fluctuation(config.metrics_history, window_size=5)
            if fluctuation_analysis.get('coefficient_of_variation', 0) > 0.1:
                suggestions = suggest_hyperparameter_adjustments(config.metrics_history, config)
                if suggestions:
                    print(f"\n💡 Hyperparameter adjustment suggestions:")
                    for key, suggestion in suggestions.items():
                        print(f"   {suggestion}")
    
    avg_loss = total_loss / len(dataloader)
    exact_match = detailed_metrics['exact_match_ratio']
    macro_f1 = detailed_metrics['macro_f1']
    
    return avg_loss, exact_match, macro_f1


def save_predictions_to_file(predictions, targets, filenames, class_names, 
                           threshold, output_dir):
    """
    Save predictions to file
    
    Args:
        predictions: Prediction probability matrix [N, num_classes]
        targets: True label matrix [N, num_classes] 
        filenames: List of filenames
        class_names: List of class names
        threshold: Classification threshold
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    pred_labels = (predictions >= threshold).astype(int)
    
    output_file = os.path.join(output_dir, 'evaluation_results.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Scene-level Multi-modal Classification Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Number of samples: {len(predictions)}\n")
        f.write(f"Number of classes: {len(class_names)}\n")
        f.write(f"Classification threshold: {threshold}\n")
        f.write(f"Class names: {', '.join(class_names)}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Detailed prediction results:\n")
        f.write(f"{'Filename':<30} {'True Labels':<30} {'Predicted Labels':<30} {'Prediction Probabilities'}\n")
        f.write("-" * 120 + "\n")
        
        for i, filename in enumerate(filenames):
            true_labels = [class_names[j] for j in range(len(class_names)) if targets[i, j] == 1]
            true_labels_str = ','.join(true_labels) if true_labels else 'None'
            
            pred_labels_list = [class_names[j] for j in range(len(class_names)) if pred_labels[i, j] == 1]
            pred_labels_str = ','.join(pred_labels_list) if pred_labels_list else 'None'
            
            prob_str = ','.join([f'{prob:.3f}' for prob in predictions[i]])
            
            f.write(f"{filename:<30} {true_labels_str:<30} {pred_labels_str:<30} {prob_str}\n")
    
    print(f"Predictions saved to: {output_file}")
    
    np.save(os.path.join(output_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(output_dir, 'targets.npy'), targets)
    print(f"Prediction data saved to: {output_dir}/predictions.npy and targets.npy")
