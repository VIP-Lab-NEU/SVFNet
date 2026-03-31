def compute_detailed_metrics(predictions, targets, threshold=0.5, class_names=None):
    """
    Compute detailed evaluation metrics for multi-label classification
    
    Args:
        predictions: Predicted probabilities, shape [N, num_classes], values in [0,1]
        targets: True labels, shape [N, num_classes], values 0 or 1
        threshold: Classification threshold
        class_names: List of class names
        
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    targets = targets.astype(int)
    
    pred_labels = (predictions >= threshold).astype(int)
    
    num_samples, num_classes = targets.shape
    
    exact_match = np.all(pred_labels == targets, axis=1).mean()
    
    hamming = hamming_loss(targets, pred_labels)
    
    jaccard_samples = []
    for i in range(num_samples):
        intersection = np.sum(pred_labels[i] & targets[i])
        union = np.sum(pred_labels[i] | targets[i])
        if union > 0:
            jaccard_samples.append(intersection / union)
        else:
            jaccard_samples.append(1.0)
    jaccard_sample_avg = np.mean(jaccard_samples)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, pred_labels, average=None, zero_division=0
    )
    
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    tp = np.sum((pred_labels == 1) & (targets == 1))
    fp = np.sum((pred_labels == 1) & (targets == 0))
    fn = np.sum((pred_labels == 0) & (targets == 1))
    
    micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    micro_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    try:
        class_ap = np.zeros(num_classes)
        class_auc = np.zeros(num_classes)
        
        for i in range(num_classes):
            if np.sum(targets[:, i]) > 0:
                ap = average_precision_score(targets[:, i], predictions[:, i])
                class_ap[i] = ap
                
                if len(np.unique(targets[:, i])) > 1:
                    auc = roc_auc_score(targets[:, i], predictions[:, i])
                    class_auc[i] = auc
        
        mean_ap = np.mean(class_ap[class_ap > 0]) if np.any(class_ap > 0) else 0.0
        mean_auc = np.mean(class_auc[class_auc > 0]) if np.any(class_auc > 0) else 0.0
    except Exception as e:
        print(f"Warning: Error computing AP/AUC: {e}")
        class_ap = np.zeros(num_classes)
        class_auc = np.zeros(num_classes)
        mean_ap = 0.0
        mean_auc = 0.0
    
    pred_positive_rate = np.mean(pred_labels)
    true_positive_rate = np.mean(targets)
    
    pred_labels_per_sample = np.sum(pred_labels, axis=1)
    true_labels_per_sample = np.sum(targets, axis=1)
    
    avg_pred_labels = np.mean(pred_labels_per_sample)
    avg_true_labels = np.mean(true_labels_per_sample)
    
    class_sensitivity = recall
    
    class_specificity = np.zeros(num_classes)
    for i in range(num_classes):
        tn = np.sum((pred_labels[:, i] == 0) & (targets[:, i] == 0))
        fp = np.sum((pred_labels[:, i] == 1) & (targets[:, i] == 0))
        class_specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    
    valid_sensitivity = class_sensitivity[class_sensitivity > 0]
    g_mean_sensitivity = np.power(np.prod(valid_sensitivity), 1.0/len(valid_sensitivity)) if len(valid_sensitivity) > 0 else 0.0
    
    class_g_mean = np.sqrt(class_sensitivity * class_specificity)
    g_mean_balanced = np.mean(class_g_mean)
    
    overall_sensitivity = micro_recall
    overall_specificity = np.sum((pred_labels == 0) & (targets == 0)) / np.sum(targets == 0) if np.sum(targets == 0) > 0 else 1.0
    g_mean_overall = np.sqrt(overall_sensitivity * overall_specificity)
    
    metrics = {
        'exact_match_ratio': exact_match,
        'hamming_loss': hamming,
        'jaccard_similarity': jaccard_sample_avg,
        
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        
        'mean_average_precision': mean_ap,
        'mean_auc': mean_auc,
        
        'g_mean_sensitivity': g_mean_sensitivity,
        'g_mean_balanced': g_mean_balanced,
        'g_mean_overall': g_mean_overall,
        
        'pred_positive_rate': pred_positive_rate,
        'true_positive_rate': true_positive_rate,
        'avg_pred_labels_per_sample': avg_pred_labels,
        'avg_true_labels_per_sample': avg_true_labels,
        
        'class_precision': precision,
        'class_recall': recall,
        'class_f1': f1,
        'class_support': support,
        'class_ap': class_ap,
        'class_auc': class_auc,
        'class_sensitivity': class_sensitivity,
        'class_specificity': class_specificity,
        'class_g_mean': class_g_mean,
        
        'predictions': predictions,
        'targets': targets,
        'pred_labels': pred_labels,
        'threshold': threshold,
        'num_samples': num_samples,
        'num_classes': num_classes
    }
    
    if class_names is not None:
        metrics['class_names'] = class_names
    
    return metrics


def print_detailed_metrics(metrics, class_names=None, compact=False):
    """
    Print detailed evaluation metrics
    
    Args:
        metrics: Metrics dictionary returned by compute_detailed_metrics
        class_names: List of class names
        compact: Whether to print compact version
    """
    if class_names is None:
        class_names = metrics.get('class_names', [f'Class_{i}' for i in range(metrics['num_classes'])])
    
    print(f"\n{'='*50}")
    print("Multi-label Classification Metrics")
    print(f"{'='*50}")
    print(f"Total samples: {metrics['num_samples']}, Classes: {metrics['num_classes']}")
    print(f"Threshold: {metrics['threshold']:.3f}")
    
    print(f"\nOverall Metrics:")
    print(f"  Exact Match Ratio (EMR): {metrics['exact_match_ratio']:.4f}")
    print(f"  Hamming Loss:            {metrics['hamming_loss']:.4f}")
    print(f"  Jaccard Similarity:      {metrics['jaccard_similarity']:.4f}")
    
    print(f"\nAverage Metrics:")
    print(f"  Macro - Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro - Recall:    {metrics['macro_recall']:.4f}")
    print(f"  Macro - F1 Score:  {metrics['macro_f1']:.4f}")
    print(f"  Micro - Precision: {metrics['micro_precision']:.4f}")
    print(f"  Micro - Recall:    {metrics['micro_recall']:.4f}")
    print(f"  Micro - F1 Score:  {metrics['micro_f1']:.4f}")
    
    print(f"\nOther Metrics:")
    print(f"  Average Precision (mAP): {metrics['mean_average_precision']:.4f}")
    print(f"  Average AUC:             {metrics['mean_auc']:.4f}")
    
    print(f"\nGeometric Mean Accuracy (G_mean):")
    print(f"  Sensitivity G-mean: {metrics['g_mean_sensitivity']:.4f}")
    print(f"  Balanced G-mean:    {metrics['g_mean_balanced']:.4f}")
    print(f"  Overall G-mean:     {metrics['g_mean_overall']:.4f}")
    
    print(f"\nLabel Distribution:")
    print(f"  True positive rate:       {metrics['true_positive_rate']:.4f}")
    print(f"  Predicted positive rate:  {metrics['pred_positive_rate']:.4f}")
    print(f"  Avg true labels per sample: {metrics['avg_true_labels_per_sample']:.2f}")
    print(f"  Avg pred labels per sample: {metrics['avg_pred_labels_per_sample']:.2f}")
    
    if not compact:
        print(f"\nPer-class Metrics:")
        print(f"{'Class':<12} {'Support':<8} {'AP':<8} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Specificity':<12} {'G_mean':<8}")
        print("-" * 89)
        
        for i, name in enumerate(class_names):
            support = int(metrics['class_support'][i])
            ap = metrics['class_ap'][i]
            precision = metrics['class_precision'][i]
            recall = metrics['class_recall'][i]
            f1 = metrics['class_f1'][i]
            specificity = metrics['class_specificity'][i]
            g_mean = metrics['class_g_mean'][i]
            
            print(f"{name:<12} {support:<8} {ap:<8.3f} {precision:<10.3f} {recall:<8.3f} {f1:<8.3f} {specificity:<12.3f} {g_mean:<8.3f}")
    else:
        print(f"\nKey Metrics: EMR={metrics['exact_match_ratio']:.3f} | "
              f"Macro F1={metrics['macro_f1']:.3f} | Micro F1={metrics['micro_f1']:.3f} | "
              f"G_mean={metrics['g_mean_balanced']:.3f}")


def analyze_prediction_errors(metrics, class_names=None, top_k=5):
    """
    Analyze prediction errors by class
    
    Args:
        metrics: Metrics dictionary
        class_names: List of class names
        top_k: Show top k error classes
    """
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(metrics['num_classes'])]
    
    targets = metrics['targets']
    pred_labels = metrics['pred_labels']
    
    false_positives = np.sum((pred_labels == 1) & (targets == 0), axis=0)
    false_negatives = np.sum((pred_labels == 0) & (targets == 1), axis=0)
    
    print(f"\nPrediction Error Analysis (Top {top_k}):")
    
    fp_indices = np.argsort(false_positives)[::-1][:top_k]
    print(f"\nMost false positives:")
    for i, idx in enumerate(fp_indices):
        print(f"    {i+1}. {class_names[idx]}: {false_positives[idx]} times")
    
    fn_indices = np.argsort(false_negatives)[::-1][:top_k]
    print(f"\n  Most false negatives:")
    for i, idx in enumerate(fn_indices):
        print(f"    {i+1}. {class_names[idx]}: {false_negatives[idx]} times")


def analyze_f1_fluctuation(metrics_history, window_size=5):
    """
    Analyze F1 score fluctuation
    
    Args:
        metrics_history: List of historical metrics
        window_size: Sliding window size
        
    Returns:
        dict: Fluctuation analysis results
    """
    if len(metrics_history) < window_size:
        return {'message': f'Need at least {window_size} data points'}
    
    recent_f1 = [m['macro_f1'] for m in metrics_history[-window_size:]]
    
    mean_f1 = np.mean(recent_f1)
    std_f1 = np.std(recent_f1)
    cv = std_f1 / mean_f1 if mean_f1 > 0 else float('inf')
    
    stability_score = max(0, 1 - cv)
    
    return {
        'recent_f1_scores': recent_f1,
        'mean': mean_f1,
        'std': std_f1,
        'coefficient_of_variation': cv,
        'stability_score': stability_score,
        'window_size': window_size
    }


def suggest_hyperparameter_adjustments(metrics_history, config=None):
    """
    Suggest hyperparameter adjustments based on historical metrics
    
    Args:
        metrics_history: List of historical metrics
        config: Current configuration object
        
    Returns:
        dict: Adjustment suggestions
    """
    if len(metrics_history) < 5:
        return {}
    
    suggestions = {}
    
    recent_f1 = [m['macro_f1'] for m in metrics_history[-5:]]
    f1_trend = np.polyfit(range(len(recent_f1)), recent_f1, 1)[0]
    
    recent_loss = [m.get('val_loss', 0) for m in metrics_history[-5:]]
    loss_trend = np.polyfit(range(len(recent_loss)), recent_loss, 1)[0]
    
    current_f1 = np.mean(recent_f1)
    
    if f1_trend < -0.01:
        suggestions['f1_declining'] = "F1 score declining, suggest reducing learning rate or increasing regularization"
    
    if current_f1 < 0.3:
        suggestions['low_f1'] = "Low F1 score, suggest adjusting classification threshold or checking data balance"
    
    if abs(loss_trend) < 0.001 and len(metrics_history) > 10:
        suggestions['loss_plateau'] = "Validation loss plateauing, may need to adjust learning rate schedule"
    
    if len(metrics_history) >= 3:
        train_losses = [m.get('train_loss', 0) for m in metrics_history[-3:]]
        val_losses = [m.get('val_loss', 0) for m in metrics_history[-3:]]
        if len(train_losses) > 0 and len(val_losses) > 0:
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            if avg_val_loss > avg_train_loss * 1.2:
                suggestions['overfitting'] = "Possible overfitting, suggest increasing dropout or data augmentation"
    
    return suggestions


def evaluate_threshold_sensitivity(predictions, targets, thresholds=None, class_names=None):
    """
    Evaluate model performance under different thresholds
    
    Args:
        predictions: Prediction probabilities
        targets: True labels
        thresholds: List of thresholds, defaults to [0.1, 0.2, ..., 0.9]
        class_names: Class names
        
    Returns:
        dict: Metrics under different thresholds
    """
    if thresholds is None:
        thresholds = [i * 0.1 for i in range(1, 10)]
    
    results = {}
    
    for threshold in thresholds:
        metrics = compute_detailed_metrics(predictions, targets, threshold, class_names)
        results[threshold] = {
            'exact_match_ratio': metrics['exact_match_ratio'],
            'macro_f1': metrics['macro_f1'],
            'micro_f1': metrics['micro_f1'],
            'hamming_loss': metrics['hamming_loss']
        }
    
    return results


def find_optimal_threshold(predictions, targets, metric='macro_f1', class_names=None):
    """
    Find optimal classification threshold
    
    Args:
        predictions: Prediction probabilities
        targets: True labels
        metric: Metric name to optimize ('macro_f1', 'micro_f1', 'exact_match_ratio')
        class_names: Class names
        
    Returns:
        tuple: (optimal_threshold, optimal_metric_value, all_results)
    """
    thresholds = [i * 0.05 for i in range(1, 20)]  # 0.05 to 0.95 with step 0.05
    
    best_threshold = 0.5
    best_score = 0
    all_results = {}
    
    for threshold in thresholds:
        metrics = compute_detailed_metrics(predictions, targets, threshold, class_names)
        score = metrics.get(metric, 0)
        all_results[threshold] = score
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score, all_results
