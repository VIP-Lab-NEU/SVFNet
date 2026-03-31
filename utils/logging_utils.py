import os
import sys
import datetime
from contextlib import contextmanager


class Tee:
    """Class for simultaneous output to console and file, filtering progress bars"""
    def __init__(self, *files, filter_progress=True):
        self.files = files
        self.filter_progress = filter_progress

    def write(self, obj):
        if self.files:
            console = self.files[0]
            console.write(obj)
            console.flush()

        if self.filter_progress and len(self.files) > 1:
            log_file = self.files[1]
            if not ('\r' in obj and '\n' not in obj):
                log_file.write(obj)
                log_file.flush()
        elif not self.filter_progress:
            for f in self.files[1:]:
                f.write(obj)
                f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def _is_progress_bar(self, text):
        """Detect if output is from progress bar"""
        return '\r' in text and '\n' not in text


class TrainingLogger:
    """Training logger manager"""
    
    def __init__(self, log_dir='work_logs', prefix='training'):
        self.log_dir = log_dir
        self.prefix = prefix
        self.log_file = None
        self.original_stdout = None
        self.original_stderr = None
        self.start_time = None
        
    def start_logging(self, model_size='tiny', additional_info=None):
        """Start logging"""
        os.makedirs(self.log_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        log_filename = f"{self.prefix}_{model_size}_{timestamp}.log"
        log_file_path = os.path.join(self.log_dir, log_filename)
        
        try:
            self.log_file = open(log_file_path, 'w', encoding='utf-8')
            
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr
            
            sys.stdout = Tee(sys.stdout, self.log_file, filter_progress=True)
            sys.stderr = Tee(sys.stderr, self.log_file, filter_progress=True)
            
            self.start_time = datetime.datetime.now()
            print(f"Log file: {log_file_path}")
            print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Model size: {model_size}")
            
            if additional_info:
                print(f"Additional info: {additional_info}")
            
            print("=" * 60)
            
            return log_file_path
            
        except Exception as e:
            print(f"Warning: Cannot create log file {log_file_path}: {e}")
            print("Continue with console output...")
            return None
    
    def stop_logging(self):
        """Stop logging"""
        if self.log_file:
            end_time = datetime.datetime.now()
            if self.start_time:
                duration = end_time - self.start_time
                print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Total training time: {duration}")
            
            print("=" * 60)
            print(f"Log saved to: {self.log_file.name}")
            
            if self.original_stdout:
                sys.stdout = self.original_stdout
            if self.original_stderr:
                sys.stderr = self.original_stderr
            
            self.log_file.close()
            self.log_file = None
    
    def log_config(self, config):
        """Log detailed configuration"""
        print("🔧 Training config:")
        print(f"   Config file: {getattr(config, 'config_path', 'config.yaml')}")
        print(f"   Dataset: {getattr(config, 'dataset_dir_train', 'data/train')}")
        print(f"   Classes: {getattr(config, 'num_classes', 'N/A')}")
        if hasattr(config, 'class_names'):
            print(f"   Class names: {config.class_names}")
        print(f"   Input shape: {getattr(config, 'input_shape', 'N/A')}")
        print(f"   Model size: {getattr(config, 'model_size', 'N/A')}")
        print(f"   Pretrained: {getattr(config, 'pretrained', 'N/A')}")
        if hasattr(config, 'local_weights_path'):
            print(f"   Local weights: {config.local_weights_path}")
        
        freeze_batch = getattr(config, 'Freeze_batch_size', getattr(config, 'freeze_batch_size', 'N/A'))
        unfreeze_batch = getattr(config, 'Unfreeze_batch_size', getattr(config, 'unfreeze_batch_size', 'N/A'))
        print(f"   Batch size: {freeze_batch} -> {unfreeze_batch}")
        
        init_lr = getattr(config, 'Init_lr', getattr(config, 'init_lr', 'N/A'))
        min_lr = getattr(config, 'Min_lr', getattr(config, 'min_lr', 'N/A'))
        print(f"   Learning rate: {init_lr} -> {min_lr}")
        
        if hasattr(config, 'backbone_lr'):
            print(f"   Backbone learning rate: {config.backbone_lr}")
        
        optimizer = getattr(config, 'optimizer_type', 'N/A')
        print(f"   Optimizer: {optimizer}")
        
        if hasattr(config, 'warmup_epochs'):
            print(f"   LR warmup epochs: {config.warmup_epochs}")
        
        lambda_cont = getattr(config, 'lambda_contrastive', 'N/A')
        print(f"   Contrastive learning weight: {lambda_cont}")
        
        if hasattr(config, 'lambda_contrastive_decay_enabled'):
            print(f"     Decay enabled: {config.lambda_contrastive_decay_enabled}")
            if config.lambda_contrastive_decay_enabled:
                print(f"     Decay epochs: {getattr(config, 'lambda_contrastive_decay_epochs', 'N/A')}")
                print(f"     Decay end value: {getattr(config, 'lambda_contrastive_end_value', 'N/A')}")
        
        lambda_cls = getattr(config, 'lambda_classification', 'N/A')
        print(f"   Classification loss weight: {lambda_cls}")
        
        if hasattr(config, 'lambda_sonar_modal'):
            print(f"   SONAR modality weight: {config.lambda_sonar_modal}")
        if hasattr(config, 'lambda_rgb_modal'):
            print(f"   RGB modality weight: {config.lambda_rgb_modal}")
        
        if hasattr(config, 'save_feature_heatmaps'):
            print(f"   Save feature heatmaps: {config.save_feature_heatmaps}")
        
        if hasattr(config, 'temperature'):
            print(f"   Temperature parameter: {config.temperature}")
        
        init_epoch = getattr(config, 'Init_Epoch', getattr(config, 'init_epoch', 0))
        freeze_epoch = getattr(config, 'Freeze_Epoch', getattr(config, 'freeze_epoch', 'N/A'))
        unfreeze_epoch = getattr(config, 'UnFreeze_Epoch', getattr(config, 'unfreeze_epoch', 'N/A'))
        print(f"   Training epochs: {init_epoch} -> {freeze_epoch} -> {unfreeze_epoch}")
        
        if hasattr(config, 'fp16'):
            print(f"   Mixed precision: {config.fp16}")
        
        if hasattr(config, 'early_stopping'):
            es_config = config.early_stopping
            if isinstance(es_config, dict):
                print(f"   Early stopping patience: {es_config.get('patience', 'N/A')}")
        
        if hasattr(config, 'baseline_mode'):
            print(f"   Baseline mode: {config.baseline_mode}")
            if hasattr(config, 'rgb_baseline'):
                print(f"     RGB baseline: {config.rgb_baseline}")
            if hasattr(config, 'sonar_baseline'):
                print(f"     SONAR baseline: {config.sonar_baseline}")
        
        if hasattr(config, 'fusion_mode'):
            print(f"   Feature fusion mode: {config.fusion_mode}")
            if hasattr(config, 'use_feature_concat'):
                print(f"     Feature concatenation: {config.use_feature_concat}")
            if hasattr(config, 'use_rgb_gqsa'):
                print(f"     RGB-GQSA: {config.use_rgb_gqsa}")
            if hasattr(config, 'use_sonar_gqsa'):
                print(f"     SONAR-GQSA: {config.use_sonar_gqsa}")
        
        if hasattr(config, 'gqsa_stages'):
            gqsa_config = config.gqsa_stages
            if isinstance(gqsa_config, dict):
                print(f"GQSA Multi-stage Fusion Config:")
                print(f"   Enable multi-stage: {gqsa_config.get('enable_multi_stage', False)}")
                print(f"   Fusion stages: {gqsa_config.get('stages', [])}")
                print(f"   Fusion method: {gqsa_config.get('fusion_method', 'N/A')}")
                print(f"   Stage weights: {gqsa_config.get('stage_weights', [])}")
                print(f"   Attention heads: {gqsa_config.get('num_heads', 'N/A')}")
                print(f"   Dropout rate: {gqsa_config.get('dropout', 'N/A')}")
        
        print("=" * 60)
    
    def log_epoch(self, epoch, total_epochs, train_loss, val_loss=None, metrics=None):
        """Log epoch information"""
        print(f"\nEpoch {epoch}/{total_epochs}:")
        print(f"   Training loss: {train_loss:.4f}")
        
        if val_loss is not None:
            print(f"   Validation loss: {val_loss:.4f}")
        
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
    
    def log_training_loss(self, epoch, losses_dict, learning_rate=None):
        """Log detailed training loss information"""
        loss_str = " | ".join([f"{k.capitalize()}: {v:.3f}" for k, v in losses_dict.items()])
        if learning_rate is not None:
            print(f"Epoch {epoch} Training Loss - {loss_str} | LR: {learning_rate:.6f}")
        else:
            print(f"Epoch {epoch} Training Loss - {loss_str}")
    
    def log_validation_results(self, epoch, val_loss, metrics_dict):
        """Log detailed validation results"""
        print(f"Epoch {epoch} Validation Results:")
        print(f"   Validation loss: {val_loss:.6f}")
        
        if 'exact_match' in metrics_dict:
            print(f"   Exact match ratio: {metrics_dict['exact_match']:.4f}")
        if 'macro_f1' in metrics_dict:
            print(f"   Macro F1: {metrics_dict['macro_f1']:.4f}")
        if 'micro_f1' in metrics_dict:
            print(f"   Micro F1: {metrics_dict['micro_f1']:.4f}")
        if 'hamming_loss' in metrics_dict:
            print(f"   Hamming loss: {metrics_dict['hamming_loss']:.4f}")
        
        if 'detailed_metrics' in metrics_dict:
            detailed = metrics_dict['detailed_metrics']
            if 'per_class_precision' in detailed and 'per_class_recall' in detailed:
                print("   Per-class detailed metrics:")
                class_names = detailed.get('class_names', [f'Class_{i}' for i in range(len(detailed['per_class_precision']))])
                for i, class_name in enumerate(class_names):
                    if i < len(detailed['per_class_precision']) and i < len(detailed['per_class_recall']):
                        precision = detailed['per_class_precision'][i]
                        recall = detailed['per_class_recall'][i]
                        f1 = detailed.get('per_class_f1', [0] * len(class_names))[i] if 'per_class_f1' in detailed else 0
                        print(f"     {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    def log_model_info(self, model_info):
        """Log model information"""
        print("Model Information:")
        if 'model_name' in model_info:
            print(f"   Model architecture: {model_info['model_name']}")
        if 'num_params' in model_info:
            print(f"   Number of parameters: {model_info['num_params']:,}")
        if 'trainable_params' in model_info:
            print(f"   Trainable parameters: {model_info['trainable_params']:,}")
        if 'frozen_params' in model_info:
            print(f"   Frozen parameters: {model_info['frozen_params']:,}")
        if 'model_size_mb' in model_info:
            print(f"   Model size: {model_info['model_size_mb']:.2f} MB")
        print("=" * 60)
    
    def log_dataset_info(self, dataset_info):
        """Log dataset information"""
        print("Dataset Information:")
        if 'train_samples' in dataset_info:
            print(f"   Training samples: {dataset_info['train_samples']}")
        if 'val_samples' in dataset_info:
            print(f"   Validation samples: {dataset_info['val_samples']}")
        if 'test_samples' in dataset_info:
            print(f"   Test samples: {dataset_info['test_samples']}")
        if 'class_distribution' in dataset_info:
            print(f"   Class distribution: {dataset_info['class_distribution']}")
        print("=" * 60)
    
    def log_optimizer_info(self, optimizer_info):
        """Log optimizer information"""
        print("Optimizer Parameter Groups:")
        if 'param_groups' in optimizer_info:
            for group_name, group_info in optimizer_info['param_groups'].items():
                params_count = group_info.get('params_count', 'N/A')
                lr = group_info.get('lr', 'N/A')
                print(f"   {group_name}: {params_count} params, learning rate: {lr}")
        elif 'total_params' in optimizer_info:
            print(f"   Total parameters: {optimizer_info['total_params']} params")
            if 'lr' in optimizer_info:
                print(f"   Learning rate: {optimizer_info['lr']}")
        print("=" * 60)
    
    def log_best_model(self, best_score, model_path):
        """Log best model information"""
        print(f"New best model!")
        print(f"Best score: {best_score:.4f}")
        print(f"Model path: {model_path}")


@contextmanager
def training_logger(log_dir='work_logs', prefix='training', model_size='tiny', additional_info=None):
    """Training logger context manager"""
    logger = TrainingLogger(log_dir, prefix)
    try:
        logger.start_logging(model_size, additional_info)
        yield logger
    finally:
        logger.stop_logging()


def create_log_filename(prefix, model_size, timestamp=None):
    """Create log filename"""
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{model_size}_{timestamp}.log"


def get_log_path(log_dir, filename):
    """Get complete log file path"""
    return os.path.join(log_dir, filename)


if __name__ == "__main__":
    with training_logger(prefix='test', model_size='tiny') as logger:
        print("This is a test log")
        logger.log_config(type('Config', (), {
            'model_size': 'tiny',
            'input_shape': [224, 224],
            'num_classes': 10,
            'Unfreeze_batch_size': 16,
            'Init_lr': 1e-3,
            'lambda_contrastive': 1.0,
            'UnFreeze_Epoch': 100
        })())
    
    logger = TrainingLogger(prefix='manual_test')
    log_path = logger.start_logging('small', 'Manual test')
    print("Manual test log")
    logger.stop_logging()
