from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, StepLR
from torch.optim import Optimizer

class GradualWarmupScheduler(_LRScheduler):
    """
    Learning rate warmup scheduler that linearly increases LR from a low value to initial LR during the first N epochs.
    After warmup, switches to another scheduler (e.g., CosineAnnealingLR).
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        """
        Args:
            optimizer (Optimizer): Optimizer
            multiplier (float): LR will be multiplied by this factor after warmup. Usually set to 1.0
            total_epoch (int): Total warmup epochs
            after_scheduler (Scheduler): Scheduler to use after warmup
        """
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater than or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        """Compute learning rate for current epoch"""
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Update learning rate"""
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

def get_lr_scheduler(optimizer: Optimizer,
                     lr_decay_type: str,
                     total_epochs: int,
                     warmup_epochs: int = 0,
                     eta_min: float = 0):
    """
    Create learning rate scheduler with warmup support
    
    Args:
        optimizer: Optimizer
        lr_decay_type: LR decay type ('cos' or 'step')
        total_epochs: Total training epochs
        warmup_epochs: Warmup epochs
        eta_min: Minimum learning rate (for 'cos' scheduler)
        
    Returns:
        Learning rate scheduler
    """
    if lr_decay_type == 'cos':
        after_scheduler = CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min
        )
    elif lr_decay_type == 'step':
        after_scheduler = StepLR(
            optimizer, step_size=int(total_epochs * 0.3), gamma=0.1
        )
    else:
        raise ValueError(f"Unsupported LR decay type: {lr_decay_type}")
    
    if warmup_epochs > 0:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1.0,
            total_epoch=warmup_epochs,
            after_scheduler=after_scheduler
        )
    else:
        scheduler = after_scheduler
        
    return scheduler 