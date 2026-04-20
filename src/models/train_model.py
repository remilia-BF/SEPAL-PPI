"""
Training utilities for SEPAL-PPI models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import logging
import math
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from ..utils.helpers import get_device

logger = logging.getLogger(__name__)


class WarmupThenReduceLROnPlateau:
    """
    Warmup for the first N epochs, then switch to ReduceLROnPlateau driven by a validation metric.

    - During warmup, LR scales from initial_lr_factor * base_lr to base_lr with either linear or exponential strategy
    - After warmup, uses torch.optim.lr_scheduler.ReduceLROnPlateau with user-configured params

    Call step(val_metric) once per epoch.
    """
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        initial_lr_factor: float = 0.1,
        warmup_strategy: str = 'exponential',
        mode: str = 'max',
        factor: float = 0.5,
        patience: int = 4,
        threshold: float = 1e-4,
        threshold_mode: str = 'rel',
        cooldown: int = 0,
        min_lr_factor: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.initial_lr_factor = initial_lr_factor
        self.warmup_strategy = warmup_strategy
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        # Compute absolute min_lrs by scaling base lrs
        self.min_lrs = [lr * min_lr_factor for lr in self.base_lrs]
        self.epoch_idx = 0

        # Plateau scheduler for post-warmup
        self._plateau = lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=self.min_lrs,
        )

    def _compute_warmup_factor(self, epoch: int) -> float:
        if self.warmup_epochs <= 0:
            return 1.0
        progress = epoch / self.warmup_epochs
        progress = max(0.0, min(1.0, progress))
        if self.warmup_strategy == 'exponential':
            # grow from initial_lr_factor to 1.0 exponentially
            return self.initial_lr_factor + (1.0 - self.initial_lr_factor) * (2 ** progress - 1) / (2 ** 1 - 1)
        else:
            # linear
            return self.initial_lr_factor + (1.0 - self.initial_lr_factor) * progress

    def _apply_factor(self, factor: float) -> None:
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group['lr'] = base_lr * factor

    def step(self, metric: Optional[float] = None) -> None:
        """
        Step scheduler at end of epoch. Pass metric when available.
        """
        if self.epoch_idx < self.warmup_epochs:
            factor = self._compute_warmup_factor(self.epoch_idx + 1)
            self._apply_factor(factor)
        else:
            # After warmup, use plateau scheduler; requires a metric
            if metric is None:
                # Gracefully fallback if metric is missing
                self._plateau.step(0.0)
            else:
                self._plateau.step(metric)

        self.epoch_idx += 1

    def get_last_lr(self) -> List[float]:
        # Return current learning rates from optimizer param groups
        return [group['lr'] for group in self.optimizer.param_groups]


class WarmupCosineAnnealingLR(lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with warmup and cosine annealing
    """
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, 
                 min_lr_factor: float = 0.01, initial_lr_factor: float = 0.1, 
                 warmup_strategy: str = 'exponential', last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_factor = min_lr_factor
        self.initial_lr_factor = initial_lr_factor
        self.warmup_strategy = warmup_strategy
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            if self.warmup_strategy == 'exponential':
                # Exponential warmup: start from initial_lr_factor and exponentially grow to 1.0
                progress = self.last_epoch / self.warmup_epochs
                factor = self.initial_lr_factor + (1.0 - self.initial_lr_factor) * (2 ** progress - 1) / (2 ** 1 - 1)
            else:
                # Linear warmup: start from initial_lr_factor and linearly grow to 1.0
                progress = self.last_epoch / self.warmup_epochs
                factor = self.initial_lr_factor + (1.0 - self.initial_lr_factor) * progress
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            factor = self.min_lr_factor + (1.0 - self.min_lr_factor) * 0.5 * (1 + math.cos(math.pi * progress))
        
        return [base_lr * factor for base_lr in self.base_lrs]


class WarmupExponentialDecayLR(lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with warmup and exponential decay
    """
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, 
                 gamma: float = 0.95, step_size: int = 5, initial_lr_factor: float = 0.1, 
                 warmup_strategy: str = 'exponential', last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.gamma = gamma
        self.step_size = step_size
        self.initial_lr_factor = initial_lr_factor
        self.warmup_strategy = warmup_strategy
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            if self.warmup_strategy == 'exponential':
                # Exponential warmup: start from initial_lr_factor and exponentially grow to 1.0
                progress = self.last_epoch / self.warmup_epochs
                factor = self.initial_lr_factor + (1.0 - self.initial_lr_factor) * (2 ** progress - 1) / (2 ** 1 - 1)
            else:
                # Linear warmup: start from initial_lr_factor and linearly grow to 1.0
                progress = self.last_epoch / self.warmup_epochs
                factor = self.initial_lr_factor + (1.0 - self.initial_lr_factor) * progress
        else:
            # Exponential decay phase
            decay_epochs = self.last_epoch - self.warmup_epochs
            decay_steps = decay_epochs // self.step_size
            factor = self.gamma ** decay_steps
        
        return [base_lr * factor for base_lr in self.base_lrs]


def create_lr_scheduler(optimizer, config: Dict) -> Optional[Any]:
    """
    Factory function to create learning rate scheduler
    
    Args:
        optimizer: Optimizer
        config (Dict): Scheduler configuration
        
    Returns:
        Optional[Any]: Configured scheduler or None
    """
    if not config:
        return None
    
    scheduler_type = config.get('type', 'none')
    total_epochs = config.get('total_epochs', 100)
    warmup_config = config.get('warmup', {})
    warmup_enabled = warmup_config.get('enabled', False)
    warmup_epochs = warmup_config.get('epochs', 0) if warmup_enabled else 0
    initial_lr_factor = warmup_config.get('initial_lr_factor', 0.1)
    warmup_strategy = warmup_config.get('strategy', 'exponential')

    # Support "warmup only" when scheduler type is none
    if scheduler_type == 'none':
        if warmup_enabled and warmup_epochs > 0:
            class WarmupHoldLR(lr_scheduler._LRScheduler):
                """
                Warm up LR for N epochs, then hold at base LR (no further scheduling).

                This allows using a fixed learning rate overall while still benefiting
                from a short warmup phase at the beginning of training.
                """
                def __init__(self, optimizer, warmup_epochs: int, initial_lr_factor: float = 0.1,
                             warmup_strategy: str = 'exponential', last_epoch: int = -1):
                    self.warmup_epochs = int(max(0, warmup_epochs))
                    self.initial_lr_factor = float(initial_lr_factor)
                    self.warmup_strategy = str(warmup_strategy)
                    self.base_lrs = [group['lr'] for group in optimizer.param_groups]
                    super().__init__(optimizer, last_epoch)

                def get_lr(self) -> List[float]:
                    if self.warmup_epochs <= 0:
                        factor = 1.0
                    elif self.last_epoch < self.warmup_epochs:
                        progress = self.last_epoch / self.warmup_epochs
                        if self.warmup_strategy == 'exponential':
                            factor = self.initial_lr_factor + (1.0 - self.initial_lr_factor) * (2 ** progress - 1) / (2 - 1)
                        else:
                            factor = self.initial_lr_factor + (1.0 - self.initial_lr_factor) * progress
                    else:
                        factor = 1.0
                    return [base_lr * factor for base_lr in self.base_lrs]

            return WarmupHoldLR(
                optimizer=optimizer,
                warmup_epochs=warmup_epochs,
                initial_lr_factor=initial_lr_factor,
                warmup_strategy=warmup_strategy,
            )
        else:
            return None
    
    if scheduler_type == 'cosine_annealing':
        cosine_config = config.get('cosine_annealing', {})
        min_lr_factor = cosine_config.get('min_lr_factor', 0.01)
        t_max = cosine_config.get('t_max', total_epochs)
        
        if warmup_enabled:
            return WarmupCosineAnnealingLR(
                optimizer=optimizer,
                warmup_epochs=warmup_epochs,
                total_epochs=t_max,
                min_lr_factor=min_lr_factor,
                initial_lr_factor=initial_lr_factor,
                warmup_strategy=warmup_strategy
            )
        else:
            return lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=t_max,
                eta_min=optimizer.param_groups[0]['lr'] * min_lr_factor
            )
    
    elif scheduler_type == 'exponential_decay':
        exp_config = config.get('exponential_decay', {})
        gamma = exp_config.get('gamma', 0.95)
        step_size = exp_config.get('step_size', 5)
        
        if warmup_enabled:
            return WarmupExponentialDecayLR(
                optimizer=optimizer,
                warmup_epochs=warmup_epochs,
                total_epochs=total_epochs,
                gamma=gamma,
                step_size=step_size,
                initial_lr_factor=initial_lr_factor,
                warmup_strategy=warmup_strategy
            )
        else:
            return lr_scheduler.ExponentialLR(
                optimizer=optimizer,
                gamma=gamma
            )
    
    elif scheduler_type == 'step':
        step_config = config.get('step_decay', {})
        gamma = step_config.get('gamma', 0.5)
        step_size = step_config.get('step_size', 10)
        
        if warmup_enabled:
            # For step decay with warmup, we'll use a custom implementation
            return WarmupExponentialDecayLR(
                optimizer=optimizer,
                warmup_epochs=warmup_epochs,
                total_epochs=total_epochs,
                gamma=gamma,
                step_size=step_size,
                initial_lr_factor=initial_lr_factor,
                warmup_strategy=warmup_strategy
            )
        else:
            return lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=step_size,
                gamma=gamma
            )
    
    elif scheduler_type == 'adaptive':
        # Warmup + ReduceLROnPlateau
        adaptive_cfg = config.get('adaptive', {})
        mode = adaptive_cfg.get('mode', 'max')
        factor = adaptive_cfg.get('factor', 0.5)
        patience = adaptive_cfg.get('patience', 4)
        threshold = adaptive_cfg.get('threshold', 1e-4)
        threshold_mode = adaptive_cfg.get('threshold_mode', 'rel')
        cooldown = adaptive_cfg.get('cooldown', 0)
        min_lr_factor = adaptive_cfg.get('min_lr_factor', 0.0)

        return WarmupThenReduceLROnPlateau(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            initial_lr_factor=initial_lr_factor,
            warmup_strategy=warmup_strategy,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr_factor=min_lr_factor,
        )

    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}")
        return None


def create_optimizer(model: nn.Module, 
                    optimizer_type: str = 'adam',
                    learning_rate: float = 0.001,
                    **kwargs):
    """
    Factory function to create optimizer
    
    Args:
        model (nn.Module): Model to optimize
        optimizer_type (str): Type of optimizer ('adam', 'adamw', 'sgd', 'rmsprop')
        learning_rate (float): Learning rate
        **kwargs: Additional optimizer parameters
    
    Returns:
        Optimizer: Configured optimizer
    """
    if optimizer_type.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_type.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_type.lower() == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=learning_rate, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_loss_function(loss_type: str = 'bce', **kwargs) -> nn.Module:
    """
    Factory function to create loss function
    
    Args:
        loss_type (str): Type of loss function ('bce', 'focal', 'weighted_bce')
        **kwargs: Additional loss function parameters
    
    Returns:
        nn.Module: Loss function
    """
    if loss_type.lower() == 'bce':
        return nn.BCELoss(**kwargs)
    elif loss_type.lower() == 'mse':
        return nn.MSELoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_single_epoch(model: nn.Module,
                      train_loader,
                      optimizer,
                      loss_function: nn.Module,
                      device: torch.device,
                      epoch: int,
                      total_start_time: float,
                      scheduler: Optional[Any] = None,
                      verbose: bool = True) -> float:
    """
    Train model for a single epoch
    
    Args:
        model (nn.Module): Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        loss_function (nn.Module): Loss function
        device (torch.device): Device to train on
        epoch (int): Current epoch number
        total_start_time (float): Training start time for progress tracking
        scheduler (Optional[Any]): Learning rate scheduler
        verbose (bool): Whether to print progress
    
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    epoch_start_time = time.time()
    
    for step, (batch_x, batch_y) in enumerate(train_loader):
        # Move data to device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        output = model(batch_x)
        
        # 计算主要的PPI预测损失
        ppi_loss = loss_function(torch.flatten(output).float(), batch_y.float())
        
        # 获取重构损失（如果有）
        reconstruction_loss = 0.0
        if hasattr(model, '_reconstruction_loss'):
            reconstruction_loss = model._reconstruction_loss
        
        # 总损失 = PPI损失 + α * 重构损失
        loss = ppi_loss + reconstruction_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print progress
        if verbose:
            def print_progress(epoch, step, num_batches, epoch_start_time, total_start_time):
                percent = 100 * (step + 1) / num_batches
                elapsed = time.time() - epoch_start_time
                total_elapsed = time.time() - total_start_time
                # 优先通过调度器查询学习率
                try:
                    # scheduler 只在外层可见，这里通过 optimizer 回落
                    # 优先尝试从 PyTorch 提供的最近学习率接口获取
                    if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
                        current_lr = optimizer.param_groups[0]['lr']
                    else:
                        current_lr = 0.0
                except Exception:
                    current_lr = 0.0
                print(f"\rEpoch {epoch+1} [{step+1}/{num_batches}] {percent:.1f}% - "
                      f"Elapsed: {elapsed:.1f}s, Total: {total_elapsed:.1f}s, LR: {current_lr:.6f}", 
                      end='', flush=True)
            print_progress(epoch, step, num_batches, epoch_start_time, total_start_time)
    
    if verbose:
        pass  # Progress bar completed
    
    # Step the scheduler at the end of the epoch
    if scheduler is not None:
        scheduler.step()
    
    return total_loss / num_batches


def train_model(model: nn.Module,
               train_loader,
               config: Dict,
               device: Optional[torch.device] = None) -> Dict:
    """
    Train the model with given configuration
    
    Args:
        model (nn.Module): Model to train
        train_loader: Training data
        config (Dict): Training configuration
        device (Optional[torch.device]): Device to train on
    
    Returns:
        Dict: Training results and metadata
    """
    if device is None:
        device = get_device()
    
    # Move model to device
    model.to(device)
    
    # Create optimizer and loss function
    optimizer = create_optimizer(
        model, 
        config.get('optimizer', 'adam'),
        config.get('learning_rate', 0.001)
    )
    
    loss_function = create_loss_function(config.get('loss_type', 'bce'))
    loss_function.to(device)
    
    # Create scheduler
    scheduler = create_lr_scheduler(optimizer, config.get('lr_scheduler', {}))
    
    # Training parameters
    epochs = config.get('epochs', 40)
    verbose = config.get('verbose', True)
    
    if verbose:
        logger.info("=== Training Started ===")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Device: {device}")
        logger.info(f"Optimizer: {config.get('optimizer', 'adam')}")
        logger.info(f"Learning rate: {config.get('learning_rate', 0.001)}")
        logger.info(f"Loss function: {config.get('loss_type', 'bce')}")
        if scheduler:
            logger.info(f"Scheduler: {config.get('lr_scheduler', {}).get('type', 'unknown')}")
    
    # Training loop
    training_start_time = time.time()
    epoch_losses = []
    
    for epoch in range(epochs):
        epoch_loss = train_single_epoch(
            model, train_loader, optimizer, loss_function,
            device, epoch, training_start_time, scheduler, verbose
        )
        epoch_losses.append(epoch_loss)
        
        if verbose and epoch % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
    
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    if verbose:
        logger.info("=== Training Completed ===")
        logger.info(f"Total training time: {total_training_time:.2f}s")
        logger.info(f"Average epoch time: {total_training_time/epochs:.2f}s")
        logger.info(f"Final loss: {epoch_losses[-1]:.6f}")
    
    # Return training results
    results = {
        'model': model,
        'epoch_losses': epoch_losses,
        'total_training_time': total_training_time,
        'final_loss': epoch_losses[-1],
        'config': config
    }
    
    return results


def get_default_training_config():
    """
    Get default training configuration
    
    Returns:
        Dict: Default training configuration
    """
    return {
        'epochs': 40,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'loss_type': 'bce',
        'verbose': True
    }


def train_with_defaults(model: nn.Module, train_loader) -> Dict:
    """
    Train model with default configuration
    
    Args:
        model (nn.Module): Model to train
        train_loader: Training data
    
    Returns:
        Dict: Training results
    """
    config = get_default_training_config()
    return train_model(model, train_loader, config)


def save_model(model: nn.Module, filepath: str, additional_data: Optional[Dict] = None):
    """
    Save trained model and additional data
    
    Args:
        model (nn.Module): Trained model
        filepath (str): Path to save the model
        additional_data (Optional[Dict]): Additional data to save with model
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }
    
    if additional_data:
        save_dict.update(additional_data)
    
    torch.save(save_dict, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(model: nn.Module, filepath: str, device: Optional[torch.device] = None) -> nn.Module:
    """
    Load trained model
    
    Args:
        model (nn.Module): Model architecture to load weights into
        filepath (str): Path to saved model
        device (Optional[torch.device]): Device to load model on
    
    Returns:
        nn.Module: Loaded model
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    logger.info(f"Model loaded from {filepath}")
    return model