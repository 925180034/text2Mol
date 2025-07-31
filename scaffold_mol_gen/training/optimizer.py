"""
Optimizer and scheduler utilities for scaffold-based molecular generation.

This module provides utilities for creating optimizers and learning rate
schedulers with appropriate configurations for molecular generation tasks.
"""

import logging
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> Optimizer:
    """
    Create optimizer with model-specific parameter grouping.
    
    Args:
        model: Model to optimize
        config: Optimizer configuration
        
    Returns:
        Configured optimizer
    """
    optimizer_type = config.get('type', 'adamw').lower()
    learning_rate = config.get('learning_rate', 5e-5)
    weight_decay = config.get('weight_decay', 0.01)
    
    # Create parameter groups with different learning rates
    param_groups = create_parameter_groups(model, config)
    
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=config.get('eps', 1e-8),
            betas=config.get('betas', (0.9, 0.999))
        )
    
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=config.get('eps', 1e-8),
            betas=config.get('betas', (0.9, 0.999))
        )
    
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=config.get('momentum', 0.9),
            nesterov=config.get('nesterov', True)
        )
    
    elif optimizer_type == 'adafactor':
        try:
            from transformers import Adafactor
            optimizer = Adafactor(
                param_groups,
                lr=learning_rate,
                weight_decay=weight_decay,
                relative_step_size=config.get('relative_step_size', False),
                scale_parameter=config.get('scale_parameter', True),
                warmup_init=config.get('warmup_init', False)
            )
        except ImportError:
            logger.warning("Adafactor not available, falling back to AdamW")
            optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, weight_decay=weight_decay)
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    logger.info(f"Created {optimizer_type} optimizer with {len(param_groups)} parameter groups")
    
    return optimizer


def create_parameter_groups(model: nn.Module, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create parameter groups with different learning rates for different components.
    
    Args:
        model: Model to create parameter groups for
        config: Configuration containing learning rate settings
        
    Returns:
        List of parameter group dictionaries
    """
    base_lr = config.get('learning_rate', 5e-5)
    weight_decay = config.get('weight_decay', 0.01)
    
    # Component-specific learning rate multipliers
    lr_multipliers = config.get('lr_multipliers', {
        'molt5': 0.1,      # Lower LR for pre-trained MolT5
        'encoders': 1.0,   # Standard LR for encoders
        'fusion': 1.0,     # Standard LR for fusion
        'decoders': 1.0,   # Standard LR for decoders
        'adapters': 2.0    # Higher LR for adaptation layers
    })
    
    # Parameter groups
    param_groups = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Determine component
        component = 'default'
        for comp_name in lr_multipliers.keys():
            if comp_name in name.lower():
                component = comp_name
                break
        
        if component not in param_groups:
            param_groups[component] = {
                'params': [],
                'lr': base_lr * lr_multipliers.get(component, 1.0),
                'weight_decay': weight_decay
            }
        
        param_groups[component]['params'].append(param)
    
    # Apply no weight decay to bias and layer norm parameters
    final_groups = []
    for component, group in param_groups.items():
        # Separate parameters with and without weight decay
        decay_params = []
        no_decay_params = []
        
        for param in group['params']:
            # Find parameter name
            param_name = None
            for name, p in model.named_parameters():
                if p is param:
                    param_name = name
                    break
            
            if param_name and ('bias' in param_name or 'LayerNorm' in param_name or 'layer_norm' in param_name):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        # Add groups
        if decay_params:
            final_groups.append({
                'params': decay_params,
                'lr': group['lr'],
                'weight_decay': group['weight_decay']
            })
        
        if no_decay_params:
            final_groups.append({
                'params': no_decay_params,
                'lr': group['lr'],
                'weight_decay': 0.0
            })
    
    # Log parameter group information
    total_params = 0
    for i, group in enumerate(final_groups):
        group_params = sum(p.numel() for p in group['params'])
        total_params += group_params
        logger.info(f"Parameter group {i}: {group_params:,} parameters, "
                   f"lr={group['lr']:.2e}, weight_decay={group['weight_decay']}")
    
    logger.info(f"Total trainable parameters: {total_params:,}")
    
    return final_groups


def get_scheduler(optimizer: Optimizer, config: Dict[str, Any], 
                 num_training_steps: int) -> Optional[_LRScheduler]:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        config: Scheduler configuration
        num_training_steps: Total number of training steps
        
    Returns:
        Configured scheduler or None
    """
    scheduler_type = config.get('type', 'linear').lower()
    
    if scheduler_type == 'none':
        return None
    
    warmup_steps = config.get('warmup_steps', int(0.1 * num_training_steps))
    
    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    
    elif scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=config.get('num_cycles', 0.5)
        )
    
    elif scheduler_type == 'polynomial':
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            power=config.get('power', 1.0)
        )
    
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', num_training_steps // 3),
            gamma=config.get('gamma', 0.1)
        )
    
    elif scheduler_type == 'multistep':
        milestones = config.get('milestones', [num_training_steps // 3, 2 * num_training_steps // 3])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=config.get('gamma', 0.1)
        )
    
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.get('gamma', 0.95)
        )
    
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.get('mode', 'min'),
            factor=config.get('factor', 0.5),
            patience=config.get('patience', 10),
            min_lr=config.get('min_lr', 1e-8)
        )
    
    elif scheduler_type == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=config.get('base_lr', 1e-6),
            max_lr=config.get('max_lr', 1e-3),
            step_size_up=config.get('step_size_up', num_training_steps // 20),
            mode=config.get('mode', 'triangular2')
        )
    
    elif scheduler_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.get('max_lr', 1e-3),
            total_steps=num_training_steps,
            pct_start=config.get('pct_start', 0.3),
            anneal_strategy=config.get('anneal_strategy', 'cos')
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    logger.info(f"Created {scheduler_type} scheduler with {warmup_steps} warmup steps")
    
    return scheduler


def get_polynomial_decay_schedule_with_warmup(optimizer: Optimizer,
                                            num_warmup_steps: int,
                                            num_training_steps: int,
                                            power: float = 1.0,
                                            last_epoch: int = -1) -> _LRScheduler:
    """
    Create polynomial decay schedule with warmup.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        power: Polynomial power
        last_epoch: Last epoch
        
    Returns:
        Polynomial decay scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, (1.0 - progress) ** power)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class WarmupScheduler(_LRScheduler):
    """
    Custom warmup scheduler that can wrap other schedulers.
    """
    
    def __init__(self, optimizer: Optimizer, warmup_steps: int,
                 base_scheduler: Optional[_LRScheduler] = None,
                 warmup_factor: float = 1.0, last_epoch: int = -1):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            base_scheduler: Base scheduler to use after warmup
            warmup_factor: Factor to scale learning rate during warmup
            last_epoch: Last epoch
        """
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        self.warmup_factor = warmup_factor
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            warmup_progress = self.last_epoch / self.warmup_steps
            return [base_lr * self.warmup_factor * warmup_progress 
                   for base_lr in self.base_lrs]
        else:
            # Use base scheduler if available
            if self.base_scheduler:
                self.base_scheduler.last_epoch = self.last_epoch - self.warmup_steps
                return self.base_scheduler.get_lr()
            else:
                return self.base_lrs


class GradualUnfreezing:
    """
    Utility for gradual unfreezing of model parameters during training.
    """
    
    def __init__(self, model: nn.Module, unfreeze_schedule: Dict[str, int]):
        """
        Initialize gradual unfreezing.
        
        Args:
            model: Model to unfreeze
            unfreeze_schedule: Dictionary mapping component names to unfreeze steps
        """
        self.model = model
        self.unfreeze_schedule = unfreeze_schedule
        self.unfrozen_components = set()
        
        # Initially freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        logger.info(f"Initialized gradual unfreezing with schedule: {unfreeze_schedule}")
    
    def step(self, current_step: int, optimizer: Optimizer):
        """
        Check if any components should be unfrozen at current step.
        
        Args:
            current_step: Current training step
            optimizer: Optimizer to update parameter groups
        """
        newly_unfrozen = []
        
        for component, unfreeze_step in self.unfreeze_schedule.items():
            if (current_step >= unfreeze_step and 
                component not in self.unfrozen_components):
                
                # Unfreeze component parameters
                self._unfreeze_component(component)
                self.unfrozen_components.add(component)
                newly_unfrozen.append(component)
        
        # Update optimizer parameter groups if needed
        if newly_unfrozen:
            self._update_optimizer_groups(optimizer)
            logger.info(f"Step {current_step}: Unfroze components {newly_unfrozen}")
    
    def _unfreeze_component(self, component: str):
        """Unfreeze parameters for a specific component."""
        unfrozen_params = 0
        
        for name, param in self.model.named_parameters():
            if component.lower() in name.lower():
                param.requires_grad = True
                unfrozen_params += param.numel()
        
        logger.info(f"Unfroze {unfrozen_params:,} parameters for component '{component}'")
    
    def _update_optimizer_groups(self, optimizer: Optimizer):
        """Update optimizer parameter groups to include newly unfrozen parameters."""
        # This is a simplified implementation
        # In practice, you might want to create new parameter groups
        # or update existing ones with appropriate learning rates
        
        # Add newly unfrozen parameters to existing groups
        for group in optimizer.param_groups:
            trainable_params = [p for p in group['params'] if p.requires_grad]
            group['params'] = trainable_params


def create_optimizer_and_scheduler(model: nn.Module, 
                                 train_dataloader,
                                 config: Dict[str, Any]) -> tuple:
    """
    Convenience function to create both optimizer and scheduler.
    
    Args:
        model: Model to optimize
        train_dataloader: Training data loader
        config: Configuration dictionary
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Calculate training steps
    num_epochs = config.get('num_epochs', 100)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    
    num_training_steps = (len(train_dataloader) * num_epochs) // gradient_accumulation_steps
    
    # Create optimizer
    optimizer_config = config.get('optimizer', {})
    optimizer = get_optimizer(model, optimizer_config)
    
    # Create scheduler
    scheduler_config = config.get('scheduler', {})
    scheduler = get_scheduler(optimizer, scheduler_config, num_training_steps)
    
    # Setup gradual unfreezing if specified
    unfreeze_schedule = config.get('gradual_unfreezing')
    gradual_unfreezing = None
    if unfreeze_schedule:
        gradual_unfreezing = GradualUnfreezing(model, unfreeze_schedule)
    
    return optimizer, scheduler, gradual_unfreezing