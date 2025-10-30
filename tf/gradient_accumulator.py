import tensorflow as tf
from typing import List, Optional, Tuple

class GradientAccumulator:
    def __init__(self, optimizer, accumulation_steps: int, 
                 use_mixed_precision: bool = False):
        """
        Gradient accumulation optimizer that simulates larger batch sizes.
        
        Args:
            optimizer: Base optimizer (e.g., SGD, Adam)
            accumulation_steps: Number of steps to accumulate gradients
            use_mixed_precision: Whether to use mixed precision
        """
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.use_mixed_precision = use_mixed_precision
        
        # Gradient accumulation variables
        self.accumulated_gradients = None
        self.step_count = 0
        
        # For mixed precision
        self.loss_scale = 128.0 if use_mixed_precision else 1.0
        
        # Statistics
        self.total_steps = 0
        self.actual_updates = 0
        
    def reset_gradients(self):
        """Reset accumulated gradients"""
        self.accumulated_gradients = None
        self.step_count = 0
        
    def accumulate_gradients(self, gradients: List[tf.Tensor]):
        """Accumulate gradients from current step"""
        if self.accumulated_gradients is None:
            # Initialize accumulated gradients
            self.accumulated_gradients = [
                tf.zeros_like(g) if g is not None else None 
                for g in gradients
            ]
        
        # Add current gradients to accumulated gradients
        self.accumulated_gradients = [
            (acc + g) if acc is not None and g is not None 
            else None if acc is None and g is None 
            else acc if g is None 
            else g
            for acc, g in zip(self.accumulated_gradients, gradients)
        ]
        
        self.step_count += 1
        
    def should_apply_gradients(self) -> bool:
        """Check if gradients should be applied"""
        return self.step_count >= self.accumulation_steps
        
    def apply_gradients(self, variables: List[tf.Variable]) -> Optional[tf.Tensor]:
        """Apply accumulated gradients"""
        if not self.should_apply_gradients():
            return None
        
        # Scale gradients by accumulation steps
        if self.accumulated_gradients:
            scaled_gradients = [
                g / tf.cast(self.accumulation_steps, g.dtype) 
                if g is not None else None
                for g in self.accumulated_gradients
            ]
            
            # Apply gradients
            self.optimizer.apply_gradients(
                zip(scaled_gradients, variables)
            )
            
            self.actual_updates += 1
        
        # Reset for next accumulation
        self.reset_gradients()
        self.total_steps += 1
        
        return tf.constant(self.actual_updates, dtype=tf.int32)
    
    def get_stats(self) -> dict:
        """Get accumulation statistics"""
        return {
            'accumulation_steps': self.accumulation_steps,
            'current_step': self.step_count,
            'total_steps': self.total_steps,
            'actual_updates': self.actual_updates,
            'effective_batch_size_multiplier': self.accumulation_steps
        }

class AdaptiveGradientAccumulator(GradientAccumulator):
    def __init__(self, optimizer, initial_accumulation_steps: int = 4,
                 min_accumulation_steps: int = 1, max_accumulation_steps: int = 16,
                 memory_threshold: float = 0.8, **kwargs):
        """
        Adaptive gradient accumulator that adjusts accumulation steps based on memory.
        
        Args:
            optimizer: Base optimizer
            initial_accumulation_steps: Starting accumulation steps
            min_accumulation_steps: Minimum accumulation steps
            max_accumulation_steps: Maximum accumulation steps
            memory_threshold: Memory usage threshold for adjustment
        """
        super().__init__(optimizer, initial_accumulation_steps, **kwargs)
        
        self.min_accumulation_steps = min_accumulation_steps
        self.max_accumulation_steps = max_accumulation_steps
        self.memory_threshold = memory_threshold
        
        # Memory tracking
        self.memory_history = []
        self.adjustment_interval = 100
        self.last_adjustment_step = 0
        
    def get_memory_usage(self) -> float:
        """Get current memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent / 100.0
        except:
            return 0.5  # Default fallback
    
    def should_adjust_accumulation(self, step: int) -> bool:
        """Check if accumulation steps should be adjusted"""
        if step - self.last_adjustment_step < self.adjustment_interval:
            return False
        return len(self.memory_history) >= 10
    
    def adjust_accumulation_steps(self, step: int):
        """Adjust accumulation steps based on memory usage"""
        if not self.should_adjust_accumulation(step):
            return
        
        current_memory = self.get_memory_usage()
        self.memory_history.append(current_memory)
        
        # Keep only recent history
        if len(self.memory_history) > 20:
            self.memory_history = self.memory_history[-20:]
        
        avg_memory = sum(self.memory_history) / len(self.memory_history)
        
        old_steps = self.accumulation_steps
        
        # Adjust based on memory usage
        if avg_memory > self.memory_threshold:
            # High memory usage - increase accumulation steps
            new_steps = min(self.accumulation_steps * 2, self.max_accumulation_steps)
        elif avg_memory < self.memory_threshold * 0.6:
            # Low memory usage - decrease accumulation steps
            new_steps = max(self.accumulation_steps // 2, self.min_accumulation_steps)
        else:
            return
        
        if new_steps != old_steps:
            print(f"Adjusting accumulation steps from {old_steps} to {new_steps} "
                  f"(memory: {avg_memory:.1%})")
            self.accumulation_steps = new_steps
            self.last_adjustment_step = step
    
    def apply_gradients(self, variables: List[tf.Variable]) -> Optional[tf.Tensor]:
        """Apply accumulated gradients with adaptive adjustment"""
        result = super().apply_gradients(variables)
        
        # Check if we should adjust accumulation
        if result is not None:
            self.adjust_accumulation_steps(self.total_steps)
        
        return result
