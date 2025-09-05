"""
DeepSpeed CUDA compatibility fix.
This module sets environment variables and patches DeepSpeed to prevent CUDA version mismatch errors.
"""

import os
import sys
import warnings


def setup_deepspeed_env():
    """
    Set environment variables to bypass DeepSpeed's CUDA compilation checks.
    This prevents errors when system CUDA version differs from PyTorch's CUDA version.
    """
    # Skip CUDA version checks
    os.environ["DS_SKIP_CUDA_CHECK"] = "1"
    
    # Disable ALL JIT compilation of CUDA extensions
    os.environ["DS_BUILD_CPU_ADAM"] = "0"
    os.environ["DS_BUILD_FUSED_ADAM"] = "0"
    os.environ["DS_BUILD_UTILS"] = "0"
    os.environ["DS_BUILD_AIO"] = "0"
    os.environ["DS_BUILD_SPARSE_ATTN"] = "0"
    os.environ["DS_BUILD_TRANSFORMER"] = "0"
    os.environ["DS_BUILD_TRANSFORMER_INFERENCE"] = "0"
    os.environ["DS_BUILD_OPS"] = "0"
    os.environ["DS_BUILD_LAMB"] = "0"
    os.environ["DS_BUILD_FUSED_LAMB"] = "0"
    
    # Additional flags to prevent any compilation
    os.environ["DS_BUILD_CUTLASS_OPS"] = "0"
    os.environ["DS_BUILD_RAGGED_DEVICE_OPS"] = "0"
    os.environ["DS_BUILD_EVOFORMER_ATTN"] = "0"
    
    # Clear CUDA_HOME to prevent version detection
    os.environ["CUDA_HOME"] = ""
    
    # Force disable JIT compilation
    os.environ["TORCH_CUDA_ARCH_LIST"] = ""


class FusedAdamWrapper:
    """
    Wrapper that creates PyTorch AdamW optimizer with FusedAdam-compatible interface.
    Filters out DeepSpeed-specific parameters that PyTorch doesn't understand.
    """
    def __init__(self, params, lr=1e-3, bias_correction=True, betas=(0.9, 0.999), 
                 eps=1e-8, adam_w_mode=True, weight_decay=0.01, amsgrad=False, **kwargs):
        from torch.optim import AdamW, Adam
        
        # Filter out DeepSpeed-specific parameters
        deepspeed_params = ['adam_w_mode', 'bias_correction']
        
        # Prepare kwargs for PyTorch optimizer
        pytorch_kwargs = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'amsgrad': amsgrad
        }
        
        # Add any additional kwargs that PyTorch understands
        for key, value in kwargs.items():
            if key not in deepspeed_params:
                pytorch_kwargs[key] = value
        
        # Use AdamW if adam_w_mode is True, else use Adam
        if adam_w_mode:
            self.optimizer = AdamW(params, **pytorch_kwargs)
        else:
            self.optimizer = Adam(params, **pytorch_kwargs)
        
        warnings.warn(f"Using PyTorch {'AdamW' if adam_w_mode else 'Adam'} instead of FusedAdam due to CUDA mismatch", UserWarning)
    
    def __getattr__(self, name):
        """Forward all other method calls to the underlying optimizer."""
        return getattr(self.optimizer, name)
    
    def step(self, closure=None):
        """Forward step to underlying optimizer."""
        return self.optimizer.step(closure)
    
    def zero_grad(self, set_to_none=True):
        """Forward zero_grad to underlying optimizer."""
        return self.optimizer.zero_grad(set_to_none)
    
    def state_dict(self):
        """Forward state_dict to underlying optimizer."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Forward load_state_dict to underlying optimizer."""
        return self.optimizer.load_state_dict(state_dict)


def monkey_patch_deepspeed():
    """
    Monkey patch DeepSpeed to use PyTorch optimizers instead of trying to compile FusedAdam.
    """
    # Pre-emptively create fake modules for FusedAdam
    class FakeFusedAdamModule:
        """Fake FusedAdam module that redirects to our wrapper."""
        FusedAdam = FusedAdamWrapper
    
    # Set fake modules before deepspeed imports them
    sys.modules['deepspeed.ops.adam.fused_adam'] = FakeFusedAdamModule()
    sys.modules['fused_adam'] = FakeFusedAdamModule()
    
    # If deepspeed is already imported, patch it
    if 'deepspeed' in sys.modules:
        import deepspeed
        
        # Replace FusedAdam with our wrapper
        if hasattr(deepspeed, 'ops'):
            if not hasattr(deepspeed.ops, 'adam'):
                deepspeed.ops.adam = type('adam', (), {})()
            deepspeed.ops.adam.FusedAdam = FusedAdamWrapper
            
            # Also patch the module
            if hasattr(deepspeed.ops, 'adam') and hasattr(deepspeed.ops.adam, 'fused_adam'):
                deepspeed.ops.adam.fused_adam.FusedAdam = FusedAdamWrapper
        
        # Patch builders to return our wrapper
        if hasattr(deepspeed, 'ops') and hasattr(deepspeed.ops, 'op_builder'):
            import deepspeed.ops.op_builder as op_builder
            
            if hasattr(op_builder, 'FusedAdamBuilder'):
                def patched_load(self):
                    # Return our wrapper class
                    return FusedAdamWrapper
                
                op_builder.FusedAdamBuilder.load = patched_load
            
            if hasattr(op_builder, 'CPUAdamBuilder'):
                def patched_load(self):
                    # For CPU Adam, also return our wrapper
                    return FusedAdamWrapper
                
                op_builder.CPUAdamBuilder.load = patched_load


# Automatically apply fixes when module is imported
setup_deepspeed_env()
monkey_patch_deepspeed()