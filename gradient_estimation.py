from typing import *
import torch

'''
Provides a gradient estimator for blackbox objective functions

Options for improvement
- Use better (unbiased) baselines to reduce variance
- Reuse samples via importance sampling
'''

class BlackboxGradientEstimator:
    ''' 
    This is implemented as a class so we can have a memory/state
    (which we might need at a later point when implementing improvements).
    
    Args:
        bb_func : The black box function bb_func : R^n -> R
                  operating on a flattened parameter vector of `network`.
    '''
    def __init__(
        self,
        black_box_function: Callable[[torch.Tensor], float]
    ):
        self.black_box_function = black_box_function
    
    def add_estimated_grad_to_network(
        self,
        network: torch.nn.Module,
        num_samples: int=10,
        sigma: float=1.,
        bb_objective_coef: float=1.,
        clip_norm: float=1.
    ) -> None:
        ''' 
        Estimates the gradient using `num_samples` using a 
        Gaussian measure with standdeviation of `sigma`.
        The gradient is additionally multiplied with `bb_objective_coef`.
        This serves as a coefficient to balance the influence of
        multiple objectives.
        The estimated gradient is then added to the gradients
        of the parameters of `network`.
        '''
        
        flat_mu = torch.cat([p.detach().flatten() for p in network.parameters()])

        # sample \theta ~ N(mu, sigma^2)
        theta_samples = sigma * torch.randn((num_samples, len(flat_mu)))
        theta_samples += flat_mu

        # Estimate gradient from samples via average
        # does likely not benefit much from vectorization if bb_func( )
        # evalation are the bottleneck
        estimated_grad = torch.zeros_like(flat_mu)

        # using this as baseline is technically not allowed
        # as it has a gradient wrt mu and biases the expected value
        baseline = self.black_box_function(flat_mu) # <- biased estimate! (dirty hack)

        for s in theta_samples:
            loss = self.black_box_function(s) - baseline
            estimated_grad += loss * (1/sigma * (s - flat_mu))
        estimated_grad /= num_samples

        # clipping gradient norm (dirty hack)
        estimated_grad /= torch.linalg.norm(estimated_grad)*clip_norm
        
        # scale to balance objectives
        estimated_grad *= bb_objective_coef 
        
        # add estimated grad to network parameter .grads
        current_index = 0
        for p in network.parameters():
            grad = estimated_grad[current_index:current_index + p.numel()]
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)
            p.grad += grad.view(*p.shape)
            current_index += p.numel()
        return None