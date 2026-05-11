import torch
import math
import random
import numpy as np

def apply_softmax(x: torch.Tensor, dim: int):     
    shifted = x - x.max(dim = dim, keepdim = True).values #Subtract the max value to avoid numerical instability
    exp_x = shifted.exp()
    return exp_x/exp_x.sum(dim = dim, keepdim = True)

def cross_entropy(predicted_logits: torch.Tensor, targets: torch.Tensor):
    """
    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len)
    """
    
    # tensor.max() returns a namedtuple (values, indices), need .values if explicitly wants values
    max_values = predicted_logits.max(dim = -1, keepdim= True).values
    predicted_logits = predicted_logits - max_values
    
    # torch.gather() gathers values along an axis specified by dim; input and index must have the same number of dimensions, hence the unsqueeze()
    target_logits = predicted_logits.gather(dim = -1, index = targets.unsqueeze(-1))
    
    # log and exp cancels out for the numerater
    return -(target_logits - torch.log(predicted_logits.exp().sum(dim = -1, keepdim= True))).mean()


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        
        # defaults is a dict of hyperparams that gets stored in each param group.
        defaults = {'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay}
        super().__init__(params, defaults)
    
    def step(self):
        for group in self.param_groups:
            for theta in group['params']:
                
                # Avoids frozen layer
                if theta.grad is None:
                    continue
                
                state = self.state[theta]
                
                # Initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(theta)
                    state['v'] = torch.zeros_like(theta)
                
                # Increment first, avoid division by 0
                state['step'] += 1
                
                # adjust learning rate
                # Actually, applies bias correction to m and v
                # This is just algebraic rearrangement
                lr = group['lr'] * ((1-group['betas'][1]**state['step']) ** 0.5) / (1-group['betas'][0]**state['step'])

                # Weight decay
                theta.data -= group['lr'] * group['weight_decay'] * theta.data
                
                m = group['betas'][0] * state['m'] + (1- group['betas'][0]) * theta.grad
                v = group['betas'][1] * state['v'] + (1- group['betas'][1]) * theta.grad ** 2
                theta.data -= lr * m / (v ** 0.5 + group['eps'])
                
                state['m'] = m
                state['v'] = v
                

def learning_rate_schedule(t, alpha_max, alpha_min, Tw, Tc):
    if t < Tw:
        return t/Tw * alpha_max
    elif Tw <= t <= Tc:
        return alpha_min + 1/2 * (1 + math.cos((t-Tw)/(Tc - Tw) * math.pi)) * (alpha_max - alpha_min)

    else:
        return alpha_min
    

def gradient_clipping(parameters, maximum, eps = 1e-6):
    norm = 0
    for theta in parameters:
        if theta.grad is not None:
            norm += torch.linalg.vector_norm(theta.grad) ** 2
    
    norm = math.sqrt(norm)    
    if norm >= maximum:
        for theta in parameters:
            if theta.grad is not None:
                theta.grad *= maximum / (norm + eps)
                
                
def data_loading(x, batch_size, context_length, device: str):
    n = len(x)
    indices = np.random.randint(0, n - context_length, batch_size)
    
    rolling_indices = np.array(indices)[:, None] + np.arange(context_length)
    
    inputs = torch.from_numpy(x[rolling_indices]).to(device)
    targets = torch.from_numpy(x[rolling_indices+1]).to(device)
    
    return inputs, targets

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str):
    torch.save({'Model': model.state_dict(), 'Optimizer': optimizer.state_dict(), 'Iteration': iteration}, out)

def load_checkpoint(src: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    d = torch.load(src, map_location='cpu')
    model.load_state_dict(d['Model'])    
    optimizer.load_state_dict(d['Optimizer'])
    return d['Iteration']




def decode(inputs, model, max_context_window, temperature, p, eos_token_id = None):
    
    len_inputs = len(inputs)
    start = len_inputs
    latest_token_id = -1
    while start < max_context_window and (eos_token_id is None or latest_token_id != eos_token_id):
        with torch.no_grad():
            logits = model(torch.tensor(inputs).unsqueeze(0))[0, -1, :]
    
        probs = apply_softmax(logits/temperature, -1)
        
        # torch.sort gives both values and original vocab indices
        sorted_probs, sorted_indices = torch.sort(probs, descending= True) 

        
        cumsum = torch.cumsum(sorted_probs, dim = 0)
        
        to_remove = (cumsum - sorted_probs) >= p #happens when cumsum does not need this prob to be larger than p, meaning cumsum >= p already reached.
        sorted_probs[to_remove] = 0.0
        
        sampled = torch.multinomial(sorted_probs, num_samples= 1)
        latest_token_id = sorted_indices[sampled].item()
        inputs.append(latest_token_id)
        start += 1
        
    return inputs[len_inputs:]
    