import torch
import math
import random
import numpy as np
import copy

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
    # model.parameters() returns a one-shot iterator. Materialize it so the
    # same parameters are available when computing and applying the clip.
    parameters = list(parameters)

    norm = 0
    for theta in parameters:
        if theta.grad is not None:
            norm += torch.linalg.vector_norm(theta.grad) ** 2
    
    norm = math.sqrt(norm)    
    if norm >= maximum:
        for theta in parameters:
            if theta.grad is not None:
                theta.grad *= maximum / (norm + eps)

    return norm
                
                
def data_loading(
    x,
    batch_size,
    context_length,
    device: str,
    rng: np.random.Generator | None = None,
):
    n = len(x)
    if rng is None:
        indices = np.random.randint(0, n - context_length, batch_size)
    else:
        indices = rng.integers(0, n - context_length, batch_size)
    
    rolling_indices = np.array(indices)[:, None] + np.arange(context_length)
    
    inputs = torch.from_numpy(x[rolling_indices]).to(device)
    targets = torch.from_numpy(x[rolling_indices+1]).to(device)
    
    return inputs, targets

def capture_rng_state(train_rng: np.random.Generator | None = None):
    numpy_state = np.random.get_state()
    state = {
        "python_random": random.getstate(),
        "numpy_random": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if train_rng is not None:
        state["train_numpy_generator"] = copy.deepcopy(train_rng.bit_generator.state)
    return state


def restore_rng_state(state, train_rng: np.random.Generator | None = None):
    if not state:
        return

    random.setstate(state["python_random"])
    numpy_state = state["numpy_random"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["state"].cpu().numpy().astype(np.uint32, copy=True),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["torch_cuda"]])
    if train_rng is not None and "train_numpy_generator" in state:
        train_rng.bit_generator.state = copy.deepcopy(state["train_numpy_generator"])


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str,
    rng_state=None,
    training_state=None,
):
    checkpoint = {
        "Model": model.state_dict(),
        "Optimizer": optimizer.state_dict(),
        "Iteration": iteration,
    }
    if rng_state is not None:
        checkpoint["RNG"] = rng_state
    if training_state is not None:
        checkpoint["TrainingState"] = training_state
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    return_extra_state: bool = False,
):
    d = torch.load(src, map_location="cpu", weights_only=True)
    model.load_state_dict(d['Model'])    
    optimizer.load_state_dict(d['Optimizer'])
    if return_extra_state:
        return d['Iteration'], d.get("RNG"), d.get("TrainingState", {})
    return d['Iteration']




def decode(
    inputs,
    model,
    max_context_window,
    temperature,
    p,
    eos_token_id=None,
    max_new_tokens=None,
):
    if not inputs:
        raise ValueError("Generation requires at least one input token.")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")
    if not 0 < p <= 1:
        raise ValueError("top-p must be in the interval (0, 1].")
    if max_new_tokens is None:
        max_new_tokens = max(0, max_context_window - len(inputs))
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")

    generated = list(inputs)
    prompt_length = len(generated)
    device = next(model.parameters()).device

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # Once the sequence exceeds the model context, retain the most
            # recent tokens rather than silently stopping generation.
            context = generated[-max_context_window:]
            model_input = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(model_input)[0, -1, :]
            probs = apply_softmax(logits / temperature, -1)

            # Keep the smallest high-probability prefix whose mass reaches p.
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            to_remove = (cumsum - sorted_probs) >= p
            sorted_probs[to_remove] = 0.0

            sampled = torch.multinomial(sorted_probs, num_samples=1)
            latest_token_id = sorted_indices[sampled].item()
            generated.append(latest_token_id)
            if eos_token_id is not None and latest_token_id == eos_token_id:
                break

    return generated[prompt_length:]
