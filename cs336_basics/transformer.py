from torch import nn
import torch
from einops import rearrange, einsum
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.W = nn.Parameter(torch.empty(self.out_features, self.in_features, device = device, dtype = dtype))
        init_std = math.sqrt(2/(self.in_features+self.out_features))
        
        """
        trunc_normal_ modifies W inplace -- no need to assign back.
        """
        nn.init.trunc_normal_(self.W, mean = 0.0, std = init_std, a = -3 * init_std, b = 3 * init_std)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")
    
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.W = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device = device, dtype = dtype))
        nn.init.trunc_normal_(self.W, mean= 0.0, std = 1, a = -3, b = 3)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]
    

    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.W = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x of shape (batch_size, seq_length, d_model)
        in_dtype = x.dtype
        
        # Convert to float32 for better precision.
        x = x.to(torch.float32)
        
        """
        A few subtlety: 
        1. x**2 or pow(x, 2) is scaler operation, so it applies to all elements of x.
        2. x.mean would collapse whatever dimension(s) it's running mean for, so in this case need to use keepdim = True.
        """
        RMS_x = torch.sqrt((x**2).mean(dim = -1, keepdim = True) + self.eps)
        
        # W is of dim (d,), so want to multiple to each element of normalized x, so use scalar multiplication.
        return (x/RMS_x * self.W).to(in_dtype)

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        self.d_model = d_model
        """
        canonically, d_ff should be 8/3 * d_model
        Also ensuring the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of hardware.
        """
        if not d_ff:
            self.d_ff = d_model * 8 / 3 //64 * 64
        else:
            self.d_ff = d_ff
        self.W1 = Linear(self.d_model, self.d_ff)
        self.W2 = Linear(self.d_ff, self.d_model)
        self.W3 = Linear(self.d_model, self.d_ff)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        gate = self.W1(x) # to save compute
        return self.W2(gate*torch.sigmoid(gate) * self.W3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        inv_freq = torch.tensor([1/(theta**((2*k - 2)/d_k)) for k in range(1, d_k//2 + 1)]).unsqueeze(0)
        positions =  torch.arange(0, max_seq_len).unsqueeze(1)
        # Note: use * instead of @ as this is element-wise broadcast
        # persistent = False means buffer not saved to state_dict()
        self.register_buffer('sin_buffer', torch.sin(positions * inv_freq), persistent= False)
        self.register_buffer('cos_buffer', torch.cos(positions * inv_freq), persistent= False)
        self.d_k = d_k                                                                                 
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        R = torch.zeros((len(token_positions), self.d_k, self.d_k))
        cos = self.cos_buffer[token_positions]
        sin = self.sin_buffer[token_positions]
        for i in range(len(R)):
            for j in range(0, self.d_k, 2):
                R[i][j][j] = cos[i][j//2]
                R[i][j+1][j+1] = cos[i][j//2]
                R[i][j][j+1] = -sin[i][j//2]
                R[i][j+1][j] = sin[i][j//2]
        
        # dim(R) = (len(token_positions)  d_k  d_k)
        # dim(x) = (batch  seq_len  d_k )
        # R @ x -> (batch  seq_len  d_k )?
        return einsum(R, x, "seq_len d_k1 d_k2, batch seq_len d_k2 -> batch seq_len d_k1")
        
        