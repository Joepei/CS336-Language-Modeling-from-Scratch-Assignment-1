from torch import nn
import torch
from einops import rearrange, einsum
import math
from cs336_basics.utils import apply_softmax

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
        self.W = nn.Parameter(torch.ones(d_model, device = device))
    
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
            self.d_ff = int(d_model * 8 / 3 //64 * 64)
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
        
        cos = self.cos_buffer[token_positions]
        sin = self.sin_buffer[token_positions]
        
       
        # ## Slow way of constructing R
        # R = torch.zeros((len(token_positions), self.d_k, self.d_k))

        # for i in range(len(R)):
        #     for j in range(0, self.d_k, 2):
        #         R[i][j][j] = cos[i][j//2]
        #         R[i][j+1][j+1] = cos[i][j//2]
        #         R[i][j][j+1] = -sin[i][j//2]
        #         R[i][j+1][j] = sin[i][j//2]
        # dim(R) = (len(token_positions)  d_k  d_k)
        # dim(x) = (batch  seq_len  d_k )
        # R @ x -> (batch  seq_len  d_k )?
        # return einsum(R, x, "seq_len d_k1 d_k2, batch seq_len d_k2 -> batch seq_len d_k1")
        x = x.clone()  # So you don't mutate x. It might be used in other functions as well.
        x_even = x[..., ::2].clone()
        x_odd = x[..., 1::2].clone()
        x[..., ::2] = x_even * cos - x_odd * sin # PyTorch broadcasts from the right, so not need to unsqueeze sin and cos buffers.
        x[..., 1::2]= x_even * sin + x_odd * cos
        return x




def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    d_k = Q.shape[-1]
    pre_softmax = einsum(Q, K, 'batch ... len_query d_k, batch ... len_key d_k -> batch ... len_query len_key')/(d_k ** 0.5)
    if mask is not None:
        pre_softmax = torch.where(mask, pre_softmax, -torch.inf)
    
    normalized = apply_softmax(pre_softmax, -1)
    return einsum(normalized, V, "batch ... len_query len_key, batch ... len_key d_v -> batch ... len_query d_v")

class Multihead_self_attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float = None, max_seq_len: int = None, device = None):
        super().__init__()
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.WQ = Linear(in_features = d_model, out_features = num_heads * self.d_k, device = device)
        self.WK = Linear(in_features = d_model, out_features = num_heads * self.d_k, device = device)
        self.WV = Linear(in_features = d_model, out_features = num_heads * self.d_v, device = device)
        self.WO = Linear(in_features = num_heads * self.d_v, out_features = d_model, device = device)
        self.num_heads = num_heads
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device)
        else:
            self.rope = None
        
    def forward(self, x: torch.Tensor, token_positions = None) -> torch.Tensor:
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)
        """
        rearrange does not affect backpropagation
        it is just a view operation - just reshapes the tensor without copying data.
        PyTorch's autograd tracks it transparently, so gradients flow back through it correctly as if it never happened.
        """
        Q = rearrange(Q, "batch seq_len (h d_k) -> batch h seq_len d_k", h = self.num_heads)
        K = rearrange(K, "batch seq_len (h d_k) -> batch h seq_len d_k", h = self.num_heads)
        V = rearrange(V, "batch seq_len (h d_v) -> batch h seq_len d_v", h = self.num_heads)
        if self.rope is not None:
            if token_positions is None:
                # token_positions is range(0, seq_len)
                # Need this because if token_positions = None, then in RoPE, cos_buffer[None] and sin_buffer[None] would create a new empty dimension at the beginniner (similar to how tensor.unsqueeze(0) works, which creates shape mismatch)
                token_positions = torch.arange(x.shape[-2], device= x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
            
        seq_len = x.shape[-2]
        
        """
        Note that here uses lower triangle, not upper. 
        To understand, need to refer back to pre_softmax in scaled_dot_product_attention
        The last two dimension for pre_softmax is (len_query, len_key). So the ij^th position 
        of the pre_softmax attention matrix is treating each row as each token in the query. 
        So first row should keep only the first attn, second row keeps two, etcs. Hence the lower-triangular structure.
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device = x.device)).bool() #0.0 from torch.tril would be evaluated to True, need this .bool() to covnert to True/False.
        attn = scaled_dot_product_attention(Q, K, V, mask)
        concat_attn = rearrange(attn, "batch h seq_len d_model -> batch seq_len (h d_model)")
        return self.WO(concat_attn)
    

class Transformer_block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta: float = None, max_seq_len: int = None, device = None):
        super().__init__()
        self.norm_1 = RMSNorm(d_model= d_model, device= device)
        self.multi_head_attention = Multihead_self_attention(d_model= d_model, num_heads= num_heads, theta= theta, max_seq_len= max_seq_len, device=device)
        self.norm_2 = RMSNorm(d_model= d_model, device= device)
        self.ff = FFN(d_model= d_model, d_ff = d_ff)
    
    def forward(self, x: torch.Tensor, token_positions = None) -> torch.Tensor:
        x = x + self.multi_head_attention(self.norm_1(x), token_positions = token_positions)
        x = x + self.ff(self.norm_2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta: float = None,  device = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context = context_length
        self.embedding = Embedding(num_embeddings= vocab_size, embedding_dim= d_model, device=device)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(Transformer_block(d_model= d_model, num_heads= num_heads, d_ff = d_ff, theta= theta, max_seq_len = context_length, device= device))
        
        self.norm_pre_out = RMSNorm(d_model= d_model, device= device)
        self.output_embedding = Linear(in_features= d_model, out_features= vocab_size, device= device)
    
    def forward(self, x: torch.Tensor, token_positions = None) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x, token_positions)
        x = self.norm_pre_out(x)
        x = self.output_embedding(x)
        return x
        