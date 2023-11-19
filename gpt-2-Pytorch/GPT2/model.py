'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import copy
import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter

def gelu(x):
    # Gaussian Error Linear Unit,在神经网络中用于引入非线性性
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True) # [bs, seq_len, hidden_size] --> [bs, seq_len, 1]
        s = (x - u).pow(2).mean(-1, keepdim=True) # [bs, seq_len, 1]
        x = (x - u) / torch.sqrt(s + self.variance_epsilon) # [bs, seq_len, hidden_size] 
        return self.weight * x + self.bias

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02) 
        self.weight = Parameter(w) # 
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)  #  [bs, sq_len, nf]
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight) # [bs * seq_len, nf]
        x = x.view(*size_out) #   [bs * seq_len, nf]-->[bs, sq_len, nf]
        return x

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        '''
        nx表示输入的特征维度，
        n_ctx表示上下文的最大序列长度，
        scale是一个标志，用于控制是否要缩放（scale） 
        '''
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0 
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))#  下三角矩阵，用于attention mask [1,1,n_ctx,n_ctx]
        self.n_head = config.n_head
        self.split_size = n_state #  分割的大小，通常等于输入特征维度。这个值将用于将输入分成查询（query）、键（key）和值（value）
        self.scale = scale  
        self.c_attn = Conv1D(n_state * 3, nx) # 卷积层，用于变换自注意力的输入
        self.c_proj = Conv1D(n_state, nx) # 卷积层，用于变换自注意力的输出

    def _attn(self, q, k, v):
        '''
        ATTENTION(Q,K,V) = SOFTMAX(QK^T/sqrt(dk))V
        Args:
            q: [bs, n_head, seq_len, head_features]
            k: [bs, n_head, head_features, cur_seq_len], split_heads过程中已经transpose
            v: [bs, n_head, cur_seq_len, head_features]
        Returns:
            [bs, n_head, seq_len, head_features]
        '''
        w = torch.matmul(q, k) # [bs,n_head, seq_len, cur_seq_len]，split_heads过程中k已经transpose，所以这里不需要transpose
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1) # nd: seq_len, ns: cur_seq_len
        b = self.bias[:, :, ns-nd:ns, :ns] # [1,1, seq_len, cur_seq_len] 下三角矩阵，用于attention mask，会broadcast 
        w = w * b - 1e10 * (1 - b) # 将矩阵中为0的值设置为-1e10(-inf)，这样在softmax不会有影响
        w = nn.Softmax(dim=-1)(w) # SOFTMAX(QK^T/sqrt(dk)) 
        return torch.matmul(w, v)  # [bs, n_head, seq_len, head_features]

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous() # 交换第一二维  [bs, n_head, seq_len, head_features]  --> [bs, seq_len, n_head, head_features]
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),) # [bs, seq_len, hidden_size]
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        '''
        拆分multi-head
        本质上就是将一个大向量拆分为多个小向量
        Args:
            x: [bs, seq_len, hidden_size] 
            k: whether is key, transpose 
            head_features = hidden_size // n_head, assert hidden_size % n_head==0
        Returns:
            if is key: [bs, n_head, head_features, seq_len]
            else: [bs, n_head, seq_len, head_features]
        '''
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head) #  [bs, seq_len, n_head, head_features]
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states;  [bs, seq_len, n_head, head_features]
        if k:
            return x.permute(0, 2, 3, 1)  # (bs, n_head, head_features, seq_len)
        else:
            return x.permute(0, 2, 1, 3)  # (bs, n_head, seq_len, head_features)

    def forward(self, x, layer_past=None):
        '''
        Args:
            x: [bs , seq_len, hidden_size]
            laye_past: 过去所有时刻的层的中间状态信息
                    key = laye_past[0] 和 value = laye_past[1]  
                    [2, bs, n_head, pre_seq_len, head_features]
        Returns:
            a: [bs, seq_len, hidden_size]
            present: 当前和过去所有时刻的层的中间状态信息, 
                    key  = present[0] 和 value = present[1]
                    [2, bs, n_head, cur_seq_len, head_features]
        '''
        x = self.c_attn(x) #  这个向量是 Query、Key 和 Value 向量的拼接  [bs , seq_len, hidden_size] --> [bs, seq_len, 3*hidden_size]
        query, key, value = x.split(self.split_size, dim=2) #split_size=hidden_size,  q,k,v: [bs, seq_len, hidden_size]
        query = self.split_heads(query) # [bs, seq_len, hidden_size] --> [bs, n_head, seq_len, head_features]
        key = self.split_heads(key, k=True) # K已经transpose : [bs, seq_len, hidden_size] -->  [bs, n_head, head_features, seq_len] 
        value = self.split_heads(value) # [bs, seq_len, hidden_size] --> [bs, n_head, seq_len, head_features]
        if layer_past is not None: # 如果有过去的隐藏状态信息，将过去的隐藏状态信息与当前的隐藏状态信息拼接起来
            # past_key : [bs, n_head, head_features,pre_seq_len] 
            # past_value : [bs, n_head, pre_seq_len, head_features] 
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below，
            key = torch.cat((past_key, key), dim=-1)   # [bs, n_head,head_features,cur_seq_len  ] 
            value = torch.cat((past_value, value), dim=-2) # [bs, n_head, cur_seq_len, head_features]
        present = torch.stack((key.transpose(-2, -1), value))  #  当前及过去的[k,v] 信息 ,transpose to have same shapes for stacking,[2, bs, n_head, cur_seq_len, head_features]
        a = self._attn(query, key, value) #ATTENTION 计算  [bs, n_head, seq_len, head_features]  
        a = self.merge_heads(a) #  [bs, n_head, seq_len, head_features] -> [bs, seq_len, hidden_size]
        a = self.c_proj(a) #   投影得到想要的维度  [bs, seq_len, hidden_size]
        return a, present

class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd # 模型的隐藏层的维度 hidden_size
        self.c_fc = Conv1D(n_state, nx) 
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x)) # [bs, sq_len, hidden_size] --> [bs, sq_len,n_state]
        h2 = self.c_proj(h) # [bs, sq_len,n_state] --> [bs, sq_len,hidden_size]
        return h2 # [bs, sq_len,hidden_size]

class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        '''
        模型中的一个 Transformer 块, 自注意力层 (Attention) 和前馈神经网络 (MLP) 层
        n_ctx : 模型的上下文长度（context length）
        '''
        super(Block, self).__init__()
        nx = config.n_embd # 模型的隐藏层的维度 hidden_size
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        '''
        Args:
            x: [bs, seq_len, hidden_size]
            layer_past： 过去所有时刻的中间状态信息
            layer_past[0] 为 key, layer_past[1] 为 value
            [2, bs, n_head, pre_seq_len, head_features]
        Returns:
            x: [bs, seq_len, hidden_size]
            present: 这一层当前和过去所有时刻的层的中间状态信息 
            present[0] 为 key, present[1] 为 value
            [2, bs, n_head, cur_seq_len, head_features]
        '''
        a, present = self.attn(self.ln_1(x), layer_past=layer_past) # X_attention = ATTENTION(LayerNorm(x)) 
        x = x + a # 残差连接  X_attention = X + X_attention
        m = self.mlp(self.ln_2(x)) # X_mlp = MLP(LayerNorm(X_attention)), [bs , seq_len, hidden_size]
        x = x + m # 残差连接 X_mlp = X_attention + X_mlp 
        return x, present # [bs , seq_len, hidden_size],[2, bs, n_head, cur_seq_len, head_features]

class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.n_layer = config.n_layer # Transformer层的数量
        self.n_embd = config.n_embd # 模型的隐藏层的维度 hidden_size
        self.n_vocab = config.vocab_size # 词表大小 

        self.wte = nn.Embedding(config.vocab_size, config.n_embd) # word embedding
        self.wpe = nn.Embedding(config.n_positions, config.n_embd) # position embedding
        block = Block(config.n_ctx, config, scale=True) # Transformer block
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)]) # Transformer block * n_layer
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def set_embeddings_weights(self, model_embeddings_weights):
        '''
        设置词嵌入层的权重，通常用于加载预训练模型的嵌入层权重。
        '''
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)  
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None):
        '''
        Args:
            input_ids: 输入文本的token IDs, [bs, seq_len]
            position_ids: 位置信息，用于表示每个token在输入文本中的位置，[bs, seq_len]
            token_type_ids: 用于区分两个句子的token类型，[bs, seq_len]
            past: 之前所有时刻的隐藏状态信息，
                  如果为None，则初始化为[None] * config.n_layer (time step = 0)
                  len(past) = config.n_layer, 
                  past[0] 为 key, past[1] 为 value
                  past[i].shape = [2, bs, n_head, pre_seq_len, head_features]
        Returns: 
            hidden_states: 经过 Transformer 层处理后的隐藏状态信息，[bs, seq_len, hidden_size]
            presents: 包含当前时间步及之前所有Transformer 层  key 和 value 的列表，用于上下文信息传递。
                      len(presents) = config.n_layer, 
                      presents[i].shape = [2, bs, n_head, cur_seq_len, head_features]
                      presents[0] 为 key, presents[1] 为 value
        '''
        if past is None: # 如果没有过去的隐藏状态信息，将 past_length 设为0，并将 past 初始化为与模型层数相同数量的 None 值。
            past_length = 0
            past = [None] * len(self.h) # [None] * len(self.h) -->  [None] * config.n_layer
        else:
            past_length = past[0][0].size(-2) # 用于计算当前的pos
        if position_ids is None:
            '''
            past_length 是过去的序列长度。在GPT-2模型中，为了处理连续的文本，通常需要将文本分成多个片段（例如，前一段文本和当前段文本），
            此处根据输入文本进行文本生成，pos和此前的文本有关
            所以，pos的范围是[past_length, input_ids.size(-1) + past_length)
            '''
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                        device=input_ids.device) # [input_ids.size(-1)] == [seq_len]
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)# [seq_len] --> [1, seq_len] --> [bs, seq_len]

        input_shape = input_ids.size() # [bs, seq_len]
        input_ids = input_ids.view(-1, input_ids.size(-1)) # [bs, seq_len]  
        position_ids = position_ids.view(-1, position_ids.size(-1)) # [bs, seq_len]  

        inputs_embeds = self.wte(input_ids) # [bs, seq_len] --> [bs, seq_len, hidden_size]
        position_embeds = self.wpe(position_ids) # [bs, seq_len] --> [bs, seq_len, hidden_size]
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) # [bs, seq_len] 
            token_type_embeds = self.wte(token_type_ids) # [bs, seq_len]  --> [bs , seq_len, hidden_size]
        else:
            token_type_embeds = 0 
        hidden_states = inputs_embeds + position_embeds + token_type_embeds # [bs , seq_len, hidden_size]
        presents = []  
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)  
            presents.append(present)
        hidden_states = self.ln_f(hidden_states) # LayerNorm [bs , seq_len, hidden_size]
        output_shape = input_shape + (hidden_states.size(-1),) # [bs, seq_len, hidden_size]
        return hidden_states.view(*output_shape), presents

class GPT2LMHead(nn.Module):
    '''
    负责生成文本的下一个单词预测（文本生成）
    '''
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd # 模型的隐藏层的维度 hidden_size
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False) # 输入维度为 embed_shape[1]（hidden_size），输出维度为 embed_shape[0]（vocab_size）
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        '''
        根据输入的隐藏状态 hidden_state，生成文本的下一个单词的预测
        Args:
            hidden_state: 输入的隐藏状态，经过模型处理后的结果, [bs, seq_len, hidden_size]
        Returns:
            lm_logits: [bs, seq_len, vocab_size]
        '''

        lm_logits = self.decoder(hidden_state) # [bs, seq_len, hidden_size] --> [bs, seq_len, vocab_size]
        return lm_logits # logits用于生成文本的下一个单词预测

class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__()
        self.transformer = GPT2Model(config) # 处理输入数据的转换
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config) # 生成文本的下一个单词预测（文本生成）

    def set_tied(self):
        """ 
        Make sure we are sharing the embeddings
        生成的文本与输入数据共享相同的嵌入空间
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        '''
        执行文本生成任务，生成下一个单词的预测或计算文本生成的损失。
        Args:
            input_ids: 输入文本的token IDs, [bs, seq_len]
            position_ids: 位置信息，用于表示每个token在输入文本中的位置，[bs, seq_len]
            token_type_ids: 用于区分两个句子的token类型，[bs, seq_len]
            lm_labels: 用于训练模型的标签，[bs, seq_len]
            past: 用于存储过去所有时刻的隐藏状态信息，
                  len(past) = config.n_layer, 
                  past[i].shape = [2, bs, n_head, pre_seq_len, head_features]
                  past[0] 为 key, past[1] 为 value
        Returns:
            lm_logits: 生成文本的下一个单词预测（文本生成）, [bs, seq_len, vocab_size]
            presents: 包含当前时间步及之前每个 Transformer 层 block 的 key 和 value 的列表，用于上下文信息传递。
                      len(presents) = config.n_layer, 
                      presents[i].shape = [2, bs, n_head, cur_seq_len, head_features]
                      presents[0] 为 key, presents[1] 为 value
        '''

        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past) # hidden_states: [bs, seq_len, hidden_size], presents: len(presents) = config.n_layer, presents[i].shape = [2, bs, n_head, current_seq_len, head_features]
        lm_logits = self.lm_head(hidden_states) # [bs, seq_len, hidden_size] --> [bs, seq_len, vocab_size]
        if lm_labels is not None: # training mode
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss
        return lm_logits, presents