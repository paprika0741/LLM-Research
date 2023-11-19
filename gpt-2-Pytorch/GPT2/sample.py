'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import torch
import torch.nn.functional as F
from tqdm import trange

def top_k_logits(logits, k):
    '''
    确保生成的文本中只包含前 k 个最高概率的 token，而其他 token 的概率会被大大减小 (-1e10)
    logits : [bs, vocab_size] 
    '''
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k) # 用于获取 logits 张量中前 k 个最大的值和对应的索引 [bs, k]
    min_values = values[:, -1]  # 获取前 k 个最大值中的最小值 [bs]   
    return torch.where(logits < min_values.view(-1, 1), torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)  #  将 logits 张量中小于第k大值的替换为 -1e10

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True):
    '''
    从预训练模型中生成文本序列的函数
    Args:
        model (nn.Module)：预训练模型
        length (int): 生成文本的长度
        start_token: t_0输入的token, 
                    如果为None，则执行有条件生成，需要提供context。
                    如果不为None，则执行无条件生成，生成与输入无关的文本,context为None。
        batch_size (int): 生成文本btach的大小
        context: input context, 其中元素为token id， [bs, seq_len]
        temperature (float, optional): 控制生成文本的多样性，较高的值会增加随机性，较低的值会减少随机性。
        top_k (int, optional): 用于限制生成的token范围，只选择概率最高的top_k个token。
        device (str, optional): 执行生成的设备
        sample (bool, optional): 是否执行随机采样，如果为False，则每次选择概率最高的token。  
    Returns:
        output: 生成的文本序列  [bs, input_seq_len + args.length ]
    '''
    if start_token is None: # 有条件生成
        assert context is not None, 'Specify exactly one of start_token and context!'
        # list [seq_len] --> tensor [seq_len] --> [1, seq_len] --> [bs, seq_len] 
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else: # 无条件生成，生成与输入无关的文本
        assert context is None, 'Specify exactly one of start_token and context!'
        # [bs, 1]
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context # [bs, seq_len]
    output = context # [bs, seq_len]
    past = None
    with torch.no_grad():
        for i in trange(length): # tqdm range
            logits, past = model(prev, past=past) # logits: [bs, seq_len, vocab_size], past: len(past) = config.n_layer, past[i].shape = [2, bs, n_head, cur_seq_len, head_features]
            logits = logits[:, -1, :] / temperature# [bs, seq_len, vocab_size] --> [bs, vocab_size] 只关注最后一个 token 的 logits
            logits = top_k_logits(logits, k=top_k) # [bs, vocab_size] --> [bs, vocab_size]
            log_probs = F.softmax(logits, dim=-1) # [bs, vocab_size] --> [bs, vocab_size]
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1) # 在多项式分布中进行多项式采样 [bs, 1]
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1) # 每个位置上选择概率最高的 [bs, 1]
            output = torch.cat((output, prev), dim=1) # [bs, input_seq_len + i + 1 ]
    return output   # [bs, input_seq_len + args.length ]