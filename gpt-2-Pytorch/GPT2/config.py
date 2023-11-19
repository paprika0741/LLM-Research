'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
class GPT2Config(object):
    def __init__(
            self,
            vocab_size_or_config_json_file=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
    ):
        self.vocab_size = vocab_size_or_config_json_file # 用于指定词汇表的大小
        self.n_ctx = n_ctx # 模型的上下文长度（context length）
        self.n_positions = n_positions # 模型的最大位置编码（position encoding）的数量
        self.n_embd = n_embd # 模型的隐藏层的维度
        self.n_layer = n_layer # Transformer层的数量
        self.n_head = n_head # 注意力头的数量
        self.layer_norm_epsilon = layer_norm_epsilon # LayerNorm层中的epsilon参数（避免除0）
        self.initializer_range = initializer_range # 始化模型权重的范围
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.n_embd, self.n_head))