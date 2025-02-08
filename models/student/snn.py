import transformers
from transformers import AutoTokenizer,AutoModel,AutoConfig
import torch.nn as nn
import torch
from train_dcidsfpos import logger

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        '''
        Copy from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        '''
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # print('............  ',pe.shape)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1),:]
        return self.dropout(x)

class SimpleNN(nn.Module):
    save_path = 'models/student'
    def __init__(self, config,params):
        super(SimpleNN, self).__init__()
        self.global_config = config
        self.params = params
        self.init_attr_from_config()
        self.init_model()
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("pytorch_total_params: ",pytorch_total_params)

    def init_attr_from_config(self):
        model_config = self.global_config['STUDENT']
        self.hidden_dim = model_config.get('output_dim',1024)
        self.layers = model_config.get('layers',1)
        self.dropout = model_config.get('dropout',0.1)
        self.emb_dim = model_config.get('emb_dim', 768)
        self.pe_type = model_config.get('pe_type', 'none')

        self.teacher_config = self.global_config['TEACHERS']
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('{}/{}'.format(self.save_path,'t5-base'+"_tokenizer"))
        except:
            self.tokenizer = AutoTokenizer.from_pretrained('t5-base')
            self.tokenizer.save_pretrained('{}/{}'.format(self.save_path,'t5-base'+"_tokenizer"))

    def init_model(self):
        self.emb_layer = nn.Embedding(num_embeddings = len(self.tokenizer.get_vocab()),embedding_dim = self.emb_dim,\
                                      padding_idx=self.tokenizer.pad_token_id)
        # print('init use_tr_tokenizer done')
        self.embedding_dim = self.emb_layer.weight.shape[1]
        
        self.linear_relu_stack = nn.Sequential()
        init_dim = self.embedding_dim
        for i in range(self.layers):
            self.linear_relu_stack.append(nn.Linear(init_dim, self.hidden_dim))
            self.linear_relu_stack.append(nn.ReLU())
            init_dim = self.hidden_dim
           
        self.linear_relu_stack.append(nn.Dropout(p=self.dropout))

        if self.pe_type == 'absolute_sin':
            self.pe = PositionalEncoding(d_model=self.embedding_dim, dropout=self.dropout, max_len=5000)
        

        self.task_heads = {}
        for teacher_name in self.teacher_config['teacher_list']:
            te_cfg = self.teacher_config[teacher_name]
            if te_cfg['head']['type'] not in self.task_heads:
                self.task_heads[te_cfg['head']['type']] = nn.Linear(self.hidden_dim,te_cfg['head']['nclasses'])
        self.task_heads = nn.ModuleDict(self.task_heads)
            

    def forward(self, input_ids, attention_mask=None):
        out = self.emb_layer(input_ids)
        if self.pe_type == 'absolute_sin':
            out = self.pe(out)
        out = self.linear_relu_stack(out)
        out1 = out
        out2 = {}
        for task, head in self.task_heads.items():
            out2[task] = head(out1)
        return out1, out2
    