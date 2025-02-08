import transformers
from transformers import AutoTokenizer,AutoModel,AutoConfig
import torch.nn as nn
import torch
from train_dcidsfpos import logger
class Student_BERT(nn.Module):
    save_path = 'models/student'
    def __init__(self, config,params):
        super(Student_BERT, self).__init__()
        self.global_config = config
        self.params = params
        self.init_attr_from_config()
        self.init_model()
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("pytorch_total_params: ",pytorch_total_params)

    def init_attr_from_config(self):
        # print("init_attr_from_config")
        model_config = self.global_config['STUDENT']
        self.bert_name = model_config['bert_name']
        self.teacher_config = self.global_config['TEACHERS']
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('{}/{}'.format(self.save_path,self.bert_name.split('/')[1]+"_tokenizer"))
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_name)
            self.tokenizer.save_pretrained('{}/{}'.format(self.save_path,self.bert_name.split('/')[1]+"_tokenizer"))

    def init_model(self):
        self.bert_config = AutoConfig.from_pretrained(self.bert_name)
        print("bert config", self.bert_config)
        logger.info(self.bert_config)
        self.model = AutoModel.from_config(self.bert_config)  
        

        self.task_heads = {}
        for teacher_name in self.teacher_config['teacher_list']:
            te_cfg = self.teacher_config[teacher_name]
            if te_cfg['head']['type'] not in self.task_heads:
                self.task_heads[te_cfg['head']['type']] = nn.Linear(self.bert_config.hidden_size,te_cfg['head']['nclasses'])
        self.task_heads = nn.ModuleDict(self.task_heads)
            

    def forward(self, input_ids, attention_mask=None):
        out = self.model(input_ids = input_ids,attention_mask = attention_mask)
        out1 = out['last_hidden_state']
        out2 = {}
        for task, head in self.task_heads.items():
            out2[task] = head(out1)
        return out1, out2