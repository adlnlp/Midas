import transformers
from transformers import AutoTokenizer,AutoModel
import torch.nn as nn
import torch
from train_dcidsfpos import logger
class BERT(nn.Module):
    save_path = 'models/teacher'
    def __init__(self, name='bert-base-uncased',head={'type':'dc','nclasses':2},freeze = True,load_ft=None):
        super(BERT, self).__init__()
        self.model_name = name
        self.freeze = freeze
        self.head_cfg = head
        self.load_ft = load_ft
        try:
            self.model = AutoModel.from_pretrained('{}/{}'.format(self.save_path,self.model_name))
        except:
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.save_pretrained('{}/{}'.format(self.save_path,self.model_name), from_pt=True)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained('{}/{}'.format(self.save_path,self.model_name+"_tokenizer"))
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.save_pretrained('{}/{}'.format(self.save_path,self.model_name+"_tokenizer"))
        
        for param in self.model.parameters():
            param.requires_grad = not self.freeze

        # build head:
        if self.head_cfg['type'] in ('dc','id','sf','pos','cdc'):
            self.head = nn.Linear(self.model.config.hidden_size,self.head_cfg['nclasses'])

        if self.load_ft is not None:
            logger.info("Loading fine tuned state dict {}...".format(self.load_ft))
            sd = torch.load(self.load_ft)
            # neglect the pretrinaed head
            for name in list(sd.keys()):
                if name.startswith("head"):
                    logger.info("Neglect: {}".format(name))
                    sd.pop(name)
            # won't change the requires_grad
            self.load_state_dict(sd,strict=False)
        
        # for param in self.model.parameters():
        #     print(param.requires_grad)
            

    def forward(self, inputs):
        out = self.model(**inputs)
        if self.head_cfg['type'] in ('dc','id','cdc'):
            out1 = out['last_hidden_state'][:,0,:]
            out2 = self.head(out1)
        elif self.head_cfg['type'] in ('sf','pos'):
            out1 = out['last_hidden_state']
            out2 = self.head(out1)
            
        return out1, out2