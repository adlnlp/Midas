'''
code from https://github.com/DSKSD/RNN-for-Joint-NLU/blob/master/model.py
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer,AutoModel
import torch
from train_dcidsfpos import logger

class Encoder(nn.Module):
    def __init__(self, input_size,embedding_size, hidden_size ,n_layers=1):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, batch_first=True,bidirectional=True)
    
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        #self.lstm.weight.data.
    
    def init_hidden(self,input):
        hidden = Variable(torch.zeros(self.n_layers*2, input.size(0), self.hidden_size)).cuda() if USE_CUDA else Variable(torch.zeros(self.n_layers*2, input.size(0), self.hidden_size))
        context = Variable(torch.zeros(self.n_layers*2, input.size(0), self.hidden_size)).cuda() if USE_CUDA else Variable(torch.zeros(self.n_layers*2, input.size(0), self.hidden_size))
        return (hidden,context)
     
    def forward(self, input,input_masking):
        """
        input : B,T (LongTensor)
        input_masking : B,T (PAD 마스킹한 ByteTensor)
        
        <PAD> 제외한 리얼 Context를 다시 만들어서 아웃풋으로
        """
        
        self.hidden = self.init_hidden(input)
        
        embedded = self.embedding(input)
        output, self.hidden = self.lstm(embedded, self.hidden)
        
        real_context=[]
        
        for i,o in enumerate(output): # B,T,D
            real_length = input_masking[i].data.tolist().count(0) # 실제 길이
            real_context.append(o[real_length-1])
            
        return output, torch.cat(real_context).view(input.size(0),-1).unsqueeze(1)

class Decoder(nn.Module):
    
    def __init__(self,head_cfg,embedding_size,hidden_size,n_layers=1,dropout_p=0.1):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        # self.slot_size = slot_size
        # self.intent_size = intent_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.embedding_size = embedding_size
        self.head_cfg = head_cfg
        # Define the layers
        self.embedding = nn.Embedding(self.slot_size, self.embedding_size) #TODO encoder와 공유하도록 하고 학습되지 않게..

        #self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.embedding_size+self.hidden_size*2, self.hidden_size, self.n_layers, batch_first=True)
        self.attn = nn.Linear(self.hidden_size,self.hidden_size) # Attention

        self.slot_out = nn.Linear(self.hidden_size*2, self.head_cfg['sf'])
        self.intent_out = nn.Linear(self.hidden_size*2,self.head_cfg['id'])
        self.domain_out = nn.Linear(self.hidden_size*2,self.head_cfg['dx'])
    
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        #self.out.bias.data.fill_(0)
        #self.out.weight.data.uniform_(-0.1, 0.1)
        #self.lstm.weight.data.
    
    def Attention(self, hidden, encoder_outputs, encoder_maskings):
        """
        hidden : 1,B,D
        encoder_outputs : B,T,D
        encoder_maskings : B,T # ByteTensor
        """
        
        hidden = hidden.squeeze(0).unsqueeze(2)  # 히든 : (1,배치,차원) -> (배치,차원,1)
        
        batch_size = encoder_outputs.size(0) # B
        max_len = encoder_outputs.size(1) # T
        energies = self.attn(encoder_outputs.contiguous().view(batch_size*max_len,-1)) # B*T,D -> B*T,D
        energies = energies.view(batch_size,max_len,-1) # B,T,D (배치,타임,차원)
        attn_energies = energies.bmm(hidden).transpose(1,2) # B,T,D * B,D,1 --> B,1,T
        attn_energies = attn_energies.squeeze(1).masked_fill(encoder_maskings,-1e12) # PAD masking
        
        alpha = F.softmax(attn_energies) # B,T
        alpha = alpha.unsqueeze(1) # B,1,T
        context = alpha.bmm(encoder_outputs) # B,1,T * B,T,D => B,1,D
        
        return context # B,1,D
    
    def init_hidden(self,input):
        hidden = Variable(torch.zeros(self.n_layers*1, input.size(0), self.hidden_size)).cuda()
        context = Variable(torch.zeros(self.n_layers*1, input.size(0), self.hidden_size)).cuda()
        return (hidden,context)
    
    def forward(self, input,context,encoder_outputs,encoder_maskings,head_type):
        """
        input : B,L(length)
        enc_context : B,1,D
        """
        # Get the embedding of the current input word
        embedded = self.embedding(input)
        hidden = self.init_hidden(input)
        before_logits = {'dc':None,'id':None,'sf':[]}
        logits = {'dc':None,'id':None,'sf':[]}
        aligns = encoder_outputs.transpose(0,1)
        length = encoder_outputs.size(1)
        # if head_type in ('dc','cdc','id'):
        cls_hidden = hidden[0].clone() 
        cls_context = self.Attention(cls_hidden, encoder_outputs,encoder_maskings) 
        concated = torch.cat((cls_hidden,cls_context.transpose(0,1)),2).squeeze(0) # 1,B,D
        id_logits = self.intent_out(concated) # B,D
        dc_logits = self.domain_out(concated) # B,D
        before_logits['dc'] = concated
        before_logits['id'] = concated
        logits['dc'] = dc_logits
        logits['id'] = id_logits
        # if head_type in ('sf','pos'):
        for i in range(length): # Input_sequence와 Output_sequence의 길이가 같기 때문..
            aligned = aligns[i].unsqueeze(1)# B,1,D
            _, hidden = self.lstm(torch.cat((embedded,context,aligned),2), hidden) # input, context, aligned encoder hidden, hidden

            concated = torch.cat((hidden[0],context.transpose(0,1)),2).squeeze(0)
            score = self.head(concated)
            softmaxed = F.log_softmax(score)
            before_logits['sf'].append(concated)
            logits['sf'].append(score)
            _,input = torch.max(softmaxed,1)
            embedded = self.embedding(input.unsqueeze(1))
            
            # 그 다음 Context Vector를 Attention으로 계산
            context = self.Attention(hidden[0], encoder_outputs,encoder_maskings) 
        # 요고 주의! time-step을 column-wise concat한 후, reshape!!
        logits['sf'] = torch.cat(logits,1)
        before_logits['sf'] = torch.cat(before_logits,1)
        return before_logits, logits
    

class Seq2Seq(nn.Module):
    save_path = 'models/teacher'
    def __init__(self,name='seq2seq',tokenizer = 'bert-base-uncased',head={'dc':2,'sf':2,'id':2},emb_size=64,hidden_size = 64): #64,64 from the original code
        super(Seq2Seq, self).__init__()
        self.model_name = name
        self.tokenizer_name = tokenizer    
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('{}/{}'.format(self.save_path,self.tokenizer_name+"_tokenizer"))
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.tokenizer.save_pretrained('{}/{}'.format(self.save_path,self.tokenizer_name+"_tokenizer"))

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.head_cfg = head
        self.encoder = Encoder(len(self.tokenizer.get_vocab()),self.emb_size,self.hidden_size)
        self.decoder = Decoder(self.head_cfg,self.emb_size,self.hidden_size*2)

        self.encoder.init_weights()
        self.decoder.init_weights()

    def forward(self, inputs):
        # x,y_1,y_2 = zip(*batch)
        # x = torch.cat(x)
        # tag_target = torch.cat(y_1)
        # intent_target = torch.cat(y_2)
        x = inputs.input_ids
        x_mask = inputs.attention_mask
        # torch.cat([Variable(torch.ByteTensor(tuple(map(lambda s: s ==0, t.data)))).cuda() if USE_CUDA else Variable(torch.ByteTensor(tuple(map(lambda s: s ==0, t.data)))) for t in x]).view(BATCH_SIZE,-1)
        # y_1_mask = torch.cat([Variable(torch.ByteTensor(tuple(map(lambda s: s ==0, t.data)))).cuda() if USE_CUDA else Variable(torch.ByteTensor(tuple(map(lambda s: s ==0, t.data)))) for t in tag_target]).view(BATCH_SIZE,-1)
 
        # encoder.zero_grad()
        # decoder.zero_grad()

        output, hidden_c = self.encoder(x,x_mask)
        start_decode = torch.LongTensor([[self.tokenizer.bos_token_id]*x.size[0]]).cuda().transpose(1,0)
        before_logits, logits = self.decoder(start_decode,hidden_c,output,x_mask)
        return before_logits, logits
        # loss_1 = loss_function_1(tag_score,tag_target.view(-1))
        # loss_2 = loss_function_2(intent_score,intent_target)

        # loss = loss_1+loss_2
        # losses.append(loss.data.cpu().numpy()[0] if USE_CUDA else loss.data.numpy()[0])
        # loss.backward()

        # torch.nn.utils.clip_grad_norm(encoder.parameters(), 5.0)
        # torch.nn.utils.clip_grad_norm(decoder.parameters(), 5.0)

        # enc_optim.step()
        # dec_optim.step()

        # if i % 100==0:
        #     print("Step",step," epoch",i," : ",np.mean(losses))
        #     losses=[]
       