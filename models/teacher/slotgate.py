'''
code from Tri-JNLU https://github.com/DSKSD/RNN-for-Joint-NLU/blob/master/model.py
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

class SlotGate(nn.Module):
    def __init__(self, hidden_dim, activation_func):
        super(SlotGate, self).__init__()
        self.fc_intent_context = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.activation_func = activation_func
    def forward(self, slot_context, intent_context): # For each time stamp
        '''
        slot_context: [batch_size, hidden_dim] at each time stamp
        intent_context:[batch_size, hidden_dim]
        '''
        # intent_context_linear: [batch_size, hidden_dim]
        intent_context_linear = self.fc_intent_context(intent_context)
        
        # sum_intent_slot_context: [batch_size, hidden_dim]
        sum_intent_slot_context = slot_context + intent_context_linear
        
        scaled = None
        if self.activation_func == 'tanh':
          scaled = torch.tanh(sum_intent_slot_context)
        elif self.activation_func == 'sigmoid':
          scaled = torch.sigmoid(sum_intent_slot_context)

        # fc_linear: [batch_size, hidden_dim]
        fc_linear = self.fc_v(scaled)
        
        # sum_gate_vec: [batch_size]
        sum_gate_vec = torch.sum(fc_linear, dim=1) # This is the 'g' in paper
        
        return sum_gate_vec
    
# To calculate the slot contexts and intent context (use matrix multiplication)
class AttnContext(nn.Module):
    def __init__(self, hidden_dim):
        super(AttnContext, self).__init__()

    def forward(self, hidden, source_output_hidden):
        # source_output_hidden: [batch_size, seq_len, hidden_size], is the output label sequence
        # hidden: [batch_size, hidden_size]
        hidden = hidden.unsqueeze(1) # [batch_size, 1, hidden_size]
        
        attn_weight = torch.sum(hidden * source_output_hidden, dim=2) # [batch_size, seq_len]
        attn_weight = F.softmax(attn_weight, dim=1).unsqueeze(1) # [batch_size, 1, seq_len]
        
        attn_vector = attn_weight.bmm(source_output_hidden) # [batch_size, 1, hidden_size]
        
        return attn_vector.squeeze(1) # [batch_size, hidden_size]

class SGJointModel(nn.Module):
    def __init__(self, source_input_dim, source_emb_dim, hidden_dim, 
                n_layers, dropout, pad_index, slot_output_size, intent_output_size, 
                seq_len, activation_func, slot_attention_flag=True):
        super(SGJointModel, self).__init__()
        self.pad_index = pad_index
        self.hidden_dim = hidden_dim//2 # bilstm
        self.n_layers = n_layers
        self.slot_output_size = slot_output_size
        # whether to predict or not
        self.predict_flag = None
        # Full attention/ Intent attention
        self.slot_attention_flag = slot_attention_flag
        self.source_embedding = nn.Embedding(source_input_dim, source_emb_dim, padding_idx=pad_index)
        self.source_lstm = nn.LSTM(source_emb_dim, self.hidden_dim, 
                                    n_layers, dropout=dropout, bidirectional=True, batch_first=True)
        
        # slot context
        self.slot_context = AttnContext(hidden_dim)
        
        # intent context
        self.intent_context = AttnContext(hidden_dim)
        
        # slotgate 
        self.slotGate = SlotGate(hidden_dim, activation_func)
        
        # intent prediction
        self.intent_output = nn.Linear(hidden_dim, intent_output_size)

        # intent prediction
        self.intent_output = nn.Linear(hidden_dim, intent_output_size)
        
        # slot prediction
        self.slot_output = nn.Linear(hidden_dim, slot_output_size)
        
        
    def forward(self, source_input, source_len , predict_flag=False):
        '''
        source_input: [batch_size, seq_len]
        source_len: [batch_size]
        '''
        self.predict_flag = predict_flag
        if self.predict_flag:
            assert len(source_input) == 1 #One sentence per iteration
            seq_len = source_len[0]
            
            'Encoder:'
            # source_embedded: [batch_size, seq_len, source_emb_dim]
            source_embedded = self.source_embedding(source_input)
            packed = torch.nn.utils.rnn.pack_padded_sequence(source_embedded, 
                                                            source_len, batch_first=True, enforce_sorted=False) 
            source_output, hidden = self.source_lstm(packed)
            # source_output=[batch_size, seq_len, 2 * self.hidden_size]
            # hidden=[n_layers * 2, batch_size, self.hidden_size]
            source_output, _ = torch.nn.utils.rnn.pad_packed_sequence(source_output, 
                                                                        batch_first=True, padding_value=self.pad_index, total_length=len(source_input[0])) 
            
            batch_size = source_input.shape[0]
            seq_len = source_input.shape[1]

            # save the slot context vectors
            slot_outputs = torch.zeros(batch_size, seq_len, self.slot_output_size).to(device)       
                
            aligns = source_output.transpose(0,1) # The encder hidden states (the  alignment information)
            
            output_tokens =[]

            'Decoder'    
            # slot filling
            for t in range(seq_len):
                '''
                hi at time stamp i
                '''
                aligned = aligns[t] # [batch_size, hidden_size]
                    
                # Full attention? 
                if self.slot_attention_flag:
                    
                    # [batch_size, hidden_size]
                    slot_context = self.slot_context(aligned, source_output)
                    
                    # [batch_size, hidden_size]，the last hidden states from the encoder (bilstm)
                    intent_context = self.intent_context(source_output[:,-1,:], source_output)
                    
                    # gate mechanism，[batch_size]
                    slot_gate = self.slotGate(slot_context, intent_context)
                    
                    # slot_gate: [batch_size, 1]
                    slot_gate = slot_gate.unsqueeze(1)
                    
                    # slot_context_gate: [batch_size, hidden_dim]
                    slot_context_gate = slot_gate * slot_context
                    
                # intent attention only
                else:
                        # [batch_size, hidden_size]
                    intent_context = self.intent_context(source_output[:,-1,:], source_output)
                    
                    # gate，[batch_size]
                    slot_gate = self.slotGate(source_output[:,t,:], intent_context)
                    
                        # slot_gate: [batch_size, 1]
                    slot_gate = slot_gate.unsqueeze(1)
                    
                    # slot_context_gate: [batch_size, hidden_dim]
                    slot_context_gate = slot_gate * source_output[:,t,:]
                
                
                # slot prediction, [batch_size, slot_output_size]
                slot_prediction = self.slot_output(slot_context_gate + source_output[:,t,:])
                slot_outputs[:, t, :] = slot_prediction
                
                
            #intent prediction
            intent_outputs = self.intent_output(intent_context + source_output[:,-1,:])

            return slot_outputs, intent_outputs
            
        # Training step 
        else:
            # source_embedded:[batch_size, seq_len, source_emb_dim]
            source_embedded = self.source_embedding(source_input)
            packed = torch.nn.utils.rnn.pack_padded_sequence(source_embedded, 
                                                            source_len, batch_first=True, enforce_sorted=False) 
            source_output, hidden = self.source_lstm(packed)
            # source_output=[batch_size, seq_len, 2 * self.hidden_size]
            # hidden=[n_layers * 2, batch_size, self.hidden_size]
            source_output, _ = torch.nn.utils.rnn.pad_packed_sequence(source_output, batch_first=True, padding_value=self.pad_index, total_length=len(source_input[0])) 
            
            batch_size = source_input.shape[0]
            seq_len = source_input.shape[1]
            # save the slot context vectors
            slot_outputs = torch.zeros(batch_size, seq_len, self.slot_output_size).to(device)              
            aligns = source_output.transpose(0,1) 
                
            # slot filling 
            for t in range(seq_len):
                '''
                hidden state hi at this time stamp i
                '''
                aligned = aligns[t]# [batch_size, hidden_size]
                    
                # whether to calculate the slot attention
                if self.slot_attention_flag:
                    
                    # [batch_size, hidden_size]
                    slot_context = self.slot_context(aligned, source_output)
                    
                    # [batch_size, hidden_size]
                    intent_context = self.intent_context(source_output[:,-1,:], source_output)
                    
                    # gate，[batch_size]
                    slot_gate = self.slotGate(slot_context, intent_context)
                    
                    # slot_gate: [batch_size, 1]
                    slot_gate = slot_gate.unsqueeze(1)
                    
                    # slot_context_gate:[batch_size, hidden_dim]
                    slot_context_gate = slot_gate * slot_context
                    
                # intent attention using intent context and hidden states from the encoder to calculate the slot gate
                else:
                        # [batch_size, hidden_size]
                    intent_context = self.intent_context(source_output[:,-1,:], source_output)
                    
                    # gate，[batch_size]
                    slot_gate = self.slotGate(source_output[:,t,:], intent_context)
                    
                        # slot_gate: [batch_size, 1]
                    slot_gate = slot_gate.unsqueeze(1)
                    
                    # slot_context_gate: [batch_size, hidden_dim]
                    slot_context_gate = slot_gate * source_output[:,t,:]
                
                # slot prediction, [batch_size, slot_output_size]
                slot_prediction = self.slot_output(slot_context_gate + source_output[:,t,:])
                slot_outputs[:, t, :] = slot_prediction
                
                
            #intent prediction
            intent_outputs = self.intent_output(intent_context + source_output[:,-1,:]).squeeze().unsqueeze(0) 
            intent = torch.broadcast_to(intent_outputs, (slot_outputs.size()[1], intent_outputs.size()[1], intent_outputs.size()[2])).permute(1,0,2)  # [batch_size, trg_len, output_dim]

            return slot_outputs, intent