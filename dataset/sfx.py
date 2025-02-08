import numpy as np
import json
from torch.utils.data import Dataset
import torch
import random
import re
import os
from copy import deepcopy
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')

def normalize_text(text):
    text = re.sub(r'child -s', 'children', text)
    text = re.sub(r' -s', 's', text)
    text = re.sub(r' -ly', 'ly', text)
    return text


class SFXDCIDSFPOSDataset(Dataset):
    split2file = {
        'restaurant':'sfxrestaurant/train+valid+test.json',
        'hotel':'sfxhotel/train+valid+test.json',
    }
    split_flie = "train_val_test.pt"
    # labels = ['restaurant','hotel']
    # len all_slots: 14, len all_states: 21, len all_domains: 2
    def __init__(self, split, cc = False, data_dir = 'data/sfx'):
        self.data_dir = data_dir
        self.split = split
        self.cc = cc

        self.init_data()

    def get_slot_intent_domain(self,split_dials):
        all_turns = []
        all_domains = []
        all_states = []
        slot_value_pairs = []
        slot_value_dicts = []
        all_conversations = []
        all_conversation_domains = [] # single label task
        
        for dial_data in split_dials:
            all_conversations.append([])
            all_conversation_domains.append(set())
            for ti, turn in enumerate(dial_data['dial']): 
                for utterance, side in zip((turn["S"]['base'],turn["U"]['hyp']),('S','U')):
                    if len(utterance) > 0:
                        all_turns.append(normalize_text(utterance).split())
                        all_domains.append(dial_data['domain'].lower())
                        all_conversations[-1].append(all_turns[-1])
                        all_conversation_domains[-1].add(all_domains[-1])
                        one_value_slot_dict = {}
                        if len(turn[side]['dact']) > 0:
                            # follow m2m, only use the first one
                            actions = turn[side]['dact'].split('|')
                            if actions[0] == 'Unrecognizable':
                                all_states.append("unrecognizable")
                            else:
                                all_states.append(actions[0].split('(',1)[0].lower())
                                for action_data in actions:
                                    slotvalues = action_data.split('(',1)[1][:-1]
                                    if len(slotvalues) == 0:
                                        continue
                                    for slotvalue in slotvalues.split(';'):
                                        # print("slotvalue. ",slotvalue)
                                        if '=' in slotvalue:
                                            slot, value = slotvalue.split('=')
                                            if slot == 'pricerange':
                                                slot = 'price_range'
                                            elif slot == 'goodformeal':
                                                slot = 'good_for_meal'
                                            if slot == '' or value == '':
                                                continue
                                            one_value_slot_dict[value] = slot
                                        else:
                                            continue
                        else:
                            all_states.append("unrecognizable")
                        slot_value_dicts.append(one_value_slot_dict)

        for turn_id in range(len(all_turns)):
            slot_label = ['O'] * len(all_turns[turn_id])
            one_value_slot_dict = slot_value_dicts[turn_id]
            text = ' '.join(all_turns[turn_id])
            for value, slot in one_value_slot_dict.items():
                slot = slot.lower()
                len_value = len(value.split())
                start_poss = [substr.start() for substr in re.finditer(value, text)]
                for start in start_poss:
                    start_token_idx = len(text[:start].split())
                    if slot_label[start_token_idx] != 'O':
                        # To check whether here is already another slot?
                        pass
                    slot_label[start_token_idx] = 'B-'+slot
                    for i in range(start_token_idx+1,start_token_idx+len_value):
                        slot_label[i] = 'I-'+slot
            
            slot_value_pairs.append(slot_label)
        # print("len( )",len(all_turns),len(slot_value_pairs),len(all_states),len(all_domains))
        return all_turns, slot_value_pairs, all_states, all_domains, all_conversations, all_conversation_domains
    
    def get_pos_tags(self):
        all_pos_tags = []
        for token_list in self.turns:
            one_pos_tags = [x[1] for x in pos_tag(token_list)] 
            all_pos_tags.append(one_pos_tags)
        return all_pos_tags

    def init_data(self):
        all_dials = []
        for domain in ['restaurant', 'hotel']:
            data_path = self.split2file[domain]
            raw_data = json.load(open(os.path.join(self.data_dir,data_path), "r"))
            for dial_dict in raw_data:
                dial_dict['domain'] = domain
                all_dials.append(dial_dict)
               
        split_file_path = os.path.join(self.data_dir,self.split_flie)
        if os.path.exists(split_file_path):
            split_ids = torch.load(split_file_path)
        else:
            all_ids = range(len(all_dials)) 
            train, test = train_test_split(all_ids,train_size=0.8)
            train, val = train_test_split(train,train_size=0.9)
            split_ids = {'train':train,'val':val,'test':test}
            torch.save(split_ids,split_file_path)

        split_dials = []
        # print("split_ids ",split_ids)
        # print("all_dials ",len(all_dials),max(split_ids[self.split]))
        for i in split_ids[self.split]:
            split_dials.append(all_dials[i])
        
        turns, slot_value_pairs, states, domains, conversations, conversation_domains = self.get_slot_intent_domain(split_dials)
        self.turns = turns
        self.slot_value_pairs = slot_value_pairs
        self.states = states
        self.domains = domains
        self.conversations = conversations
        self.conversation_domains = conversation_domains
        if self.cc == True:
            self.len = len(self.conversations)
        else:
            self.len = len(self.turns)     

        split_pos_flie = os.path.join(self.data_dir,'{}_pos.pt'.format(self.split))
        if os.path.exists(split_pos_flie):
            self.pos_tags = torch.load(split_pos_flie)
        else:
            self.pos_tags = self.get_pos_tags()
            torch.save(self.pos_tags, split_pos_flie)  

        # check
        for a,b in zip(self.slot_value_pairs, self.turns):
            assert len(a) == len(b), "ERROR! {}:{}".format(len(a),len(b)) 

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.cc == True:
            return (self.conversations[idx], self.conversation_domains[idx])
        else:
            return (self.turns[idx], self.slot_value_pairs[idx], self.states[idx], self.domains[idx], self.pos_tags[idx])