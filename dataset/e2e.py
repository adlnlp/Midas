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
import pandas as pd
import nltk
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')

class E2ESFPOSDataset(Dataset):
    split2file = {
        'train':'train-fixed.no-ol.csv',
        'val':'devel-fixed.no-ol.csv',
        'test':'test-fixed.csv'
    }
    # len all_slots: 23, len all_states: 1, len all_domains: 1
    # labels = ['restaurant','hotel']
    def __init__(self, split, data_dir = 'data/e2e/cleaned-data'):
        self.data_dir = data_dir
        self.split = split

        self.init_dc_data()

    def get_slot_intent_domain(self,utterances,slots):
        all_turns = []
        tmp_slots = []
        slot_value_pairs = []

        for sentence, slot_values in zip(utterances,slots):
            sentence = sentence.lower()
            slot_values = slot_values.lower()
            all_turns.append(sentence.split())
            slot_label = ['O'] * len(all_turns[-1])
            slot_value_list = slot_values.split(',')
            # print("all_turns ",all_turns[-1])
            for one_slot_value in slot_value_list:
                split_one_slot_value = one_slot_value.split('[')
                slot = split_one_slot_value[0].lower()
                value = split_one_slot_value[1][:-1]

                len_value = len(value.split())
                # print("value ",value)
                start_poss = [substr.start() for substr in re.finditer(value, sentence)]
                for start in start_poss:
                    start_token_idx = len(sentence[:start].rsplit(' ')[0].split())
                    if slot_label[start_token_idx] != 'O':
                        # To check whether here is already another slot?
                        pass
                    slot_label[start_token_idx] = 'B-'+slot.strip()
                    for i in range(start_token_idx+1,start_token_idx+len_value):
                        slot_label[i] = 'I-'+slot.strip()
            
            slot_value_pairs.append(slot_label)
            assert len(slot_value_pairs[-1]) == len(all_turns[-1]), "????????"
        # print("len( )",len(all_turns),len(slot_value_pairs),len(all_states),len(all_domains))
        return all_turns, slot_value_pairs

    def get_pos_tags(self):
        all_pos_tags = []
        for token_list in self.turns:
            one_pos_tags = [x[1] for x in pos_tag(token_list)] 
            all_pos_tags.append(one_pos_tags)
        return all_pos_tags
    
    def init_dc_data(self):
        all_dials = []
        split_flie =pd.read_csv(os.path.join(self.data_dir,self.split2file[self.split]))
        slots = split_flie['mr'].tolist()
        utterances = split_flie['ref'].tolist()
        
        turns, slot_value_pairs = self.get_slot_intent_domain(utterances,slots)
        self.turns = turns
        self.slot_value_pairs = slot_value_pairs
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
        # useless attribution
        self.states = ['none']
        self.domains = ['restaurant']      

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.turns[idx], self.slot_value_pairs[idx], 'none', 'restaurant', self.pos_tags[idx])