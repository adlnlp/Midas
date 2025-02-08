import numpy as np
import json
from torch.utils.data import Dataset
import torch
import random
import re
import os
from copy import deepcopy
import pandas as pd
import nltk
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')

class M2MDCIDSFPOSDataset(Dataset):
    split2file = {
        'restaurant':{'train':'sim-R/train.json','val':'sim-R/dev.json','test':'sim-R/test.json'},
        'movie':{'train':'sim-M/train.json','val':'sim-M/dev.json','test':'sim-M/test.json'},
    }
    # labels = ['restaurant','movie']
    # len all_slots: 21, len all_states: 15, len all_domains: 2, len all_postags: 40
    def __init__(self, split, cc=False, data_dir = 'data/m2m'):
        self.data_dir = data_dir
        self.split = split
        self.cc = cc
        self.init_data()
    
    def get_slot_intent_domain(self,dataset):
        domains = []
        slots = []
        all_turns = []
        all_states = []
        all_domains = []
        slot_loc = []
        all_conversations = []
        all_conversation_domains = [] # single label task

        for i in range(len(dataset)):
            slots.append([[x['slot'] for x in y['dialogue_state']] for y in dataset.iloc[i]['turns']])

            domains += [y['user_intents'] for y in dataset.iloc[i]['turns'] if 'user_intents' in y]
            all_conversations.append([])
            all_conversation_domains.append(set())
            for utterance in dataset['turns'][i]:
                if 'system_utterance' in utterance:
                    all_turns.append(utterance['system_utterance']['tokens'])
                    slot_loc.append(utterance['system_utterance']['slots'])
                    slot_value = []
                    for dic in utterance['system_acts']:
                        if 'value' in dic:
                            slot_value.append({dic['slot']:dic['value']})
                        else:
                            slot_value.append({})
                    all_states.append(utterance['system_acts'][0]['type'].lower())
                    all_domains.append(domains[-1])

                    all_conversations[-1].append(all_turns[-1])
                    all_conversation_domains[-1].update(domains[-1])
                if len(utterance['user_acts']):
                    all_turns.append(utterance['user_utterance']['tokens'])
                    all_states.append(utterance['user_acts'][0]['type'].lower())
                    all_domains.append(domains[-1])

                    all_conversations[-1].append(all_turns[-1])
                    all_conversation_domains[-1].update(domains[-1])
                    slot_loc.append(utterance['user_utterance']['slots'])
            # print("all_conversations   ",all_conversations[-1])
        all_domains = [x.split('_')[1].lower() for xs in all_domains for x in xs]
        all_conversation_domains = [list(x) for x in all_conversation_domains]
        for conversation_domains in all_conversation_domains:
            for i in range(len(conversation_domains)):
                conversation_domains[i] = conversation_domains[i].split('_')[1].lower()
        all_conversation_domains = [list(set(x)) for x in all_conversation_domains]
        slots = [x for xs in slots for x in xs]

        slot_value_pairs = []
        for seq, slots in zip(all_turns, slot_loc):
            slot_value_pairs.append(['O']*len(seq))
            for slot_dic in slots:
                if slot_dic != {}:
                    start_loc = slot_dic['start']
                    end_loc = slot_dic['exclusive_end']
                    this_slot = slot_dic['slot']
                    slot_value_pairs[-1][start_loc] = 'B-' + this_slot
                    slot_value_pairs[-1][start_loc+1:end_loc] = ['I-' + this_slot]*(end_loc - start_loc - 1)
        return all_turns, slot_value_pairs, all_states, all_domains,all_conversations,all_conversation_domains
    
    def get_pos_tags(self):
        all_pos_tags = []
        for token_list in self.turns:
            one_pos_tags = [x[1] for x in pos_tag(token_list)] 
            all_pos_tags.append(one_pos_tags)
        return all_pos_tags

    def init_data(self):
        self.data = []
        raw_data = []
        for domain, data_path in self.split2file.items():
            file_path = os.path.join(self.data_dir,data_path[self.split])
            raw_data.append(pd.read_json(file_path))
        raw_data = pd.concat(raw_data,ignore_index=True)

        turns, slot_value_pairs, states, domains, conversations,conversation_domains = self.get_slot_intent_domain(raw_data)
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