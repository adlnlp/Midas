### from __future__ import print_function
'''
This code is based on https://github.com/FLHonker/AMTML-KD-code
'''

import os
import time
import logging
import argparse
import random
import numpy as np
# from visdom import Visdom
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from utils import *
import dataset
from utils.params import get_params
from utils.early_stop import EarlyStopping
from sklearn.preprocessing import LabelEncoder,MultiLabelBinarizer
from sklearn.model_selection import ParameterGrid
from lightning_fabric.utilities.seed import seed_everything
import shutil
import sys
import json
from loguru import logger
import torchmetrics
from tqdm.autonotebook import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
import math
import dadaptation
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.metrics import f1_score as seqval_f1_score
# Teacher models:
# VGG11/VGG13/VGG16/VGG19, GoogLeNet, AlxNet, ResNet18, ResNet34, 
# ResNet50, ResNet101, ResNet152, ResNeXt29_2x64d, ResNeXt29_4x64d, 
# ResNeXt29_8x64d, ResNeXt29_32x64d, PreActResNet18, PreActResNet34, 
# PreActResNet50, PreActResNet101, PreActResNet152, 
# DenseNet121, DenseNet161, DenseNet169, DenseNet201, 
# BERT, RoBERTa
import models

g_exp_config = None

# Student models:
# myNet, LeNet, FitNet
# Transformer Encoder

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # Shouldn't occur unseen I-xxx label
            # # If the label is B-XXX we change it to I-XXX
            # if label % 2 == 1:
            #     label += 1
            new_labels.append(label)

    return new_labels

# def truncate_slot_labels(labels, word_ids):

def collate_fn_cdc(batch, tokenizer, max_seq_length, lEncs):
    texts = []
    cd_labels = []
    # print(len(batch),len(batch[0]))
    for tokens_list, domain_labels in zip(batch[0],batch[1]):
        if "Llama" in tokenizer.__class__.__name__:
            sep = [tokenizer.eos_token]
        else:
            sep = [tokenizer.sep_token]
        concat_tokens = []
        # print("tokens_list ",tokens_list)
        # print("tokens_list ",tokens_list)
        for tokens in tokens_list:
            concat_tokens += tokens + sep
        concat_tokens = concat_tokens[:-1]
        texts.append(concat_tokens)
        cd_labels.append(domain_labels)
    # print("texts ",texts[0])
    # print("cd_labels 1",cd_labels)
    cd_labels = lEncs['cdc'].transform(cd_labels)
    # print("cd_labels 2",cd_labels)
    # print("batch ",batch)
    # print("texts ",texts[0])
    inputs = tokenizer(texts,is_split_into_words=True,padding=True,\
                       truncation=True,max_length=max_seq_length,return_tensors='pt')
    inputs = inputs.to('cuda')
    cd_labels = torch.tensor(cd_labels).cuda()
    labels = {'cdc':cd_labels}
    return inputs, labels

# collate batch data for differnet models
def collate_fn_dc_sf_di(batch, tokenizer, max_seq_length,lEncs):
    texts = []
    s_labels = []
    tmp_s_labels = []
    i_labels = []
    d_labels = []
    p_labels = []
    tmp_p_labels = []
    # print(len(batch),len(batch[0]))
    for tokens,slot_label, intent_label, domain_label, pos_label in zip(batch[0],batch[1],batch[2],batch[3],batch[4]):
        texts.append(tokens)
        tmp_s_labels.append(slot_label)
        assert len(tokens) == len(slot_label), "ERROR! {}:{}".format(len(tokens),len(slot_label))
        i_labels.append(intent_label)
        d_labels.append(domain_label)
        tmp_p_labels.append(pos_label)
    tmp_s_labels = [lEncs['sf'].transform(one_s_labels) for one_s_labels in tmp_s_labels]
    tmp_p_labels = [lEncs['pos'].transform(one_p_labels) for one_p_labels in tmp_p_labels]
    # print("batch ",batch)
    inputs = tokenizer(texts,is_split_into_words=True,padding=True,\
                       truncation=True,max_length=max_seq_length,return_tensors='pt')
    # keep the word-wise labels but flatten here, because we will merge the token to words later
    # for i, labels in enumerate(tmp_s_labels):
    #     word_ids = inputs.word_ids(i)
    #     s_labels.append(align_labels_with_tokens(labels, word_ids))
    tmp_s_labels = [l for labels in tmp_s_labels for l in labels]
    tmp_p_labels = [l for labels in tmp_p_labels for l in labels]

    inputs = inputs.to('cuda')
    s_labels = torch.tensor(tmp_s_labels).cuda()
    i_labels = torch.tensor(lEncs['id'].transform(i_labels)).cuda()
    d_labels = torch.tensor(lEncs['dc'].transform(d_labels)).cuda()
    p_labels = torch.tensor(tmp_p_labels).cuda()
    labels = {'sf':s_labels,'dc':d_labels,'id':i_labels,'pos':p_labels}
    return inputs, labels

def collate_fn_default(samples):
    tokens,slot_label, intent_label, domain_label, pos_label = [],[],[],[],[]
    for turn, slot_value_pairs, state, domain, pos_tags in samples:
        tokens.append(turn)
        slot_label.append(slot_value_pairs)
        intent_label.append(state)
        domain_label.append(domain)
        pos_label.append(pos_tags)
    return (tokens,slot_label, intent_label, domain_label, pos_label)

def collate_fn_default_cc(samples):
    tokens, cc_domain_label = [],[]
    # print("collate_fn_default_cc   -----------")
    for conv, domains in samples:
        tokens.append(conv)
        cc_domain_label.append(domains)
    # print("tokens ",tokens[0])
    return (tokens,cc_domain_label)

# avg diatill
def distillation_loss(y, labels, logits, T, alpha=0.7):
    return nn.KLDivLoss(reduction="batchmean")(F.log_softmax(y/T,dim=1), logits) * (T*T * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)

# triplet loss
triplet_loss = nn.TripletMarginLoss(margin=0.2, p=2).cuda()

# get max infoentropy scores
# input: Tensor[3, 128, 10]
def maxInfo_logits(te_scores_Tensor):
    used_score = torch.FloatTensor(te_scores_Tensor.size(1), te_scores_Tensor.size(2)).cuda()
    ents = torch.FloatTensor(te_scores_Tensor.size(0), te_scores_Tensor.size(1)).cuda()
    logp = torch.log2(te_scores_Tensor)
    plogp = -logp.mul(te_scores_Tensor)
    for i,te in enumerate(plogp):
        ents[i] = torch.sum(te, dim=1)
    max_ent_index = torch.max(ents, dim=0).indices   # 取每一列最大值index
#     print(max_ent_index)
    for i in range(max_ent_index.size(0)):
        used_score[i] = te_scores_Tensor[max_ent_index[i].item()][i]
#     print(used_score)

    return used_score
    
# avg logits
# input: Tensor[3, 128, 10]
def avg_logits(te_scores_Tensor):
#     print(te_scores_Tensor.size())
    mean_Tensor = torch.mean(te_scores_Tensor, dim=1)
#     print(mean_Tensor)
    return mean_Tensor
    
# random logits
def random_logits(te_scores_Tensor):
    return te_scores_Tensor[np.random.randint(0, 1, 1)]

# input: t1, t2 - triplet pair
def triplet_distance(t1, t2):
    return (t1 - t2).pow(2).sum()
    
# get triplets
def random_triplets(st_maps, te_maps):
    conflict = 0
    st_triplet_list = []
    triplet_set_size = st_maps.size(0)
    batch_list = [x for x in range(triplet_set_size)]
    for i in range(triplet_set_size):
        triplet_index = random.sample(batch_list, 3)
        anchor_index = triplet_index[0]  # denote the 1st triplet item as anchor
        st_triplet = st_maps[triplet_index]
        te_triplet = te_maps[triplet_index]
        distance_01 = triplet_distance(te_triplet[0], te_triplet[1])
        distance_02 = triplet_distance(te_triplet[0], te_triplet[2])
        if distance_01 > distance_02:
            conflict += 1
            # swap postive and negative
            st_triplet[1], st_triplet[2] = st_triplet[2], st_triplet[1]
        st_triplet_list.append(st_triplet)
    
    st_triplet_batch = torch.stack(st_triplet_list, dim=1)
    return st_triplet_batch
    
# get the smallest conflicts index
def smallest_conflict_teacher(st_maps, te_maps_list):
    # print("st_maps ",st_maps.shape)
    index = 0
    triplet_set_size = st_maps.size(0)
    min_conflict = 1
    batch_list = [x for x in range(triplet_set_size)]
    triplet_index = random.sample(batch_list, 3)
    anchor_index = triplet_index[0]  # denote the 1st triplet item as anchor
    for idx, te_maps in enumerate(te_maps_list):
        # print("te_maps ",te_maps.shape)
        conflict = 0
        for i in range(triplet_set_size):
            st_triplet = st_maps[triplet_index]
            te_triplet = te_maps[triplet_index]
            # print("(t1.shape, t2.shape)",te_triplet.shape, te_triplet.shape)
            distance_01 = triplet_distance(te_triplet[0], te_triplet[1])
            distance_02 = triplet_distance(te_triplet[0], te_triplet[2])
            if distance_01 > distance_02:
                conflict += 1
        conflict /= triplet_set_size
        conflict = min(conflict, (1-conflict))
        if conflict < min_conflict:
            index = idx
    return index

def configure_optimizer(exp_cfg,all_models,total_steps):
    all_model_params = []
    lr = exp_cfg['lr']
    pytorch_total_params = 0
    for model in all_models:
        all_model_params.append(dict(params=model.parameters(), lr=lr))
        pytorch_total_params += sum(p.numel() for p in model.parameters())
    logger.info("Total Parameters: {} M".format(pytorch_total_params*1e-6))

    if exp_cfg['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(all_model_params, lr=lr)
    elif exp_cfg['optimizer']== 'adam':
        optimizer = torch.optim.Adam(all_model_params, lr=lr)
    elif exp_cfg['optimizer']== 'rmsprop':
        optimizer = torch.optim.RMSprop(all_model_params, lr=lr)
    elif exp_cfg['optimizer']== 'sgd':
        for model_params in all_model_params:
            model_params.update(dict(momentum=0.9, weight_decay=5e-4))
        optimizer = torch.optim.SGD(all_model_params, lr=lr, momentum=0.9)
    # elif exp_cfg.optimizer == 'd_adaptation':
    #     # By setting decouple=True, it use AdamW style weight decay
    #     # lr is needed, see https://github.com/facebookresearch/dadaptation
    #     optimizer = dadaptation.DAdaptAdam(all_model_params, lr=lr, \
    #                                         decouple=True,log_every=10) 

    if exp_cfg['warmup'] < 1:
        warmup = int(exp_cfg['warmup'] * total_steps)
    else:
        warmup = exp_cfg['warmup']

    if exp_cfg['lrscheduler'] == 'cosinewarmup':
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps, last_epoch=-1)
    elif exp_cfg['lrscheduler'] == 'constantwarmup':
        scheduler = get_constant_schedule_with_warmup(optimizer,num_warmup_steps=warmup,last_epoch=-1)
    return optimizer, scheduler

def merge_logits_for_sf(before_classifier, logits, inputs):
    # print("\n\n============================================================")
    # print("before_classifier, logits ",before_classifier.shape, logits.shape)
    scatter_index = []
    select_index = []
    seq_len = before_classifier.shape[1]
    new_pos = 0
    for i in range(len(logits)):
        word_ids = inputs.word_ids(i)
        last_w_id = None
        for w_pos, w_id in enumerate(word_ids):
            if w_id is None:
                continue
            select_index.append(i*seq_len + w_pos)
            if last_w_id != w_id:
                scatter_index.append(new_pos)
                new_pos += 1
            else:
                scatter_index.append(scatter_index[-1])
            last_w_id = w_id
    select_index = torch.tensor(select_index).cuda()
    # print("select_index new_pos scatter_index",select_index.shape,new_pos, len(scatter_index))
    wt_before_classifier = torch.index_select(before_classifier.flatten(0,1),dim=0,index=select_index)
    wt_logits = torch.index_select(logits.flatten(0,1),dim=0,index=select_index)
    # print("2 wt_before_classifier, logits ",wt_before_classifier.shape, wt_logits.shape)
    word_before_classifier = torch.zeros(new_pos,before_classifier.shape[-1],dtype=logits.dtype).cuda()
    word_logits = torch.zeros(new_pos,logits.shape[-1],dtype=logits.dtype).cuda()
    # print("word_before_classifier, word_logits ",word_before_classifier.shape, word_logits.shape)
    scatter_index_b = torch.tensor(scatter_index).reshape(-1,1).repeat_interleave(before_classifier.shape[-1],dim=1).cuda()
    scatter_index_l = torch.tensor(scatter_index).reshape(-1,1).repeat_interleave(logits.shape[-1],dim=1).cuda()
    # print("scatter_index_b scatter_index_l",scatter_index_b.shape, scatter_index_l.shape)
    word_before_classifier.scatter_add_(0, scatter_index_b, wt_before_classifier)
    word_logits.scatter_add_(0, scatter_index_l, wt_logits)
    
    # print("f word_before_classifier, word_logits", word_before_classifier.shape,word_logits.shape)
    return word_before_classifier, word_logits

# def val_or_test_compute_task_metrics(task, task_metrics, labels, st_before_classifier, st_logits, st_inputs):
#     if task in ('dc','id','cdc'):
#         st_before_classifier = st_before_classifier[:,0,:]
#         st_logits = st_logits[:,0,:]
#     elif task in ('sf','pos'):
#         st_before_classifier, st_logits = merge_logits_for_sf(st_before_classifier, st_logits, st_inputs)

#     out_for_record = st_logits.detach().cpu()
#     labels_for_record = labels.cpu()
#     acc = task_metrics['acc'](out_for_record, labels_for_record)
#     f1 = task_metrics['f1'](out_for_record, labels_for_record)
#     # compute loss
#     loss = F.cross_entropy(st_logits, labels)
#     loss_value = loss.detach().item()
#     task_metrics['ce'].update(torch.tensor([loss_value]*len(st_logits)))
#     metric_resutls = {'acc':acc,'f1':f1,'ce':loss_value}
#     return metric_resutls, out_for_record, labels_for_record

def init_all_task_metrics_train(all_te_models,data_cfg):
    all_task_metrics = {}
    params = data_cfg.get('params',{})
    if params.get('cc', False):
        for task, te_models in all_te_models.items():
            nclasses = te_models[0].head_cfg['nclasses']
            mif1_metric = torchmetrics.F1Score(task='multilabel', average = 'micro', num_labels=nclasses)
            loss_metric = torchmetrics.aggregation.MeanMetric()
            all_task_metrics[task] = dict(mif1 = mif1_metric, bce=loss_metric)
    else:
        for task, te_models in all_te_models.items():
            nclasses = te_models[0].head_cfg['nclasses']
            acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=nclasses)
            f1_metric = torchmetrics.F1Score(task='multiclass', average = 'macro', num_classes=nclasses)
            mif1_metric = torchmetrics.F1Score(task='multiclass', average = 'micro', num_classes=nclasses)
            loss_metric = torchmetrics.aggregation.MeanMetric()
            all_task_metrics[task] = dict(acc = acc_metric, f1 =f1_metric, mif1 = mif1_metric, ce=loss_metric)
    return all_task_metrics

def init_all_task_metrics_val_or_test(all_te_models,data_cfg):
    all_task_metrics = {}
    params = data_cfg.get('params',{})
    if params.get('cc', False):
        for task, te_models in all_te_models.items():
            nclasses = te_models[0].head_cfg['nclasses']
            mif1_metric = torchmetrics.F1Score(task='multilabel', average = 'micro', num_labels=nclasses)
            loss_metric = torchmetrics.aggregation.MeanMetric()
            all_task_metrics[task] = dict(mif1 = mif1_metric, bce=loss_metric)
    else:
        for task, te_models in all_te_models.items():
            nclasses = te_models[0].head_cfg['nclasses']
            acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=nclasses)
            f1_metric = torchmetrics.F1Score(task='multiclass', average = 'macro', num_classes=nclasses)
            mif1_metric = torchmetrics.F1Score(task='multiclass', average = 'micro', num_classes=nclasses)
            loss_metric = torchmetrics.aggregation.MeanMetric()
            all_task_metrics[task] = dict(acc = acc_metric, f1 =f1_metric, mif1 = mif1_metric, ce=loss_metric)
    return all_task_metrics

def make_log(task_results_dict):
    templete = "Task: [{}]:"
    str_log = ''
    for task, task_metric_resutls in task_results_dict.items():
        if task == 'total_loss':
            continue
        # print(task_metric_resutls)
        keys = sorted(list(task_metric_resutls.keys()))
        str_log += templete.format(task)
        for key in keys:
            str_log += " {}: {},".format(key, task_metric_resutls[key])
        str_log = str_log[:-1] + "\n"
    str_log = f"Total Loss: {task_results_dict['total_loss']} \n" +  str_log
    return str_log

def metrics_compute_epoch(all_task_metrics):
    task_results_dict = {}
    for task, task_metrics in all_task_metrics.items():
        task_results = {}
        for metric_name, metric_obj in task_metrics.items():
            task_results[metric_name] = metric_obj.compute()
            metric_obj.reset()
        task_results_dict[task] = task_results
    return task_results_dict

def get_save_model_name(data_cfg, exp_cfg, all_te_models):
    temodel = None
    for k, v in all_te_models.items():
        temodel = v[0]
        break # should break here, because only support fine tune one teacher each time
    s_name = temodel.model_name
    s_name = s_name.replace('/','_')
    s_name = '{}_{}'.format(s_name, exp_cfg['str_name'])
    # for name in data_cfg['names']:
    #     s_name += f'_{name}'
    return s_name

# train with multi-teacher
def train(exp_cfg, data_cfg, all_te_models, train_loader, lEncs, exp_root_path, val_loader=None):
    logger.info('Training:')
    
    all_task_metrics = init_all_task_metrics_train(all_te_models,data_cfg)

    total_loss_metric = torchmetrics.aggregation.MeanMetric()
    s_name = get_save_model_name(data_cfg, exp_cfg, all_te_models)
    stop_criteria = EarlyStopping(minmax = exp_cfg['mode'],patience=exp_cfg['stop_patience'], \
                                  delta=1e-6, path='results/teachers', trace_func=logger.info, save_every_eposh=True, model_name=s_name)

    max_seq_len = data_cfg['max_seq_len']
    Temp = exp_cfg['temperature']
    

    save_stats = {'train':{'batch':[],'epoch':[]},'val':{'batch':[],'epoch':[]},'test':{'batch':[],'epoch':[]}}
    train_data_len = len(train_loader.dataset)
    total_steps = int(exp_cfg['epochs'] * train_data_len / exp_cfg['batch_size'])
    all_models =[]
    for k, v in all_te_models.items():
        all_models += v
        for te in v:
            te.train()
    params = data_cfg.get('params',{})
    optimizer, scheduler = configure_optimizer(exp_cfg,all_models,total_steps)
    with tqdm(range(exp_cfg['epochs']),desc='Train Epoch Loop') as tbar2:
        for epoch_idx in tbar2:
    # for epoch_idx in range(exp_cfg['epochs']):
            with tqdm(train_loader,desc='Train Batch Loop') as tbar3:
                for batch_idx, raw_batch in enumerate(tbar3):
                    optimizer.zero_grad()
                    bs = len(raw_batch)
                    
                    total_loss = None
                    task_results_dict = {}
                    for task, te_models in all_te_models.items():
                        task_metrics = all_task_metrics[task]
                        te_model = te_models[0]

                        if params.get('cc', False):
                            inputs, labels = collate_fn_cdc(raw_batch, te_model.tokenizer,max_seq_len,lEncs)
                        else:
                            inputs, labels = collate_fn_dc_sf_di(raw_batch, te_model.tokenizer,max_seq_len,lEncs)
                        labels = labels[task]
                        # with torch.no_grad(): # the last linear layer should have grad
                        t_before_classifier, t_logits = te_model(inputs)
                        # print("1 t_before_classifier ",t_before_classifier.shape, t_logits.shape, len(labels))
                        if task in ('sf','pos'):
                            t_before_classifier, t_logits = merge_logits_for_sf(t_before_classifier, t_logits, inputs)

                        # print("2 t_before_classifier ",t_before_classifier.shape, t_logits.shape, len(labels))
                        out_for_record = t_logits.detach().cpu()
                        labels_for_record = labels.cpu()
                        if params.get('cc', False):
                            preds = nn.functional.sigmoid(out_for_record)
                            preds = torch.where(preds>0.5,1,0)
                            mif1 = task_metrics['mif1'](preds, labels_for_record)
                            loss = nn.functional.binary_cross_entropy_with_logits(t_logits, labels.float(), weight=None, size_average=None,reduce=None, reduction='mean')
                            loss_value = loss.detach().item()
                            metric_resutls =  {'mif1':mif1,'bce':loss_value}
                        else:
                            acc = task_metrics['acc'](out_for_record, labels_for_record)
                            f1 = task_metrics['f1'](out_for_record, labels_for_record)
                            mif1 = task_metrics['mif1'](out_for_record, labels_for_record)
                            # compute loss
                            loss = F.cross_entropy(t_logits, labels)
                            loss_value = loss.detach().item()
                            task_metrics['ce'].update(torch.tensor([loss_value]*len(t_logits)))
                            metric_resutls = {'acc':acc,'f1':f1,'ce':loss_value,'mif1':mif1}
                        task_results_dict[task] = metric_resutls
                        task_results_dict['total_loss'] = loss_value
                        break

                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    save_stats['train']['batch'].append(task_results_dict)
                    batch_log = make_log(task_results_dict)

                    # print("batch_log ",batch_log)
                    tbar3.set_description(batch_log)
        
            epoch_task_results_dict = metrics_compute_epoch(all_task_metrics)
            if params.get('cc', False):
                epoch_task_results_dict['total_loss'] = task_results_dict[task]['bce']
            else:
                epoch_task_results_dict['total_loss'] = task_results_dict[task]['ce']
            save_stats['train']['epoch'].append(epoch_task_results_dict)
            train_epoch_log = make_log(epoch_task_results_dict)
        
            if val_loader is not None:
                val_test_epoch_log, epoch_task_results_dict = val_or_test(exp_cfg, data_cfg, all_te_models,
                                                                            val_loader, lEncs, save_stats, split='val')
            epoch_log = 'Epoch: {}, Train: {} \nVal: {}'.format(epoch_idx, train_epoch_log, val_test_epoch_log)
            tbar2.set_description(epoch_log)
            logger.info(epoch_log)

            # Check Stop Criterian
            if exp_cfg['stop_strategy'] == 'early_stop':
                if exp_cfg['monitor'] == 'val_loss':
                    stop = stop_criteria(epoch_task_results_dict['total_loss'],te_model)
            elif exp_cfg['stop_strategy'] == 'epoch_stop':
                stop_criteria(epoch_task_results_dict['total_loss'],te_model)
                stop = False
            if stop == True:
                best_model = stop_criteria.load_best_checkpoint(te_model)
                return save_stats,best_model
    if exp_cfg['stop_strategy'] == 'early_stop':
        best_model = stop_criteria.load_best_checkpoint(te_model)
    else:
        best_model = te_model
    return save_stats,best_model


def val_or_test(exp_cfg, data_cfg, all_te_models, data_loader, lEncs, save_stats, split='val'):
    desc = 'Validation' if split == 'val' else 'Test'
    all_task_metrics = init_all_task_metrics_val_or_test(all_te_models,data_cfg)
    total_loss_metric = torchmetrics.aggregation.MeanMetric()
    params = data_cfg.get('params',{})
    max_seq_len = data_cfg['max_seq_len']
    if split == 'test':
        all_report_data = {}
    with torch.no_grad():
        with tqdm(data_loader,desc=desc) as tbar4:
            for batch_idx, raw_batch in enumerate(tbar4):
                bs = len(raw_batch)
              
                task_results_dict = {}
                for task, te_models in all_te_models.items():
                    task_metrics = all_task_metrics[task]
                    te_model = te_models[0]
                
                    if params.get('cc', False):
                        inputs, labels = collate_fn_cdc(raw_batch, te_model.tokenizer,max_seq_len,lEncs)
                    else:
                        inputs, labels = collate_fn_dc_sf_di(raw_batch, te_model.tokenizer,max_seq_len,lEncs)
                    labels = labels[task]
                    # with torch.no_grad(): # the last linear layer should have grad
                    t_before_classifier, t_logits = te_model(inputs)
                    if task in ('sf','pos'):
                        t_before_classifier, t_logits = merge_logits_for_sf(t_before_classifier, t_logits, inputs)
                
                    out_for_record = t_logits.detach().cpu()
                    labels_for_record = labels.cpu()
                    if params.get('cc', False):
                            preds = nn.functional.sigmoid(out_for_record)
                            preds = torch.where(preds>0.5,1,0)
                            mif1 = task_metrics['mif1'](preds, labels_for_record)
                            loss = nn.functional.binary_cross_entropy_with_logits(t_logits, labels.float(), weight=None, size_average=None,reduce=None, reduction='mean')
                            loss_value = loss.detach().item()
                            metric_resutls =  {'mif1':mif1,'bce':loss_value}
                    else:
                        acc = task_metrics['acc'](out_for_record, labels_for_record)
                        f1 = task_metrics['f1'](out_for_record, labels_for_record)
                        mif1 = task_metrics['mif1'](out_for_record, labels_for_record)
                        # compute loss
                        loss = F.cross_entropy(t_logits, labels)
                        loss_value = loss.detach().item()
                        task_metrics['ce'].update(torch.tensor([loss_value]*len(t_logits)))
                        metric_resutls = {'acc':acc,'f1':f1,'ce':loss_value,'total_loss':loss_value,'mif1':mif1}
                    task_results_dict[task] = metric_resutls
                    if split == 'test':
                        if task not in all_report_data:
                            all_report_data[task] = {'preds':[],'targets':[]}   
                        all_report_data[task]['preds'].append(out_for_record)
                        all_report_data[task]['targets'].append(labels_for_record)
                    task_results_dict['total_loss'] = loss_value
                    break
                save_stats[split]['batch'].append(task_results_dict)
                test_batch_log = make_log(task_results_dict)
                # tbar4.set_description(test_batch_log)
        
        epoch_task_results_dict = metrics_compute_epoch(all_task_metrics)
        if params.get('cc', False):
            epoch_task_results_dict['total_loss'] = task_results_dict[task]['bce']
        else:
            epoch_task_results_dict['total_loss'] = task_results_dict[task]['ce']
        # epoch_task_results_dict['total_loss'] = epoch_task_results_dict[task]['ce']
        
        # generate classification report
        # print("report_data['targets'] ",torch.cat(report_data['preds'],dim=0).shape)
        if split == 'test':
            for task, report_data in all_report_data.items():
                if task == 'cdc':
                    all_preds = nn.functional.sigmoid(out_for_record).tolist()
                    all_targets = torch.cat(report_data['targets'],dim=0).tolist()
                else:
                    all_preds = torch.argmax(torch.cat(report_data['preds'],dim=0),dim=1).tolist()
                    all_targets = torch.cat(report_data['targets'],dim=0).tolist()
                class_names = list(lEncs[task].classes_)
                report = classification_report(all_targets, all_preds, labels=lEncs[task].transform(class_names), target_names=class_names)
                
                logger.info(f"{task} Task Report")
                logger.info(report)
                if task in ('sf'):
                    all_preds = lEncs[task].inverse_transform(all_preds).tolist()
                    all_targets = lEncs[task].inverse_transform(all_targets).tolist()
                    report = seqeval_classification_report([all_targets], [all_preds])
                    logger.info("slot based Report")
                    logger.info(report)
                    f1 = seqval_f1_score([all_targets], [all_preds], average='macro')
                    logger.info("Seqval Macro f1 {}".format(f1))
                    epoch_task_results_dict[task]['Seqval Macro f1'] = f1
        save_stats[split]['epoch'].append(epoch_task_results_dict)
        val_test_epoch_log = make_log(epoch_task_results_dict)


    return val_test_epoch_log, epoch_task_results_dict


def current_exp_count(exp_result_dir,ori_exp_name):
    '''
    Experiment count starts from 0
    '''
    all_exps = os.listdir(exp_result_dir)
    max_count = -1
    for exp_floder_name in all_exps:
        exp_name,str_cound_id = exp_floder_name.rsplit("_",1)
        if exp_name != ori_exp_name:
            continue
        
        count_id = int(str_cound_id)
        if count_id > max_count:
            max_count = count_id
    count = max_count + 1
    return count

def calculate_test_avg_std(save_stats):
    all_task_test_epoch_results = save_stats['test']['epoch']
    tmp = {}
    for all_task_test_results in all_task_test_epoch_results:
        for task,test_results in all_task_test_results.items():
            # print("task,test_results ",task,test_results)
            if task not in tmp:
                tmp[task] = {} 
            # print("result ",result)
            if task == 'total_loss':
                tmp[task][len(tmp[task])] = test_results
            else:
                for k,v in test_results.items():
                    if k not in tmp:
                        tmp[task][k] = [v]
                    else:
                        tmp[task][k].append(v)
    ret_str = ''
    for task, task_stat in tmp.items():
        logger.info(f'Task: {task} ')
        ret_str += f'Task: {task} '
        if task == 'total_loss':
            values = list(task_stat.values())
            str1 = 'key: {}, mean: {}, std: {} |'.format(task,np.mean(values),np.std(values))
            ret_str += str1
            str2 = 'all values: {}'.format(values)
            logger.info(str1)
            logger.info(str2)
        else:
            for k,v in task_stat.items():
                str1 = 'key: {}, mean: {}, std: {} |'.format(k,np.mean(v),np.std(v))
                ret_str += str1
                str2 = 'all values: {}'.format(v)
                logger.info(str1)
                logger.info(str2)
    return ret_str[:-1]

def get_label_encoders(all_slots_list, all_states_list, all_domains_list, all_postags_list):
    all_slots = sorted(set([label for for_one_data in all_slots_list for label in for_one_data]))
    all_states = sorted(set([label for for_one_data in all_states_list for label in for_one_data]))
    all_domains = sorted(set([label for for_one_data in all_domains_list for label in for_one_data]))
    all_postags = sorted(set([label for for_one_data in all_postags_list for label in for_one_data]))
    logger.info('''Merged labels of all of the data: \n len all_slots: {} \n {} \n len all_states: {} \n {}
                 \n len all_domains: {} \n {} len all_postags: {} \n {}'''.format(len(all_slots),all_slots,\
                   len(all_states),all_states, len(all_domains), all_domains, len(all_postags), all_postags))
    lEnc_slot = LabelEncoder()
    lEnc_intent = LabelEncoder()
    lEnc_domain = LabelEncoder()
    lEnc_pos = LabelEncoder()
    lEnc_conv_domain = MultiLabelBinarizer()
    lEnc_slot.fit(all_slots)
    lEnc_intent.fit(all_states)
    lEnc_domain.fit(all_domains)
    lEnc_pos.fit(all_postags)
    # print("all_domains  ",all_domains)
    lEnc_conv_domain.fit([all_domains])
    return lEnc_slot,lEnc_intent,lEnc_domain,lEnc_pos,lEnc_conv_domain

def get_all_labels_of_one_dataset(train, val, test):
    all_slots = []
    all_postags = []
    for xs in train.slot_value_pairs + val.slot_value_pairs + test.slot_value_pairs:
        all_slots += xs
    for xs in train.pos_tags + val.pos_tags + test.pos_tags:
        all_postags += xs
    all_slots = sorted(set(all_slots))
    all_states = sorted(set(train.states + val.states + test.states))
    all_domains = sorted(set(train.domains + val.domains + test.domains))
    all_postags = sorted(set(all_postags))
    logger.info("len all_slots: {} \n {} \n len all_states: {} \n {} \n len all_domains: {} \n {} \n all_postags: {} \n {}"\
                .format(len(all_slots),all_slots,len(all_states),all_states, len(all_domains), all_domains, len(all_postags),all_postags))
    return all_slots, all_states, all_domains, all_postags

def get_fintuned_data_loader_and_lenc(data_cfg, exp_cfg):
    # data
    data_names = data_cfg['names']
    train_set_list = []
    val_set_list = []
    test_set_list = []
    all_slots_list = []
    all_states_list = []
    all_domains_list = []
    all_postags_list = []
    params = data_cfg.get('params',{})
    for name in data_names:
        train_set = getattr(dataset,name)(split='train',**params)
        val_set = getattr(dataset,name)(split='val',**params)
        test_set = getattr(dataset,name)(split='test',**params)
        train_set_list.append(train_set)
        val_set_list.append(val_set)
        test_set_list.append(test_set)
        logger.info("In {} Data, labels of all tasks are: ".format(name))
        all_slots, all_states, all_domains, all_postags = get_all_labels_of_one_dataset(train_set, val_set, test_set)
        all_slots_list.append(all_slots)
        all_states_list.append(all_states)
        all_domains_list.append(all_domains)
        all_postags_list.append(all_postags)
    
    save_label_file = 'results/teachers/{}_labels.pt'.format('_'.join(data_names))
    if not os.path.exists(save_label_file):
        # save the labels for later teaching student
        torch.save(dict(all_slots_list=all_slots_list, all_states_list=all_states_list, all_domains_list=all_domains_list,all_postags_list=all_postags_list),save_label_file)
    lEnc_slot,lEnc_intent,lEnc_domain, lEnc_pos, lEnc_conv_domain = get_label_encoders(all_slots_list, all_states_list, \
                                                                     all_domains_list, all_postags_list)
    # print("lEnc_conv_domain ",lEnc_conv_domain,lEnc_conv_domain.classes_)
    lEncs = {'sf':lEnc_slot,'id':lEnc_intent,'dc':lEnc_domain,'pos':lEnc_pos,'cdc':lEnc_conv_domain}

    train_set = ConcatDataset(train_set_list)
    val_set = ConcatDataset(val_set_list)
    test_set = ConcatDataset(test_set_list)
    collate_fn = collate_fn_default_cc if params.get('cc',False) else collate_fn_default
    train_loader = DataLoader(train_set, batch_size=exp_cfg['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=exp_cfg['batch_size'], shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=exp_cfg['batch_size'], shuffle=False, collate_fn=collate_fn)

    return lEncs, train_loader, val_loader, test_loader

def run_one_exp(config,args,test_params = None, times = 1):
    data_cfg = config['DATA']
    teacher_cfg = config['TEACHERS']
    # student_cfg = config['STUDENT']
    exp_cfg = config['EXPERIMENT']
    global g_exp_config
    g_exp_config = exp_cfg
    retain_ckp = config.get('retain_ckp', True)
    ori_exp_name = config.get('name',"default_name")
    stop_strategy = exp_cfg.get('stop_strategy','early_stop')
    exp_result_dir = exp_cfg.get('save_path',"results/experiments/")

    if not args.test_only:
        if os.path.exists(exp_result_dir):
            current_time = current_exp_count(exp_result_dir,ori_exp_name)
        else:
            current_time = 0
        if args.resume is not None: # resume from last experiment
            current_time -= 1
        time_exp_name = '{}_{}'.format(ori_exp_name,current_time)
    else:
        time_exp_name, ckpt_name = args.test_ckpt.split('/')
    exp_root_path = os.path.join(exp_result_dir,time_exp_name)
    # logger.remove(handler_id=None)
    if args.test_only == True:
        logger.add(os.path.join(exp_root_path,"test_out.log"))
    else:
        logger.add(os.path.join(exp_root_path,"out.log"))
    logger.info("torch.cuda.device_count: {}".format(torch.cuda.device_count()))
    logger.info('config: ' + json.dumps(config, indent=4))
    if test_params is not None:
        logger.info('Test params: '+str(test_params))

    with tqdm(range(times),desc='EXP Times Loop') as tbar1:
        for i in tbar1:
            cur_seed = config['seed']*i+1
            logger.info('Start experiment: {}, seed: {}, time: {}'.format(config['name'],cur_seed,i))
            seed_everything(cur_seed,True)

            # teacher model, grouped by task
            teacher_models = {}
            # name='bert-base-uncased',head={type='dc',nclasses=48},freeze=false}
            for te_cfg_name in teacher_cfg['teacher_list']:
                te_params = teacher_cfg[te_cfg_name]
                task = te_params['head']['type']
                if task not in teacher_models:
                    teacher_models[task] = []
                model_list = teacher_models[task]
                te_type = te_params.pop('type')
                te_model = getattr(models, te_type)(**te_params)
                te_model.cuda()
                te_model.eval()  # eval mode
                model_list.append(te_model)
    
            # student model
            st_model = None
            
            # data
            # train_set = getattr(dataset, data_cfg['name'])(split='train')
            # val_set = getattr(dataset, data_cfg['name'])(split='val')
            # test_set = getattr(dataset, data_cfg['name'])(split='test')
            # lEnc_slot,lEnc_intent,lEnc_domain = get_label_encoders(train_set, val_set, test_set)
            # lEncs = {'sf':lEnc_slot,'id':lEnc_intent,'dc':lEnc_domain}
            # train_loader = DataLoader(train_set, batch_size=exp_cfg['batch_size'], shuffle=True, collate_fn=collate_fn_default)
            # val_loader = DataLoader(val_set, batch_size=exp_cfg['batch_size'], shuffle=True, collate_fn=collate_fn_default)
            # test_loader = DataLoader(test_set, batch_size=exp_cfg['batch_size'], shuffle=False, collate_fn=collate_fn_default)
            lEncs, train_loader, val_loader, test_loader = get_fintuned_data_loader_and_lenc(data_cfg, exp_cfg)
            save_stats_path = os.path.join(exp_root_path,"stats_record.pt")
            if not args.test_only:
                save_stats,best_st_model = train(exp_cfg,data_cfg,teacher_models,train_loader,lEncs,exp_root_path,val_loader)
            else:
                st_model.load_state_dict(torch.load(os.path.join(exp_root_path,ckpt_name)))
                best_st_model = st_model
                if os.path.exists(save_stats_path):
                    save_stats = torch.load(save_stats_path)
                else:
                    save_stats = {'train':{'batch':[],'epoch':[]},'val':{'batch':[],'epoch':[]},'test':{'batch':[],'epoch':[]}}
            teacher_models[task][0] = best_st_model
            test_epoch_log, epoch_task_results_dict = val_or_test(exp_cfg, data_cfg, teacher_models, \
                                                        test_loader, lEncs, save_stats, split='test')

            torch.save(save_stats,os.path.join(exp_root_path,"stats_record.pt"))
            logger.info("Test: "+test_epoch_log)
            time_exp_log = calculate_test_avg_std(save_stats)
            tbar1.set_description(time_exp_log)

def prepare_envs():
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if not os.path.exists('results/teachers'):
        os.mkdir('results/teachers')
    # if not os.path.exists('results/cache/'):
    #     os.mkdir('results/cache/')
    # if not os.path.exists('results/analysis/'):
    #     os.mkdir('results/analysis/')
    # if not os.path.exists('results/cache/key_phrase_split/'):
    #     os.mkdir('results/cache/key_phrase_split/')
    # if not os.path.exists('results/cache/tokenized_results/'):
    #     os.mkdir('results/cache/tokenized_results/')
    # if not os.path.exists('results/cache/vocabs/'):
    #     os.mkdir('results/cache/vocabs/')
    if not os.path.exists('results/experiments/'):
        os.mkdir('results/experiments/')

if __name__=="__main__":
    prepare_envs()
    # Training settings

    parser = argparse.ArgumentParser(description='PyTorch multi_teacher_avg_distill')  
    parser.add_argument('--config', default='ERROR')
    parser.add_argument('--start_version', type=int, default=0)
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--run_times',type=int, default=1)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--test_ckpt', default=None)
    args = parser.parse_args()

    config_file = args.config
    run_times = args.run_times
    config = get_params(config_file)

    exp_result_dir = config['EXPERIMENT'].get('save_path',"results/experiments/")
    if os.path.exists(exp_result_dir) and args.clean == True:
        shutil.rmtree(exp_result_dir)
    if 'PARAMSGRID' in config:
        print("This is a hyper paremeters grid search experiment: {}, seed: {}!!".format(config['name'],config['seed']))
        params_grid = list(ParameterGrid(config['PARAMSGRID']))
        start_version = args.start_version # the version increases from 0
        print(start_version, len(params_grid))
        for i in range(start_version, len(params_grid)):
            params = params_grid[i]
            for combination_name, value in params.items():
                all_names = combination_name.split('_-')
                param_name = all_names[-1]
                sub_config = config
                for p_name in all_names[:-1]:
                    if p_name in sub_config:
                        sub_config = sub_config[p_name]
                    else:
                        print('ERROR config of ',combination_name)
                        sys.exit(0)
                sub_config[param_name] = value
            print("---------------------")
            print('Total param groups: {}, current: {}'.format(len(params_grid), i+1))
            # when searchning the parameters, run_times should be 1
            # with torch.autograd.set_detect_anomaly(True):
            run_one_exp(config,args,params,times=1)
    elif 'PARAMSLIST' in config:
        for combination_name, value_list in config['PARAMSLIST'].items():
            all_names = combination_name.split('_-')
            param_name = all_names[-1]
            sub_config = config
            for p_name in all_names[:-1]:
                if p_name in sub_config:
                    sub_config = sub_config[p_name]
                else:
                    print('ERROR config of ',combination_name)
                    sys.exit(0)
            for i in range(len(value_list)):
                sub_config[param_name] = value_list[i]
                print("---------------------")
                print('current param groups: {}, current: {}'.format(len(value_list), i+1))
                # when searchning the parameters, run_times should be 1
                run_one_exp(config,args,value_list[i],times=1)
    else:
        run_one_exp(config,args,None,times=run_times)
