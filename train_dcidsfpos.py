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
from torch.utils.data import DataLoader
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
import copy

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
        # print("tokenizer.__class__.__name__ ",tokenizer.__class__.__name__)
        if "Llama" in tokenizer.__class__.__name__ or "T5" in tokenizer.__class__.__name__:
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
    for conv, domains in samples:
        tokens.append(conv)
        cc_domain_label.append(domains)
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
# input: Tensor[128, 3, 5]
def avg_score(te_scores_Tensor):
#     print(te_scores_Tensor.size())
    mean_Tensor = torch.mean(te_scores_Tensor, dim=1)
#     print(mean_Tensor)
    return mean_Tensor #[120, 5]
    
# random logits
# def random_logits(te_scores_Tensor):
#     return te_scores_Tensor[np.random.randint(0, 1, 1)]

# input: t1, t2 - triplet pair
def triplet_distance(t1, t2):
    return (t1 - t2).pow(2).sum()
    
# get triplets, according to the relation in teachers'view
def random_triplets(st_maps, te_maps_list):
    st_triplet_list = []
    triplet_set_size = st_maps.size(0)
    batch_list = [x for x in range(triplet_set_size)]
    if (len(te_maps_list) % 2) == 0:
        for i in range(triplet_set_size):
            triplet_index = random.sample(batch_list, 3)
            anchor_index = triplet_index[0]  # denote the 1st triplet item as anchor
            st_triplet = st_maps[triplet_index]
            reverse = 0
            for te_maps in te_maps_list:
                te_triplet = te_maps[triplet_index]
                distance_01 = triplet_distance(anchor_index, te_triplet[1]) # positive
                distance_02 = triplet_distance(anchor_index, te_triplet[2]) # negative
                dif = distance_01-distance_02
                reverse += dif
            if reverse > 0:
                # swap postive and negative
                st_triplet[1], st_triplet[2] = st_triplet[2], st_triplet[1]
            st_triplet_list.append(st_triplet)
    else:
        for i in range(triplet_set_size):
            triplet_index = random.sample(batch_list, 3)
            anchor_index = triplet_index[0]  # denote the 1st triplet item as anchor
            st_triplet = st_maps[triplet_index]
            reverse = 0
            for te_maps in te_maps_list:
                te_triplet = te_maps[triplet_index]
                distance_01 = triplet_distance(anchor_index, te_triplet[1]) # positive
                distance_02 = triplet_distance(anchor_index, te_triplet[2]) # negative
                if distance_01 > distance_02:
                    reverse += 1
                else:
                    reverse -= 1
            if reverse > 0:
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

def val_or_test_compute_task_metrics(task, task_metrics, labels, st_before_classifier, st_logits, st_inputs, data_cfg):
    if task in ('dc','id','cdc'):
        st_before_classifier = st_before_classifier[:,0,:]
        st_logits = st_logits[:,0,:]
    elif task in ('sf','pos'):
        st_before_classifier, st_logits = merge_logits_for_sf(st_before_classifier, st_logits, st_inputs)
    params = data_cfg.get('params',{})
    out_for_record = st_logits.detach().cpu()
    labels_for_record = labels.cpu()
    if params.get('cc', False):
        nclasses = st_logits.shape[-1]
        if nclasses > 2:
            preds = nn.functional.sigmoid(out_for_record)
            preds = torch.where(preds>0.5,1,0)
            acc = task_metrics['acc'](preds, labels_for_record)
            f1 = task_metrics['f1'](preds, labels_for_record)
            mif1 = task_metrics['mif1'](preds, labels_for_record)
            # compute loss
            loss = F.binary_cross_entropy_with_logits(st_logits, labels_for_record)
            loss_value = loss.detach().item()
            task_metrics['bce'].update(torch.tensor([loss_value]*len(st_logits)))
            metric_resutls = {'acc':acc,'f1':f1,'bce':loss_value,'mif1':mif1}
        else:
            labels_for_record = torch.argmax(labels_for_record,dim=1)
            acc = task_metrics['acc'](out_for_record, labels_for_record)
            f1 = task_metrics['f1'](out_for_record, labels_for_record)
            mif1 = task_metrics['mif1'](out_for_record, labels_for_record)
            # compute loss
            loss = F.cross_entropy(st_logits, torch.argmax(labels,dim=1))
            loss_value = loss.detach().item()
            task_metrics['ce'].update(torch.tensor([loss_value]*len(st_logits)))
            metric_resutls = {'acc':acc,'f1':f1,'ce':loss_value,'mif1':mif1}
    else:
        acc = task_metrics['acc'](out_for_record, labels_for_record)
        f1 = task_metrics['f1'](out_for_record, labels_for_record)
        mif1 = task_metrics['mif1'](out_for_record, labels_for_record)
        # compute loss
        loss = F.cross_entropy(st_logits, labels)
        loss_value = loss.detach().item()
        task_metrics['ce'].update(torch.tensor([loss_value]*len(st_logits)))
        metric_resutls = {'acc':acc,'f1':f1,'ce':loss_value,'mif1':mif1}
    return metric_resutls, out_for_record, labels_for_record

def cos_sim_loss(teacher_logits_list, student_logits):
    loss = 0
    for teacher_logits in teacher_logits_list:
        loss += F.cosine_similarity(teacher_logits,student_logits).mean()
    return -loss

def student_ce_loss(student_logits, y_true):
    return F.cross_entropy(student_logits, y_true)

def teacher_pred_loss(teacher_logits_list, student_logits):
    loss = 0
    for teacher_logits in teacher_logits_list:
        loss += F.cross_entropy(student_logits, torch.argmax(teacher_logits,dim=-1))
    return loss

def teacher_pred_regression_loss(teacher_logits_list, student_logits):
    loss = 0
    for teacher_logits in teacher_logits_list:
        loss += F.cross_entropy(student_logits, F.softmax(teacher_logits,dim=-1))
    return loss

def train_compute_task_metrics(task, task_metrics, labels, st_before_classifier, st_logits, st_inputs, te_models, raw_batch, max_seq_len, lEncs, Temp, data_cfg):
    if task in ('dc','id','cdc'):
        st_before_classifier = st_before_classifier[:,0,:]
        st_logits = st_logits[:,0,:]
    elif task in ('sf','pos'):
        st_before_classifier, st_logits = merge_logits_for_sf(st_before_classifier, st_logits, st_inputs)
    
    bs = len(st_logits) # unnecessarily to be equal to len(raw_batch)
    t_scores_list = []
    t_logits_list = []
    t_hint_maps = []
    params = data_cfg.get('params',{})
    for i,te_model in enumerate(te_models):
        if params.get('cc', False):
            inputs, _ = collate_fn_cdc(raw_batch, te_model.tokenizer,max_seq_len,lEncs)
        else:
            inputs, _ = collate_fn_dc_sf_di(raw_batch, te_model.tokenizer,max_seq_len,lEncs)
        # inputs, _ = collate_fn_dc_sf_di(raw_batch, te_model.tokenizer,max_seq_len,lEncs)
        # with torch.no_grad(): # the last linear layer should have grad
        t_before_classifier, t_logits = te_model(inputs)
        if task in ('sf','pos'):
            t_before_classifier, t_logits = merge_logits_for_sf(t_before_classifier, t_logits, inputs)
        t_logits_list.append(t_logits)
        # print(t_logits_list)
        t_hint_maps.append(t_before_classifier)
        t_output = F.softmax(t_logits/Temp,dim=1)
        t_scores_list.append(t_output)
    te_scores_Tensor = torch.stack(t_scores_list, dim=1)  # size: [128, 3, 10]
    mean_score = avg_score(te_scores_Tensor)
    
    # te_index = smallest_conflict_teacher(st_before_classifier, t_hint_maps)
    st_tripets = random_triplets(st_before_classifier, t_hint_maps)

    metric_resutls = {}
    # compute accuracy and f1 score
    out_for_record = st_logits.detach().cpu()
    
    if params.get('cc', False):
        nclasses = te_models[0].head_cfg['nclasses']
        if nclasses > 2:
            preds = nn.functional.sigmoid(out_for_record)
            preds = torch.where(preds>0.5,1,0)
            metric_resutls['acc'] = task_metrics['acc'](preds, labels.cpu())
            metric_resutls['f1'] = task_metrics['f1'](preds, labels.cpu())
            metric_resutls['mif1'] = task_metrics['mif1'](preds, labels.cpu())
        else:
            labels = torch.argmax(labels,dim=1)
            # print("labels_for_record ",labels.shape, labels_for_record.shape,out_for_record.shape)
            metric_resutls['acc'] = task_metrics['acc'](out_for_record, labels.cpu())
            metric_resutls['f1'] = task_metrics['f1'](out_for_record, labels.cpu())
            metric_resutls['mif1'] = task_metrics['mif1'](out_for_record, labels.cpu())
    else:
        metric_resutls['acc'] = task_metrics['acc'](out_for_record, labels.cpu())
        metric_resutls['f1'] = task_metrics['f1'](out_for_record, labels.cpu())
        metric_resutls['mif1'] = task_metrics['mif1'](out_for_record, labels.cpu())

    global g_exp_config
    loss_version = g_exp_config['loss_version']
    # compute gradient and update parameters, 0.7 is from the code https://github.com/FLHonker/AMTML-KD-code
    # Mean teacher loss
    if params.get('cc', False):
        if nclasses > 2:
            mean_te_loss = F.binary_cross_entropy_with_logits(mean_score, labels)
            mean_te_loss_value = mean_te_loss.detach().item()
        else:
            mean_te_loss = F.cross_entropy(mean_score, labels)
            mean_te_loss_value = mean_te_loss.detach().item()
    else:
        mean_te_loss = F.cross_entropy(mean_score, labels)
        mean_te_loss_value = mean_te_loss.detach().item()
    metric_resutls['meante'] = mean_te_loss_value
    task_metrics['meante'].update(torch.tensor([mean_te_loss_value]*bs))

    if loss_version == "student_only_without_detach_mean_score":
        kd_loss = distillation_loss(st_logits, labels, mean_score, T=Temp, alpha=0.7)
        kd_loss_value = kd_loss.detach().item()
        metric_resutls['kd'] = kd_loss_value
        task_metrics['kd'].update(torch.tensor([kd_loss_value]*bs))

        relation_loss = triplet_loss(st_tripets[0], st_tripets[1], st_tripets[2])
        rel_loss_value = relation_loss.detach().item()
        metric_resutls['rel'] = rel_loss_value
        task_metrics['rel'].update(torch.tensor([rel_loss_value]*bs))
    elif loss_version == "student_without_teacher":
        ce_loss = F.cross_entropy(st_logits, labels)
        ce_loss_value = ce_loss.detach().item()
        metric_resutls['ce'] = ce_loss_value
        task_metrics['ce'].update(torch.tensor([ce_loss_value]*bs))

        relation_loss = triplet_loss(st_tripets[0], st_tripets[1], st_tripets[2])
        rel_loss_value = relation_loss.detach().item()
        metric_resutls['rel'] = rel_loss_value
        task_metrics['rel'].update(torch.tensor([rel_loss_value]*bs))
    elif loss_version in ("rel_kd_sim_sce_tp", "rel_kd_sim_sce_tpr", "kd_sim_sce_tp","kd_sim_sce_tpr",'rel_kd_sce_tp_tpr','rel_kd_sim_sce','kd_sce_tp_tpr','kd_sim_sce','rel_kd_sim'):
        relation_loss = triplet_loss(st_tripets[0], st_tripets[1], st_tripets[2])
        rel_loss_value = relation_loss.detach().item()
        metric_resutls['rel'] = rel_loss_value
        task_metrics['rel'].update(torch.tensor([rel_loss_value]*bs))

        kd_loss = distillation_loss(st_logits, labels, mean_score, T=Temp, alpha=0.7)
        kd_loss_value = kd_loss.detach().item()
        metric_resutls['kd'] = kd_loss_value
        task_metrics['kd'].update(torch.tensor([kd_loss_value]*bs))

        sim_loss = cos_sim_loss(t_logits_list,st_logits)
        sim_loss_value = sim_loss.detach().item()
        metric_resutls['sim'] = sim_loss_value
        task_metrics['sim'].update(torch.tensor([sim_loss_value]*bs))

        sce_loss = student_ce_loss(st_logits,labels)
        sce_loss_value = sce_loss.detach().item()
        metric_resutls['sce'] = sce_loss_value
        task_metrics['sce'].update(torch.tensor([sce_loss_value]*bs))

        tp_loss = teacher_pred_loss(t_logits_list, st_logits)
        tp_loss_value = tp_loss.detach().item()
        metric_resutls['tp'] = tp_loss_value
        task_metrics['tp'].update(torch.tensor([tp_loss_value]*bs))

        tpr_loss = teacher_pred_regression_loss(t_logits_list, st_logits)
        tpr_loss_value = tpr_loss.detach().item()
        metric_resutls['tpr'] = tpr_loss_value
        task_metrics['tpr'].update(torch.tensor([tpr_loss_value]*bs))

    else:
        # in this step, teacher mean_score shouldn't produce gradient,i.e., teacher's parameters shouldn't be updated by this loss
        kd_loss = distillation_loss(st_logits, labels, mean_score.detach(), T=Temp, alpha=0.7)
        kd_loss_value = kd_loss.detach().item()
        metric_resutls['kd'] = kd_loss_value
        task_metrics['kd'].update(torch.tensor([kd_loss_value]*bs))
        relation_loss = triplet_loss(st_tripets[0], st_tripets[1], st_tripets[2])
        rel_loss_value = relation_loss.detach().item()
        metric_resutls['rel'] = rel_loss_value
        task_metrics['rel'].update(torch.tensor([rel_loss_value]*bs))
   


    if loss_version == "student_only_without_detach_mean_score":
        total_loss = kd_loss + relation_loss
        total_loss_value = kd_loss_value + rel_loss_value
    elif loss_version == "student_without_teacher":
        total_loss = ce_loss
        total_loss_value = ce_loss_value
    elif loss_version == "rel_kd_sim_sce_tp":
        total_loss = kd_loss + relation_loss + sce_loss + sim_loss + tp_loss
        total_loss_value = kd_loss_value + rel_loss_value + sce_loss_value + sim_loss_value + tp_loss_value
    elif loss_version == "rel_kd_sim_sce_tp":
        total_loss = kd_loss + relation_loss + sce_loss + sim_loss + tpr_loss
        total_loss_value = kd_loss_value + rel_loss_value + sce_loss_value + sim_loss_value + tpr_loss_value
    elif loss_version == "kd_sim_sce_tp":
        total_loss = kd_loss + sce_loss + sim_loss + tp_loss
        total_loss_value = kd_loss_value + sce_loss_value + sim_loss_value + tp_loss_value
    elif loss_version == "kd_sim_sce_tpr":
        total_loss = kd_loss + sce_loss + sim_loss + tpr_loss
        total_loss_value = kd_loss_value + sce_loss_value + sim_loss_value + tpr_loss_value
    elif loss_version == 'rel_kd_sce_tp_tpr':
        total_loss = kd_loss + relation_loss +  sce_loss  + tp_loss + tpr_loss
        total_loss_value = kd_loss_value + rel_loss_value + sce_loss_value + tp_loss_value + tpr_loss_value
    elif loss_version == 'rel_kd_sim_sce':
        total_loss = kd_loss + relation_loss +  sce_loss  + sim_loss
        total_loss_value = kd_loss_value + rel_loss_value + sce_loss_value + sim_loss_value
    elif loss_version == 'kd_sce_tp_tpr':
        total_loss = kd_loss + sce_loss + tp_loss + tpr_loss
        total_loss_value = kd_loss_value + sce_loss_value + tp_loss_value + tpr_loss_value
    elif loss_version == 'kd_sim_sce':
        total_loss = kd_loss + sim_loss + sce_loss
        total_loss_value = kd_loss_value + sim_loss_value + sce_loss_value
    elif loss_version == 'rel_kd_sim':
        total_loss = kd_loss + relation_loss + sim_loss
        total_loss_value = kd_loss_value + rel_loss_value + sim_loss_value
    else:
        total_loss = mean_te_loss + kd_loss + relation_loss
        total_loss_value = mean_te_loss_value + kd_loss_value + rel_loss_value
    # total_loss = kd_loss + relation_loss
    # total_loss_value = kd_loss_value + rel_loss_value
    metric_resutls['tot'] = total_loss_value
    task_metrics['tot'].update(torch.tensor([total_loss_value]*bs))

    return total_loss, metric_resutls

def init_all_task_metrics_train(all_te_models, data_cfg):
    all_task_metrics = {}
    params = data_cfg.get('params',{})
    if params.get('cc', False):
         # if the nclasses==2, accuracy = microf1 = multiclass case
        for task, te_models in all_te_models.items():
            nclasses = te_models[0].head_cfg['nclasses']
            if nclasses > 2: 
            # only multiwoz has more than 2 classes and more than one 
            # domain for each conversation, other datasets only has one domain for each conversation:
                acc_metric = torchmetrics.Accuracy(task='multilabel', num_labels=nclasses)
                f1_metric = torchmetrics.F1Score(task='multilabel', average = 'macro', num_labels=nclasses)
                mif1_metric = torchmetrics.F1Score(task='multilabel', average = 'micro', num_labels=nclasses)
            else:
                acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=nclasses)
                f1_metric = torchmetrics.F1Score(task='multiclass', average = 'macro', num_classes=nclasses)
                mif1_metric = torchmetrics.F1Score(task='multiclass', average = 'micro', num_classes=nclasses)
            mean_te_loss_metric = torchmetrics.aggregation.MeanMetric()
            kd_loss_metric = torchmetrics.aggregation.MeanMetric()
            rel_loss_metric = torchmetrics.aggregation.MeanMetric()
            sim_loss_metric = torchmetrics.aggregation.MeanMetric()
            tp_loss_metric = torchmetrics.aggregation.MeanMetric()
            tpr_loss_metric = torchmetrics.aggregation.MeanMetric()
            sce_loss_metric = torchmetrics.aggregation.MeanMetric()
            total_loss_metric = torchmetrics.aggregation.MeanMetric()
            all_task_metrics[task] = dict(acc = acc_metric, f1 =f1_metric, mif1=mif1_metric, meante = mean_te_loss_metric, kd = kd_loss_metric, rel=rel_loss_metric, tot=total_loss_metric, sim=sim_loss_metric, tp=tp_loss_metric,tpr=tpr_loss_metric,sce=sce_loss_metric)
    else:
        for task, te_models in all_te_models.items():
            nclasses = te_models[0].head_cfg['nclasses']
            acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=nclasses)
            f1_metric = torchmetrics.F1Score(task='multiclass', average = 'macro', num_classes=nclasses)
            mif1_metric = torchmetrics.F1Score(task='multiclass', average = 'micro', num_classes=nclasses)
            mean_te_loss_metric = torchmetrics.aggregation.MeanMetric()
            kd_loss_metric = torchmetrics.aggregation.MeanMetric()
            rel_loss_metric = torchmetrics.aggregation.MeanMetric()
            sim_loss_metric = torchmetrics.aggregation.MeanMetric()
            tp_loss_metric = torchmetrics.aggregation.MeanMetric()
            tpr_loss_metric = torchmetrics.aggregation.MeanMetric()
            sce_loss_metric = torchmetrics.aggregation.MeanMetric()
            total_loss_metric = torchmetrics.aggregation.MeanMetric()
            all_task_metrics[task] = dict(acc = acc_metric, f1 =f1_metric, mif1=mif1_metric, meante = mean_te_loss_metric, kd = kd_loss_metric, rel=rel_loss_metric, tot=total_loss_metric,sim=sim_loss_metric, tp=tp_loss_metric,tpr=tpr_loss_metric,sce=sce_loss_metric)
    return all_task_metrics 

def init_all_task_metrics_val_or_test(all_te_models,data_cfg):
    all_task_metrics = {}
    params = data_cfg.get('params',{})
    if params.get('cc', False):
        for task, te_models in all_te_models.items():
            nclasses = te_models[0].head_cfg['nclasses']
            if nclasses > 2: 
            # only multiwoz has more than 2 classes and more than one 
            # domain for each conversation, other datasets only has one domain for each conversation:
                acc_metric = torchmetrics.Accuracy(task='multilabel', num_labels=nclasses)
                f1_metric = torchmetrics.F1Score(task='multilabel', average = 'macro', num_labels=nclasses)
                mif1_metric = torchmetrics.F1Score(task='multilabel', average = 'micro', num_labels=nclasses)
                loss_metric = torchmetrics.aggregation.MeanMetric()
                all_task_metrics[task] = dict(acc = acc_metric, f1 =f1_metric, mif1=mif1_metric, bce=loss_metric)
            else:
                acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=nclasses)
                f1_metric = torchmetrics.F1Score(task='multiclass', average = 'macro', num_classes=nclasses)
                mif1_metric = torchmetrics.F1Score(task='multiclass', average = 'micro', num_classes=nclasses)
                loss_metric = torchmetrics.aggregation.MeanMetric()
                all_task_metrics[task] = dict(acc = acc_metric, f1 =f1_metric, mif1=mif1_metric, ce=loss_metric)
    else:
        for task, te_models in all_te_models.items():
            nclasses = te_models[0].head_cfg['nclasses']
            acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=nclasses)
            f1_metric = torchmetrics.F1Score(task='multiclass', average = 'macro', num_classes=nclasses)
            mif1_metric = torchmetrics.F1Score(task='multiclass', average = 'micro', num_classes=nclasses)
            loss_metric = torchmetrics.aggregation.MeanMetric()
            all_task_metrics[task] = dict(acc = acc_metric, f1 =f1_metric, mif1=mif1_metric, ce=loss_metric)
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


# train with multi-teacher
def train(exp_cfg, data_cfg, all_te_models, st_model, train_loader, lEncs, exp_root_path, val_loader=None):
    logger.info('Training:')
    
    all_task_metrics = init_all_task_metrics_train(all_te_models,data_cfg)

    total_loss_metric = torchmetrics.aggregation.MeanMetric()
    stop_criteria = EarlyStopping(minmax = exp_cfg['mode'],patience=exp_cfg['stop_patience'], \
                                  delta=1e-6, path=exp_root_path, trace_func=logger.info)

    max_seq_len = data_cfg['max_seq_len']
    Temp = exp_cfg['temperature']
    

    save_stats = {'train':{'batch':[],'epoch':[]},'val':{'batch':[],'epoch':[]},'test':{'batch':[],'epoch':[]}}
    train_data_len = len(train_loader.dataset)
    total_steps = int(exp_cfg['epochs'] * train_data_len / exp_cfg['batch_size'])
    all_models =[st_model]
    for k, v in all_te_models.items():
        all_models += v
        for te in v:
            te.train()

    params = data_cfg.get('params',{})
    optimizer, scheduler = configure_optimizer(exp_cfg,all_models,total_steps)
    with tqdm(range(exp_cfg['epochs']),desc='Train Epoch Loop') as tbar2:
        for epoch_idx in tbar2:
    # for epoch_idx in range(exp_cfg['epochs']):
            # switch to train mode
            st_model.train()
            # with tqdm(train_loader,desc='Train Batch Loop') as tbar3:
            #     for batch_idx, raw_batch in enumerate(tbar3):
            for batch_idx, raw_batch in enumerate(train_loader):
                optimizer.zero_grad()
                bs = len(raw_batch)
                # compute student outputs
                if params.get('cc', False):
                    st_inputs, labels = collate_fn_cdc(raw_batch, st_model.tokenizer,max_seq_len,lEncs)
                else:
                    st_inputs, labels = collate_fn_dc_sf_di(raw_batch, st_model.tokenizer,max_seq_len,lEncs)
                # st_inputs, labels = collate_fn_dc_sf_di(raw_batch, st_model.tokenizer,max_seq_len,lEncs)
                st_before_classifier, all_task_st_logits = st_model(st_inputs['input_ids'],st_inputs['attention_mask'])
                total_loss = None
                task_results_dict = {}
                for task, te_models in all_te_models.items():
                    task_total_loss, task_metric_resutls = train_compute_task_metrics(task, all_task_metrics[task], labels[task],\
                                                            st_before_classifier, all_task_st_logits[task], st_inputs, te_models,\
                                                                raw_batch, max_seq_len, lEncs, Temp, data_cfg)
                    if total_loss is None:
                        total_loss = task_total_loss
                    else:
                        total_loss += task_total_loss
                    task_results_dict[task] = task_metric_resutls

                total_loss = total_loss/len(all_te_models.keys())
                total_loss_value = total_loss.detach().item()
                total_loss_metric.update(torch.tensor([total_loss_value]*bs))
                task_results_dict['total_loss'] = total_loss_value
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                save_stats['train']['batch'].append(task_results_dict)
                batch_log = make_log(task_results_dict)

                # print("batch_log ",batch_log)
                # tbar3.set_description(batch_log)
            
            epoch_task_results_dict = metrics_compute_epoch(all_task_metrics)
            epoch_total_loss = total_loss_metric.compute()
            epoch_task_results_dict['total_loss'] = epoch_total_loss.item()
            total_loss_metric.reset()
            save_stats['train']['epoch'].append(epoch_task_results_dict)
            train_epoch_log = make_log(epoch_task_results_dict)
        
            if val_loader is not None:
                val_test_epoch_log, epoch_task_results_dict = val_or_test(exp_cfg, data_cfg, all_te_models, st_model, \
                                                                            val_loader, lEncs, save_stats, split='val')
            epoch_log = 'Train: {} \nVal: {}'.format(train_epoch_log, val_test_epoch_log)
            # tbar2.set_description(epoch_log)
            logger.info(epoch_log)

            # Check Stop Criterian
            if exp_cfg['stop_strategy'] == 'early_stop':
                if exp_cfg['monitor'] == 'val_loss':
                    stop = stop_criteria(epoch_task_results_dict['total_loss'],st_model)
            if stop == True:
                best_model = stop_criteria.load_best_checkpoint(st_model)
                return save_stats,best_model
        best_model = stop_criteria.load_best_checkpoint(st_model)
    return save_stats,best_model


def val_or_test(exp_cfg, data_cfg, all_te_models, st_model, data_loader, lEncs, save_stats, split='val'):
    st_model.eval()
    desc = 'Validation' if split == 'val' else 'Test'
    all_task_metrics = init_all_task_metrics_val_or_test(all_te_models,data_cfg)
    total_loss_metric = torchmetrics.aggregation.MeanMetric()

    max_seq_len = data_cfg['max_seq_len']
    if split == 'test':
        all_report_data = {}
    params = data_cfg.get('params',{})
    with torch.no_grad():
        with tqdm(data_loader,desc=desc) as tbar4:
            for batch_idx, raw_batch in enumerate(tbar4):
                bs = len(raw_batch)
                # compute student outputs
                if params.get('cc', False):
                    st_inputs, labels = collate_fn_cdc(raw_batch, st_model.tokenizer,max_seq_len,lEncs)
                else:
                    st_inputs, labels = collate_fn_dc_sf_di(raw_batch, st_model.tokenizer,max_seq_len,lEncs)
                # st_inputs, labels = collate_fn_dc_sf_di(raw_batch, st_model.tokenizer,max_seq_len,lEncs)
                st_before_classifier, all_task_st_logits = st_model(st_inputs['input_ids'],st_inputs['attention_mask'])
                task_results_dict = {}
                total_loss_value = 0
                for task, te_models in all_te_models.items():
                    metric_resutls, out_for_record, labels_for_record = val_or_test_compute_task_metrics(task, all_task_metrics[task], \
                                                                labels[task], st_before_classifier, all_task_st_logits[task], st_inputs,data_cfg)
                    task_results_dict[task] = metric_resutls
                    total_loss_value += metric_resutls['ce']
                    if split == 'test':
                        if task not in all_report_data:
                            all_report_data[task] = {'preds':[],'targets':[]}   
                        all_report_data[task]['preds'].append(out_for_record)
                        all_report_data[task]['targets'].append(labels_for_record)
               
                total_loss_value = total_loss_value/len(all_te_models.keys())
                task_results_dict['total_loss'] = total_loss_value
                total_loss_metric.update(torch.tensor([total_loss_value]*bs))
                save_stats[split]['batch'].append(task_results_dict)
                test_batch_log = make_log(task_results_dict)
                # tbar4.set_description(test_batch_log)
        
        epoch_task_results_dict = metrics_compute_epoch(all_task_metrics)
        epoch_total_loss = total_loss_metric.compute()
        total_loss_metric.reset()
        epoch_task_results_dict['total_loss'] = epoch_total_loss.item()
        
        # generate classification report
        # print("report_data['targets'] ",torch.cat(report_data['preds'],dim=0).shape)
        if split == 'test':
            for task, report_data in all_report_data.items():
                class_names = list(lEncs[task].classes_)
                if task == 'cdc':
                    all_preds = torch.cat(report_data['preds'],dim=0)
                    all_preds = nn.functional.sigmoid(all_preds) # sigmoid doesn't chenge the predict relation
                    if all_preds.shape[-1] > 2:
                        all_preds = torch.where(all_preds>0.5,1,0).tolist()
                    else:
                        all_preds = torch.argmax(all_preds,dim=-1)
                    all_targets = torch.cat(report_data['targets'],dim=0).long().tolist()
                    # print("all_preds ",all_preds[0], all_targets)

                    cls_labels=np.array([np.argmax(lEncs[task].transform([[one]]),axis=1).item() for one in class_names])
                else:
                    all_preds = torch.argmax(torch.cat(report_data['preds'],dim=0),dim=1).tolist()
                    all_targets = torch.cat(report_data['targets'],dim=0).tolist()
                    cls_labels=lEncs[task].transform(class_names)
                # all_preds = torch.argmax(torch.cat(report_data['preds'],dim=0),dim=1).tolist()
                # all_targets = torch.cat(report_data['targets'],dim=0).tolist()
                report = classification_report(all_targets, all_preds, labels=cls_labels, target_names=class_names)
                
                logger.info(f"{task} Task Report")
                logger.info(report)
                print("Classification report")
                if task == 'sf':
                    all_preds = lEncs[task].inverse_transform(all_preds).tolist()
                    all_targets = lEncs[task].inverse_transform(all_targets).tolist()
                    report = seqeval_classification_report([all_targets], [all_preds])
                    logger.info("slot based Report")
                    logger.info(report)
                    print("Seqval Classification report")
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

def get_label_encoders(train, val, test):
    all_slots = []
    all_postags = []
    for xs in train.slot_value_pairs + val.slot_value_pairs + test.slot_value_pairs:
        all_slots += xs
    for xs in train.pos_tags + val.pos_tags + test.pos_tags:
        all_postags += xs
    all_slots = sorted(set(all_slots))
    all_states = sorted(set(train.states + val.states + test.states))
    all_domains = sorted(set(train.domains + val.domains + test.domains))
    all_conv_domains = train.conversation_domains + val.conversation_domains + test.conversation_domains
    all_postags = sorted(set(all_postags))
    # print("all_postags ",all_postags)
    logger.info("len all_slots: {}, len all_states: {}, len all_domains: {}, len all_postags: {}".format(len(all_slots),len(all_states),len(all_domains),len(all_postags)))
    lEnc_slot = LabelEncoder()
    lEnc_intent = LabelEncoder()
    lEnc_domain = LabelEncoder()
    lEnc_pos = LabelEncoder()
    lEnc_conv_domain = MultiLabelBinarizer()

    lEnc_slot.fit(all_slots)
    lEnc_intent.fit(all_states)
    lEnc_domain.fit(all_domains)
    lEnc_pos.fit(all_postags)
    lEnc_conv_domain.fit(all_conv_domains)
    return lEnc_slot,lEnc_intent,lEnc_domain, lEnc_pos, lEnc_conv_domain

def run_one_exp(config,args,test_params = None, times = 1):
    data_cfg = config['DATA']
    teacher_cfg = config['TEACHERS']
    student_cfg = config['STUDENT']
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
    logger.remove(handler_id=None) # let logger only write the file, not the std input
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
            st_model = getattr(models, student_cfg['name'])(config,student_cfg)  # args.student()
            st_model.cuda()
            
            # data
            params = data_cfg.get('params',{})
            train_set = getattr(dataset, data_cfg['name'])(split='train',**params)
            val_set = getattr(dataset, data_cfg['name'])(split='val',**params)
            test_set = getattr(dataset, data_cfg['name'])(split='test',**params)
            print("Train, val, test split: {}/{}/{}".format(len(train_set),len(val_set),len(test_set)))
            logger.info("Train, val, test split: {}/{}/{}".format(len(train_set),len(val_set),len(test_set)))
            lEnc_slot,lEnc_intent,lEnc_domain, lEnc_pos, lEnc_conv_domain = get_label_encoders(train_set, val_set, test_set)
            lEncs = {'sf':lEnc_slot,'id':lEnc_intent,'dc':lEnc_domain, 'pos':lEnc_pos,'cdc':lEnc_conv_domain}
            collate_fn = collate_fn_default_cc if params.get('cc',False) else collate_fn_default
            train_loader = DataLoader(train_set, batch_size=exp_cfg['batch_size'], shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_set, batch_size=exp_cfg['batch_size'], shuffle=True, collate_fn=collate_fn)
            test_loader = DataLoader(test_set, batch_size=exp_cfg['batch_size'], shuffle=False, collate_fn=collate_fn)

            save_stats_path = os.path.join(exp_root_path,"stats_record.pt")
            if not args.test_only:
                save_stats,best_st_model = train(exp_cfg,data_cfg,teacher_models,st_model,train_loader,lEncs,exp_root_path,val_loader)
            else:
                st_model.load_state_dict(torch.load(os.path.join(exp_root_path,ckpt_name)))
                best_st_model = st_model
                if os.path.exists(save_stats_path):
                    save_stats = torch.load(save_stats_path)
                else:
                    save_stats = {'train':{'batch':[],'epoch':[]},'val':{'batch':[],'epoch':[]},'test':{'batch':[],'epoch':[]}}

            test_epoch_log, epoch_task_results_dict = val_or_test(exp_cfg, data_cfg, teacher_models, best_st_model, \
                                                        test_loader, lEncs, save_stats, split='test')

            torch.save(save_stats,os.path.join(exp_root_path,"stats_record.pt"))
            logger.info("Test: "+test_epoch_log)
            time_exp_log = calculate_test_avg_std(save_stats)
            logger.info(teacher_cfg)
            tbar1.set_description(time_exp_log)
            

def prepare_envs():
    if not os.path.exists('results/'):
        os.mkdir('results/')
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
            used_config =  copy.deepcopy(config)
            for combination_name, value in params.items():
                all_names = combination_name.split('_-')
                param_name = all_names[-1]
                sub_config =  used_config
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
            run_one_exp(used_config,args,params,times=1)
    elif 'PARAMSLIST' in config:
        for combination_name, value_list in config['PARAMSLIST'].items():
            all_names = combination_name.split('_-')
            param_name = all_names[-1]
            sub_config =  copy.deepcopy(config)
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
                run_one_exp(sub_config,args,value_list[i],times=1)
    else:
        run_one_exp(config,args,None,times=run_times)
