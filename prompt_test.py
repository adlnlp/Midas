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
from utils import *
import dataset
from utils.params import get_params
from utils.early_stop import EarlyStopping
from sklearn.preprocessing import LabelEncoder,MultiLabelBinarizer
from sklearn.model_selection import ParameterGrid
import shutil
import sys
import json
import torchmetrics
from tqdm.autonotebook import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
import math
from sklearn.metrics import classification_report
# Teacher models:
# VGG11/VGG13/VGG16/VGG19, GoogLeNet, AlxNet, ResNet18, ResNet34,
# ResNet50, ResNet101, ResNet152, ResNeXt29_2x64d, ResNeXt29_4x64d,
# ResNeXt29_8x64d, ResNeXt29_32x64d, PreActResNet18, PreActResNet34,
# PreActResNet50, PreActResNet101, PreActResNet152,
# DenseNet121, DenseNet161, DenseNet169, DenseNet201,
# BERT, RoBERTa
# import models
import copy
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM
import transformers
import openai
import datetime
import google.generativeai as genai


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
    print("len all_slots: {}, len all_states: {}, len all_domains: {}, len all_postags: {}".format(len(all_slots),len(all_states),len(all_domains),len(all_postags)))
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

def get_datasets(name):
    if name == 'm2m':
        data_cfg = dict(
        name = 'M2MDCIDSFPOSDataset',
        num_workers = 4,
        max_seq_len = 512)
        params = {}
    elif name == 'multiwoz':
        data_cfg = dict(
        name = 'MultiWozDCIDSFPOSDataset',
        num_workers = 4,
        max_seq_len = 512)
        params = {}
    else:
        raise "data name error"


    train_set = getattr(dataset, data_cfg['name'])(split='train',**params)
    val_set = getattr(dataset, data_cfg['name'])(split='val',**params)
    test_set = getattr(dataset, data_cfg['name'])(split='test',**params)
    return train_set, val_set, test_set

def decoder_run(model,tokenizer,test_set,prompt,save_name,task_name,max_new_tokens=10):
    llm_all_answers = []
    total_test = len(test_set)
    for words, _, intent, domain, _ in tqdm(test_set):
        text = prompt.format(' '.join(words))
        inputs = tokenizer(text,return_tensors='pt')
        if task_name == 'sf':
            mds = 12*len(words)
        else:
            mds = max_new_tokens
        generate_ids = model.generate(inputs.input_ids.cuda(),max_new_tokens=mds)
        answer = tokenizer.batch_decode(generate_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # print("answer ",answer)
        
        llm_all_answers.append(answer)

    torch.save({"llm_all_answers":llm_all_answers},save_name+"_prompt_results.pt")

def construct_in_context_samples(train_set, lEnc, task_name):
    total_train = len(train_set)
    if task_name == 'id':
        aim_idx = 2
    elif task_name == 'dc':
        aim_idx = 3
    elif task_name == 'sf':
        aim_idx = 1
    in_context_prompt = ""
    if task_name == 'sf':
        class_samples_count = {}
        for i in range(total_train):
            # words, _, intent, domain, _ = test_set[i]
            sample = train_set[i]
            use_this_sample = False
            finished_caching = True
            for one_word_tag in sample[aim_idx]:
                if class_samples_count.get(one_word_tag, 0) < 3:
                    use_this_sample = True
                    break

            if use_this_sample == True:
                in_context_prompt += '''\nDialogue: "{}", slot types: {}'''.format(' '.join(sample[0]), sample[aim_idx])

            for one_word_tag in sample[aim_idx]: 
                used_count = class_samples_count.get(one_word_tag, 0)
                class_samples_count[one_word_tag] = used_count + 1
            for tag_name in lEnc.classes_:
                if class_samples_count.get(tag_name, 0) < 3:
                    finished_caching = False
                    break
            if finished_caching == True:
                break


    else:
        label_name = "Intent" if task_name == 'id' else "Domain"
        class_samples_cache = {}
        for i in range(total_train):
            # words, _, intent, domain, _ = test_set[i]
            sample = train_set[i]
            use_this_sample = False
            finished_caching = True
            label = sample[aim_idx]
            class_samples = class_samples_cache.get(label, [])
            used_count = len(class_samples)
            if used_count < 3:
                use_this_sample = True

            if use_this_sample == True:
                class_samples.append(' '.join(sample[0]))
                class_samples_cache[label] = class_samples

            for label in lEnc.classes_:
                if len(class_samples_cache.get(label, [])) < 3:
                    finished_caching = False
                    break
            if finished_caching == True:
                break

        for label, samples in class_samples_cache.items():
            in_context_prompt += "\n{}: {} \nDialogues: \n1. {}\n2. {}\n3. {}".format(label_name, label, samples[0], samples[1], samples[2])
    
    return in_context_prompt


        



def analyse_decoder(answers, test_set, lEnc, task_name, acc_metric, print_c = 10):
    llm_all_preds = []
    llm_all_labels = []
    count_pred_out_of_the_range = 0
    acc_metric.reset()
    total_test = len(test_set)
    if task_name == 'id':
        aim_idx = 2
    elif task_name == 'dc':
        aim_idx = 3
    elif task_name == 'sf':
        aim_idx = 1
    unfinished_generation = 0
    for i in range(total_test):
        # words, _, intent, domain, _ = test_set[i]
        sample = test_set[i]
        answer = answers["llm_all_answers"][i]

        if task_name == 'sf':
            llm_all_labels.append(lEnc.transform(sample[aim_idx]))
            l = answer.find('[')+1
            r = answer.find(']')
            if r == -1:
                unfinished_generation += 1
                # print(len(sample[0]), answer)
                r = len(answer)
            answer = answer[l:r].split(',')
            # some answers contains the slot value
            answer = [slot.strip().split(':')[0].strip('\'"') for slot in answer]
            # answer = ast.literal_eval(answer)
            one_pred = []
            answer = answer + (len(llm_all_labels[-1]) - len(answer))*['error']
            # print("answer ",answer)
            # print("label ",sample[aim_idx])
            for p_slot_type_word, t_slot_type in zip(answer,llm_all_labels[-1]):
                if p_slot_type_word in lEnc.classes_:
                    one_pred.append(lEnc.transform([p_slot_type_word])[0])
                else:
                    count_pred_out_of_the_range += 1
                    one_pred.append(len(lEnc.classes_)%(t_slot_type+1))
            
            llm_all_preds.append(one_pred)
        else:
            llm_all_labels.append(lEnc.transform([sample[aim_idx]])[0])
            pred_in_the_range = False
            for label_words in lEnc.classes_:
                if label_words in answer:
                    if label_words == 'request':
                        if 'request_alts' in answer:
                            label_words = 'request_alts'
                    pred_in_the_range = True
                    llm_all_preds.append(lEnc.transform([label_words])[0])
                    break

            if pred_in_the_range == False:
                count_pred_out_of_the_range += 1
                llm_all_preds.append(len(lEnc.classes_) - 1 - lEnc.transform([sample[aim_idx]])[0])

    print(f"total test: {total_test}, prediction out of the range: {count_pred_out_of_the_range}, unfinished_generation:{unfinished_generation}")
    if task_name == 'sf':
        c_llm_all_preds = [l for labels in llm_all_preds for l in labels]
        c_llm_all_labels = [l for labels in llm_all_labels for l in labels]
        print(f"Accuracy: {acc_metric(torch.tensor(c_llm_all_preds),torch.tensor(c_llm_all_labels))}")
    else:
        print(f"Accuracy: {acc_metric(torch.tensor(llm_all_preds),torch.tensor(llm_all_labels))}")
    print("**********************************************")
    c = 0
    for i in range(total_test):
        # print(llm_all_preds[i],llm_all_labels[i])
        # print(type(llm_all_preds[i]),type(llm_all_labels[i]))
        if llm_all_preds[i] != llm_all_labels[i].tolist():
            c+=1
            if task_name == 'sf':
                label = lEnc.inverse_transform(llm_all_labels[i])
            else:
                label = lEnc.inverse_transform([llm_all_labels[i]])[0]
            print("----------------------------")
            print("sample: ",test_set[i][0])
            print("answer: {}, label: {}".format(answers["llm_all_answers"][i], label))
        if c>print_c:
            break
# gpt-4o-2024-05-13
def query_gpt_model(
    gpt_client,
    prompt: str,
    # lm: str = 'gpt-3.5-turbo-1106',
    lm: str = 'gpt-4o-2024-05-13',
    temperature: float = 1.0,
    max_decode_steps: int = 512,
    seconds_to_reset_tokens: float = 30.0,
) -> str:
  while True:
    try:
      raw_response = gpt_client.chat.completions.with_raw_response.create(
        model=lm,
        max_tokens=max_decode_steps,
        temperature=temperature,
        messages=[
          {'role': 'user', 'content': prompt},
        ]
      )
      completion = raw_response.parse()
      return completion.choices[0].message.content
    except openai.RateLimitError as e:
      print(f'{datetime.datetime.now()}: query_gpt_model: RateLimitError {e.message}: {e}')
      time.sleep(seconds_to_reset_tokens)
    except openai.APIError as e:
      print(f'{datetime.datetime.now()}: query_gpt_model: APIError {e.message}: {e}')
      print(f'{datetime.datetime.now()}: query_gpt_model: Retrying after 5 seconds...')
      time.sleep(5)

def decoder_run_gpt(gpt_client, test_set,prompt,save_name,task_name, punc=' ', max_decode_steps=10):
    llm_all_answers = []
    total_test = len(test_set)
    for words, _, intent, domain, _ in tqdm(test_set):
        text = prompt.format(punc.join(words))
        if task_name == 'sf':
            mds = 12*len(words)
        else:
            mds = max_decode_steps
        answer = query_gpt_model(gpt_client, text,max_decode_steps=mds)
        # print("answer ",answer)

        llm_all_answers.append(answer)

    torch.save({"llm_all_answers":llm_all_answers},save_name+"_prompt_results.pt")

def decoder_run_gemini(gemini_model, test_set,prompt,save_name,task_name, punc=' ', max_decode_steps=10):
    llm_all_answers = []
    total_test = len(test_set)
    for words, _, intent, domain, _ in tqdm(test_set):
        text = prompt.format(punc.join(words))
        if task_name == 'sf':
            mds = 12*len(words)
        else:
            mds = max_decode_steps
        response = gemini_model.generate_content([text],generation_config= {"candidate_count":1, "max_output_tokens":mds,"temperature":0})
        answer = response.text
        # print("answer ",answer)

        llm_all_answers.append(answer)

    torch.save({"llm_all_answers":llm_all_answers},save_name+"_prompt_results.pt")
    

if __name__=="__main__":
    # Training settings

    parser = argparse.ArgumentParser(description='Prompt test')  
    parser.add_argument('--data', default='m2m')
    parser.add_argument('--mname', default='None')
    parser.add_argument('--task', default='id')
    parser.add_argument('--punc', default=' ')
    parser.add_argument('--incontext', action='store_true')
    args = parser.parse_args()

    data_name = args.data
    model_name = args.mname
    task_name = args.task
    incontext = args.incontext
    punc = args.punc
    use_gpt = False
    use_gemini = False

    if "Llama" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/{model_name}", token=' ')
        if "70" in model_name:
            bnb_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            model = AutoModelForCausalLM.from_pretrained(f"meta-llama/{model_name}", quantization_config=bnb_config,token=' ').eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(f"meta-llama/{model_name}", token=' ').eval().bfloat16().cuda()
        key_word = model_name.split('-')[2]
        save_name = f"llama2-{key_word}-chat-{data_name}-{task_name}"
    elif "Qwen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{model_name}")
        model = AutoModelForCausalLM.from_pretrained(f"Qwen/{model_name}",torch_dtype="auto").eval().cuda()
        key_word = model_name.split('-')[2]
        save_name = f"qwen2-{key_word}-chat-{data_name}-{task_name}"
    elif "gemma" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(f"google/{model_name}", token=' ')
        model = AutoModelForCausalLM.from_pretrained(f"google/{model_name}", token=' ').eval().cuda()
        key_word = model_name.split('-')[1]
        save_name = f"gemma-{key_word}-{data_name}-{task_name}"
    elif "bart" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}", token=' ')
        model = AutoModelForCausalLM.from_pretrained(f"facebook/{model_name}", token=' ').eval().cuda()
        key_word = model_name.split('-')[1]
        save_name = f"{model_name}-{data_name}-{task_name}"

    elif "t5" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(f"google/{model_name}", token=' ')
        model = AutoModelForSeq2SeqLM.from_pretrained(f"google/{model_name}", token=' ').eval().cuda()
        key_word = model_name.split('-')[1]
        save_name = f"{model_name}-{data_name}-{task_name}"
    # elif 'gpt' in model_name:
    #     use_gpt = True
    #     save_name = f"gpt3-5-{data_name}-{task_name}"
    elif 'gpt4o' in model_name:
        use_gpt = True
        save_name = f"gpt4o-{data_name}-{task_name}"
    elif 'gemini' in model_name:
        use_gemini = True
        genai.configure(api_key=" ")
        save_name = f"gemini-{data_name}-{task_name}"
    
    if incontext == True:
        save_name = f"{save_name}_dem"

    train_set, val_set, test_set = get_datasets(data_name)
    lEnc_slot,lEnc_intent,lEnc_domain, lEnc_pos, lEnc_conv_domain = get_label_encoders(train_set, val_set, test_set)

    if task_name == 'id':
        lEnc = lEnc_intent
        if data_name == 'm2m':
            num = "fifteen"
        elif data_name == 'multiwoz':
            num = "eleven"
        if incontext == False:
            prompt = f"""Definition: In this task, you are given a dialogue. Your job is to classify the following dialogue into one of the {num} different intents. The intents are: """
            for label_words in lEnc.classes_:
                prompt += f'''"{label_words}", '''
            prompt = prompt[:-2] + ". Input: [{}]. Output(only output the intent):"
        else:
            in_context_prompt = construct_in_context_samples(train_set, lEnc, task_name)
            prompt = """Definition: In this task, you are given a dialogue. Your job is to classify the following dialogue into one of the {} different intents. The intents and examples are: {}""".format(num, in_context_prompt)
            prompt += "\nInput: [{}]. Output(only output the intent):"
    elif task_name == 'dc':
        lEnc = lEnc_domain
        if data_name == 'm2m':
            num = "two"
        elif data_name == 'multiwoz':
            num = "eight"
        if incontext == False:
            prompt = f"""Definition: In this task, you are given a dialogue. Your job is to classify the following dialogue into one of the {num} different domains. The domains are: """
            for label_words in lEnc.classes_:
                prompt += f'''"{label_words}", '''
            prompt = prompt[:-2] + ". Input: [{}]. Output(only output the domain):"
        else:
            in_context_prompt = construct_in_context_samples(train_set, lEnc, task_name)
            prompt = """Definition: In this task, you are given a dialogue. Your job is to classify the following dialogue into one of the {} different intents. The domains and examples are: {}""".format(num, in_context_prompt)
            prompt += "\nInput: [{}]. Output(only output the domain):"
    elif task_name == 'sf':
        lEnc = lEnc_slot
        if data_name == 'm2m':
            num = "twenty-one"
        elif data_name == 'multiwoz':
            num = "thirty"
        if incontext == True:
            in_context_prompt = construct_in_context_samples(train_set, lEnc, task_name)
            prompt= f"""In the task of slot filling, the B-, I-, and O- prefixes are commonly used to annotate slot types, indicating the boundaries and types of slots. These labels typically represent:
            B- (Begin): Signifies the beginning of a slot, marking the start of a new slot.
            I- (Inside): Represents the interior of a slot, indicating a continuation of the slot.
            O (Outside): Denotes parts of the input that are not part of any slot.
            For instance, in a sentence where we want to label a "date" slot, words containing date information might be tagged as "B-date" (indicating the beginning of a date slot), followed by consecutive words carrying date information tagged as "I-date" (indicating the continuation of the date slot), while words not containing date information would be tagged as "O" (indicating they are outside any slot). Here are some examples:{in_context_prompt}

    Definition: In this task, you are given a dialogue. Your job is to classify the words in the following dialogue into one of the {num} different slots. The slots are: """
        
        elif incontext == False:
            prompt= f"""In the task of slot filling, the B-, I-, and O- prefixes are commonly used to annotate slot types, indicating the boundaries and types of slots. These labels typically represent:
            B- (Begin): Signifies the beginning of a slot, marking the start of a new slot.
            I- (Inside): Represents the interior of a slot, indicating a continuation of the slot.
            O (Outside): Denotes parts of the input that are not part of any slot.
            For instance, in a sentence where we want to label a "date" slot, words containing date information might be tagged as "B-date" (indicating the beginning of a date slot), followed by consecutive words carrying date information tagged as "I-date" (indicating the continuation of the date slot), while words not containing date information would be tagged as "O" (indicating they are outside any slot).

            Definition: In this task, you are given a dialogue. Your job is to classify the words in the following dialogue into one of the {num} different slots. The slots are: """

        for label_words in lEnc.classes_:
            prompt += f'''"{label_words}", '''
        prompt = prompt[:-2] + ". Input: [{}]. Output(Only output slot types. And the slot types should be output as a list without any explanation):"
    
    print(f"task: {task_name}, classes:")
    print(lEnc.classes_)
    print(lEnc.transform(lEnc.classes_))
    print("==========================")
    
    print("prompt ",prompt)
    print("==========================")

    if task_name == 'sf':
        acc_metric = torchmetrics.F1Score(task='multiclass', average = 'micro', num_classes=len(lEnc.classes_))
    else:    
        acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=len(lEnc.classes_))

    if use_gpt == False and use_gemini == False:
        decoder_run(model,tokenizer,test_set,prompt,save_name, task_name, max_new_tokens=10)
    elif use_gpt == True:
        key = ' '
        gpt_client = openai.OpenAI(api_key=key)
        decoder_run_gpt(gpt_client, test_set, prompt, save_name, task_name, punc, max_decode_steps=10)
    elif use_gemini == True:
        model = genai.GenerativeModel("models/gemini-1.5-pro")
        decoder_run_gemini(model, test_set, prompt, save_name, task_name, punc, max_decode_steps=10)

    answers = torch.load(os.path.join("results/prompt/",save_name + "_prompt_results.pt"))
    analyse_decoder(answers, test_set, lEnc, task_name, acc_metric, print_c = 10)
