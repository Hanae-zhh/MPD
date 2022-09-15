#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2022/01/03 15:54:20
@Author      :zhangh
@version      :1.0
'''
from __future__ import absolute_import, division, print_function

import os
import math
import random
from re import sub
from threading import main_thread
from tkinter.messagebox import NO
import torch
import logging
import numpy as np
import torch.nn as nn
from overrides import overrides
from typing import List, Any, Dict, Union, Tuple
from tqdm import tqdm
from transformers.models import bert
from data.attackreader import AttackReader,AttackInstance
from utils import mask
from utils.config import PRETRAINED_MODEL_TYPE, DATASET_TYPE
from data.processor import DataProcessor
from data.instance import InputInstance
from data.dataset import ListDataset
from args import ClassifierArgs
from data.reader import DataReader
from predictor import Predictor

from typing import Sized
import warnings
import json
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertConfig, BertTokenizer
from transformers import BertForSequenceClassification, BertForMaskedLM
import copy
import numpy as np 
from transformers import PreTrainedTokenizer
from nltk.tokenize import word_tokenize
import time

filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves',',','.','(',')', "'s", '-', '[UNK]', '"', "'",'','[',']',"!"]
filter_words = set(filter_words)


class Data(object):
    def __init__(self, seq_a, label):
        self.label = label
        self.seq = seq_a
        self.final_adverse = seq_a
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.changes = []

class PurifyFeature(object):
    def __init__(self):
       
        # self.orig_purified_text = ''
        # self.perd_purified_text = ''
        
        self.orig_pred_labels = []
        self.orig_pred_probs = []
        self.perd_pred_labels = []
        self.perd_pred_probs = []
        self.orig_vote_pred = ""
        self.orig_avg_pred = ""
        self.perd_vote_pred = ""
        self.perd_avg_pred = ""

        self.orig_output = ''
        self.perd_output = '' 
        self.orig_changes  = []
        self.perd_changes  = []


class Purifier:
    def __init__(self, args: ClassifierArgs):
        
        # self.orig_pury_text = ""
        # self.orig_pury_label = ""
        # self.attk_pury_text = ""
        # self.attk_pury_label = ""
        self.mlm_model = ""
        self.mlm_tokenizer  = ""
        self.mask_rate = args.mask_rate
        self.top_k = 0
        self.tgt_model = self.build_model(args)
        self.data_reader = ''
        self.tgt_tokenizer = ''
        self.data_processor = ''
        self.use_mask_flag = 1 #whether use trained mlm model, else bert-base-uncased
        self.dataset_dir = "/data/ZhanghData/MaskDefense/dataset"
        self.model_dir = os.path.join(args.mlm_save_dir,args.dataset_name)
        self.purify_num = args.purify_num
        self.c_loss = nn.CrossEntropyLoss(reduction='none')
        self.mlm_params_path = ""
        self.valid_vocab = [] #load when run

        #training parameter
        self.max_ngram_length = args.max_ngram_length
        self.ngram_size = args.ngram_size
        self.mlm_batch_size = args.mlm_batch_size
        self.epochs = args.mlm_epochs

    def get_modelfile_name(self,epoch:int=None,best:bool=False):
        hyper_parameter_dict = \
                    {'ngram':self.ngram_size,'len': self.max_ngram_length, \
                     'epo': self.epochs, 'batch': self.mlm_batch_size}
        file_name = '{}'.format("-".join(["{}{}".\
                format(key, value) for key, value in hyper_parameter_dict.items()]))
        if epoch is not None:
            file_name = "{}-epoch{}".format(file_name,epoch)
        if best :
            file_name = "{}-{}".format(file_name,"best")
        return file_name

    def build_model(self, args: ClassifierArgs) -> nn.Module:
        # config_class: PreTrainedConfig
        # model_class: PreTrainedModel
        config_class, model_class, _ = PRETRAINED_MODEL_TYPE.MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(
            args.model_name_or_path,
            num_labels=args.class_num,
            finetuning_task=args.dataset_name,
            output_hidden_states=True,
        )
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool('ckpt' in args.model_name_or_path),
            config=config
        ).cuda()
        return model

    def load_mlm(self):
        config_atk = BertConfig.from_pretrained("bert-base-uncased")
        self.mlm_model = BertForMaskedLM.from_pretrained("bert-base-uncased", config=config_atk)
        self.mlm_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        if args.use_trained_mlm == True:
            self.mlm_model.load_state_dict(torch.load(self.mlm_params_path))
            logging.info("Loaded mask model from:\\{}".format(self.mlm_params_path))
    
    def loading_model_from_file(self, save_dir: str, file_name: str):
        load_file_name = '{}.pth'.format(file_name)
        load_path = os.path.join(save_dir, load_file_name)
        assert os.path.exists(load_path) and os.path.isfile(load_path), '{} not exits'.format(load_path)
        self.tgt_model.load_state_dict(torch.load(load_path), strict=False)
        logging.info('Loading model from {}'.format(load_path))

    def build_data_processor(self, args: ClassifierArgs, **kwargs) -> List[Union[DataReader, PreTrainedTokenizer, DataProcessor]]:
        data_reader = DATASET_TYPE.DATA_READER[args.dataset_name]()
        #for example data_reader = BinarySentimentAnalysisDataReader()
        _, _, tokenizer_class = PRETRAINED_MODEL_TYPE.MODEL_CLASSES[args.model_type]
        #for example, with model_type=bert,return (BertConfig, BertForSequenceClassification, BertTokenizer)
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path,
            do_lower_case=args.do_lower_case
        )
        data_processor = DataProcessor(data_reader=data_reader,
                                       tokenizer=tokenizer,
                                       model_type=args.model_type,
                                       max_seq_length=args.max_seq_length)

        return [data_reader, tokenizer, data_processor]

    def mlm_tokenize(self,seq, tokenizer):
        seq = seq.replace('\n', '').lower()
        try:
            words = word_tokenize(seq)
        except IndexError as e :
            logging.info("{}".format(e))
            words = seq.split(" ")
        sub_words = []
        sub_index = []
        index = 0
        for word in words:
            sub = tokenizer.tokenize(word)
            sub_words += sub
            sub_index.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, sub_index
    
    def mask_words(self,words):
        #
        masked_words =[]
        masked_indexs=[]
        
        #过滤掉无效词
        valid_words =[]
        valid_index =[]
        for i,word in enumerate(words[:args.max_seq_length]):
            if word not in filter_words and len(word)<25 and word in self.valid_vocab:
                valid_words.append(word)
                valid_index .append(i)
        mask_num = int(self.mask_rate* len(valid_index))  
        masked_words = list(np.random.choice(valid_words, size=mask_num,replace=False))
        masked_indexs =[valid_index[valid_words.index(w)] for w in masked_words]
        
        return masked_words, masked_indexs

   

    def word_replace(self, masked_words, masked_indexs,sub_indexs,top_pred_scores, top_pred_words ):
        replaced_words = []
        count_ = 0
        for idx in masked_indexs:
            substitutes = top_pred_words[sub_indexs[idx][0]:sub_indexs[idx][1]]
            pred_scores = top_pred_scores[sub_indexs[idx][0]:sub_indexs[idx][1]]
            replace_word = self.get_substitues(substitutes,pred_scores)
            if replace_word == '':
                replace_word = masked_words[count_]
            # if replace_word.type is not str:
            #     replaced_words.append(masked_words[idx])
            # else:
            #     replaced_words.append(replace_word)
            replaced_words.append(replace_word)
            count_ += 1
        print(f"replaced_words:{replaced_words}")
        return replaced_words
    
    @torch.no_grad()
    def reference(self,
                  text_a: str,
                  text_b: str = None,
                  label: str = None,
                  ) -> List:
        ''''''
        instance = InputInstance(guid='predict',
                               text_a=text_a,
                               text_b=text_b,
                               label=label)
        
        pred_probs = self.predictor.predict(instance)
        pred_label = np.argmax(pred_probs)
        pred_label = self.data_reader.get_idx_to_label(pred_label)
        if pred_label == '100':
            pred_label = '0'
        elif pred_label == '101':
            pred_label = '1'

        return [pred_label, pred_probs]

    def make_mlm_input(self,text):
        tokens = self.mlm_tokenizer.tokenize(text)
        input_tokens = ['[CLS]']+tokens+['[SEP]']
        mask_pos = input_tokens.index('[MASK]')

        input_ids = self.mlm_tokenizer.convert_tokens_to_ids(input_tokens)
        mask_ids = [1]*len(input_ids)
        if len(input_ids)>self.max_ngram_length:
            input_ids= input_ids[:self.max_ngram_length]
            mask_ids = mask_ids[:self.max_ngram_length]
        
        padding = [0]*(self.max_ngram_length-len(input_ids))
        input_ids += padding
        mask_ids += padding
        input_ids,mask_ids =  torch.tensor([input_ids]),torch.tensor([input_ids])
        # print(f"input_ids,mask_ids, mask_pos:\n{input_ids,mask_ids, mask_pos}")
        return input_ids,mask_ids, mask_pos

    def softmax(self, v):
        l1 = list(map(lambda x: np.exp(x), v))
        return list(map(lambda x: x / sum(l1), l1))

    def predict(self,logits,idx,top_k:int=0):
        probs = logits[0,idx,:]
        
        if  top_k == 0:
            sorted_probs, pred = torch.topk(probs, 1000, -1)
            sorted_probs, pred = sorted_probs.detach().cpu().numpy().tolist(), pred.detach().cpu().numpy().tolist()
            for prob, pred_id in zip(sorted_probs, pred):
                #print(f"prob, pred_id:{prob}, {pred_id}")
                substitute = self.mlm_tokenizer.convert_ids_to_tokens(pred_id)
                if substitute in self.valid_vocab:
                    return substitute
        else:
            sorted_probs, pred = torch.topk(probs, top_k, -1)
            sorted_probs, pred = sorted_probs.detach().cpu().numpy().tolist(), pred.detach().cpu().numpy().tolist()
            probs = np.array(self.softmax(sorted_probs))
            try_time = 0 
            substitute=''
            while substitute not in self.valid_vocab and try_time < 10:
                pred_id= np.random.choice(a=pred, size=1, replace=True, p=probs)[0]
                substitute = self.mlm_tokenizer.convert_ids_to_tokens(int(pred_id))
                try_time += 1
            return substitute
        return ""

    @torch.no_grad()
    def purify_text(self, text):
        
        if text == None:
            return None,None
        seq = text.replace('\n', '').lower()
        try:
            words = word_tokenize(seq)
        except IndexError as e :
            logging.info("{}".format(e))
            words = seq.split(" ")

        purified_words = copy.deepcopy(words)
        #mask_num = int(self.mask_rate*len(words))

        masked_words, masked_indexs = self.mask_words(words)
        if len(masked_words) < 1:
            return " ",[]
     
        # print(f"words:{words}")
        # print(f"masked_words, masked_indexs :\n{masked_words}, \n{masked_indexs }")
        
        #mlm model predict
        words_pad =  ["[CLS]"]*self.ngram_size + words + ["[SEP]"]*self.ngram_size 
        changes = []
        for w,idx in zip (masked_words,masked_indexs):
            input_seq = words_pad[idx:idx+2*self.ngram_size+1]
            input_seq [self.ngram_size]='[MASK]'
            #print(f"input_seq:{input_seq}")
            input_text = " ".join(input_seq)
            
            input_ids,mask_ids,mask_pos = self.make_mlm_input(input_text)
            if mask_pos >= self.max_ngram_length:
                changes.append((idx,w,w))
                logging.info("Failed for  mask pos out of range:{}".format(w))
                continue
            input_ids,mask_ids = input_ids.to("cuda"),mask_ids.to("cuda")
            logits = self.mlm_model(input_ids=input_ids , attention_mask=mask_ids).logits
            replace_w = self.predict(logits,mask_pos,top_k=self.top_k)
            if replace_w == "":
                replace_w = w
                logging.info("Failed to replace [{}]".format(w))
                print("Failed to replace [{}]".format(w))
            purified_words[idx] = replace_w
            changes.append((idx,w,replace_w))
    
        return " ".join(purified_words), changes
    
    def ensemble_probs(self,probs_list):
        probs = np.array(probs_list)
        probs = probs.reshape(int(self.purify_num),args.class_num).sum(axis=0)/(self.purify_num)
        return np.argmax(probs)

    def load_valid(self):
        file_path = os.path.join(self.dataset_dir,\
            ("{}_valid_words.json").format(args.dataset_name))
        with open (file_path,'r') as  f:
            temp = json.load(f)
        return list(temp.keys())

    @torch.no_grad()
    def run(self,args):
        print("Loading attack file...")
        
        if args.dataset_name == 'snli':
            pair_flag = True
        else:
            pair_flag =False
        
        attacker_log_path = args.attacking_dir 
        file_path = os.path.join(attacker_log_path, args.build_saving_file_name(),\
                                '{}_{}.csv'.format(args.attack_method,args.attack_numbers))
        
        attackreader = AttackReader()
        instances = attackreader.get_instances(file_path,pair_flag)
        logging.info("Load isntances:{}".format(len(instances)))

        print('Loading data_processor and predictor...')
        self.data_reader, self.tgt_tokenizer, self.data_processor = self.build_data_processor(args)
        self.predictor = Predictor(self.tgt_model, self.data_processor, args.model_type)
        self.valid_vocab = self.load_valid()
        print("Loading mlm model...")
        self.mlm_params_path = os.path.join(self.model_dir,"{}.pth".\
                                    format(self.get_modelfile_name(best=True)))
        self.load_mlm()#self.mlm_params_path needed to be altered
        self.mlm_model.to(args.device)
        self.mlm_model.eval()

        print("Loading target model...")
        self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description='best'))
        self.tgt_model.eval()

        count = 0
        valid_instance = 0 
        orig_acc_num = 0
        orig_attk_num = 0
        pury_orig_acc_num_vote = 0
        pury_orig_acc_num_avg = 0
        pury_attk_defense_num_vote = 0
        pury_attk_defense_num_avg = 0
        orig_changs_count = 0 
        perd_changs_count = 0

        
        for instance in instances:
            #print(instance)
            count += 1 
            logging.info("********************* instance {} *********************".format(count))
            print("********* instance {} *********".format(count))
            purd_feat = PurifyFeature()
            
            non_purify = False
            purd_feat.orig_output,_ = self.reference(instance.orig_text_a,instance.orig_text_b)
            #
            purd_feat.perd_output,_ = self.reference(instance.perd_text_a,instance.perd_text_b)
            # purd_feat.orig_purified_label,_ = self.reference(purd_feat.orig_purified_text)
            # purd_feat.perd_purified_label,_ = self.reference(purd_feat.perd_purified_text)
            count_orig = 0
            count_perd = 0
            for i in range(self.purify_num):
                logging.info("********* Purify Time: {} *********".format(i+1))
                orig_purified_text_a, purd_feat.orig_changes = \
                                            self.purify_text(instance.orig_text_a)
                
                perd_purified_text_a, purd_feat.perd_changes = \
                                            self.purify_text(instance.perd_text_a)
                # orig_purified_text_a, purd_feat.orig_changes = instance.orig_text_a,[]
                # perd_purified_text_a, purd_feat.perd_changes = instance.perd_text_a,[]

                # if pair_flag :
                # orig_purified_text_b, orig_changes_b = \
                #                         self.purify_text(instance.orig_text_b)
                # purd_feat.orig_changes.append(orig_changes_b)
                # perd_purified_text_b, perd_changes_b = \
                #                         self.purify_text(instance.perd_text_b)       
                # purd_feat.perd_changes.append(perd_changes_b)
                orig_purified_text_b = instance.orig_text_b
                perd_purified_text_b = instance.perd_text_b
                                                             
                if i == 0 and (len(purd_feat.perd_changes) ==0 or len(purd_feat.orig_changes) ==0):
                    non_purify = True
                    break

                logging.info("orig text changes:{}\n perd text changes::{}".format(\
                purd_feat.orig_changes,purd_feat.perd_changes))

                orig_pred_label,orig_pred_prob = self.reference(orig_purified_text_a,orig_purified_text_b)
                perd_pred_label,perd_pred_prob = self.reference(perd_purified_text_a,perd_purified_text_b)   
                purd_feat.orig_pred_labels.append(int(orig_pred_label))
                purd_feat.orig_pred_probs.append(orig_pred_prob)
                purd_feat.perd_pred_labels.append(int(perd_pred_label))
                purd_feat.perd_pred_probs.append(perd_pred_prob)
                count_orig += len( purd_feat.orig_changes)
                count_perd += len( purd_feat.perd_changes)

            if non_purify == True:
                continue  
            
            orig_changs_count += count_orig/self.purify_num
            perd_changs_count = count_perd/self.purify_num
        
            # print(f"purd_feat.orig_pred_labels:\n{purd_feat.orig_pred_labels}")
            # print(f"purd_feat.orig_pred_probs:\n{purd_feat.orig_pred_probs}")
            purd_feat.orig_vote_pred = 1 if sum(purd_feat.orig_pred_labels) >= self.purify_num/2 else 0
            purd_feat.orig_avg_pred =  self.ensemble_probs(purd_feat.orig_pred_probs)
            purd_feat.perd_vote_pred = 1 if sum(purd_feat.perd_pred_labels) >= self.purify_num/2 else 0
            purd_feat.perd_avg_pred = self.ensemble_probs(purd_feat.perd_pred_probs)
            
            if int(purd_feat.orig_output) != instance.orig_label or int(purd_feat.perd_output) != instance.perd_label :
                logging.info("instance.orig_label :{}, purd_feat.orig_output : {}".\
                    format(instance.orig_label, purd_feat.orig_output))
                logging.info("instance.perd_label: {}, purd_feat.perd_output : {}".\
                    format(instance.perd_label, purd_feat.perd_output))
                # logging.info("purd_feat.orig_output : {}".format(purd_feat.orig_output))
                # logging.info("purd_feat.perd_output : {}".format(purd_feat.perd_output))
                continue

            valid_instance += 1
            if instance.ground == instance.orig_label:
                orig_acc_num += 1
            if instance.ground != instance.perd_label:
                orig_attk_num += 1

            if instance.ground == purd_feat.orig_vote_pred:
                pury_orig_acc_num_vote += 1
            if instance.ground == purd_feat.orig_avg_pred:
                pury_orig_acc_num_avg += 1

            if instance.ground == purd_feat.perd_vote_pred:
                pury_attk_defense_num_vote += 1
            if instance.ground == purd_feat.perd_avg_pred:
                pury_attk_defense_num_avg += 1
            
            logging.info("ground_truth: {},  orig_label: {}, perd_label: {}".format\
                (instance.ground,instance.orig_label,instance.perd_label))
            logging.info("orig_output:{} → vote:{} avg:{}, perd_output:{} → vote:{} avg:{}".format(\
                purd_feat.orig_output, purd_feat.orig_vote_pred,purd_feat.orig_avg_pred,\
                    purd_feat.perd_output,purd_feat.perd_vote_pred,purd_feat.perd_avg_pred))
            logging.info("orig_changes_num: {:.3f}, perd_changes_num: {:3f}".format\
                            (count_orig/self.purify_num,count_perd/self.purify_num))


        logging.info("=====================Final result=====================") 
        logging.info("Number of valid instance: {}".format(valid_instance))
        logging.info("Original accuracy: {:.2f}%".format(float(orig_acc_num/valid_instance)*100))
        logging.info("Clean: Accuracy under purify(vote): {:.2f}%".format(float(pury_orig_acc_num_vote/valid_instance)*100))
        logging.info("Clean: Accuracy under purify(avg): {:.2f}%".format(float(pury_orig_acc_num_avg/valid_instance)*100))
        logging.info("Accuracy under attack: {:.2f}%".format(float((valid_instance-orig_attk_num)/valid_instance)*100))
        logging.info("Perturbation: Accuracy under purify(vote): {:.2f}%".format(float(pury_attk_defense_num_vote/valid_instance)*100))
        logging.info("Perturbation: Accuracy under purify(avg): {:.2f}%".format(float(pury_attk_defense_num_avg/valid_instance)*100))
        logging.info("orig_changes_num: {:.3f}, perd_changes_num: {:.3f}".format\
                            (orig_changs_count/valid_instance,perd_changs_count/valid_instance))
            
     

if __name__ == '__main__':
    
    #pwws/pso/ga/'fga'/'textfooler'/'bae'/'deepwordbug'/'textbugger'
  
    # python test_classifier.py --model_type bert --dataset_name snli
    # CUDA_VISIBLE_DEVICES=5 python purify.py --mode purify --model_type roberta --dataset_name imdb --attack_method bae --mask_rate 0.7 --purify_num 10
    # CUDA_VISIBLE_DEVICES=1 python3 purify.py --mode purify --model_type roberta --dataset_name snli --attack_method bae --mask_rate 0.6 --purify_num 10

    args = ClassifierArgs()._parse_args()
    #print(args)
    logging.info(args)
    
    #参数重置
    if args.model_type =='bert':
            # print("******here bert******")
            args.model_name_or_path = '/data/ZhanghData/Pretrained_Models/bert-base-uncased'
    elif args.model_type =='roberta':
            
            if args.dataset_name == 'snli':
                #args.model_name_or_path = '/data/ZhanghData/Pretrained_Models/roberta-large-snli'
                args.model_name_or_path = '/data/ZhanghData/Pretrained_Models/nli-MiniLM2-L6-H768'
            else:
                args.model_name_or_path = '/data/ZhanghData/Pretrained_Models/roberta-base'
        #args.model_name_or_path = '/data/ZhanghData/Pretrained_Models/roberta-base'
    
    #args.max_seq_length = 256 if args.dataset_name in ['imdb','snli'] else 128
    args.max_seq_length = 256 if args.dataset_name in ['imdb'] else 128
    if args.dataset_name in ['agnews', 'snli']:
        args.keep_sentiment_word = False
    #args.batch_size = 16 if args.dataset_name in ['imdb', 'snli'] else 32  # batch size
    if args.dataset_name in ['imdb'] :
        # if args.model_type =='roberta':
        #     args.batch_size = 8
        # else:
        #     args.batch_size = 16 
        args.batch_size = 16
    elif args.dataset_name in ['snli']:
        args.class_num = 3
        if args.model_type =='roberta':
                args.batch_size = 16
        else:
                args.batch_size = 32
    else:
        args.batch_size = 32 
    

    #build logging
    # including check logging path, and set logging config
    args.build_logging_dir()
    args.build_logging()

    logging.info(args)

    args.build_environment()
    # check dataset and its path
    args.build_dataset_dir()

    args.build_saving_dir()
    args.build_caching_dir()

    if args.dataset_name in ['agnews', 'snli']:
        args.keep_sentiment_word = False

    purifier=Purifier(args)
    start = time.time()
    purifier.run(args)
    end = time.time()
    logging.info("Running time:{:.4f} minutes={:.2f} hours".format((end-start)/60,(end-start)/3600))
