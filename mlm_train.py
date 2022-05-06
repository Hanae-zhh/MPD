#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2022/01/05 20:45:27
@Author      :zhangh
@version      :1.0
'''
from __future__ import absolute_import, division, print_function

import os
import math
import random
from threading import main_thread
import torch
import logging
import numpy as np
import torch.nn as nn
from overrides import overrides
from typing import List, Any, Dict, Union, Tuple
from tqdm import tqdm
from data.attackreader import AttackReader,AttackInstance
from utils.config import PRETRAINED_MODEL_TYPE, DATASET_TYPE
from data.processor import DataProcessor
from data.instance import InputInstance
from data.dataset import ListDataset
from data.reader import DataReader
from args import ClassifierArgs
from data.reader import DataReader
from predictor import Predictor
from trainer import MlmTrainer
from utils.utils import convert_batch_to_bert_input_dict
from utils.metrics import Metric, RandomSmoothAccuracyMetrics, RandomAblationCertifyMetric
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.utils import convert_batch_to_mlm_input_dict
from torch.optim.adamw import AdamW
from nltk.tokenize import word_tokenize
class InputFeatures_mlm_train(object):

    def __init__(self, ngram_ids, ngram_labels, ngram_masks, num):

        self.ngram_ids = ngram_ids
        self.ngram_labels = ngram_labels
        self.ngram_masks = ngram_masks
        self.num = num
        #self.ngram_embeddings = ngram_embeddings

class Mlm_Train(object):
    
    def __init__(self):
        ''''''
        self.data_reader = ""
        self.tokenizer = ""
        self.data_processor = ""
        self.model = ""
        self.loss_function = self.build_criterion(args.dataset_name)
        
        self.max_ngram_length = args.max_ngram_length
        self.ngram_size = args.ngram_size
        self.mlm_batch_size = args.mlm_batch_size
        self.epochs = args.mlm_epochs
        self.valid_vocab = []
        self.dataset_dir = "/data/ZhanghData/MaskDefense/dataset"
        self.saving_dir = os.path.join(args.mlm_save_dir,args.dataset_name)

    def load_mlm(self):
        config_atk = BertConfig.from_pretrained("bert-base-uncased")
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased", config=config_atk)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    
    def build_criterion(self, dataset):
        return DATASET_TYPE.get_loss_function(dataset)

    # def build_data_processor(self, args: ClassifierArgs, **kwargs) -> List[Union[DataReader, PreTrainedTokenizer, DataProcessor]]:
    #     data_reader = DATASET_TYPE.DATA_READER[args.dataset_name]()
    #     #for example data_reader = BinarySentimentAnalysisDataReader()
    #     _, _, tokenizer_class = PRETRAINED_MODEL_TYPE.MODEL_CLASSES[args.model_type]
    #     #for example, with model_type=bert,return (BertConfig, BertForSequenceClassification, BertTokenizer)
    #     tokenizer = tokenizer_class.from_pretrained(
    #         args.model_name_or_path,
    #         do_lower_case=args.do_lower_case
    #     )
    #     data_processor = DataProcessor(data_reader=data_reader,
    #                                     tokenizer=tokenizer,
    #                                     model_type=args.model_type,
    #                                     max_seq_length=args.max_seq_length)

    #     return [data_reader, tokenizer, data_processor]
    
    def convert_tokens_to_ngram(self,words):
        '''
        create mlm-data: the number of mlm-data for one text is sequence_length 
        return:
            features: ngram_input_ids (batch_size, sequence_length, ngram_length)
            labels: the word-level ids of the token to predict (batch_size, sequence_length)
            masks: ngram_input_mask (batch_size, sequence_length, ngram_length)
        '''
        # padding for collecting n-grams
        words_pad =  ["[CLS]"]*self.ngram_size + words + ["[SEP]"]*self.ngram_size
        
        inputs, labels, masks = [], [], []
        num = 0
        # length of tokens :len(words)
        for i in range(len(words)):
            # if len(words) > self.max_seq_length and train and random.random() > 0.25: 
            #     continue
            # two situations the ngram would be created:
            # 1. no flaw labels are given, should generate ngrams for all the tokens to train 
            # 2. flaw labels are given, should generate ngrams for those flaw tokens to test
            # if (flaw_labels is not None and i in flaw_labels) or (flaw_labels is None):
            tgt_word = words_pad[i+self.ngram_size]
            if tgt_word not in self.valid_vocab:
                continue
            output = words_pad[i:(i+1+2*self.ngram_size)]
            input = copy.deepcopy(output)
            input[self.ngram_size] = "[MASK]"

            # tokens = ' '.join(tokens)
            input_tokens = self.tokenizer.tokenize(' '.join(input))
            output_tokens = self.tokenizer.tokenize(' '.join(output))
            input_tokens = ['[CLS]']+input_tokens+['[SEP]']
            output_tokens = ['[CLS]']+output_tokens+['[SEP]']

            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            output_ids = self.tokenizer.convert_tokens_to_ids(output_tokens)
            
            if len(input_ids) != len(output_ids):
                print("Error for different length of input and output")
                logging.info("Error for different length of input and output")
                continue

            if len(input_ids) > self.max_ngram_length:
                input_ids = input_ids[:self.max_ngram_length]
                output_ids = output_ids[:self.max_ngram_length]

            mask_ids = [1] * len(input_ids)
            padding = [0] * (self.max_ngram_length - len(input_ids))
            input_ids += padding
            output_ids += padding
            mask_ids += padding
            
            inputs.append(input_ids)
            masks.append(mask_ids)
            labels.append(output_ids)
            num += 1
        
        return inputs, labels, masks, num

    def convert_instances_to_features_mlm_train(self,instances):
    
        features = []
        num_sum = 0
        for (idx, instances) in enumerate(instances):
            #words = word_tokenize(instances.text_a)
            seq = instances.text_a.replace('\n', '').lower()
            words = word_tokenize(seq)

            ngram_ids, ngram_labels, ngram_masks,num = self.convert_tokens_to_ngram(words)
            # if idx < 3:
            #     logging.info("*** Example ***")
            #     logging.info("tokens: %s" % " ".join([str(x) for x in words]))
            #     logging.info("token_ids: %s" % " ".join([str(x) for x in token_ids]))
            #     logging.info("ngram_ids: %s" % " ".join([str(x) for x in ngram_ids]))
            #     logging.info("ngram_labels: %s" % " ".join([str(x) for x in ngram_labels]))

            features.append(
                    InputFeatures_mlm_train(ngram_ids=ngram_ids,
                                            ngram_labels=ngram_labels,
                                            ngram_masks=ngram_masks,
                                            num= num))
            num_sum+= num 
            
        # all_ngram_ids = torch.tensor([f.ngram_ids for f in features], dtype=torch.long)
        # # #size: [len(instances),seq_length,ngram_length]
        # all_ngram_labels = torch.tensor([f.ngram_labels for f in features], dtype=torch.long)
        # all_ngram_masks = torch.tensor([f.ngram_masks for f in features], dtype=torch.long)
        all_ngram_ids = []
        # # #size: [len(instances),seq_length,ngram_length]
        all_ngram_labels = []
        all_ngram_masks = []

        for f in features:
            all_ngram_ids += f.ngram_ids
            all_ngram_labels += f.ngram_labels
            all_ngram_masks += f.ngram_masks

        logging.info("Total train data: {}, avg for one sentence: {:.4f} ".\
            format(len(all_ngram_ids),len(all_ngram_ids)/idx))
        
        inputs_ids = torch.tensor(all_ngram_ids, dtype=torch.long)
        masks_ids = torch.tensor(all_ngram_masks, dtype=torch.long)
        outputs_ids = torch.tensor(all_ngram_labels, dtype=torch.long)
        dataset = TensorDataset(inputs_ids,masks_ids,outputs_ids)

        return dataset

    def build_optimizer(self, args: ClassifierArgs, **kwargs):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        return optimizer



    def build_trainer(self,dataset, data_loader):
         # get optimizer
        optimizer = self.build_optimizer(args)

        # get learning rate decay
        lr_scheduler = CosineAnnealingLR(optimizer, len(dataset) // self.mlm_batch_size * self.epochs)

        trainer = MlmTrainer(data_loader, self.model, None, optimizer, lr_scheduler)
        return trainer

    def load_valid(self):
        file_path = os.path.join(self.dataset_dir,\
            ("{}_valid_words.json").format(args.dataset_name))
        with open (file_path,'r') as  f:
            temp = json.load(f)
        return list(temp.keys())

    def save_model_to_file(self, save_dir: str, file_name: str):
        save_file_name = '{}.pth'.format(file_name)
        save_path = os.path.join(save_dir, save_file_name)
        torch.save(self.model.state_dict(), save_path)
        logging.info('Saving model to {}'.format(save_path))
    
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

    def saving_model_by_epoch(self, args: ClassifierArgs, epoch: int):
        # saving
        if args.saving_step is not None and args.saving_step != 0:
            if (epoch - 1) % args.saving_step == 0:
                file_name = self.get_modelfile_name(epoch=epoch)
                save_dir = os.path.join(self.saving_dir,self.get_modelfile_name())
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                self.save_model_to_file(save_dir, file_name)

    @torch.no_grad()
    def evaluate(self, args: ClassifierArgs,data_loader, is_training=False) -> Metric:
        
        self.model.eval()
        epoch_iterator = tqdm(data_loader)
        
        metric = DATASET_TYPE.get_evaluation_metric(args.dataset_name,compare_key=args.compare_key)
        #metric = DATASET_TYPE.get_mlm_evaluation_metric(args.dataset_name,compare_key=args.compare_key)
        for step, batch in enumerate(epoch_iterator):
            assert isinstance(batch[0], torch.Tensor)
            batch = tuple(t.cuda() for t in batch)
            golds = batch[2]
            
            inputs, golds = convert_batch_to_mlm_input_dict(batch, self.max_ngram_length)
            # outputs = self.model(**inputs, labels=golds)
            # #losses,logits = outputs[0],outputs[1]
            # loss,logits = outputs.loss,outputs.logits
            # print(f"Output[0].size:{outputs[0].size()},Output[0].size:{outputs[1].size()}")
            # print(f"loss:{loss}")
            logits = self.model.forward(**inputs)[0]
            losses = self.loss_function(logits.view(-1, self.tokenizer.vocab_size), golds.view(-1))
            epoch_iterator.set_description('loss: {:.4f}'.format(torch.mean(losses)))
            metric(losses, logits, golds)
            # print(f"logits.size:{logits.size()}")
            # print(f"loss.size:{losses.size()}")
            # print(f"loss:{losses}")
            #print(metric)
            # epoch_iterator.set_description('loss: {:.4f}'.format(loss))
            # metric(loss, logits, golds)

        print(f"metric-print:{metric}")
        logging.info("metric:{} ".format(metric))
        return metric

    def collate_fn(self,batch):
        """ 
        batch should be a list of (sequence, target, length) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest,
        """
        if len(batch[0]) == 4:
            all_input_ids, all_attention_mask, all_token_type_ids, all_lens = map(torch.stack, zip(*batch))
        else:
            all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))

        max_len = max(all_lens).item()
        #print(f"max_len in collate_fn:{max_len}")
        all_input_ids = all_input_ids[:, :max_len]
        all_attention_mask = all_attention_mask[:, :max_len]
        all_token_type_ids = all_token_type_ids[:, :max_len]
        if len(batch[0]) == 4:
            return all_input_ids, all_attention_mask, all_token_type_ids
        else:
            return all_input_ids, all_attention_mask, all_token_type_ids, all_labels

    def train(self):

        print("Loading initinal mlm model...")

        self.load_mlm()
        self.model.to(args.device)

        print('Loading dataset...')
        self.data_reader = DataReader()
        self.valid_vocab = self.load_valid()

        data_file_path = os.path.join(self.dataset_dir,"{}_mlm_data.txt".format(args.dataset_name))
        instances = self.data_reader.get_mlm_instance(data_file_path)
        train_dataset =self.convert_instances_to_features_mlm_train(instances)
        
        train_dataloader = DataLoader(train_dataset,  batch_size=self.mlm_batch_size,shuffle=True)
        
        print('Bulider trainier...')
        trainer = self.build_trainer(train_dataset,train_dataloader)
        
        print("Trainging...")
        best_metric = None
        for epoch in range(self.epochs):
            trainer.train_epoch(args, epoch)

            # saving model according to epoch_time
            self.saving_model_by_epoch(args, epoch)

            #evaluate model according to epoch_time
            metric = self.evaluate(args, train_dataloader,is_training=True)

            #update best metric
            #if best_metric is None, update it with epoch metric directly, otherwise compare it with epoch_metric
            if best_metric is None or metric > best_metric:
                logging.info("Update the best trained model:{}".format(epoch))
                best_metric = metric
                self.save_model_to_file(self.saving_dir,self.get_modelfile_name(best=True))
        
        self.evaluate(args,train_dataloader)

            
if __name__ == '__main__':
    
    #pwws/pso/ga/'fga'/'textfooler'/'bae'/'deepwordbug'/'textbugger'
  
    # python test_classifier.py --model_type bert --dataset_name snli
    # CUDA_VISIBLE_DEVICES=0 python3 mlm_train.py  --model_type bert --saving_step 1 --dataset_name sst2
    args = ClassifierArgs()._parse_args()
    #print(args)
    logging.info(args)
    
    #参数重置
    # if args.model_type =='bert':
    #         # print("******here bert******")
    #         args.model_name_or_path = '/data/ZhanghData/Pretrained_Models/bert-base-uncased'
    # elif args.model_type =='roberta':
    #         print("******here******")
    #         if args.dataset_name == 'snli':
    #             #args.model_name_or_path = '/data/ZhanghData/Pretrained_Models/roberta-large-snli'
    #             args.model_name_or_path = '/data/ZhanghData/Pretrained_Models/nli-MiniLM2-L6-H768'
    #         else:
    #             args.model_name_or_path = '/data/ZhanghData/Pretrained_Models/roberta-base'
        #args.model_name_or_path = '/data/ZhanghData/Pretrained_Models/roberta-base'


    #build logging
    # including check logging path, and set logging config
    args.build_mlm_logging_dir()
    args.build_mlm_logging()

    logging.info(args)

    args.build_environment()
    # check dataset and its path
    

    #args.build_mlm_saving_dir()
    args.build_caching_dir()

    mlm_trainer=Mlm_Train()
    mlm_trainer.train()
