# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import math
import torch
import logging
import numpy as np
import torch.nn as nn
from overrides import overrides
from typing import List, Any, Dict, Union, Tuple
from tqdm import tqdm
#from torchnlp.samplers.bucket_batch_sampler import BucketBatchSampler
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizer

from args import ClassifierArgs
from utils.config import PRETRAINED_MODEL_TYPE, DATASET_TYPE
from data.reader import DataReader
from data.processor import DataProcessor
from data.instance import InputInstance
from data.dataset import ListDataset
from utils.metrics import Metric, RandomSmoothAccuracyMetrics, RandomAblationCertifyMetric
from utils.loss import ContrastiveLearningLoss, UnsupervisedCircleLoss
from utils.mask import mask_instance, mask_forbidden_index
from predictor import Predictor
from utils.utils import collate_fn, xlnet_collate_fn, convert_batch_to_bert_input_dict, build_forbidden_mask_words
from utils.hook import EmbeddingHook
from trainer import (BaseTrainer,
                    FreeLBTrainer,
                    PGDTrainer,
                    HotflipTrainer,
                    EmbeddingLevelMetricTrainer,
                    TokenLevelMetricTrainer,
                    RepresentationLearningTrainer,
                    MaskTrainer,
                    SAFERTrainer
                    )
from utils.textattack import build_english_attacker
from utils.textattack import CustomTextAttackDataset, SimplifidResult
from textattack.models.wrappers import HuggingFaceModelWrapper, HuggingFaceModelMaskEnsembleWrapper, HuggingFaceModelSaferEnsembleWrapper
from textattack.loggers.attack_log_manager import AttackLogManager
#from textattack import AttackArgs
from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult
from utils.public import auto_create
from utils.certify import predict, lc_bound, population_radius_for_majority, population_radius_for_majority_by_estimating_lambda, population_lambda
from torch.optim.adamw import AdamW

import logging
import os
# os.environ["CUDA_VISIBLE_DEVICES"] =  "1,3"
#os.environ['TFHUB_CACHE_DIR'] = '/home/zhangh/.cache/tfhub_modules'
os.environ['TFHUB_CACHE_DIR'] = '/home/zhangh/.cache'
class Classifier:
    def __init__(self, args: ClassifierArgs):

        # for data_reader and processing
        self.data_reader, self.tokenizer, self.data_processor = self.build_data_processor(args)
        #data_reader, tokenizer, data_processor
        self.model = self.build_model(args)
        self.type_accept_instance_as_input = ['conat', 'sparse', 'safer']
        self.loss_function = self.build_criterion(args.dataset_name)
        
        self.forbidden_words = None
        if args.keep_sentiment_word:
            self.forbidden_words = build_forbidden_mask_words(args.sentiment_path)

    def save_model_to_file(self, save_dir: str, file_name: str):
        save_file_name = '{}.pth'.format(file_name)
        save_path = os.path.join(save_dir, save_file_name)
        torch.save(self.model.state_dict(), save_path)
        logging.info('Saving model to {}'.format(save_path))

    def loading_model_from_file(self, save_dir: str, file_name: str):
        load_file_name = '{}.pth'.format(file_name)
        load_path = os.path.join(save_dir, load_file_name)
        assert os.path.exists(load_path) and os.path.isfile(load_path), '{} not exits'.format(load_path)
        self.model.load_state_dict(torch.load(load_path), strict=False)
        logging.info('Loading model from {}'.format(load_path))

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

    def build_model(self, args: ClassifierArgs) -> nn.Module:
        # config_class: PreTrainedConfig
        # model_class: PreTrainedModel
        config_class, model_class, _ = PRETRAINED_MODEL_TYPE.MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(
            args.model_name_or_path,
            num_labels=self.data_reader.NUM_LABELS,
            finetuning_task=args.dataset_name,
            output_hidden_states=True,
        )
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool('ckpt' in args.model_name_or_path),
            config=config
        ).cuda()
        return model

    def build_data_processor(self, args: ClassifierArgs, **kwargs) -> List[Union[DataReader, PreTrainedTokenizer, DataProcessor]]:
        data_reader = DATASET_TYPE.DATA_READER[args.dataset_name]()
        #for example data_reader = BinarySentimentAnalysisDataReader()
        _, _, tokenizer_class = PRETRAINED_MODEL_TYPE.MODEL_CLASSES[args.model_type]
        #print("to")
        #for example, with model_type=bert,return (BertConfig, BertForSequenceClassification, BertTokenizer)
        print(f" **********args.model_name_or_path: { args.model_name_or_path}")
   
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path,
            do_lower_case=args.do_lower_case
        )
        data_processor = DataProcessor(data_reader=data_reader,
                                       tokenizer=tokenizer,
                                       model_type=args.model_type,
                                       max_seq_length=args.max_seq_length)

        return [data_reader, tokenizer, data_processor]

    def build_criterion(self, dataset):
        return DATASET_TYPE.get_loss_function(dataset)

    def build_data_loader(self, args: ClassifierArgs, data_type: str, use_tokenizer: bool = True, **kwargs) -> List[Union[Dataset, DataLoader]]:
        
        # for some training type, when training, the inputs type is Inputstance
        #print(f"use_tokenizer:{use_tokenizer}")
        if data_type == 'train' and args.training_type in self.type_accept_instance_as_input:
            use_tokenizer = False
            #print(f"use_tokenizer in of :{use_tokenizer}")
        shuffle = True if data_type == 'train' else False
        file_name = data_type
        if file_name == 'train' and args.file_name is not None:
            file_name = args.file_name
        #print(f"file_name:{file_name}") #train
        name = '{}_max{}{}'.format(file_name, args.max_seq_length, '_tokenizer' if use_tokenizer else '')
        #print(f"name:{name}")# train_max256_tokenizer
        # print(f"use_tokenizer:{use_tokenizer}")
        # print(f"file_name:{file_name}")
        # print(f"name:{name}")
        dataset = auto_create(name,
                            lambda: self.data_processor.read_from_file(args.dataset_dir, file_name, tokenizer=use_tokenizer),
                            #args.dataset_dir = /data/ZhanghData/MaskDefense/dataset/[dataset]
                            True, args.caching_dir)#args.caching_dir =/data/ZhanghData/MaskDefense/caches/[dataset]_[model]
        #dataset:Dataset类型,已经是tensor了
        
        # for collate function
        if use_tokenizer:
            collate_function = xlnet_collate_fn if args.model_type in ['xlnet'] else collate_fn
        else:
            collate_function = lambda x: x
      
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_function)
        #collate_fn 将一个list的sample组成一个mini-batch的函数
        #关于DataLoader：https://blog.csdn.net/g11d111/article/details/81504637
        return [dataset, data_loader]


    def build_attacker(self, args: ClassifierArgs, **kwargs):
        if args.training_type == 'sparse' or args.training_type == 'safer':
            if args.dataset_name in ['agnews', 'imdb']:
                batch_size = 300
            else:
                batch_size = 600
            if args.training_type == 'sparse':
                model_wrapper = HuggingFaceModelMaskEnsembleWrapper(args, 
                                                                    self.model, 
                                                                    self.tokenizer, 
                                                                    batch_size=batch_size)
            else:
                model_wrapper = HuggingFaceModelSaferEnsembleWrapper(args, 
                                                                    self.model, 
                                                                    self.tokenizer, 
                                                                    batch_size=batch_size)
        else:
            model_wrapper = HuggingFaceModelWrapper(self.model, self.tokenizer, batch_size=args.batch_size)
        

        attacker = build_english_attacker(args, model_wrapper)
        return attacker

    def build_writer(self, args: ClassifierArgs, **kwargs) -> Union[SummaryWriter, None]:
        writer = None
        if args.tensorboard == 'yes':
            tensorboard_file_name = '{}-tensorboard'.format(args.build_logging_path())
            tensorboard_path = os.path.join(args.logging_dir, tensorboard_file_name)
            writer = SummaryWriter(tensorboard_path)
        return writer


    @torch.no_grad()
    def evaluate(self, args: ClassifierArgs, is_training=False) -> Metric:
        if is_training:
            logging.info('Using current modeling parameter to evaluate.')
            data_type = 'validation' if args.dataset_name == 'snli' else 'dev' 
        else:
            self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description='best'))
            data_type = args.evaluation_data_type
        self.model.eval()

        dataset, data_loader = self.build_data_loader(args, data_type)
        epoch_iterator = tqdm(data_loader)

        metric = DATASET_TYPE.get_evaluation_metric(args.dataset_name,compare_key=args.compare_key)
        for step, batch in enumerate(epoch_iterator):
            assert isinstance(batch[0], torch.Tensor)
            batch = tuple(t.cuda() for t in batch)
            golds = batch[3]
            inputs = convert_batch_to_bert_input_dict(batch, args.model_type)
            logits = self.model.forward(**inputs)[0]
            losses = self.loss_function(logits.view(-1, self.data_reader.NUM_LABELS), golds.view(-1))
            epoch_iterator.set_description('loss: {:.4f}'.format(torch.mean(losses)))
            metric(losses, logits, golds)

        print(metric)
        logging.info(metric)
        return metric
    
    def attack(self, args: ClassifierArgs, **kwargs):
        # self.evaluate(args, is_training=False)
        # self.evaluate(args, is_training=False)
        self.loading_model_from_file(args.saving_dir, args.build_saving_file_name(description='best'))
        self.model.eval()
        print(f'************Model loaded!')
        # build test dataset 
        
        dataset, _ = self.build_data_loader(args, args.evaluation_data_type, use_tokenizer=False)
        test_instances = dataset.data
        print(f"***********Total instances: {len(test_instances)}")
        print(f'***********Data loaded!')

        # print(f'************Type of test instance: {type(test_instances)}')
        # print(f'{type(dataset)}')
        # print(f"test_instance 1 :{test_instances[1]}")
        # print(f"len of test instance:{len(test_instances)}")
        
        #build attacker
        attacker = self.build_attacker(args)

        attacker_log_path = '{}'.format(args.build_logging_path())
        attacker_log_path = os.path.join(args.attacking_dir, attacker_log_path)
        #attacker_log_path = os.path.join(args.logging_dir, attacker_log_path)
        attacker_log_manager = AttackLogManager()
        # attacker_log_manager.enable_stdout()
        attacker_log_manager.add_output_file(os.path.join(attacker_log_path, '{}_{}.txt'.format(args.attack_method,args.attack_numbers)))
        attacker_log_manager.add_output_csv(os.path.join(attacker_log_path, '{}_{}.csv'.format(args.attack_method,args.attack_numbers)),None)
        #'plain'
        
        for i in range(args.attack_times):
            print("Attack time {}".format(i))
            
            choice_instances = np.random.choice(test_instances, size=(args.attack_numbers,),replace=False)
            dataset = CustomTextAttackDataset.from_instances(args.dataset_name, choice_instances, self.data_reader.get_labels())
            results_iterable = attacker.attack_dataset(dataset)
            description = tqdm(results_iterable, total=len(choice_instances))
            result_statistics = SimplifidResult()
            for result in description:
                try:
                    attacker_log_manager.log_result(result)
                    result_statistics(result)
                    description.set_description(result_statistics.__str__())
                except RuntimeError as e:
                    print('error in process')

        attacker_log_manager.enable_stdout()
        attacker_log_manager.log_summary()



    def mask_instance_decorator(self, args: ClassifierArgs, instance:InputInstance, numbers:int=1, return_indexes:bool=False):
        if self.forbidden_words is not None:
            forbidden_index = mask_forbidden_index(instance.perturbable_sentence(), self.forbidden_words)
            return mask_instance(instance, args.sparse_mask_rate, self.tokenizer.mask_token, numbers, return_indexes, forbidden_index)
        else:
            return mask_instance(instance, args.sparse_mask_rate, self.tokenizer.mask_token, numbers, return_indexes)

if __name__ == '__main__':
    
    #pwws/pso/ga/'fga'/'textfooler'/'bae'/'deepwordbug'/'textbugger'
  
    #CUDA_VISIBLE_DEVICES=1 python test_classifier.py --model_type bert --dataset_name snli
    #CUDA_VISIBLE_DEVICES=3 python attack.py --mode attack --model_type bert --dataset_name snli --attack_method bae 
    args = ClassifierArgs()._parse_args()
    #print(args)
    logging.info(args)
    
#参数重置
    if args.model_type =='bert':
            # print("******here bert******")
            args.model_name_or_path = '/data/ZhanghData/Pretrained_Models/bert-base-uncased'
    elif args.model_type =='roberta':
            print("******here******")
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

    classifier = Classifier(args)
    classifier.attack(args)
