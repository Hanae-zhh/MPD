#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2021/12/03 14:57:30
@Author      :zhangh
@version      :1.0
'''
import argparse
import os


class Config:
    parser = argparse.ArgumentParser(description="args for experiments")
    
    parser.add_argument(
        "-visible_devices",
        dest="visible_devices",
        default="3,5",
        type=str,
        help="which GPUs to use",
    )
   
    parser.add_argument(
        "-num_epoch",
        dest="num_epoch",
        default="20",
        type=int,
        help="number of epochs to train for",
    )
    parser.add_argument(
        "-dataset",
        dest="dataset",
        default="imdb",
        type=str,
        help="dataset to use; one of imdb, sst2",
    )

    parser.add_argument(
        "-gpu",
        dest="gpu",
        action="store_true",
        help="flag indicating whether to use GPU",
    )
    parser.add_argument(
        "-ngram",
        dest="ngram",
        default=2,
        type=int,
        action="store_true",
        help="the windows of input",
    )

    args = parser.parse_args()

    gpu = args.gpu
    visible_devices = args.visible_devices
    dataset = args.dataset
    n_gram = args.ngram

    num_epoch = args.num_epoch
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    project_root_path = "."
    data_root_path ="{}/data" .format(os.path.dirname(os.getcwd()).format())
    model_name_suffix = ""


    seq_max_len = 64 

    path_to_dist_mat = data_root_path + "/data/{}/dist_mat.npy".format(dataset)
    path_to_idx_to_dist_mat_idx_dict = (
        data_root_path + "/data/{}/idx_to_dist_mat_idx.pkl".format(dataset)
    )
    path_to_dist_mat_idx_to_idx_dict = (
        data_root_path + "/data/{}/dist_mat_idx_to_idx.pkl".format(dataset)
    )

    # Training params
    # batch_size_train = 100
    # batch_size_val = 100
    # batch_size_test = 100
    learning_rate = 1e-5
    dropout_rate = 0.1

    batch_size = 16 if dataset == "sst2" else 8
    epoch = 20
    learning_rate = 1e-5

    #  # Params for RoBERTa
    # if use_BERT:
    #     # RoBERTa params from https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.glue.md
    #     batch_size_train = 32 if dataset == "sst2" else 16
    #     batch_size_val = 32 if dataset == "sst2" else 16
    #     batch_size_test = 32 if dataset == "sst2" else 16
    #     weight_decay = 0.1
    #     warmup_percent = 0.06
    #     adam_eps = 1e-6
    #     learning_rate = 1e-5
    #     num_epoch = 10
    #     clip_norm = 0.0


    # # Params for purifier
    base_model_path='/data/ZhanghData/Pretrained_Models/bert-base-uncased'
    purify_params_dir = '/data/ZhanghData/MaskDefense/save_models/mask_language'
    
    # masked_rate = 0.15
    # masked_max_pred = 20
    masked_multiple = 1
    
    #purify_params_path = "/home/zhangh/workspace/FGWS-Detection/data/models/masklm/bert_base_LM_IMDB_0.2only_old.pkl"
    '''bert_base_LM_IMDB.pkl : mask以bertmask原则进行，每个数据mask一次'''
    '''bert_base_LM_IMDB_0.2only.pkl : mask以only mask原则进行，每个数据mask 4次'''


    

    