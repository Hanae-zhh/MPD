#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2021/11/29 09:51:05
@Author      :zhangh
@version      :1.0
'''
# Predicting neighbors to a word in sentence using BERTMaskedLM. 
# Neighbors are from BERT vocab (which includes subwords and full words) 
from attr import mutable
from jsonschema.exceptions import best_match
import torch
# from torch._C import LongTensor
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict
import logging
import random
import sys,os
import copy
import numpy as np
from transformers.utils.dummy_pt_objects import LongformerForMaskedLM

p = os.path.dirname(os.getcwd())  #获取要导入模块的上上级目录
sys.path.append(p)
print(f"p:{p}")

from configLM import Config
import process_imdb
import process_sst
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from logger import Logger

#os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
DEFAULT_MODEL_PATH= config.base_model_path
#params_dir ='/home/zhangh/workspace/Attack-Word/data/model/bert_base_uncased/bert_base_model_beta.pkl'
params_dir = config.purify_params_dir

def load_stopwords():
    file_path = "/home/zhangh/dataset/english_stopwords.txt"
    stop_words = []
    with open(file_path,'r') as f:
        file = f.readlines()
        for line in file:
            stop_words.append(line.strip())
    return stop_words
stop_words = load_stopwords()
def init_model(model_path,to_lower):
    """
    Initiate BERTForMaskedLm model.
    """
    logging.basicConfig(level=logging.INFO)
    tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case=to_lower)
    model = BertForMaskedLM.from_pretrained(model_path)
    #model.load_state_dict(torch.load(params_dir))
    return model,tokenizer

def mask_words(words_list, tokenizer,max_pred=20, valid_max_pos=300, masked_rate=0.15, only_masked=True):
    """
    @description: 将文本中部分单词masked
    ---------
    @param: 
        words_list 存放一个文本中所有单词，开头是'[CLS]',结尾是'[SEP]'
        max_pred 最大mask数量
        valid_max_pos 最大mask下标，也就是最大文本长度
        masked_rate mask的数量
        only_masked 只mask不随机替换
    -------
    @Returns:
        words masked了之后的文本 格式等于word_list
         masked_tokens, masked_pos
    -------
    """
    words = copy.deepcopy(words_list)
    invalid_tokens =  ['[CLS]', '[SEP]', '[PAD]']
    n_pred =  min(max_pred, max(1, int(len(words) * masked_rate))) 
    #number of tokens in one sentence
    cand_maked_pos = [i for i, token in enumerate(words)
                          if token not in invalid_tokens and i<valid_max_pos]
    
    # candidate masked position
    random.shuffle(cand_maked_pos)
    masked_tokens, masked_pos = [], []
    invalid_replaced = []
    for token in invalid_tokens:    
        invalid_replaced.append(tokenizer.convert_tokens_to_ids(token))
    if only_masked:
        ''''''
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(words[pos])
            words[pos] = '[MASK]' # make mask
    
    else:
        for pos in cand_maked_pos[:n_pred]:
                masked_pos.append(pos)
                masked_tokens.append(words[pos])
                if random.random() < 0.8:  # 80%
                    words[pos] = '[MASK]' # make mask
                elif random.random() > 0.5:  # 10%
                    index = random.randint(0, tokenizer.vocab_size-1) 
                    # random index in vocabulary
                    while index in invalid_replaced: # can't involve 'CLS', 'SEP', 'PAD'
                        index = random.randint(0, tokenizer.vocab_size-1) 
                        words[pos] = tokenizer.convert_ids_to_tokens(index) # replace
    
    return words, masked_tokens, masked_pos

def create_dataset(traindata,tokenizer,max_len):
    '''
    imput
    traindata:原始输入数据，存放str文本，list
    tokenizer:模型的tokenizer
    output
    input_idxs:模型训练的输入indexs
    target_idxs:模型训练的输出indexs
    masked_tokens:masked的token
    masked_pos:masked的token对应文本中的位置
    '''
    input_idxs = []
    target_idxs = []
    masked_tokens_all = []
    masked_pos_all = []
    attention_mask = []
    masked_num = 0

    for text in traindata:
        input_tokens = []
        label_tokens = []
        org_tokens = tokenizer.tokenize(text)
        org_tokens = ['[CLS]']*2 + org_tokens + ['[SEP]']*2
        for i in range(0,len(org_tokens)-2):

            if org_tokens[i+2] not in stop_words:
                new_token = org_tokens[i:i+2]+['[MASK]']+org_tokens[i+2:i+5]
                input_tokens.append(new_token)
                label_tokens.append(org_tokens[i+2])
        
        #Padding
        if len(org_tokenized_text)<max_len:
            tmp=['[PAD]']*(max_len-len(org_tokenized_text))
            org_tokenized_text.extend(tmp)
            masked_tokenized_text.extend(tmp)
        elif len(org_tokenized_text)>max_len:
            org_tokenized_text = org_tokenized_text[:max_len]
            masked_tokenized_text = masked_tokenized_text[:max_len]

        attention_mask.append([1 if token!='[PAD]' else 0 for token in masked_tokenized_text ])

        target_indexed_tokens = tokenizer.convert_tokens_to_ids(org_tokenized_text)
        input_indexed_tokens = tokenizer.convert_tokens_to_ids(masked_tokenized_text)
        target_idxs.append(target_indexed_tokens)
        input_idxs.append(input_indexed_tokens)
        masked_tokens_all.append(masked_tokens)
        masked_pos_all.append(masked_pos)

    logger.log.info("Avg masked num: {}".format(masked_num/len(traindata)))
    return input_idxs,target_idxs,attention_mask,masked_tokens_all,masked_pos_all


def loader(datas,tokenizer, batch_size=8, multiple=1):
    """
    @description  :
    将原始数据转为bert模型输入数据
    ---------
    @param  :
        data: 
       
    -------
    @Returns  :
    -------
    """
    
    
    total_input_idxs=[]
    total_target_idxs=[]
    total_masked_tokens=[]
    total_masked_pos=[]
    total_attention_mask = []

    
    input_idxs,target_idxs,attention_mask,masked_tokens,masked_pos = \
        create_dataset(datas,tokenizer,max_len=300)
    total_input_idxs.extend(input_idxs)
    total_target_idxs.extend(target_idxs)
    total_masked_tokens.extend(masked_tokens)
    total_masked_pos.extend(masked_pos)
    total_attention_mask.extend(attention_mask)


    # logger.log.info('Loaded dataset:{} ; avg mask num: {:.3f}'.
    #                 format(len(total_input_idxs),masked_num/len(total_input_idxs)))
    train_set = TensorDataset(torch.LongTensor(total_input_idxs), \
                              torch.LongTensor(total_target_idxs),\
                              torch.LongTensor(total_attention_mask))
    train_loader = DataLoader(dataset=train_set,
                          batch_size=batch_size,
                          shuffle=True
                          )
    
    return train_loader,masked_tokens,masked_pos
    
def predict(tokenizer,logits,masked_pos,topK_sample=0,max_sample=1):
    """
    Guess masked tokens.
    """
    #result_idx_tokens = {}
    result = []
    #print(f"masked_pos: {masked_pos}")
    for pos in masked_pos:
        #print(f"pos:{pos}")
        # print(len(logits[pos])-1)
        # print(len(logits[pos].tolist()))
        preds = dict(zip(range(0,len(logits[pos])-1),logits[pos].tolist()))
        #print(f"preds: {preds}")
        #print(predictions.logits[0][idx])
        #print("Average score: ",torch.mean(predictions.logits[0][idx],dim=0))
        sorted_pred = OrderedDict(sorted(preds.items(), 
            key=lambda kv: kv[1], reverse=True))
        idx = sample(sorted_pred,topK_sample,max_sample)
        result.append(tokenizer.convert_ids_to_tokens(idx))
        #result_idx_tokens[pos]=tokenizer.convert_ids_to_tokens(idx)
    return result

def train_model(model,tokenizer,dataset,epoch=4,batch_size=8):
    ''''''
    model.train()  # 将模型设置为训练模式
    model.to(device)

    train_loader,masked_tokens,masked_pos = loader(dataset,tokenizer,batch_size,config.masked_multiple)

    avg_loss = []
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    for e in range(epoch):
        for batch_idx,(input_idxs,target_idxs,attention_masked) in enumerate(train_loader):
            input_idxs,target_idxs,attention_masked=\
                input_idxs.to(device),target_idxs.to(device),attention_masked.to(device)
            #print(input_idxs.shape,target_idxs.shape)
            output = model(input_idxs,attention_mask = attention_masked, labels = target_idxs)
            loss,logits = output[0],output[1]
            loss = loss / batch_size  # 梯度积累
            avg_loss.append(loss.item())
            loss.backward()
            if ((batch_idx + 1) % batch_size) == 0:
                # 每 8 次更新一下网络中的参数
                optimizer.step()
                optimizer.zero_grad()
            if batch_idx % 1000 == 0:
                logger.log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                    e + 1, batch_idx, len(train_loader), 100. *
                    batch_idx / len(train_loader), np.array(avg_loss).mean()
                ))
    logger.log.info('Finished Training')
    return model

'''测试'''

def softmax(v):
    l1 = list(map(lambda x: np.exp(x), v))
    return list(map(lambda x: x / sum(l1), l1))

def sample(score_dict,topK_sample,max_sample):
    '''
    通过模型输出进行采样
    score_dict,topK_sample,max_sample
    '''
    
    if max_sample > 0 :#最大值采样
        return list(score_dict.keys())[0]
    else:#top_k 采样
        pred_scores = list(score_dict.values())[:topK_sample]
        pred_idxs = list(score_dict.keys())[:topK_sample]
        probs = np.array(softmax(pred_scores))
        pred_id = np.random.choice(a=pred_idxs, size=1, replace=True, p=probs)[0]
        return pred_id

def test_model(model,tokenizer,test_set):
    '''
    test_text:list[str,str,...]
    '''
    model.eval()
    model= model.to(device)
    print("Loading test dataset...")
    test_loader,masked_tokens,masked_pos = loader(test_set,tokenizer,batch_size=config.batch_size)
    print("Load test dataset!")
    #print(len(test_loader), len(masked_tokens),len(masked_posed))
    total = 0 #测试机masked 总数
    correct = 0 #预测正确的数量
    acc_sum = 0

    with torch.no_grad():
        for idx,(input,_,attention_mask) in enumerate(test_loader):
            text_cor = 0
            input = input.to(device)
            attention_mask= attention_mask.to(device)
            output = model(input, attention_mask=attention_mask)
            pred_tokens = predict(tokenizer,output.logits[0],masked_pos[idx])#
            # print(f"mask num : {masked_pos[idx]}")
            # print(f"result: {result_idx_tokens}")
            # print(f"masked_token: {masked_tokens[idx]}")
            # print(f"length of masked_tokens[idx] {len(masked_tokens[idx])},\
            #      length of pred_tokens[i] :{len(pred_tokens)}")
            
            for i in range(len(masked_tokens[idx])):
                if masked_tokens[idx][i]==pred_tokens[i]: 
                    text_cor+= 1 
            #text_acc = text_cor/len(input)
            #print(f"")
            correct+= text_cor
            acc_sum += float(text_cor/len(masked_pos[idx]))
            total += (len(masked_pos[idx]))

    logger.log.info("correctly predict: {}，masked tokens: {},accuracy: {} %".format(correct, total,np.round(100*(correct/total),2)) )
    logger.log.info("Avg accuracy for every text: {}%".format(np.round((acc_sum/idx)*100,2) ) )

if __name__ == "__main__":
    
    #config.mode = ''
    if config.dataset == 'imdb':

        # config.masked_max_len = 300
        #config.masked_max_len = 256
        # config.masked_rate = 0.2
        # config.masked_max_pred = 50
        # config.masked_multiple = 4
        train_texts,_ = process_imdb.read_file('test',clean_flag=False)
        
    
    elif config.dataset == 'sst2':
        #config.masked_max_len = 128
        # config.masked_rate = 0.2
        # config.masked_max_pred = 50
        # config.masked_multiple = 4
        train_texts,_ = process_sst.load_sst("test",clean_flag=False)
    params_path = params_dir+"/bert_base_mlm_{}_{}.pkl".\
                                             format(config.dataset,config.n_gram)

    logger = Logger(config,path_dir=params_dir,adding='_{}_{}'.\
                                             format(config.dataset,config.n_gram))

    logger.log.info("config.m_gram: {}".format(config.n_gram))

    
    model, tokenizer = init_model(DEFAULT_MODEL_PATH,to_lower=False)
 

    #train
    model = train_model(model,tokenizer,train_texts,epoch=config.epoch,batch_size= config.batch_size)
    torch.save(model.state_dict(), params_path )
    logger.log.info("Model saved, path:\n{}".format(params_path))
    
    #test
    #model.load_state_dict(torch.load(params_path))
    logger.log.info("------Test infomation------")
    test_model(model,tokenizer,train_texts)
    
