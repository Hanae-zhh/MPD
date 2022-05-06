import os
import numpy as np
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
import sys

from configLM import Config
sys.path.append("..")
config = Config()
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


def replace_abbreviations(text):
    '''
    '''
    texts = []
    for item in text:
        item = item.lower().replace("it's", "it is").replace("i'm", "i am").replace("he's", "he is").replace("she's", "she is")\
            .replace("we're", "we are").replace("they're", "they are").replace("you're", "you are").replace("that's", "that is")\
            .replace("this's", "this is").replace("can't", "can not").replace("don't", "do not").replace("doesn't", "does not")\
            .replace("we've", "we have").replace("i've", " i have").replace("isn't", "is not").replace("won't", "will not")\
            .replace("hasn't", "has not").replace("wasn't", "was not").replace("weren't", "were not").replace("let's", "let us")\
            .replace("didn't", "did not").replace("hadn't", "had not").replace("waht's", "what is").replace("couldn't", "could not")\
            .replace("you'll", "you will").replace("you've", "you have")
        item = item.replace("'s", "")
        texts.append(item)
    return texts

def load_stopwords():
    file_path = "/home/zhangh/dataset/english_stopwords.txt"
    stop_words = []
    with open(file_path,'r') as f:
        file = f.readlines()
        for line in file:
            stop_words.append(line.strip())
    return stop_words

def clear_review(text):
    
    texts = []
    for item in text:
        item = item.replace("<br /><br />", "")
        item = re.sub("[^a-zA-Z]", " ", item.lower())
        texts.append(" ".join(item.split()))
    return texts

def stemed_words(text):
   
    stop_words = load_stopwords()
    lemma = WordNetLemmatizer()
    texts = []
    for item in text:
        words = [lemma.lemmatize(w, pos='v') for w in item.split() if w not in stop_words ]
        #words = [lemma.lemmatize(w, pos='v') for w in item.split() if w not in stop_words and len(w)>1]
        texts.append(" ".join(words))
    return texts
            
def preprocess(text):
    
    text = replace_abbreviations(text)
    text = clear_review(text)
    text = stemed_words(text)
    return text

#读取数据
def read_file(filetype, file_path = None, clean_flag = None):
    '''
    filetype: train or test
    clean_flag: True——进行去停用词、词源统一、缩写还原处理
    '''
    #file_path = "/home/zhangh/dataset/imdb_dataset/aclImdb/"
    root_dir = os.path.dirname(os.getcwd()) 
    if file_path is None:
        file_path = root_dir+"/"+config.path_to_imdb
    file_list = []
    positive_path = file_path+"/"+filetype+"/pos/"
    for f in os.listdir(positive_path):
        file_list += [positive_path+f]
    negative_path = file_path+"/"+filetype+"/neg/"
    for f in os.listdir(negative_path):
        file_list += [negative_path+f]
    print('read', filetype, 'files:', len(file_list))
    all_labels = np.array(([1] * 12500 + [0] * 12500))
    all_texts = []

    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            filelines = file_input.readlines()
            all_texts += [rm_tags(filelines[0])]

    all_texts = [z.lower().replace('\n', '') for z in all_texts]
    all_texts = [z.replace('<br />', ' ') for z in all_texts]
    
    if clean_flag == True:
        all_texts = preprocess(all_texts)
    
    # all_texts = np.array(all_texts)
    return all_texts, all_labels
    # if filetype == 'train':
    #     return pd.DataFrame({'text': pd.Series(all_texts), 'label': pd.Series(all_labels)})
    # else:
    #     return pd.DataFrame({'text': pd.Series(all_texts), 'label': pd.Series(all_labels)})