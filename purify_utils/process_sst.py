from spacy.lang.en import English
import re
import sys
import csv
sys.path.append("..")
from configLM import Config
config = Config()
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)

# from configLM import Config
config=Config()

nlp = English()
spacy_tokenizer = nlp.tokenizer

def clean_str(string, tokenizer=None):
    """
    Parts adapted from https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/mydatasets.py
    """
    assert isinstance(string, str)
    string = string.replace("<br />", "")
    string = re.sub(r"[^a-zA-Z0-9.]+", " ", string)

    return (
        string.strip().lower().split()
        if tokenizer is None
        else [t.text.lower() for t in tokenizer(string.strip())]
    )

def cut_raw(seq, max_len):
    assert isinstance(seq, list)
    return seq[:max_len]

def load_sst(clean_flag = False):
    texts = []
    labels = []

    with open("{}/{}.tsv".format(config.path_to_sst2, "train")) as f:
        #print("self.config.path_to_sst2:\n{self.config.path_to_sst2}")
        data = csv.reader(f, delimiter="\t")
        header = False
        
        for line in data:
            if not header:
                header = True
            else:
                [text, label] = line
                if clean_flag == True:
                    cleaned = cut_raw(
                    clean_str(text, tokenizer=spacy_tokenizer),
                    config.max_len,)
                    texts.append(cleaned)
                else:
                    texts.append(text)
                labels.append(int(label.replace("\n", "").strip()))
    return texts, labels