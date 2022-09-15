# MPD Defense

Source code for paper: Masking and Purifying Inputs for Blocking Textual Adversarial Attacks

## Requirements
* pytorch (1.2.0)
* numpy (1.20.2)
* scikit-learn (0.24.1)
* nltk (3.6.2)
* transformers (4.18)
* spacy (1.5.4)

The code is built in Python 3.6.13. To install all required packages, run 
```
pip install -r requirements.txt
```

## Obtaining the data
To download the necessary datasets and pre-trained embeddings, run
```
cd data
sh download_data.sh
```

## Train/test a model
Now you can train a model by running 
```
CUDA_VISIBLE_DEVICES=1 python3 train.py --model_type bert --dataset_name imdb
```

A trained model can be tested by running 
```
CUDA_VISIBLE_DEVICES=1 python3 train.py --model_type bert --dataset_name imdb --mode test
```

## Attack a model
Once you have trained a model, run 
```
CUDA_VISIBLE_DEVICES=1 python attack.py --mode attack --model_type bert --dataset_name snli --attack_method bae
```

## Masking and Purifying for Defense
Finally, text-refactored detection can be done. 
```
 CUDA_VISIBLE_DEVICES=1 python3 purify.py --mode purify --model_type bert --dataset_name imdb --attack_method bae --mask_rate 0.6 --purify_num 10
```

## References
If you find this repository useful, please consider citing our paper:
```
***
```