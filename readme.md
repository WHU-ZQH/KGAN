# KGAN: Knowledge Graph Augmented Network Towards Multi-level Representation Learning for Aspect-based Sentiment Analysis

___

## Requirements

* python 3.7
* pytorch-gpu 1.7 
* numpy 1.19.4
* pytorch_pretrained_bert 0.6.2
* nltk 3.3 
* spacy 2.2.0
* en_core_web_sm 2.1.0
* GloVe.840B.300d
* bert-base-uncased

## Environment

- OS: Windows 10
- GPU: NVIDIA GeForce GTX 1660 SUPER
- CUDA: 11.0
- cuDNN: v8.0.4

## Dataset

* raw data: "./dataset/"
* processing data: "./dataset_npy/"
* word embedding file: "./embeddings/"

## Training options

- **ds_name**: the name of target dataset, ['14semeval_laptop', '14semeval_rest'], default='14semeval_rest'
- **bs**: batch size to use during training, [32, 64], default=64
- **learning_rate**: learning rate to use, [0.001, 0.0005], default=0.001
- **n_epoch**: number of epoch to use, default=20
- **model**: the name of model, default='KGNN'
- **dim_w**: the dimension of word embeddings, default=300
- **dim_k**: the dimension of graph embeddings, [200,400],  default=200
- **is_test**:  train or test the model, [0, 1], default=1
- **is_bert**: GloVe-based or BERT-based, [0, 1], default=0

## Running

#### training based on GloVe: 

* python ./main.py   -ds_name 14semeval_laptop   -bs 32   -learning_rate 0.001   -n_epoch 20   -model KGNN -dim_w 300 -dim_k 400  -is_test 0   -is_bert 0
* python ./main.py   -ds_name 14semeval_rest   -bs 64   -learning_rate 0.001   -n_epoch 20   -model KGNN -dim_w 300 -dim_k 200  -is_test 0   -is_bert 0

## Evaluation

To have a quick look, we saved the best model weight trained on the evaluated datasets in the "./model_weight/best_model_weight". You can easily load them and test the performance. You can evaluate the model weight with:

- python ./main.py   -ds_name 14semeval_laptop   -bs 32  -model KGNN -dim_w 300 -dim_k 400 -is_test 1 
- python ./main.py   -ds_name 14semeval_rest   -bs 64  -model KGNN -dim_w 300 -dim_k 200 -is_test 1 

## Notes

- The datasets and more than 50% of the code are borrowed from TNet-ATT (Tang et.al, ACL2019).

  
