# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os
from model.rgat_file.read_rgat_data import *
from model.rgat_file.read_dep_graph import *
import json
import random
from pytorch_pretrained_bert import BertModel, BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained(r'./bert-base-uncased/vocab.txt')
from transformers import RobertaTokenizer, RobertaModel
roberta_tokenizer = RobertaTokenizer.from_pretrained(r'./roberta-base')
import copy

def pad_dataset(dataset, bs):
    n_records = len(dataset)
    n_padded = bs - n_records % bs
    new_dataset = [t for t in dataset]
    new_dataset.extend(dataset[:n_padded])
    return new_dataset

def pad_seq(dataset, field, max_len, symbol):
    n_records = len(dataset)
    for i in range(n_records):
        if field in ['adj',"know_adj"]:
            dataset[i][field]=np.pad(dataset[i][field], \
                ((0,max_len-dataset[i][field].shape[0]),(0,max_len-dataset[i][field].shape[0])), 'constant')
        else:
            assert isinstance(dataset[i][field], list)
            while len(dataset[i][field]) < max_len:
                dataset[i][field].append(symbol)
    return dataset

def read(path):
    dataset = []
    sid = 0 # id
    with open(path, encoding='utf-8') as fp:
        for line in fp:
            record = {}
            tokens = line.strip().split()
            words, target_words = [], []
            d = []
            find_label = False
            for t in tokens:
                if '/p' in t or '/n' in t or '/0' in t:
                    end = 'xx'
                    y = 0
                    if '/p' in t:
                        end = '/p'
                        y = 0
                    elif '/n' in t:
                        end = '/n'
                        y = 1
                    elif '/0' in t:
                        end = '/0'
                        y = 2
                    words.append(t.strip(end))
                    target_words.append(t.strip(end))

                    if not find_label:
                        find_label = True
                        record['y'] = y
                        left_most = right_most = tokens.index(t)
                    else:
                        right_most += 1
                else:
                    words.append(t)
            if not find_label:
                record['y'] = None
            for pos in range(len(tokens)):
                if pos < left_most:
                    d.append(right_most - pos)
                else:
                    d.append(pos - left_most)
            record['sent'] = line.strip()
            record['words'] = words.copy()
            record['twords'] = target_words.copy()
            record['wc'] = len(words)
            record['wct'] = len(record['twords'])
            record['dist'] = d.copy()
            record['sid'] = sid
            record['beg'] = left_most
            record['end'] = right_most + 1
            sid += 1
            if record['y'] is not None:
                dataset.append(record)
    return dataset

def load_data(ds_name):
    data_npz = 'dataset_npy/dataset_%s.npz' % ds_name
    vocab_npy='dataset_npy/vocab_%s.npy' % ds_name
    if not os.path.exists(data_npz):
        train_file = './dataset/%s/train.txt' % ds_name
        test_file = './dataset/%s/test.txt' % ds_name
        train_set = read(path=train_file)
        test_set = read(path=test_file)
        train_wc = [t['wc'] for t in train_set]
        test_wc = [t['wc'] for t in test_set]
        max_len = max(train_wc) if max(train_wc) > max(test_wc) else max(test_wc)
        train_t_wc = [t['wct'] for t in train_set]
        test_t_wc = [t['wct'] for t in test_set]
        max_len_target = max(train_t_wc) if max(train_t_wc) > max(test_t_wc) else max(test_t_wc)
        train_set = pad_seq(dataset=train_set, field='dist', max_len=max_len, symbol=-1)
        test_set = pad_seq(dataset=test_set, field='dist', max_len=max_len, symbol=-1)
        train_set = calculate_position_weight(dataset=train_set)
        test_set = calculate_position_weight(dataset=test_set)
        vocab = build_vocab(dataset=train_set+test_set)
        train_set = set_wid(dataset=train_set, vocab=vocab, max_len=max_len)
        test_set = set_wid(dataset=test_set, vocab=vocab, max_len=max_len)
        train_set = set_tid(dataset=train_set, vocab=vocab, max_len=max_len_target)
        test_set = set_tid(dataset=test_set, vocab=vocab, max_len=max_len_target)
        dataset = [train_set,test_set]
        np.savez(data_npz,train=train_set,test=test_set)
        np.save(vocab_npy,vocab)
    else:
        dataset=np.load(data_npz,allow_pickle=True)
        train_set,test_set=dataset['train'],dataset['test']
        train_set,test_set=train_set.tolist(),test_set.tolist()
        dataset=[train_set,test_set]
        vocab=np.load(vocab_npy,allow_pickle=True).tolist()
    return dataset, vocab

def read_dep(path, args=None):
    dataset = []
    if args.gcn==2:
        if "laptop" in path:
            graph_path="AR-BERT/data/laptop/laptop_graph_embedding.npy"
        elif "rest" in path:
            graph_path="AR-BERT/data/rest/rest_graph_embedding.npy"
        elif "Twitter" in path:
            graph_path="AR-BERT/data/twitter/twitter_graph_embedding.npy"
        graph_embeddings=np.load(graph_path, allow_pickle=True).tolist()
    sid = 0  # id
    with open(path, encoding='utf-8') as fp:
        for line in fp:
            record = {}
            tokens = line.strip().split()
            words, target_words = [], []
            d = []
            mask=[]
            find_label = False
            for t in tokens:
                if '/p' in t or '/n' in t or '/0' in t:
                    end = 'xx'
                    y = 0
                    if '/p' in t:
                        end = '/p'
                        y = 0
                    elif '/n' in t:
                        end = '/n'
                        y = 1
                    elif '/0' in t:
                        end = '/0'
                        y = 2
                    words.append(t.strip(end))
                    target_words.append(t.strip(end))
                    mask.append(1)

                    if not find_label:
                        find_label = True
                        record['y'] = y
                        left_most = right_most = tokens.index(t)
                    else:
                        right_most += 1
                else:
                    words.append(t)
                    mask.append(0)
            if not find_label:
                record['y'] = None
            for pos in range(len(tokens)):
                if pos < left_most:
                    d.append(right_most - pos)
                else:
                    d.append(pos - left_most)
            record['sent'] = line.strip()
            record['words'] = words.copy()
            record['twords'] = target_words.copy()
            sentence=' '.join(words)
            
            syntactic_adj_matrix, common_adj_matrix, words_know=process_graph(sentence)
            record['adj'] = syntactic_adj_matrix
            record['mask']=mask
            common_mask=mask.copy()
            record['know_adj'] = common_adj_matrix
            record['know_mask']=common_mask
            record["words_know"]= words_know.copy()
            if args.gcn==2:
                record["graph_embedding"]=graph_embeddings["/".join(target_words)] if "/".join(target_words) in graph_embeddings.keys() else np.random.uniform(-0.25, 0.25, 256)
            record['wc'] = len(words)
            record['wct'] = len(record['twords'])
            record['dist'] = d.copy()
            record['sid'] = sid
            record['beg'] = left_most
            record['end'] = right_most + 1
            sid += 1
            if record['y'] is not None:
                dataset.append(record)
    return dataset

def load_data_dep(ds_name,args=None):
    if args.gcn==1:
        data_npz = 'dataset_npy/dataset_%s_dep_gcn.npz' % ds_name
        vocab_npy = 'dataset_npy/vocab_%s_dep_gcn.npy' % ds_name
    elif args.gcn==2:
        data_npz = 'dataset_npy/dataset_%s_dep-graph.npz' % ds_name
        vocab_npy = 'dataset_npy/vocab_%s_dep-graph.npy' % ds_name
    else:
        data_npz = 'dataset_npy/dataset_%s_dep.npz' % ds_name
        vocab_npy = 'dataset_npy/vocab_%s_dep.npy' % ds_name
    if not os.path.exists(data_npz):
        train_file = './dataset/%s/train.txt' % ds_name
        test_file = './dataset/%s/test.txt' % ds_name
        train_set = read_dep(path=train_file, args=args)
        test_set = read_dep(path=test_file, args=args)
        train_wc = [t['wc'] for t in train_set]
        test_wc = [t['wc'] for t in test_set]
        max_len = max(train_wc) if max(train_wc) > max(test_wc) else max(test_wc)
        train_t_wc = [t['wct'] for t in train_set]
        test_t_wc = [t['wct'] for t in test_set]
        max_len_target = max(train_t_wc) if max(train_t_wc) > max(test_t_wc) else max(test_t_wc)
        train_set = pad_seq(dataset=train_set, field='dist', max_len=max_len, symbol=-1)
        test_set = pad_seq(dataset=test_set, field='dist', max_len=max_len, symbol=-1)
        train_set = pad_seq(dataset=train_set, field='adj', max_len=max_len, symbol=-1)
        test_set = pad_seq(dataset=test_set, field='adj', max_len=max_len, symbol=-1)
        train_set = pad_seq(dataset=train_set, field='mask', max_len=max_len, symbol=0)
        test_set = pad_seq(dataset=test_set, field='mask', max_len=max_len, symbol=0)
        ###
        if args.gcn==1:
            train_know = [t['know_adj'].shape[0] for t in train_set]
            test_know = [t['know_adj'].shape[0] for t in test_set]
            max_len_know = max(train_know) if max(train_know) > max(test_know) else max(test_know)
            train_set = pad_seq(dataset=train_set, field='know_adj', max_len=max_len_know, symbol=-1)
            test_set = pad_seq(dataset=test_set, field='know_adj', max_len=max_len_know, symbol=-1)
            train_set = pad_seq(dataset=train_set, field='know_mask', max_len=max_len_know, symbol=0)
            test_set = pad_seq(dataset=test_set, field='know_mask', max_len=max_len_know, symbol=0)
        ### 
        train_set = calculate_position_weight(dataset=train_set, re_aspect=True)
        test_set = calculate_position_weight(dataset=test_set, re_aspect=True)
        vocab = build_vocab(dataset=train_set + test_set)
        train_set = set_wid(dataset=train_set, vocab=vocab, max_len=max_len)
        test_set = set_wid(dataset=test_set, vocab=vocab, max_len=max_len)
        if args.gcn==1:
            train_set = set_wid_know(dataset=train_set, vocab=vocab, max_len=max_len_know)
            test_set = set_wid_know(dataset=test_set, vocab=vocab, max_len=max_len_know)
        train_set = set_tid(dataset=train_set, vocab=vocab, max_len=max_len_target)
        test_set = set_tid(dataset=test_set, vocab=vocab, max_len=max_len_target)
        dataset = [train_set, test_set]
        np.savez(data_npz, train=train_set, test=test_set)
        np.save(vocab_npy, vocab)
    else:
        dataset = np.load(data_npz, allow_pickle=True)
        train_set, test_set = dataset['train'], dataset['test']
        train_set, test_set = train_set.tolist(), test_set.tolist()
        dataset = [train_set, test_set]
        vocab = np.load(vocab_npy, allow_pickle=True).tolist()
    return dataset, vocab

def load_rgat_data(ds_name):
    data_npz = './rgat_data/dataset_%s.npz' % ds_name
    vocab_npy = './rgat_data/vocab_%s.npy' % ds_name
    tag_vocab_npy = './rgat_data/tag_vocab_%s.npy' % ds_name
    if not os.path.exists(data_npz):
        train, test = get_dataset(ds_name)
        multi_hop = True
        add_non_connect = True
        max_hop = 4
        _, train_all_unrolled, _, _ = get_rolled_and_unrolled_data(train, multi_hop,add_non_connect,max_hop)
        _, test_all_unrolled, _, _ = get_rolled_and_unrolled_data(test, multi_hop,add_non_connect,max_hop)
        data=train_all_unrolled+test_all_unrolled
        if os.path.exists('./rgat_data/dep_tag_vocab_%s.pkl'% ds_name):
            with open('./rgat_data/dep_tag_vocab_%s.pkl'% ds_name, 'rb') as f:
                dep_tag_vocab = pickle.load(f)
        else:
            dep_tag_vocab = build_dep_tag_vocab(data, min_freq=0)
            with open('./rgat_data/dep_tag_vocab_%s.pkl'% ds_name, 'wb') as f:
                pickle.dump(dep_tag_vocab, f, -1)
        train_set=convert_data(train_all_unrolled)
        test_set=convert_data(test_all_unrolled)
        train_wc = [t['wc'] for t in train_set]
        test_wc = [t['wc'] for t in test_set]
        max_len = max(train_wc) if max(train_wc) > max(test_wc) else max(test_wc)
        train_t_wc = [t['wct'] for t in train_set]
        test_t_wc = [t['wct'] for t in test_set]
        max_len_target = max(train_t_wc) if max(train_t_wc) > max(test_t_wc) else max(test_t_wc)
        train_tag = [len(t['dep_tag']) for t in train_set]
        test_tag = [len(t['dep_tag']) for t in test_set]
        max_len_tag = max(train_tag) if max(train_tag) > max(test_tag) else max(test_tag)
        train_set = pad_seq(dataset=train_set, field='dist', max_len=max_len, symbol=-1)
        test_set = pad_seq(dataset=test_set, field='dist', max_len=max_len, symbol=-1)
        train_set = calculate_position_weight(dataset=train_set)
        test_set = calculate_position_weight(dataset=test_set)
        vocab = build_vocab(dataset=train_set + test_set)
        train_set = set_tag(dataset=train_set, vocab=dep_tag_vocab['stoi'], max_len=max_len_tag)
        test_set = set_tag(dataset=test_set, vocab=dep_tag_vocab['stoi'], max_len=max_len_tag)
        train_set = set_wid(dataset=train_set, vocab=vocab, max_len=max_len)
        test_set = set_wid(dataset=test_set, vocab=vocab, max_len=max_len)
        train_set = set_tid(dataset=train_set, vocab=vocab, max_len=max_len_target)
        test_set = set_tid(dataset=test_set, vocab=vocab, max_len=max_len_target)
        dataset = [train_set, test_set]
        tag_vocab=dep_tag_vocab['itos']
        np.savez(data_npz, train=train_set, test=test_set)
        np.save(vocab_npy, vocab)
        np.save(tag_vocab_npy,dep_tag_vocab['itos'])
    else:
        dataset = np.load(data_npz, allow_pickle=True)
        train_set, test_set = dataset['train'], dataset['test']
        train_set, test_set = train_set.tolist(), test_set.tolist()
        dataset = [train_set, test_set]
        vocab = np.load(vocab_npy, allow_pickle=True).tolist()
        tag_vocab = np.load(tag_vocab_npy, allow_pickle=True).tolist()
    return dataset, vocab, tag_vocab

def read_bert(path):
    dataset = []
    sid = 0  # id
    with open(path, encoding='utf-8') as fp:
        for line in fp:
            record = {}
            tokens = line.strip().split()
            words, target_words = [], []
            d = []
            find_label = False
            for t in tokens:
                if '/p' in t or '/n' in t or '/0' in t:
                    end = 'xx'
                    y = 0
                    if '/p' in t:
                        end = '/p'
                        y = 0
                    elif '/n' in t:
                        end = '/n'
                        y = 1
                    elif '/0' in t:
                        end = '/0'
                        y = 2
                    words.append(t.strip(end))
                    target_words.append(t.strip(end))

                    if not find_label:
                        find_label = True
                        record['y'] = y
                        left_most = right_most = tokens.index(t)
                    else:
                        right_most += 1
                else:
                    words.append(t)
            if not find_label:
                record['y'] = None
            for pos in range(len(tokens)):
                if pos < left_most:
                    d.append(right_most - pos)
                else:
                    d.append(pos - left_most)
            bert_sentence = bert_tokenizer.tokenize(' '.join(words.copy()))
            bert_aspect = bert_tokenizer.tokenize(' '.join(target_words.copy()))
            record['bert_token']=bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + bert_sentence + ['[SEP]'])
            record['bert_token_aspect']=bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + bert_aspect + ['[SEP]'])
            record['sent'] = line.strip()
            record['words'] = words.copy()
            record['twords'] = target_words.copy()
            record['wc'] = len(words)
            record['wct'] = len(record['twords'])
            record['bert_len']=len(record['bert_token'])
            record['bert_as_len']=len(record['bert_token_aspect'])
            record['dist'] = d.copy()
            record['dist'].append(-1)
            record['dist'].insert(0,-1)
            record['sid'] = sid
            record['beg'] = left_most
            record['end'] = right_most + 1
            sid += 1
            if record['y'] is not None:
                dataset.append(record)
    return dataset

def load_data_bert(ds_name):
    train_file = './dataset/%s/train.txt' % ds_name
    test_file = './dataset/%s/test.txt' % ds_name
    train_set = read_bert(path=train_file)
    test_set = read_bert(path=test_file)
    train_wc = [t['wc'] for t in train_set]
    test_wc = [t['wc'] for t in test_set]
    train_t_wc = [t['wct'] for t in train_set]
    test_t_wc = [t['wct'] for t in test_set]
    max_len_target = max(train_t_wc) if max(train_t_wc) > max(test_t_wc) else max(test_t_wc)
    #######bert######
    train_bert_len = [t['bert_len'] for t in train_set]
    test_bert_len = [t['bert_len'] for t in test_set]
    max_bert_len = max(train_bert_len) if max(train_bert_len) > max(test_bert_len) else max(test_bert_len)
    train_bert_as_len = [t['bert_as_len'] for t in train_set]
    test_bert_as_len = [t['bert_as_len'] for t in test_set]
    max_bert_as_len = max(train_bert_as_len) if max(train_bert_as_len) > max(test_bert_as_len) else max(test_bert_as_len)
    print(max_bert_len,max_bert_as_len)
    num1 = len(train_set)
    num2=len(test_set)
    for i in range(num1):
        train_set[i]['bert_token'].extend([0] * (max_bert_len - len(train_set[i]['bert_token'])))
        train_set[i]['bert_token_aspect'].extend([0] * (max_bert_as_len - len(train_set[i]['bert_token_aspect'])))
    for i in range(num2):
        test_set[i]['bert_token'].extend([0] * (max_bert_len - len(test_set[i]['bert_token'])))
        test_set[i]['bert_token_aspect'].extend([0] * (max_bert_as_len - len(test_set[i]['bert_token_aspect'])))
    #######bert######
    train_set = pad_seq(dataset=train_set, field='dist', max_len=max_bert_len, symbol=-1)
    test_set = pad_seq(dataset=test_set, field='dist', max_len=max_bert_len, symbol=-1)
    train_set = calculate_position_weight(dataset=train_set)
    test_set = calculate_position_weight(dataset=test_set)
    vocab = build_vocab(dataset=train_set + test_set)
    train_set = set_wid(dataset=train_set, vocab=vocab, max_len=max_bert_len)
    test_set = set_wid(dataset=test_set, vocab=vocab, max_len=max_bert_len)
    train_set = set_tid(dataset=train_set, vocab=vocab, max_len=max_len_target)
    test_set = set_tid(dataset=test_set, vocab=vocab, max_len=max_len_target)
    dataset = [train_set,test_set]
    return dataset, vocab

def read_dep_bert(path, is_bert):
    dataset = []
    sid = 0  # id
    with open(path, encoding='utf-8') as fp:
        for line in fp:
            record = {}
            tokens = line.strip().split()
            words, target_words = [], []
            d = []
            mask=[]
            find_label = False
            for t in tokens:
                if '/p' in t or '/n' in t or '/0' in t:
                    end = 'xx'
                    y = 0
                    if '/p' in t:
                        end = '/p'
                        y = 0
                    elif '/n' in t:
                        end = '/n'
                        y = 1
                    elif '/0' in t:
                        end = '/0'
                        y = 2
                    words.append(t.strip(end))
                    target_words.append(t.strip(end))
                    mask.append(1)
                    if not find_label:
                        find_label = True
                        record['y'] = y
                        left_most = right_most = tokens.index(t)
                    else:
                        right_most += 1
                else:
                    words.append(t)
                    mask.append(0)
            if not find_label:
                record['y'] = None
            for pos in range(len(tokens)):
                if pos < left_most:
                    d.append(right_most - pos)
                else:
                    d.append(pos - left_most)
            if is_bert ==1:
                bert_sentence = ['[CLS]'] + bert_tokenizer.tokenize(' '.join(words.copy())) + ['[SEP]']
                bert_aspect = ['[CLS]'] + bert_tokenizer.tokenize(' '.join(target_words.copy()))+ ['[SEP]']
                bert_concat=['[CLS]'] + bert_tokenizer.tokenize(' '.join(words.copy())) + ['[SEP]']+bert_tokenizer.tokenize(' '.join(target_words.copy()))+ ['[SEP]']
                record['bert_token']=bert_tokenizer.convert_tokens_to_ids(bert_sentence)
                record['concat_token']=bert_tokenizer.convert_tokens_to_ids(bert_concat)
                record['bert_token_aspect']=bert_tokenizer.convert_tokens_to_ids(bert_aspect)
            elif is_bert ==2:
                bert_sentence = ['<s>'] + roberta_tokenizer.tokenize(' '.join(words.copy())) + ['</s>']
                bert_aspect = ['<s>'] + roberta_tokenizer.tokenize(' '.join(target_words.copy()))+ ['</s>']
                bert_concat=['<s>'] + roberta_tokenizer.tokenize(' '.join(words.copy())) + ['</s>']+roberta_tokenizer.tokenize(' '.join(target_words.copy()))+ ['</s>']
                record['bert_token']=roberta_tokenizer.convert_tokens_to_ids(bert_sentence)
                record['concat_token']=roberta_tokenizer.convert_tokens_to_ids(bert_concat)
                record['bert_token_aspect']=roberta_tokenizer.convert_tokens_to_ids(bert_aspect)
            record['sent'] = line.strip()
            record['words'] = words.copy()
            record['twords'] = target_words.copy()
            record['adj'] = process_graph(' '.join(bert_sentence.copy()))
            record['mask']=mask
            record['mask'].append(0)
            record['mask'].insert(0,0)
            record['bert_len']=len(record['bert_token'])
            record['concat_len']=len(record['concat_token'])
            record['bert_as_len']=len(record['bert_token_aspect'])
            record['dist'] = d.copy()
            record['dist'].append(-1)
            record['dist'].insert(0,-1)
            record['sid'] = sid
            record['beg'] = left_most
            record['end'] = right_most + 1
            sid += 1
            if record['y'] is not None:
                dataset.append(record)
    return dataset

def load_data_dep_bert(ds_name, is_bert):
    if is_bert ==1:
        data_npz = 'dataset_npy/dataset_%s_dep_bert.npz' % ds_name
        vocab_npy = 'dataset_npy/vocab_%s_dep_bert.npy' % ds_name
    elif is_bert ==2:
        data_npz = 'dataset_npy/dataset_%s_dep_roberta.npz' % ds_name
        vocab_npy = 'dataset_npy/vocab_%s_dep_roberta.npy' % ds_name
    if not os.path.exists(data_npz):
        train_file = './dataset/%s/train.txt' % ds_name
        test_file = './dataset/%s/test.txt' % ds_name
        train_set = read_dep_bert(path=train_file, is_bert=is_bert)
        test_set = read_dep_bert(path=test_file, is_bert=is_bert)

        #######bert######
        train_bert_len = [t['bert_len'] for t in train_set]
        test_bert_len = [t['bert_len'] for t in test_set]
        max_bert_len = max(train_bert_len) if max(train_bert_len) > max(test_bert_len) else max(test_bert_len)
        train_concat_len = [t['concat_len'] for t in train_set]
        test_concat_len = [t['concat_len'] for t in test_set]
        max_concat_len = max(train_concat_len) if max(train_concat_len) > max(test_concat_len) else max(test_concat_len)
        train_bert_as_len = [t['bert_as_len'] for t in train_set]
        test_bert_as_len = [t['bert_as_len'] for t in test_set]
        max_bert_as_len = max(train_bert_as_len) if max(train_bert_as_len) > max(test_bert_as_len) else max(test_bert_as_len)
        print(max_bert_len,max_bert_as_len)
        num1 = len(train_set)
        num2=len(test_set)
        for i in range(num1):
            train_set[i]['bert_token'].extend([0] * (max_bert_len - len(train_set[i]['bert_token'])))
            train_set[i]['concat_token'].extend([0] * (max_concat_len - len(train_set[i]['concat_token'])))
            train_set[i]['bert_token_aspect'].extend([0] * (max_bert_as_len - len(train_set[i]['bert_token_aspect'])))
        for i in range(num2):
            test_set[i]['bert_token'].extend([0] * (max_bert_len - len(test_set[i]['bert_token'])))
            test_set[i]['concat_token'].extend([0] * (max_concat_len - len(test_set[i]['concat_token'])))
            test_set[i]['bert_token_aspect'].extend([0] * (max_bert_as_len - len(test_set[i]['bert_token_aspect'])))
        #######bert######

        train_set = pad_seq(dataset=train_set, field='dist', max_len=max_bert_len, symbol=-1)
        test_set = pad_seq(dataset=test_set, field='dist', max_len=max_bert_len, symbol=-1)
        train_set = pad_seq(dataset=train_set, field='adj', max_len=max_bert_len, symbol=-1)
        test_set = pad_seq(dataset=test_set, field='adj', max_len=max_bert_len, symbol=-1)
        train_set = pad_seq(dataset=train_set, field='mask', max_len=max_bert_len, symbol=0)
        test_set = pad_seq(dataset=test_set, field='mask', max_len=max_bert_len, symbol=0)
        train_set = calculate_position_weight(dataset=train_set)
        test_set = calculate_position_weight(dataset=test_set)
        vocab = build_vocab(dataset=train_set + test_set)
        train_set = set_wid(dataset=train_set, vocab=vocab, max_len=max_bert_len)
        test_set = set_wid(dataset=test_set, vocab=vocab, max_len=max_bert_len)
        train_set = set_tid(dataset=train_set, vocab=vocab, max_len=max_bert_as_len)
        test_set = set_tid(dataset=test_set, vocab=vocab, max_len=max_bert_as_len)
        dataset = [train_set,test_set]
        np.savez(data_npz, train=train_set, test=test_set)
        np.save(vocab_npy, vocab)
    else:
        dataset = np.load(data_npz, allow_pickle=True)
        train_set, test_set = dataset['train'], dataset['test']
        train_set, test_set = train_set.tolist(), test_set.tolist()
        dataset = [train_set, test_set]
        vocab = np.load(vocab_npy, allow_pickle=True).tolist()
    return dataset, vocab

def build_vocab(dataset):
    vocab = {}
    idx = 1
    n_records = len(dataset)
    for i in range(n_records):
        for w in dataset[i]['words']:
            if w not in vocab:
                vocab[w] = idx
                idx += 1
        for w in dataset[i]['twords']:
            if w not in vocab:
                vocab[w] = idx
                idx += 1
    return vocab

def build_vocab_pre(dataset):
    vocab = {}
    idx = 1
    n_records = len(dataset)
    for i in range(n_records):
        for w in dataset[i]['words']:
            if w not in vocab:
                vocab[w] = idx
                idx += 1
    return vocab

def set_wid(dataset, vocab, max_len):
    n_records = len(dataset)
    for i in range(n_records):
        sent = dataset[i]['words']
        dataset[i]['wids'] = word2id(vocab, sent, max_len)
    return dataset

def set_wid_know(dataset, vocab, max_len):
    n_records = len(dataset)
    for i in range(n_records):
        sent = dataset[i]['words_know']
        dataset[i]['wids_know'] = word2id(vocab, sent, max_len)
    return dataset

def set_tag(dataset, vocab, max_len):
    n_records = len(dataset)
    for i in range(n_records):
        sent = dataset[i]['dep_tag']
        dataset[i]['dep_tag_ids'] = word2id(vocab, sent, max_len)
    return dataset

def set_tid(dataset, vocab, max_len):
    n_records = len(dataset)
    for i in range(n_records):
        sent = dataset[i]['twords']
        dataset[i]['tids'] = word2id(vocab, sent, max_len)
    return dataset

def word2id(vocab, sent, max_len):
    wids=[]
    for w in sent:
        try:
            wids.append(vocab[w])
        except KeyError:
            wids.append(0)
    #wids = [vocab[w] for w in sent]
    if len(wids)> max_len:
        wids=wids[:max_len]
    while len(wids) < max_len:
        wids.append(0)
    return wids
 
def get_embedding(vocab, ds_name, args, types):
    emb_file = "./glove.840B.300d.txt"
    pkl = 'embeddings/%s_840B.pkl' % ds_name
    n_emb = 0
    graph_emb=0
    if types !='only_graph':
        if not os.path.exists(pkl):
            embeddings = np.zeros((len(vocab)+1, args.dim_w), dtype='float32')
            with open(emb_file, encoding='utf-8') as fp:
                for line in fp:
                    eles = line.strip().split()
                    w = eles[0]
                    n_emb += 1
                    if w in vocab:
                        try:
                            embeddings[vocab[w]] = [float(v) for v in eles[1:]]
                        except ValueError:
                            pass
            pickle.dump(embeddings, open(pkl, 'wb'))
        else:
            embeddings = pickle.load(open(pkl, 'rb'))

    if types == 'only_graph':
        if args.kge == "transe":
            graph_file = 'embeddings/entity_embeddings_transe_300.txt'
        elif args.kge == "complex":
            graph_file = "embeddings/entity_embeddings_complex_200.txt"
        elif args.kge == "analogy":
            graph_file = 'embeddings/entity_embeddings_analogy_400.txt'
        elif args.kge == "distmult":
            graph_file = 'embeddings/entity_embeddings_distmult_200.txt'

        if args.is_bert==0:
            graph_pkl = 'embeddings/{}_graph_{}.pkl'.format(ds_name,args.kge)
        elif args.is_bert==1:
            graph_pkl = 'embeddings/{}_graph_{}_bert.pkl'.format(ds_name,args.kge)
        elif args.is_bert==2:
            graph_pkl = 'embeddings/{}_graph_{}_roberta.pkl'.format(ds_name,args.kge)

        if not os.path.exists(graph_pkl):
            graph_embeddings = np.zeros((len(vocab)+1, args.dim_k), dtype='float32')
            with open(graph_file, encoding='utf-8') as fp:
                for line in fp:
                    eles = line.strip().split()
                    w = eles[0]
                    graph_emb += 1
                    if w in vocab:
                        try:
                            graph_embeddings[vocab[w]] = [float(eles[i+1]) for i in range(args.dim_k)]
                        except ValueError:
                            pass
            pickle.dump(graph_embeddings, open(graph_pkl, 'wb'))
        else:
            graph_embeddings = pickle.load(open(graph_pkl, 'rb'))

    if types == 'only_pre-graph':

        graph_file = 'KGE/vocab2vec-transe-wiki50.npy'
        if args.is_bert==0:
            graph_pkl = 'KGE/%s_graph_transe-wiki-50.pkl' % ds_name
        elif args.is_bert==1:
            graph_pkl = 'KGE/%s_graph_transe-wiki-50-bert.pkl' % ds_name
        elif args.is_bert==2:
            graph_pkl = 'KGE/%s_graph_transe-wiki-50-roberta.pkl' % ds_name

        if not os.path.exists(graph_pkl):
            pre_graph_embeddings = np.zeros((len(vocab)+1, args.dim_k), dtype='float32')
            wiki_embedding=np.load(graph_file, allow_pickle=True).tolist()
            for w, idx in vocab.items():
                try:
                    pre_graph_embeddings[idx] = wiki_embedding[w]
                except:
                    pass
            pickle.dump(pre_graph_embeddings, open(graph_pkl, 'wb'))
        else:
            pre_graph_embeddings = pickle.load(open(graph_pkl, 'rb'))

    if types == 'only_graph':
        return graph_embeddings
    elif types == "only_pre-graph":
        return pre_graph_embeddings
    elif types == 'only_word':
        return embeddings
    elif types == 'all':
        return embeddings, graph_embeddings
    else:
        print('error! Please input the correct types!')

import json

def build_dataset(args,is_bert=False):
    if is_bert:
        if args.model in ['ASGCN','KGNN',"BERT_vanilla"]:
            dataset, vocab = load_data_dep_bert(ds_name=args.ds_name, is_bert = args.is_bert)
        else:
            dataset, vocab = load_data_bert(ds_name=args.ds_name)
    else:
        if args.model =='RGAT':
            dataset, vocab, dep_vocab = load_rgat_data(ds_name=args.ds_name)
        elif args.model in ['ASGCN','KGNN']:
            dataset, vocab = load_data_dep(ds_name=args.ds_name,args=args)
        else:
            dataset, vocab = load_data(ds_name=args.ds_name)
    n_train = len(dataset[0])
    n_test = len(dataset[1])

    train_set = pad_dataset(dataset=dataset[0], bs=args.bs)
    test_set = pad_dataset(dataset=dataset[1], bs=args.bs)
    # assert 0
    if is_bert is False:
        embeddings = get_embedding(vocab, args.ds_name, args, types='only_word')
        for i in range(len(embeddings)):
            if i and np.count_nonzero(embeddings[i]) == 0:
                embeddings[i] = np.random.uniform(-0.25, 0.25, embeddings.shape[1])
        embeddings = np.array(embeddings, dtype='float32')
        if args.model == 'KGNN':
            graph_embeddings = get_embedding(vocab, args.ds_name, args, types='only_graph')
            for i in range(len(graph_embeddings)):
                if i and np.count_nonzero(graph_embeddings[i]) == 0:
                    graph_embeddings[i] = np.random.uniform(-0.25, 0.25, graph_embeddings.shape[1])
            graph_embeddings = np.array(graph_embeddings, dtype='float32')
            return [train_set, test_set], embeddings, graph_embeddings, n_train, n_test
        elif args.model =='RGAT':
            return [train_set,test_set], embeddings, n_train,n_test, dep_vocab
        else:
            return [train_set, test_set], embeddings, n_train, n_test
    else:
        if args.model == 'KGNN':
            bert_vocab={}
            if args.is_bert==1:
                with open(r'bert-base-uncased/vocab.txt','r', encoding='utf-8') as f:
                    for num,line in enumerate(f.readlines()):
                        bert_vocab[line.strip()]=num
            elif args.is_bert ==2:
                with open(r'./roberta-base/vocab.json', 'r', encoding='utf-8') as f:
                    line = f.readline()
                    bert_vocab = json.loads(line)
            graph_embeddings = get_embedding(bert_vocab, args.ds_name, args, types='only_graph')

            percent=0.00
            threshold=random.uniform(0,1)
            noise=int(len(vocab)*percent)
            noise_num=0
            for i in range(len(graph_embeddings)):
                if i and np.count_nonzero(graph_embeddings[i]) == 0:
                    graph_embeddings[i] = np.random.uniform(-0.25, 0.25, graph_embeddings.shape[1])
                elif random.uniform(0,1)<threshold and noise_num<noise:
                    graph_embeddings[i] = np.random.uniform(-0.25, 0.25, graph_embeddings.shape[1])
                    noise_num+=1
                elif noise_num == noise:
                    print('Introduce {} percent of noise'.format(percent))
                    noise_num+=1

            graph_embeddings = np.array(graph_embeddings, dtype='float32')
            return [train_set, test_set], graph_embeddings, n_train, n_test
        else:
            return [train_set, test_set], n_train, n_test



def calculate_position_weight(dataset,re_aspect=False):
    tmax = 40
    n_tuples = len(dataset)
    for i in range(n_tuples):
        dataset[i]['pw'] = []
        weights = []
        for w in dataset[i]['dist']:
            if re_aspect:
                if w <= 0:
                    weights.append(0.0)
                elif w > tmax:
                    weights.append(0.0)
                else:
                    weights.append(1.0 - float(w) / tmax)
            else:
                if w == -1:
                    weights.append(0.0)
                elif w > tmax:
                    weights.append(0.0)
                else:
                    weights.append(1.0 - float(w) / tmax)
        dataset[i]['pw'].extend(weights)
    return dataset

