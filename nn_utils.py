# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def get_batch_input(dataset, bs, args, idx=None,all=False):
    if all :
        batch_input = dataset
    else:
        batch_input = dataset[idx*bs:(idx+1)*bs]
    batch_data = pd.DataFrame.from_dict(batch_input)
    if args.model == 'RGAT':
        target_fields = [ 'wids', 'tids', 'dep_tag_ids', 'y', 'pw']
    elif args.model in ['ASGCN','KGNN']:
        target_fields = [ 'wids', 'tids', 'y', 'pw', 'adj', 'mask']
    else:
        target_fields = [ 'wids', 'tids', 'y', 'pw']
    batch_input_var = []
    for key in target_fields:
        data = list(batch_data[key].values)
        if key in ['pw']:
            batch_input_var.append(np.array(data, dtype='float32'))
        else:
            try:
                batch_input_var.append(np.array(data, dtype='int32'))
            except ValueError:
                print(batch_data[key].values)
    return batch_input_var

def get_batch_input_inference(dataset, bs, args, idx=None,all=False):
    if all :
        batch_input = dataset
    else:
        batch_input = dataset[idx*bs:(idx+1)*bs]
    batch_data = pd.DataFrame.from_dict(batch_input)
    if args.model == 'RGAT':
        target_fields = [ 'wids', 'tids', 'dep_tag_ids', 'y', 'pw']
    elif args.model in ['ASGCN','KGNN']:
        target_fields = [ 'words','twords','wids', 'tids', 'y', 'pw', 'adj', 'mask']
    else:
        target_fields = [ 'wids', 'tids', 'y', 'pw']
    batch_input_var = []
    for key in target_fields:
        data = list(batch_data[key].values)
        if key in ['pw']:
            batch_input_var.append(np.array(data, dtype='float32'))
        elif key in ['words','twords']:
            batch_input_var.append(data)
        else:
            try:
                batch_input_var.append(np.array(data, dtype='int32'))
            except ValueError:
                print(batch_data[key].values)
    return batch_input_var

def get_batch_input_bert(dataset, bs, args, idx=None,all=False):
    if all :
        batch_input = dataset
    else:
        batch_input = dataset[idx*bs:(idx+1)*bs]
    batch_data = pd.DataFrame.from_dict(batch_input)
    if args.model == 'RGAT':
        target_fields = [ 'bert_token', 'bert_token_aspect', 'dep_tag_ids', 'y', 'pw']
    elif args.model in ['ASGCN','KGNN']:
        target_fields = [ 'bert_token', 'bert_token_aspect', 'y', 'pw', 'adj', 'mask']
    else:
        target_fields = ['bert_token', 'bert_token_aspect', 'y', 'pw']
    batch_input_var = []
    for key in target_fields:
        data = list(batch_data[key].values)
        if key in ['pw']:
            batch_input_var.append(np.array(data, dtype='float32'))
        else:
            try:
                batch_input_var.append(np.array(data, dtype='int32'))
            except ValueError:
                print(batch_data[key].values)
    return batch_input_var


