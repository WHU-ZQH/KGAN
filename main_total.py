# -*- coding: utf-8 -*-
import argparse
import time
from model.ATAE_LSTM import ATAE_LSTM, ATAE_LSTM_Bert
from model.GCAE import GCAE,  GCAE_Bert
from model.RGAT import RGAT
from model.ASGCN import ASGCN, ASGCN_BERT
from model.KGNN import KGNN, KGNN_BERT
from model.bert_vanill import BERT_vanilla
from model.IAN import IAN
from model.TNet import TNet_LF
from utils import *
from nn_utils import *
import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn
import math
from sklearn import metrics
import warnings
import torch
from torchstat import stat
import thop
import logging
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"   #"0,1,2,3"
seed = 14
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def _reset_params(model):
    for name,p in model.named_parameters():
        if 'bert' not in name:
            if p.requires_grad:
                if len(p.shape) > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

def train(args,times=0):
    start_time=time.time()
    if args.model == 'KGNN':
        dataset, embeddings,graph_embeddings, n_train, n_test = build_dataset(args=args)
    elif args.model =='RGAT':
        dataset, embeddings, n_train, n_test, dep_vocab = build_dataset(args=args)
    else:
        dataset, embeddings, n_train, n_test = build_dataset(args=args)
    end_time=time.time()
    print("DP:%s s"%(end_time-start_time))
    flops, params=[],[]
    
    args.embeddings = embeddings
    if args.model == 'KGNN':
        args.graph_embeddings=graph_embeddings
    n_train_batches = math.ceil(n_train / args.bs)
    n_test_batches = math.ceil(n_test / args.bs)
    train_set,test_set = dataset

    if args.model == 'KGNN':
        model = KGNN(args)
        _reset_params(model)
    elif args.model == 'GCAE':
        model = GCAE(args=args)
    elif args.model == 'ATAE':
        model = ATAE_LSTM(args=args)
    elif args.model == 'ASGCN':
        model = ASGCN(args)
    elif args.model =='IAN':
        model =IAN(args)
    elif args.model =='TNet':
        model = TNet_LF(args)
    elif args.model == 'RGAT':
        model = RGAT(args, dep_tag_num=len(dep_vocab))
    else:
        print('model error')
    if torch.cuda.is_available():
        model = model.cuda()
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    # total_num = sum(p.numel() for p in model.parameters())
    # trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_time = []
    result_store_test = [[], []]
    if args.model in ['ASGCN','KGNN', 'RGAT']:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    else:
        optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    save_acc,save_f1=0.0,0.0
    model.train()
    for i in range(1, args.n_epoch + 1):
        beg = time.time()
        np.random.shuffle(train_set)
        max_acc, max_f1 = 0, 0
        for j in range(n_train_batches):
            model.train()
            if args.model in ['KGNN','ASGCN']:
                optimizer.zero_grad()
                if args.gcn == 1 :
                    train_x, train_xt, train_y, train_pw, train_adj, train_mask, train_adj_know, train_mask_know,train_x_know = get_batch_input(dataset=train_set, bs=args.bs, args=args, idx=j)
                    train_x, train_xt, train_y, train_pw, train_adj,train_mask = torch.from_numpy(train_x), torch.from_numpy(
                        train_xt),torch.from_numpy(train_y).long(), torch.from_numpy(train_pw), torch.from_numpy(train_adj),torch.from_numpy(train_mask)
                    if torch.cuda.is_available():
                        train_x, train_xt, train_y, train_pw, train_adj, train_mask = train_x.cuda(), train_xt.cuda(), \
                                                                                                    train_y.cuda(), train_pw.cuda(), train_adj.cuda(), train_mask.cuda()
                    train_adj_know, train_mask_know,train_x_know=torch.from_numpy(train_adj_know).cuda(),torch.from_numpy(train_mask_know).cuda(),torch.from_numpy(train_x_know).cuda()
                    logit = model(train_x, train_xt, train_pw, train_adj, train_mask,train_adj_know, train_mask_know,train_x_know)

                elif args.gcn==2:
                    train_x, train_xt, train_y, train_pw, train_adj, train_mask, train_graph = get_batch_input(dataset=train_set, bs=args.bs, args=args, idx=j)
                    train_x, train_xt, train_y, train_pw, train_adj,train_mask,train_graph = torch.from_numpy(train_x), torch.from_numpy(
                        train_xt),torch.from_numpy(train_y).long(), torch.from_numpy(train_pw), torch.from_numpy(train_adj),torch.from_numpy(train_mask), torch.from_numpy(train_graph)
                    if torch.cuda.is_available():
                        train_x, train_xt, train_y, train_pw, train_adj, train_mask,train_graph = train_x.cuda(), train_xt.cuda(), \
                                                                                                    train_y.cuda(), train_pw.cuda(), train_adj.cuda(), train_mask.cuda(),train_graph.cuda()
                    logit = model(train_x, train_xt, train_pw, train_adj, train_mask, train_graph)

                else:
                    train_x, train_xt, train_y, train_pw, train_adj, train_mask = get_batch_input(dataset=train_set, bs=args.bs, args=args, idx=j)
                    train_x, train_xt, train_y, train_pw, train_adj,train_mask = torch.from_numpy(train_x), torch.from_numpy(
                        train_xt),torch.from_numpy(train_y).long(), torch.from_numpy(train_pw), torch.from_numpy(train_adj),torch.from_numpy(train_mask)
                    if torch.cuda.is_available():
                        train_x, train_xt, train_y, train_pw, train_adj, train_mask = train_x.cuda(), train_xt.cuda(), \
                                                                                                    train_y.cuda(), train_pw.cuda(), train_adj.cuda(), train_mask.cuda()
                    logit = model(train_x, train_xt, train_pw, train_adj, train_mask)

                loss = F.cross_entropy(logit, train_y)
            elif args.model =='RGAT':
                optimizer.zero_grad()
                train_x, train_xt, train_tag, train_y, train_pw = get_batch_input(dataset=train_set, bs=args.bs, args=args,
                                                                                             idx=j)
                train_x, train_xt,train_tag, train_y, train_pw = torch.from_numpy(train_x), torch.from_numpy(
                    train_xt), torch.from_numpy(train_tag),torch.from_numpy(train_y).long(), torch.from_numpy(train_pw)
                if torch.cuda.is_available():
                    train_x, train_xt,train_tag, train_y, train_pw = train_x.cuda(), train_xt.cuda(),train_tag.cuda(), train_y.cuda(), train_pw.cuda()
                logit = model(train_x, train_xt, train_tag)
                loss = F.cross_entropy(logit, train_y)
            else:
                # optimizer.zero_grad()
                train_x, train_xt, train_y, train_pw = get_batch_input(dataset=train_set, bs=args.bs, args=args, idx=j)
                train_x, train_xt, train_y, train_pw = torch.from_numpy(train_x), torch.from_numpy(
                    train_xt), torch.from_numpy(train_y).long(), torch.from_numpy(train_pw)
                if torch.cuda.is_available():
                    train_x, train_xt, train_y, train_pw = train_x.cuda(), train_xt.cuda(), train_y.cuda(), train_pw.cuda()
                logit = model(train_x, train_xt, train_pw)
                loss = F.cross_entropy(logit, train_y)
            loss.backward()
            optimizer.step()
            corrects = (torch.max(logit, 1)[1].view(train_y.size()).data == train_y.data).sum()
            accuracy = 100.0 * corrects / train_y.shape[0]
            f1 = metrics.f1_score(train_y.cpu(), torch.argmax(logit, -1).cpu(), labels=[0, 1, 2], average='macro')
            # if j % (n_train_batches - 1) == 0:
            if j%10==0:
                eval_acc, eval_f1 = eval(model, args, test_set,n_test_batches)
                if max_acc < eval_acc:
                    max_acc = eval_acc
                if max_f1 < eval_f1:
                    max_f1 = eval_f1
                    
                if save_acc < eval_acc:
                    save_acc = eval_acc
                    model_dict = model.state_dict()
                    torch.save(model_dict,
                               '{}/{}_{}_time{}_bert.pth'.format(args.save_dir, args.model, args.ds_name, times))
                if save_f1 < eval_f1:
                    save_f1 = eval_f1

            if j%50==0:
                logging.info(
                    '\r - loss: {:.6f} f1:{:.4f} acc: {:.4f}%({}/{})'.format(loss.item(), f1, accuracy, corrects,
                                                                             train_y.shape[0]))

        test_acc, test_f1 = max_acc, max_f1
        end = time.time()
        train_time.append(end - beg)
        result_store_test[0].append(test_acc)
        result_store_test[1].append(test_f1)
        logging.info("In Epoch %s: test_accuracy: %.2f, test_macro-f1: %.2f\n" % (i, test_acc * 100, test_f1 * 100))
    avg_time = sum(train_time) / len(train_time)
    best_index_acc = result_store_test[0].index(max(result_store_test[0]))
    logging.info("Runs: %s Best model in Epoch %s: test accuracy: %.2f, macro-f1: %.2f ,avg_time: %.2f\n" % (times,
        best_index_acc + 1, max(result_store_test[0])*100, max(result_store_test[1])*100, avg_time))
    return max(result_store_test[0]), max(result_store_test[1]), avg_time

def train_bert(args,times=0):
    if args.model == 'KGNN':
        dataset, graph_embeddings, n_train, n_test = build_dataset(args=args, is_bert=True)
    else:
        dataset, n_train,n_test = build_dataset(args=args, is_bert=True)

    args.sent_len = len(dataset[0][0]['wids'])
    args.target_len = len(dataset[0][0]['tids'])
    if args.model == 'KGNN':
        args.graph_embeddings=graph_embeddings
    n_train_batches = math.ceil(n_train / args.bs)
    n_test_batches = math.ceil(n_test / args.bs)
    train_set,test_set = dataset
    if args.is_bert == 1:
        bert = BertModel.from_pretrained(r'./bert-base-uncased')
    elif args.is_bert ==2:
        bert = RobertaModel.from_pretrained(r'./roberta-base')

    if args.model =='ASGCN':
        model=ASGCN_BERT(bert=bert,args=args)
        _reset_params(model)
    elif args.model =='KGNN':
        model=KGNN_BERT(bert=bert,args=args)
        _reset_params(model)
    elif args.model =='BERT_vanilla':
        model=BERT_vanilla(bert=bert,args=args)

    if torch.cuda.is_available():
        model = model.cuda()

    train_time = []
    result_store_test = [[], []]

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)  #,weight_decay=0.0001
    model.train()
    save_acc,save_f1=0.0,0.0

    for i in range(1, args.n_epoch + 1):
        beg = time.time()
        np.random.shuffle(train_set)
        max_acc, max_f1 = 0, 0
        for j in range(n_train_batches):
            model.train()
            optimizer.zero_grad()
            ######bert######
            if args.model in ['ASGCN',"KGNN"]:
                train_x, train_xt, train_y, train_pw, train_adj, train_mask = get_batch_input_bert(dataset=train_set, bs=args.bs,
                                                                                            args=args, idx=j)
                train_x, train_xt,train_y, train_pw, train_adj, train_mask = torch.from_numpy(train_x), torch.from_numpy(
                    train_xt), torch.from_numpy(train_y).long(), torch.from_numpy(train_pw), torch.from_numpy(
                    train_adj), torch.from_numpy(train_mask)
                if torch.cuda.is_available():
                    train_x, train_xt, train_y, train_pw, train_adj, train_mask = train_x.cuda(), train_xt.cuda(), \
                                                                                train_y.cuda(), train_pw.cuda(), train_adj.cuda(), train_mask.cuda()
                logit = model(train_x, train_xt, train_pw, train_adj, train_mask)
                loss = F.cross_entropy(logit, train_y)
            elif args.model =='BERT_vanilla':
                train_x, train_y = get_batch_input_bert(dataset=train_set, bs=args.bs,args=args, idx=j)
                train_x, train_y = torch.from_numpy(train_x),torch.from_numpy(train_y).long()
                if torch.cuda.is_available():
                    train_x, train_y = train_x.cuda(), train_y.cuda()
                logit = model(train_x)
                loss = F.cross_entropy(logit, train_y)
            ######bert######
            loss.backward()
            optimizer.step()
            corrects = (torch.max(logit, 1)[1].view(train_y.size()).data == train_y.data).sum()
            accuracy = 100.0 * corrects / train_y.shape[0]
            f1 = metrics.f1_score(train_y.cpu(), torch.argmax(logit, -1).cpu(), labels=[0, 1, 2], average='macro')
            # if j % (n_train_batches - 1) == 0:
            if j % 10 == 0:
                eval_acc, eval_f1 = eval_bert(model, args, test_set, n_test_batches)
                if max_acc < eval_acc:
                    max_acc = eval_acc
                if max_f1 < eval_f1:
                    max_f1 = eval_f1

                if save_acc < eval_acc:
                    save_acc = eval_acc
                    model_dict = model.state_dict()
                    torch.save(model_dict,
                               '{}/{}_{}_time{}_bert.pth'.format(args.save_dir, args.model, args.ds_name, times))
                if save_f1 < eval_f1:
                    save_f1 = eval_f1
            if j % 50 == 0:
                logging.info(
                    '\r - loss: {:.6f} f1:{:.4f} acc: {:.4f}%({}/{})'.format(loss.item(), f1, accuracy, corrects,
                                                                             train_y.shape[0]))
        test_acc, test_f1 = max_acc, max_f1
        end = time.time()
        train_time.append(end - beg)
        result_store_test[0].append(test_acc)
        result_store_test[1].append(test_f1)
        logging.info("In Epoch %s: test_accuracy: %.2f, test_macro-f1: %.2f\n" % (i, test_acc * 100, test_f1 * 100))
    avg_time = sum(train_time) / len(train_time)
    best_index_acc = result_store_test[0].index(max(result_store_test[0]))
    logging.info("Runs: %s Best model in Epoch %s: test accuracy: %.2f, macro-f1: %.2f ,avg_time: %.2f\n" % (times,best_index_acc + 1,max(result_store_test[0]) * 100,
                                                                                                      max(result_store_test[1]) * 100,avg_time))
    return max(result_store_test[0]), max(result_store_test[1]), avg_time

def eval(model, args, test_set, n_test_batches):
    model.eval()
    t_targets_all, t_outputs_all = None, None
    with torch.no_grad():
        corrects, f1, avg_loss, size = 0, 0, 0, 0
        loss = None
        for j in range(n_test_batches):
            if args.model in ['KGNN','ASGCN']:
                if args.gcn ==1 :
                    test_words,test_twords, test_x, test_xt, test_y, test_pw, test_adj, test_mask, train_adj_know, train_mask_know,train_x_know = get_batch_input_inference(dataset=test_set, bs=args.bs, args=args,idx=j)
                    test_x, test_xt, test_y, test_pw, test_adj,test_mask = torch.from_numpy(test_x), torch.from_numpy(
                        test_xt),torch.from_numpy(test_y).long(), torch.from_numpy(test_pw), torch.from_numpy(test_adj),torch.from_numpy(test_mask)
                    if torch.cuda.is_available():
                        test_x, test_xt, test_y, test_pw, test_adj, test_mask= test_x.cuda(), test_xt.cuda(), test_y.cuda(), test_pw.cuda(), \
                                                                                            test_adj.cuda(), test_mask.cuda()
                    train_adj_know, train_mask_know,train_x_know=torch.from_numpy(train_adj_know).cuda(),torch.from_numpy(train_mask_know).cuda(),torch.from_numpy(train_x_know).cuda()
                    logit = model(test_x, test_xt, test_pw, test_adj, test_mask,train_adj_know, train_mask_know,train_x_know)
                elif args.gcn==2:
                    test_words,test_twords, test_x, test_xt, test_y, test_pw, test_adj, test_mask, test_graph = get_batch_input_inference(dataset=test_set, bs=args.bs, args=args,idx=j)
                    test_x, test_xt, test_y, test_pw, test_adj,test_mask,test_graph = torch.from_numpy(test_x), torch.from_numpy(
                        test_xt),torch.from_numpy(test_y).long(), torch.from_numpy(test_pw), torch.from_numpy(test_adj),torch.from_numpy(test_mask), torch.from_numpy(test_graph)
                    if torch.cuda.is_available():
                        test_x, test_xt, test_y, test_pw, test_adj, test_mask, test_graph= test_x.cuda(), test_xt.cuda(), test_y.cuda(), test_pw.cuda(), \
                                                                                            test_adj.cuda(), test_mask.cuda(),test_graph.cuda()
                    logit = model(test_x, test_xt, test_pw, test_adj, test_mask, graph_embedding=test_graph)
                else:
                    test_words,test_twords, test_x, test_xt, test_y, test_pw, test_adj, test_mask = get_batch_input_inference(dataset=test_set, bs=args.bs, args=args,idx=j)
                    test_x, test_xt, test_y, test_pw, test_adj,test_mask = torch.from_numpy(test_x), torch.from_numpy(
                        test_xt),torch.from_numpy(test_y).long(), torch.from_numpy(test_pw), torch.from_numpy(test_adj),torch.from_numpy(test_mask)
                    if torch.cuda.is_available():
                        test_x, test_xt, test_y, test_pw, test_adj, test_mask= test_x.cuda(), test_xt.cuda(), test_y.cuda(), test_pw.cuda(), \
                                                                                            test_adj.cuda(), test_mask.cuda()
                    logit = model(test_x, test_xt, test_pw, test_adj, test_mask)
                loss = F.cross_entropy(logit, test_y, reduction='sum')

            elif args.model =='RGAT':
                test_x, test_xt, test_tag, test_y, test_pw = get_batch_input(dataset=test_set, bs=args.bs, args=args,
                                                                                             idx=j)
                test_x, test_xt, test_tag, test_y, test_pw = torch.from_numpy(test_x), torch.from_numpy(test_xt),torch.from_numpy(test_tag),\
                                                   torch.from_numpy(test_y).long(), torch.from_numpy(test_pw)
                if torch.cuda.is_available():
                    test_x, test_xt, test_tag, test_y, test_pw = test_x.cuda(), test_xt.cuda(), test_tag.cuda(), test_y.cuda(), test_pw.cuda()
                logit = model(test_x, test_xt, test_tag)
                loss = F.cross_entropy(logit, test_y, reduction='sum')
            else:
                test_x, test_xt, test_y, test_pw = get_batch_input(dataset=test_set, bs=args.bs, args=args,
                                                                                             idx=j)
                test_x, test_xt, test_y, test_pw = torch.from_numpy(test_x), torch.from_numpy(test_xt), torch.from_numpy(
                    test_y).long(), torch.from_numpy(test_pw)
                if torch.cuda.is_available():
                    test_x, test_xt, test_y, test_pw = test_x.cuda(), test_xt.cuda(), test_y.cuda(), test_pw.cuda()
                logit = model(test_x, test_xt, test_pw)
                loss = F.cross_entropy(logit, test_y, reduction='sum')
            avg_loss += loss.item()
            size += test_y.size(0)
            corrects += (torch.max(logit, 1)
                         [1].view(test_y.size()).data == test_y.data).sum()
            if t_targets_all is None:
                t_targets_all = test_y
                t_outputs_all = logit
            else:
                t_targets_all = torch.cat((t_targets_all, test_y), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, logit), dim=0)

        avg_loss = loss.item() / size
        accuracy = 1.0 * corrects / size
        F1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
    return accuracy, F1


def eval_bert(model, args, test_set, n_test_batches):
    model.eval()
    t_targets_all, t_outputs_all = None, None
    with torch.no_grad():
        corrects, f1, avg_loss = 0, 0, 0
        loss = None
        for j in range(n_test_batches):
            ######bert######
            if args.model in ['ASGCN',"KGNN"]:
                test_x, test_xt, test_y, test_pw, test_adj, test_mask = get_batch_input_bert(
                    dataset=test_set, bs=args.bs, args=args, idx=j)
                test_x, test_xt, test_y, test_pw, test_adj, test_mask = torch.from_numpy(test_x), torch.from_numpy(
                    test_xt), torch.from_numpy(test_y).long(), torch.from_numpy(test_pw), torch.from_numpy(
                    test_adj), torch.from_numpy(test_mask)
                if torch.cuda.is_available():
                    test_x, test_xt, test_y, test_pw, test_adj, test_mask = test_x.cuda(), test_xt.cuda(), test_y.cuda(), test_pw.cuda(), \
                                                                            test_adj.cuda(), test_mask.cuda()
                logit= model(test_x, test_xt, test_pw, test_adj, test_mask)
                loss = F.cross_entropy(logit, test_y, reduction='sum')

            elif args.model =='BERT_vanilla':
                test_x, test_y = get_batch_input_bert(dataset=test_set, bs=args.bs,args=args, idx=j)
                test_x, test_y = torch.from_numpy(test_x),torch.from_numpy(test_y).long()
                if torch.cuda.is_available():
                    test_x, test_y = test_x.cuda(), test_y.cuda()
                logit = model(test_x)
                loss = F.cross_entropy(logit, test_y)

            avg_loss += loss.item()
            corrects += (torch.max(logit, 1)
                         [1].view(test_y.size()).data == test_y.data).sum()
            if t_targets_all is None:
                t_targets_all = test_y
                t_outputs_all = logit
            else:
                t_targets_all = torch.cat((t_targets_all, test_y), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, logit), dim=0)
        size = len(test_set)
        accuracy = 1.0 * corrects / size
        F1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
    return accuracy, F1


def inference(args, model_path=None):
    if args.is_bert == 1:
        dataset, n_train,n_test = build_dataset(ds_name=args.ds_name, bs=args.bs, dim_w=args.dim_w,
                                                        is_bert=True)

        args.sent_len = len(dataset[0][0]['wids'])
        args.target_len = len(dataset[0][0]['tids'])
        n_test_batches = math.ceil(n_test / args.bs)
        train_set, test_set = dataset
        bert = BertModel.from_pretrained(r'/project/jhliu4/ZQH/Project_zqh/MAMS-for-ABSA-master/bert-base-uncased')
        model = GCAE_Bert(bert=bert).cuda()
        model.load_state_dict(torch.load(model_path))
        test_acc, test_f1 = eval_bert(model, args, test_set, n_test_batches, is_bert=True)
    else:
        if args.model == 'KGNN':
            dataset, embeddings,graph_embeddings, n_train, n_test = build_dataset(args=args)
        elif args.model =='RGAT':
            dataset, embeddings, n_train, n_test, dep_vocab = build_dataset(args=args)
        else:
            dataset, embeddings, n_train, n_test = build_dataset(args=args)
        args.embeddings = embeddings
        if args.model =='KGNN':
            args.graph_embeddings=graph_embeddings
        n_test_batches = math.ceil(n_test / args.bs)
        train_set,test_set = dataset
        if args.model == 'KGNN':
            model = KGNN(args)
            _reset_params(model)
        elif args.model == 'GCAE':
            model = GCAE(args=args)
        elif args.model == 'ATAE':
            model = ATAE_LSTM(args=args)
        elif args.model == 'ASGCN':
            model = ASGCN(args)
        elif args.model =='IAN':
            model =IAN(args)
        elif args.model =='TNet':
            model = TNet_LF(args)
        elif args.model == 'RGAT':
            model = RGAT(args, dep_tag_num=len(dep_vocab))
        else:
            print('model error')
        if torch.cuda.is_available():
            model = model.cuda()
        save_dict = torch.load(model_path, map_location='cuda')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in save_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # model.load_state_dict(torch.load(model_path,map_location='cuda'))
        start_time=time.time()
        test_acc, test_f1 = eval(model, args, test_set,n_test_batches)
        end_time=time.time()
        print('test_time: %.4f'%(end_time-start_time))
    return test_acc, test_f1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='KGNN settings')
    parser.add_argument("-ds_name", type=str, default="14semeval_rest", help="dataset name")    ## '14semeval_laptop', '14semeval_rest','15semeval_rest','16semeval_rest','Twitter'
    parser.add_argument("-bs", type=int, default=32, help="batch size, 64 for rest, 32 for laptop")
    parser.add_argument("-dropout_rate", type=float, default=0.5, help="dropout rate for sentimental features")
    parser.add_argument("-learning_rate", type=float, default=0.00003, help="learning rate for sentimental features")
    parser.add_argument("-n_epoch", type=int, default=20, help="number of training epoch")
    parser.add_argument('-model', type=str, default="KGNN", help="model name")
    parser.add_argument("-dim_w", type=int, default=768, help="dimension of word embeddings")
    parser.add_argument("-dim_k", type=int, default=200,
                        help="dimension of knowledge graph embeddings, 400 for laptop, 200 for rest")
    parser.add_argument("-is_test", type=int, default=0, help="test the model: 1 for test")
    parser.add_argument("-is_bert", type=int, default=2, help="glove-based model: 1 for bert")
    parser.add_argument("-save_dir", type=str, default="model_weight/temp", help="save dir")
    parser.add_argument("-gcn", type=int, default=0, help="whether use other knowledge graph methods, gcn=1 for using senticnet5, gcn=2 for using AR-BERT")
    parser.add_argument("-kge", type=str, default="distmult", help="kge approach")

    args = parser.parse_args()

    acc, f1 = [], []
    train_time = []

    if args.is_test == 1:
        test_path="./model_weight/temp/KGNN_14semeval_laptop_78.91_75.21.pth"
        test_acc, test_f1 = inference(args, test_path)
        print("Test : acc: {} f1: {}".format(test_acc, test_f1))
    else:
        for i in range(5):
            if args.is_bert != 0:
                a_acc, a_f1, a_time = train_bert(args, times=i)
            else:
                a_acc, a_f1, a_time = train(args, times=i)
            acc.append(a_acc)
            f1.append(a_f1)
            train_time.append(a_time)

        best_acc = max(acc)
        best_f1 = max(f1)
        avg_acc = sum(acc) / len(acc)
        avg_f1 = sum(f1) / len(f1)
        best_time = min(train_time)
        avg_time = sum(train_time) / len(train_time)
        print('The results of {} : '.format(args.ds_name), '\n',
              'best_acc: {}  best_f1: {} min_time: {}'.format(best_acc, best_f1, best_time), '\n',
              'avg_acc: {}  avg_f1: {} avg_time: {}'.format(avg_acc, avg_f1, avg_time))
