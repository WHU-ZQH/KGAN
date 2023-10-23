import torch.nn as nn
import torch.nn.functional as F

from model.rgat_file.model_utils import LinearAttention, DotprodAttention, RelationAttention, Highway, mask_logits
from model.rgat_file.tree import *


class RGAT(nn.Module):
    """
    Full model in reshaped tree
    """
    def __init__(self, args, dep_tag_num):
        super(RGAT, self).__init__()
        self.args = args
        if torch.cuda.is_available():
            self.embed = nn.Embedding.from_pretrained(torch.from_numpy(args.embeddings).float().cuda(), freeze=True)
        else:
            self.embed = nn.Embedding.from_pretrained(torch.from_numpy(args.embeddings).float(), freeze=True)

        self.dropout = nn.Dropout(0.7)
        self.tanh = nn.Tanh()
        embedding_dim=300
        num_layer=1
        hidden_sizes=300
        num_heads=7
        final_hidden_size=300
        num_classes=3
        highway=True
        self.gat_attention_type='dotprod'
        num_mlps=2

        if highway:
            self.highway_dep = Highway(num_layer, embedding_dim)
            self.highway = Highway(num_layer, embedding_dim)

        self.text_bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_sizes,
                              bidirectional=True, batch_first=True, num_layers=num_layer)

        self.aspect_bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_sizes,
                              bidirectional=True, batch_first=True, num_layers=num_layer)

        gcn_input_dim = hidden_sizes * 2

        self.gat_dep = [RelationAttention(in_dim = embedding_dim).cuda() for i in range(num_heads)]
        if self.gat_attention_type == 'linear':
            self.gat = [LinearAttention(in_dim = gcn_input_dim, mem_dim = gcn_input_dim).cuda() for i in range(num_heads)] # we prefer to keep the dimension unchanged
        elif self.gat_attention_type == 'dotprod':
            self.gat = [DotprodAttention().cuda() for i in range(num_heads)]
        else:
            # reshaped gcn
            self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)

        self.dep_embed = nn.Embedding(dep_tag_num, embedding_dim)

        last_hidden_size = hidden_sizes * 4

        layers = [
            nn.Linear(last_hidden_size, final_hidden_size), nn.ReLU()]
        for _ in range(num_mlps-1):
            layers += [nn.Linear(final_hidden_size,
                                 final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(final_hidden_size, num_classes)

    def forward(self, sentence, aspect, dep_tags):
        sentence, aspect,dep_tags=sentence.long(),aspect.long(),dep_tags.long()
        fmask = (torch.zeros_like(sentence) != sentence).float()  # (N��L)
        dmask = (torch.zeros_like(dep_tags) != dep_tags).float()  # (N ,L)

        feature = self.embed(sentence)  # (N, L, D)
        aspect_feature = self.embed(aspect) # (N, L', D)
        feature = self.dropout(feature)
        aspect_feature = self.dropout(aspect_feature)

        feature = self.highway(feature)
        aspect_feature = self.highway(aspect_feature)

        feature, _ = self.text_bilstm(feature) # (N,L,D)
        aspect_feature, _ = self.aspect_bilstm(aspect_feature) #(N,L,D)

        aspect_feature = aspect_feature.mean(dim = 1) # (N, D)

        ############################################################################################
        # do gat thing
        dep_feature = self.dep_embed(dep_tags)
        dep_feature = self.highway_dep(dep_feature)

        dep_out = [g(feature, dep_feature, fmask).unsqueeze(1) for g in self.gat_dep]  # (N, 1, D) * num_heads
        dep_out = torch.cat(dep_out, dim = 1) # (N, H, D)
        dep_out = dep_out.mean(dim = 1) # (N, D)

        if self.gat_attention_type == 'gcn':
            gat_out = self.gat(feature) # (N, L, D)
            fmask = fmask.unsqueeze(2)
            gat_out = gat_out * fmask
            gat_out = F.relu(torch.sum(gat_out, dim = 1)) # (N, D)

        else:
            gat_out = [g(feature, aspect_feature, fmask).unsqueeze(1) for g in self.gat]
            gat_out = torch.cat(gat_out, dim=1)
            gat_out = gat_out.mean(dim=1)

        feature_out = torch.cat([dep_out,  gat_out], dim = 1) # (N, D')
        # feature_out = gat_out
        #############################################################################################
        x = self.dropout(feature_out)
        x = self.fcs(x)
        logit = self.fc_final(x)
        return logit

class GAT(nn.Module):
    """
    reshape tree in GAT only
    """
    def __init__(self, args, dep_tag_num):
        super(GAT, self).__init__()
        self.args = args
        embedding_dim=300
        num_layer=2
        hidden_sizes=300
        num_heads=7
        final_hidden_size=300
        num_classes=3
        dropout=0.8
        highway=True
        self.gat_attention_type='dotprod'
        num_mlps=2
        num_embeddings, embed_dim = args.embeddings.shape
        if torch.cuda.is_available():
            self.embed = nn.Embedding.from_pretrained(torch.from_numpy(args.embeddings).float().cuda(), freeze=False)
        else:
            self.embed = nn.Embedding.from_pretrained(torch.from_numpy(args.embeddings).float(), freeze=False)

        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

        self.highway = Highway(num_layer, embedding_dim)

        self.bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_sizes,
                                  bidirectional=True, batch_first=True, num_layers=num_layer)
        gcn_input_dim = hidden_sizes * 2

        # if args.gat:
        if self.gat_attention_type == 'linear':
            self.gat = [LinearAttention(in_dim = gcn_input_dim, mem_dim = gcn_input_dim).cuda() for i in range(num_heads)] # we prefer to keep the dimension unchanged
        elif self.gat_attention_type == 'dotprod':
            self.gat = [DotprodAttention().cuda() for i in range(num_heads)]
        else:
            # reshaped gcn
            self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)


        last_hidden_size = hidden_sizes * 2

        layers = [
            nn.Linear(last_hidden_size, final_hidden_size), nn.ReLU()]
        for _ in range(num_mlps-1):
            layers += [nn.Linear(final_hidden_size,
                                 final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(final_hidden_size, num_classes)

    def forward(self, sentence, aspect, dep_tags):
        fmask = (torch.zeros_like(sentence) != sentence).float()
        dmask = (torch.zeros_like(dep_tags) != dep_tags).float()

        feature = self.embed(sentence)  # (N, L, D)
        aspect_feature = self.embed(aspect) # (N, L', D)
        feature = self.dropout(feature)
        aspect_feature = self.dropout(aspect_feature)


        feature = self.highway(feature)
        aspect_feature = self.highway(aspect_feature)

        feature, _ = self.bilstm(feature) # (N,L,D)
        aspect_feature, _ = self.bilstm(aspect_feature) #(N,L,D)

        aspect_feature = aspect_feature.mean(dim = 1) # (N, D)

        ############################################################################################

        if self.gat_attention_type == 'gcn':
            gat_out = self.gat(feature) # (N, L, D)
            fmask = fmask.unsqueeze(2)
            gat_out = gat_out * fmask
            gat_out = F.relu(torch.sum(gat_out, dim = 1)) # (N, D)

        else:
            gat_out = [g(feature, aspect_feature, fmask).unsqueeze(1) for g in self.gat]
            gat_out = torch.cat(gat_out, dim=1)
            gat_out = gat_out.mean(dim=1)

        feature_out = gat_out # (N, D')
        # feature_out = gat_out
        #############################################################################################
        x = self.dropout(feature_out)
        x = self.fcs(x)
        logit = self.fc_final(x)
        return logit

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0
