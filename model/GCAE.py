import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.squeeze_embedding import SqueezeEmbedding

class GCAE(nn.Module):
    def __init__(self, args):
        super(GCAE, self).__init__()
        self.args = args
        C = 3
        D=300
        Co=100
        Ks=[3,4,5]
        Ka=[2,3]
        self.embed=nn.Embedding.from_pretrained(torch.from_numpy(args.embeddings).float())
        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K,padding=K-2) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K,padding=K-2) for K in Ks])
        self.convs3 = nn.ModuleList([nn.Conv1d(D, Co, K, padding=K-2) for K in [3]])

        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(len(Ks)*Co, C)
        self.fc_aspect = nn.Linear(100, Co)

    def forward(self, feature, aspect,offset=None):
        text, aspect= feature.long(), aspect.long()
        feature = self.embed(text)  # (N, L, D)
        aspect_v = self.embed(aspect)  # (N, L', D)
        aa = [F.relu(conv(aspect_v.transpose(1, 2))) for conv in self.convs3]  # [(N,Co,L), ...]*len(Ks)
        aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
        aspect_v = torch.cat(aa, 1)
        x = [torch.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        x = [i*j for i, j in zip(x, y)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N,len(Ks)*Co)
        logit = self.fc3(x)  # (N,C)
        return logit

class GCAE_Bert(nn.Module):
    def __init__(self, bert):
        super(GCAE_Bert, self).__init__()
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        C = 3
        D=768
        Co=100
        Ks=[3,4,5]
        Ka=[2,3]
        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs3 = nn.ModuleList([nn.Conv1d(D, Co, K, padding=K-2) for K in [3]])

        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(len(Ks)*Co, C)
        self.fc_aspect = nn.Linear(100, Co)
        self.drop_bert = nn.Dropout(0.2)
        
    def forward(self, bert_token, bert_token_aspect,offset):
        # BERT encoding
        bert_token, bert_token_aspect=bert_token.long(), bert_token_aspect.long()
        context_len = torch.sum(bert_token != 0, dim=-1)
        target_len = torch.sum(bert_token_aspect != 0, dim=-1)
        context = self.squeeze_embedding(bert_token, context_len)
        #context,target=bert_token, bert_token_aspect
        text_embed, _ = self.bert(context,output_all_encoded_layers=False)
        feature = self.drop_bert(text_embed)
        
        target = self.squeeze_embedding(bert_token_aspect, target_len)
        aspect_embed, _ = self.bert(target,output_all_encoded_layers=False)
        aspect_v= self.drop_bert(aspect_embed)
        
        aa = [F.relu(conv(aspect_v.transpose(1, 2))) for conv in self.convs3]  # [(N,Co,L), ...]*len(Ks)
        aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
        aspect_v = torch.cat(aa, 1)
        x = [torch.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        x = [i*j for i, j in zip(x, y)]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N,len(Ks)*Co)
        logit = self.fc2(x)  # (N,C)
        return logit
