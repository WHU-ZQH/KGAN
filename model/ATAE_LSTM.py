# -*- coding: utf-8 -*-
from layers.attention import Attention, NoQueryAttention
from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
from layers.squeeze_embedding import SqueezeEmbedding

class ATAE_LSTM(nn.Module):
    def __init__(self, args):
        super(ATAE_LSTM, self).__init__()
        self.args = args
        self.embed = nn.Embedding.from_pretrained(torch.from_numpy(args.embeddings).float().cuda(), freeze=True)
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(300*2, 300, num_layers=1, batch_first=True)
        self.attention = NoQueryAttention(300+300, score_function='bi_linear')
        self.fc3 = nn.Linear(300, 3)


    def forward(self, context,aspect,offset):
        text_raw_indices,aspect_indices,offset= context.long(),aspect.long(),offset
        x_len = torch.sum(text_raw_indices != 0, dim=-1).cpu()
        x_len_max = torch.max(x_len)
        if torch.cuda.is_available():
            aspect_len = torch.tensor(torch.sum(aspect_indices != 0, dim=-1), dtype=torch.float).cuda()
        else:
            aspect_len = torch.tensor(torch.sum(aspect_indices != 0, dim=-1), dtype=torch.float)
        x = self.embed(text_raw_indices)
        x = self.squeeze_embedding(x, x_len)
        aspect = self.embed(aspect_indices)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.view(aspect_len.size(0), 1))
        aspect = torch.unsqueeze(aspect_pool, dim=1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)

        h, (_, _) = self.lstm(x, x_len)
        ha = torch.cat((h, aspect), dim=-1)
        _, score = self.attention(ha)

        output = torch.squeeze(torch.bmm(score, h), dim=1)

        out = self.fc3(output)
        return out

class ATAE_LSTM_Bert(nn.Module):
    def __init__(self, bert):
        super(ATAE_LSTM_Bert, self).__init__()
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(768*2, 300, num_layers=1, batch_first=True)
        self.fc_aspect=nn.Linear(768,300)
        self.attention = NoQueryAttention(300+300, score_function='bi_linear')
        self.fc = nn.Linear(300, 3)
        self.drop_bert=nn.Dropout(0.2)

    def forward(self, bert_token, bert_token_aspect,offset):
        # BERT encoding
        text_raw_indices,aspect_indices= bert_token.long(), bert_token_aspect.long()
        x_len = torch.sum(text_raw_indices != 0, dim=-1).cpu()
        x_len_max = torch.max(x_len).item()
        if torch.cuda.is_available():
            aspect_len = torch.tensor(torch.sum(aspect_indices != 0, dim=-1), dtype=torch.float).cuda()
        else:
            aspect_len = torch.tensor(torch.sum(aspect_indices != 0, dim=-1), dtype=torch.float)
        x , _ = self.bert(text_raw_indices,output_all_encoded_layers=False)
        x=self.drop_bert(x)
        x = self.squeeze_embedding(x, x_len)
        aspect, _ = self.bert(aspect_indices,output_all_encoded_layers=False)
        aspect=self.drop_bert(aspect)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.view(aspect_len.size(0), 1))
        aspect = torch.unsqueeze(aspect_pool, dim=1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)

        h, (_, _) = self.lstm(x, x_len)
        aspect=self.fc_aspect(aspect)
        ha = torch.cat((h, aspect), dim=-1)
        _, score = self.attention(ha)
        output = torch.squeeze(torch.bmm(score, h), dim=1)

        out = self.fc(output)
        return out
