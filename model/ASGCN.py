import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class ASGCN(nn.Module):
    def __init__(self,  args):
        super(ASGCN, self).__init__()
        self.args = args
        if torch.cuda.is_available():
            self.embed = nn.Embedding.from_pretrained(torch.from_numpy(args.embeddings).float().cuda(), freeze=True)
        else:
            self.embed = nn.Embedding.from_pretrained(torch.from_numpy(args.embeddings).float(), freeze=True)
        self.text_lstm = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2*300, 2*300)
        self.gc2 = GraphConvolution(2*300, 2*300)
        self.fc2 = nn.Linear(2*300, 3)
        self.text_embed_dropout = nn.Dropout(0.3)

    def location_feature(self, feature, offset):
        if torch.cuda.is_available():
            weight = torch.FloatTensor(feature.shape[0], feature.shape[1]).zero_().cuda()
        else:
            weight = torch.FloatTensor(feature.shape[0], feature.shape[1]).zero_()
        for i in range(offset.shape[0]):
            weight[i] = offset[i][:feature.shape[1]]
        feature = weight.unsqueeze(2) * feature
        return feature

    def forward(self, feature, aspect, offset, adj, mask):
        feature, aspect, offset, adj, mask = feature.long(), aspect.long(), offset, adj.float(), mask.long()
        text_len = torch.sum(feature != 0, dim=-1).cpu()
        text = self.embed(feature)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        seq_len = text_out.shape[1]
        adj = adj[:, :seq_len, :seq_len]
        x = F.relu(self.gc1(self.location_feature(text_out, offset), adj))
        x = F.relu(self.gc2(self.location_feature(x, offset), adj))
        # x = F.relu(self.gc1(text_out, adj))
        # x = F.relu(self.gc2(x, adj))
        x = self.location_feature(x, mask)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc2(x)
        return output

class ASGCN_BERT(nn.Module):
    def __init__(self, bert, args):
        super(ASGCN_BERT, self).__init__()
        self.args = args
        self.bert=bert
        h_num=300
        self.text_lstm = DynamicLSTM(768, h_num, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2*h_num, 2*h_num)
        self.gc2 = GraphConvolution(2*h_num, 2*h_num)
        self.fc2 = nn.Linear(2*h_num, 3)
        self.text_embed_dropout = nn.Dropout(args.dropout_rate)

    def location_feature(self, feature, offset):
        if torch.cuda.is_available():
            weight = torch.FloatTensor(feature.shape[0], feature.shape[1]).zero_().cuda()
        else:
            weight = torch.FloatTensor(feature.shape[0], feature.shape[1]).zero_()
        for i in range(offset.shape[0]):
            weight[i] = offset[i][:feature.shape[1]]
        feature = weight.unsqueeze(2) * feature
        return feature

    def forward(self, feature, aspect, offset, adj, mask):
        feature, aspect, offset, adj, mask = feature.long(), aspect.long(), offset, adj.float(), mask.long()
        # a=np.array(mask.tolist())
        text_len = torch.sum(feature != 0, dim=-1).cpu()
        text,_ = self.bert(feature,output_all_encoded_layers=False)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        text_out = self.text_embed_dropout(text_out)
        seq_len = text_out.shape[1]
        adj = adj[:, :seq_len, :seq_len]
        x = F.relu(self.gc1(self.location_feature(text_out, offset), adj))
        x = F.relu(self.gc2(self.location_feature(x, offset), adj))
        # x = F.relu(self.gc1(text_out, adj))
        # x = F.relu(self.gc2(x, adj))
        x = self.location_feature(x, mask)
        # b=np.array(x.tolist())
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        x=self.text_embed_dropout(x)
        output = self.fc2(x)
        return output
