import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
from layers.squeeze_embedding import SqueezeEmbedding
from torch.nn.utils import weight_norm
import torch
from torch import nn


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

class KGNN(nn.Module):
    def __init__(self,  args):
        super(KGNN, self).__init__()
        self.args = args
        self.hid_num=300
        if torch.cuda.is_available():
            self.embed = nn.Embedding.from_pretrained(torch.from_numpy(args.embeddings).float().cuda(), freeze=True)
            self.graph_embed = nn.Embedding.from_pretrained(torch.from_numpy(args.graph_embeddings).float().cuda(), freeze=True)
        else:
            self.embed = nn.Embedding.from_pretrained(torch.from_numpy(args.embeddings).float(), freeze=True)
            self.graph_embed = nn.Embedding.from_pretrained(torch.from_numpy(args.graph_embeddings).float(), freeze=True)
        self.text_lstm = DynamicLSTM(args.dim_w, self.hid_num, num_layers=1, batch_first=True, bidirectional=True)
        self.aspect_lstm = DynamicLSTM(args.dim_w, self.hid_num, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2*self.hid_num, 2*self.hid_num)
        self.gc2 = GraphConvolution(2*self.hid_num, 2*self.hid_num)
        self.fc = nn.Linear(4*self.hid_num+args.dim_k, 3)
        self.fc2 = nn.Linear(4*self.hid_num+args.dim_k, 3)
        self.fc3 = nn.Linear(4*self.hid_num, 3)
        self.text_embed_dropout = nn.Dropout(args.dropout_rate)
        self.squeezeEmbedding=SqueezeEmbedding()
        if self.args.ds_name in ['14semeval_rest','Twitter']:
            self.conv=nn.Conv1d(3,3,kernel_size=3)
        else:
            self.conv=nn.Conv1d(3,3,kernel_size=4)
            self.fc_out = nn.Linear(6*self.hid_num+args.dim_k, 3)

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
        aspect_len = torch.sum(aspect != 0, dim=-1).cpu()
        text = self.embed(feature)
        text = self.location_feature(text, offset)
        text = self.text_embed_dropout(text)
        aspect_embed =self.embed(aspect)
        aspect_embed = self.text_embed_dropout(aspect_embed)
        text_knowledge = self.graph_embed(feature)
        aspect_knowledge =self.graph_embed(aspect)
        text_knowledge=self.squeezeEmbedding(text_knowledge,text_len)
        aspect_knowledge=self.squeezeEmbedding(aspect_knowledge,aspect_len)
        text_out, (_, _) = self.text_lstm(text, text_len)
        aspect_out, (_, _) = self.aspect_lstm(aspect_embed, aspect_len)

        #####    syntactic_level   ######
        seq_len = text_out.shape[1]
        adj = adj[:, :seq_len, :seq_len]
        x = F.relu(self.gc1(text_out, adj))
        x = F.relu(self.gc2(x, adj))
        x = self.location_feature(x, mask)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1)
        if self.args.ds_name in ['14semeval_rest','Twitter']:
            x=x
        else:
            x=F.relu(x)
        #####    syntactic_level   ######

        #####    context_level   ######
        self_socre=torch.bmm(text_out,text_out.transpose(1,2))
        self_socre=F.softmax(self_socre,dim=1)
        text_att=torch.bmm(self_socre,text_out)
        score=torch.bmm(aspect_out,text_out.transpose(1,2))
        score=F.softmax(score, dim=1)
        y=torch.bmm(score,text_att).squeeze(1)
        if self.args.ds_name in ['14semeval_rest','Twitter']:
            y=F.max_pool1d(y.transpose(1,2),y.shape[1]).squeeze(2)
        else:
            y=F.relu(F.max_pool1d(y.transpose(1,2),y.shape[1]).squeeze(2))
        #####    context_level   ######

        #####    knowledge_level  ######
        text_knowledge=torch.cat((text_knowledge,text_out),dim=-1)
        aspect_knowledge=torch.cat((aspect_knowledge,aspect_out),dim=-1)
        knowledge_score=torch.bmm(aspect_knowledge,text_knowledge.transpose(1,2))
        knowledge_score=F.softmax(knowledge_score, dim=1)
        knowledge_out=torch.bmm(knowledge_score,text_knowledge)
        if self.args.ds_name in ['14semeval_rest','Twitter']:
            z=F.max_pool1d(knowledge_out.transpose(1,2),knowledge_out.shape[1]).squeeze(2)
        else:
            z=F.relu(F.max_pool1d(knowledge_out.transpose(1,2),knowledge_out.shape[1]).squeeze(2))
        ####    knowledge_level  ######

        ######## feature fuse  #########


        out_xz=torch.cat((x,z),dim=-1)
        out_yz=torch.cat((y,z),dim=-1)
        out_xy=torch.cat((x,y),dim=-1)
        output_xz=self.fc(out_xz)
        output_yz=self.fc2(out_yz)
        output_xy=self.fc3(out_xy)
        if self.args.ds_name in ['14semeval_rest','Twitter']:
            output=torch.cat((output_yz.unsqueeze(2),output_xz.unsqueeze(2),output_xy.unsqueeze(2)),dim=-1)
        else:
            out_xyz=torch.cat((x,y,z),dim=-1)
            output_xyz=self.fc_out(out_xyz)
            output=torch.cat((output_xz.unsqueeze(2),output_yz.unsqueeze(2),output_xy.unsqueeze(2),output_xyz.unsqueeze(2)),dim=-1)
        output=self.conv(output).squeeze(2)
        # return output,output_yz,output_xz,output_xy
        return output, alpha.squeeze(1), F.max_pool1d(score.transpose(1,2),score.shape[1]).squeeze(2), \
               F.max_pool1d(knowledge_score.transpose(1,2),knowledge_score.shape[1]).squeeze(2)

        # return output,output_xy,output_xz,output_yz
        # out=torch.cat((x,y,z),dim=-1)
        # output=self.fc_out(out)
        # return output,x,y,z

class KGNN2(nn.Module):
    def __init__(self,  args):
        super(KGNN2, self).__init__()
        self.args = args
        self.hid_num=300
        if torch.cuda.is_available():
            self.embed = nn.Embedding.from_pretrained(torch.from_numpy(args.embeddings).float().cuda(), freeze=True)
            self.graph_embed = nn.Embedding.from_pretrained(torch.from_numpy(args.graph_embeddings).float().cuda(), freeze=True)
        else:
            self.embed = nn.Embedding.from_pretrained(torch.from_numpy(args.embeddings).float(), freeze=True)
            self.graph_embed = nn.Embedding.from_pretrained(torch.from_numpy(args.graph_embeddings).float(), freeze=True)
        self.text_lstm = DynamicLSTM(args.dim_w, self.hid_num, num_layers=1, batch_first=True, bidirectional=True)
        self.aspect_lstm = DynamicLSTM(args.dim_w, self.hid_num, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2*self.hid_num, 2*self.hid_num)
        self.gc2 = GraphConvolution(2*self.hid_num, 2*self.hid_num)
        if args.ds_name == '14semeval_rest':
            self.fc = nn.Linear(4*self.hid_num+args.dim_k, 3)
        else:
            self.fc = nn.Linear(5*self.hid_num+args.dim_k, 3)
        self.text_embed_dropout = nn.Dropout(0.5)

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
        aspect_len = torch.sum(aspect != 0, dim=-1).cpu()
        text = self.embed(feature)
        text = self.location_feature(text, offset)
        text = self.text_embed_dropout(text)
        aspect_embed =self.embed(aspect)
        aspect_embed = self.text_embed_dropout(aspect_embed)
        text_knowledge = self.graph_embed(feature)
        aspect_knowledge =self.graph_embed(aspect)
        # text_knowledge=self.squeezeEmbedding(text_knowledge,text_len)
        # aspect_knowledge=self.squeezeEmbedding(aspect_knowledge,aspect_len)
        text_out, (_, _) = self.text_lstm(text, text_len)
        aspect_out, (_, _) = self.aspect_lstm(aspect_embed, aspect_len)

        #####    syntactic_level   ######
        seq_len = text_out.shape[1]
        adj = adj[:, :seq_len, :seq_len]
        x = F.relu(self.gc1(text_out, adj))
        x = F.relu(self.gc2(x, adj))
        x = self.location_feature(x, mask)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1)
        if self.args.ds_name == '14semeval_rest':
            x=x
        else:
            x=F.relu(x)
        #####    syntactic_level   ######

        #####    context_level   ######
        self_socre=torch.bmm(text_out,text_out.transpose(1,2))
        self_socre=F.softmax(self_socre,dim=1)
        text_att=torch.bmm(self_socre,text_out)
        score=torch.bmm(aspect_out,text_out.transpose(1,2))
        score=F.softmax(score, dim=1)
        y=torch.bmm(score,text_att).squeeze(1)
        if self.args.ds_name == '14semeval_rest':
            y=F.max_pool1d(y.transpose(1,2),y.shape[1]).squeeze(2)
        else:
            y=F.relu(F.max_pool1d(y.transpose(1,2),y.shape[1]).squeeze(2))
        #####    context_level   ######

        #####    knowledge_level  ######
        if self.args.ds_name == '14semeval_laptop':
            text_knowledge=torch.cat((text_knowledge,text),dim=-1)
            aspect_knowledge=torch.cat((aspect_knowledge,aspect_embed),dim=-1)
        knowledge_score=torch.bmm(aspect_knowledge,text_knowledge.transpose(1,2))
        knowledge_score=F.softmax(knowledge_score, dim=1)
        knowledge_out=torch.bmm(knowledge_score,text_knowledge)
        if self.args.ds_name == '14semeval_rest':
            z=F.max_pool1d(knowledge_out.transpose(1,2),knowledge_out.shape[1]).squeeze(2)
        else:
            z=F.relu(F.max_pool1d(knowledge_out.transpose(1,2),knowledge_out.shape[1]).squeeze(2))
        ####    knowledge_level  ######

        ######## feature fuse  #########
        out=torch.cat((x,y,z),dim=-1)
        output=self.fc(out)
        return output, x, y, z
