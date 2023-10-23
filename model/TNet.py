import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
import torch
from torch import nn

class Absolute_Position_Embedding(nn.Module):
    def __init__(self,  size=None, mode='sum'):
        self.size = size
        self.mode = mode
        super(Absolute_Position_Embedding, self).__init__()

    def forward(self, x, offset):
        if (self.size is None) or (self.mode == 'sum'):
            self.size = int(x.size(-1))
        batch_size, seq_len = x.size()[0], x.size()[1]
        if torch.cuda.is_available():
            weight=torch.FloatTensor(x.shape[0],x.shape[1]).zero_().cuda()
        else:
            weight=torch.FloatTensor(x.shape[0],x.shape[1]).zero_()
        for i in range(offset.shape[0]):
            weight[i]=offset[i][:x.shape[1]]
        x = weight.unsqueeze(2)*x
        return x


    def weight_matrix(self, pos_inx, batch_size, seq_len):
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(pos_inx[i]):
                relative_pos = pos_inx[i] - j
                weight[i].append(1 - relative_pos / 40)
            for j in range(pos_inx[i], seq_len):
                relative_pos = j - pos_inx[i]
                weight[i].append(1 - relative_pos / 40)
        weight = torch.tensor(weight)
        return weight

class TNet_LF(nn.Module):
    def __init__(self, args):
        super(TNet_LF, self).__init__()
        print("this is TNet_LF model")
        V = 300
        D = 300
        C = 3
        self.embed = nn.Embedding.from_pretrained(torch.from_numpy(args.embeddings).float().cuda(), freeze=True)
        self.position = Absolute_Position_Embedding()
        HD = 300
        self.lstm1 = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = DynamicLSTM(300, 300, num_layers=1, batch_first=True, bidirectional=True)
        self.convs3 = nn.Conv1d(2 * HD, 50, 3, padding=1)
        self.fc1 = nn.Linear(4 * HD, 2 * HD)
        self.fc = nn.Linear(50, C)

    def get_aspect_index(self,aspect_len,td_len):
        aspect_index=[]
        aspect_len=aspect_len.cpu().numpy()
        td_len=td_len.cpu().numpy()
        for i in range(aspect_len.shape[0]):
            try:
                a=td_len[i]-aspect_len[i]
            except ValueError:
                a=0
            aspect_index.append(a)
        return aspect_index

    def forward(self, feature, aspect, offset):
        text_raw_indices, aspect_indices, offset = feature.long(), aspect.long(), offset
        feature_len = torch.sum(text_raw_indices != 0, dim=-1).cpu()
        aspect_len = torch.sum(aspect_indices != 0, dim=-1).cpu()
        feature = self.embed(text_raw_indices)
        aspect = self.embed(aspect_indices)
        v, (_, _) = self.lstm1(feature, feature_len)
        e, (_, _) = self.lstm2(aspect, aspect_len)
        v = v.transpose(1, 2)
        e = e.transpose(1, 2)
        for i in range(2):
            a = torch.bmm(e.transpose(1, 2), v)
            a = F.softmax(a, 1)  # (aspect_len,context_len)
            aspect_mid = torch.bmm(e, a)
            aspect_mid = torch.cat((aspect_mid, v), dim=1).transpose(1, 2)
            aspect_mid = F.relu(self.fc1(aspect_mid).transpose(1, 2))
            v = aspect_mid + v
            v = self.position(v.transpose(1, 2), offset).transpose(1, 2)
        z = F.relu(self.convs3(v))  # [(N,Co,L), ...]*len(Ks)
        z = F.max_pool1d(z, z.size(2)).squeeze(2)
        out = self.fc(z)
        return out
