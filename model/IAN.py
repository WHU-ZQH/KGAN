from layers.dynamic_rnn import DynamicLSTM
import torch
from torch import nn
from layers.attention import Attention
class IAN(nn.Module):
    def __init__(self, args):
        super(IAN, self).__init__()
        self.args = args
        self.embed = nn.Embedding.from_pretrained(torch.from_numpy(args.embeddings).float().cuda(), freeze=True)
        self.lstm_context = DynamicLSTM(300, 300, num_layers=1, batch_first=True)
        self.lstm_aspect = DynamicLSTM(300, 300, num_layers=1, batch_first=True)
        self.attention_aspect = Attention(300, score_function='bi_linear')
        self.attention_context = Attention(300, score_function='bi_linear')
        self.dense = nn.Linear(300*2, 3)

    def forward(self, feature, aspect, offset):
        text_raw_indices, aspect_indices, offset = feature.long(), aspect.long(), offset
        text_raw_len = torch.sum(text_raw_indices != 0, dim=-1).cpu()
        aspect_len = torch.sum(aspect_indices != 0, dim=-1).cpu()

        context = self.embed(text_raw_indices)
        aspect = self.embed(aspect_indices)
        context, (_, _) = self.lstm_context(context, text_raw_len)
        aspect, (_, _) = self.lstm_aspect(aspect, aspect_len)

        if torch.cuda.is_available():
            aspect_len = torch.tensor(aspect_len, dtype=torch.float).cuda()
        else:
            aspect_len = torch.tensor(aspect_len, dtype=torch.float)
        aspect_pool = torch.sum(aspect, dim=1)
        aspect_pool = torch.div(aspect_pool, aspect_len.view(aspect_len.size(0), 1))
        if torch.cuda.is_available():
            text_raw_len = torch.tensor(text_raw_len, dtype=torch.float).cuda()
        else:
            text_raw_len = torch.tensor(text_raw_len, dtype=torch.float)
        context_pool = torch.sum(context, dim=1)
        context_pool = torch.div(context_pool, text_raw_len.view(text_raw_len.size(0), 1))

        aspect_final, _ = self.attention_aspect(aspect, context_pool)
        aspect_final = aspect_final.squeeze(dim=1)
        context_final, _ = self.attention_context(context, aspect_pool)
        context_final = context_final.squeeze(dim=1)

        x = torch.cat((aspect_final, context_final), dim=-1)
        out = self.dense(x)
        return out
