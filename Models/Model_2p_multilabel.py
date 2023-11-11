import torch
from transformer_encoder import Transformer
from TransCNN import HARTransformer, Gaussian_Position
import torch.nn.functional as F


class HARTrans(torch.nn.Module):
    def __init__(self, args):
        super(HARTrans, self).__init__()
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.transformer = HARTransformer(270, args.hlayers, args.hheads, args.input_length//args.sample)
        self.args = args
        self.kernel_num = args.kernel_num
        self.kernel_num_v = args.kernel_num_v
        self.filter_sizes = args.filter_size
        self.filter_sizes_v = args.filter_size_v
        self.pos_encoding = Gaussian_Position(270, args.input_length//args.sample, args.K)

        # activation function to change output into multilabel
        self.activation = torch.nn.Sigmoid()

        if args.vlayers == 0:
            self.v_transformer = None
            self.dense = torch.nn.Linear(270, 5)
        else:
            self.v_transformer = Transformer(args.input_length, args.vlayers, args.vheads)
            self.dense = torch.nn.Linear(self.kernel_num * len(self.filter_sizes) + self.kernel_num_v * len(self.filter_sizes_v), 5)

        self.dense2 = torch.nn.Linear(self.kernel_num * len(self.filter_sizes), 5)
        self.dropout_rate = args.dropout_rate
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.encoders = [] # list of temporal transformer encoder
        self.encoder_v = [] # list of channel transformer encoder
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             torch.nn.Conv1d(in_channels=270,
                                       out_channels=self.kernel_num,
                                       kernel_size=filter_size).to('cuda')
                             )
            self.encoders.append(self.__getattr__(enc_attr_name))
        for i, filter_size in enumerate(self.filter_sizes_v):
            enc_attr_name_v = "encoder_v_%d" % i
            self.__setattr__(enc_attr_name_v,
                             torch.nn.Conv1d(in_channels=args.input_length,
                                       out_channels=self.kernel_num_v,
                                       kernel_size=filter_size).to('cuda')
                             )
            self.encoder_v.append(self.__getattr__(enc_attr_name_v))


    def _aggregate(self, o, v=None):
        enc_outs = []
        enc_outs_v = []
        for encoder in self.encoders:
            f_map = encoder(o.transpose(-1, -2))
            enc_ = F.relu(f_map)
            k_h = enc_.size()[-1]
            enc_ = F.max_pool1d(enc_, kernel_size=k_h)
            enc_ = enc_.squeeze(dim=-1)
            enc_outs.append(enc_)
        encoding = self.dropout(torch.cat(enc_outs, 1))
        q_re = F.relu(encoding)
        if self.v_transformer is not None:
            for encoder in self.encoder_v:
                f_map = encoder(v.transpose(-1, -2))
                enc_ = F.relu(f_map)
                k_h = enc_.size()[-1]
                enc_ = F.max_pool1d(enc_, kernel_size=k_h)
                enc_ = enc_.squeeze(dim=-1)
                enc_outs_v.append(enc_)
            encoding_v = self.dropout(torch.cat(enc_outs_v, 1))
            v_re = F.relu(encoding_v)
            q_re = torch.cat((q_re, v_re), dim=1)
        return q_re


    def forward(self, data):
        d1 = data.size(dim=0)
        d3 = data.size(2)
        x = data.unsqueeze(-2)
        x = data.view(d1, -1, self.args.sample, d3)
        x = torch.sum(x, dim=-2).squeeze(-2)
        x = torch.div(x, self.args.sample)
        x = self.pos_encoding(x)
        x = self.transformer(x)

        if self.v_transformer is not None:
            y = data.view(-1, self.args.input_length, 3, 90)
            y = torch.sum(y, dim=-2).squeeze(-2)
            y = y.transpose(-1, -2)
            y = self.v_transformer(y)
            re = self._aggregate(x, y)
            predict = self.activation(self.dense(re))
        else:
            re = self._aggregate(x)
            predict = self.activation(self.dense(re))

        return predict


class TransCNN(torch.nn.Module):
    def __init__(self, args):
        super(TransCNN, self).__init__()
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.transformer = Transformer(270, args.hlayers, 9)
        self.args = args
        self.kernel_num = 128
        self.kernel_num_v = 16
        self.filter_sizes = [10, 40]
        self.filter_sizes_v = [2, 4]

        if args.vlayers == 0:
            self.v_transformer = None
            self.dense = torch.nn.Linear(270, 7)
        else:
            self.v_transformer = Transformer(args.input_length, args.vlayers, 200)
            self.dense = torch.nn.Linear(self.kernel_num * len(self.filter_sizes) + self.kernel_num_v * len(self.filter_sizes_v), 5)

        self.dense2 = torch.nn.Linear(self.kernel_num * len(self.filter_sizes), 5)
        self.dropout_rate = 0.5
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.encoders = []
        self.encoder_v = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             torch.nn.Conv1d(in_channels=90,
                                       out_channels=self.kernel_num,
                                       kernel_size=filter_size).to('cuda')
                             )
            self.encoders.append(self.__getattr__(enc_attr_name))
        for i, filter_size in enumerate(self.filter_sizes_v):
            enc_attr_name_v = "encoder_v_%d" % i
            self.__setattr__(enc_attr_name_v,
                             torch.nn.Conv1d(in_channels=args.input_length,
                                       out_channels=self.kernel_num_v,
                                       kernel_size=filter_size).to('cuda')
                             )
            self.encoder_v.append(self.__getattr__(enc_attr_name_v))


    def _aggregate(self, o, v=None):
        enc_outs = []
        enc_outs_v = []
        for encoder in self.encoders:
            f_map = encoder(o.transpose(-1, -2))
            enc_ = F.relu(f_map)
            k_h = enc_.size()[-1]
            enc_ = F.max_pool1d(enc_, kernel_size=k_h)
            enc_ = enc_.squeeze(dim=-1)
            enc_outs.append(enc_)
        encoding = self.dropout(torch.cat(enc_outs, 1))
        q_re = F.relu(encoding)
        if self.v_transformer is not None:
            for encoder in self.encoder_v:
                f_map = encoder(v.transpose(-1, -2))
                enc_ = F.relu(f_map)
                k_h = enc_.size()[-1]
                enc_ = F.max_pool1d(enc_, kernel_size=k_h)
                enc_ = enc_.squeeze(dim=-1)
                enc_outs_v.append(enc_)
            encoding_v = self.dropout(torch.cat(enc_outs_v, 1))
            v_re = F.relu(encoding_v)
            q_re = torch.cat((q_re, v_re), dim=1)
        return q_re


    def forward(self, data):
        d1 = data.size(dim=0)
        d3 = data.size(2)
        x = data.unsqueeze(-2)
        x = data.view(d1, -1, self.args.sample, d3)
        x = torch.sum(x, dim=-2).squeeze(-2)
        x = torch.div(x, self.args.sample)
        x = self.transformer(x)

        if self.v_transformer is not None:
            y = data.view(-1, args.input_length, 3, 90)
            y = torch.sum(y, dim=-2).squeeze(-2)
            y = y.transpose(-1, -2)
            y = self.v_transformer(y)
            re = self._aggregate(x, y)
            predict = self.softmax(self.dense(re))
        else:
            re = self._aggregate(x)
            predict = self.softmax(self.dense2(re))

        return predict


class TransformerM(torch.nn.Module):
    def __init__(self, args):
        super(TransformerM, self).__init__()
        self.args = args
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.transformer = Transformer(270, args.hlayers, 9)

        if args.vlayers == 0:
            self.v_transformer = None
            self.dense = torch.nn.Linear(270, 5)
        else:
            self.v_transformer = Transformer(args.input_length, args.vlayers, 200)
            self.linear = torch.nn.Linear(args.input_length, 5)
            self.dense = torch.nn.Linear(270, 5)

        self.cls = torch.nn.Parameter(torch.zeros([1, 1, 270], dtype=torch.float, requires_grad=True))
        self.sep = torch.nn.Parameter(torch.zeros([1, 1, 270], dtype=torch.float, requires_grad=True))
        torch.nn.init.xavier_uniform_(self.cls, gain=1)
        torch.nn.init.xavier_uniform_(self.sep, gain=1)


    def fusion(self, x, y):
        y = self.softmax(self.linear(y))
        x = self.softmax(self.dense(x))
        predict = x + y
        return predict


    def forward(self, data):
        d1 = data.size(dim=0)
        d3 = data.size(2)
        x = data.unsqueeze(-2)
        x = data.view(d1, -1, self.args.sample, d3)
        x = torch.sum(x, dim=-2).squeeze(-2)
        x = torch.div(x, self.args.sample)
        dx = x.size(1)
        x = self.transformer(x)
        x = torch.div(torch.sum(x, dim=1).squeeze(dim=1), dx)
        if self.v_transformer is not None:
            y = data.view(-1, args.input_length, 3, 90)
            y = torch.sum(y, dim=-2).squeeze(-2)
            y = y.transpose(-1, -2)
            d2 = y.size(1)
            y = self.v_transformer(y)
            dy = y.size(1)*3
            y = torch.div(torch.sum(y, dim=1).squeeze(dim=1), dy)
            predict = self.fusion(x, y)
        else:
            predict = self.softmax(self.dense(x))

        return predict