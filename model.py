import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math
import numpy as np


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        """
        # data_ptr 返回首元素内存位置
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()

        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        # aved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)  # time, batch, dim
            k = self.in_proj_k(key)  # time, batch, dim
            v = self.in_proj_v(value)  # time, batch, dim

        q = q * self.scaling  # time, batch, dim

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])  # 1, batch, dim+dim
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])  # 1, batch, dim+dim

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0,
                                                                                        1)  # batch*heads, time1, h_dim
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0,
                                                                                       1)  # batch*heads, time2, h_dim
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0,
                                                                                       1)  # batch*heads, time2, h_dim

        src_len = k.size(1)  # time

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)  # batch*heads, time+1, h_dim
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)  # batch*heads, time+1, h_dim

        attn_weights = torch.bmm(q, k.transpose(1,
                                                2))  # batch*heads, time1, dim || batch*heads, dim, time2 --> batch*heads, time1, time2
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights,
                         v)  # batch*heads, time1, time2 || batch*heads, time2, dim --> batch*heads, time1, dim
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)  # time1, batch, embed_dim
        attn = self.out_proj(attn)  # time1, batch, dim

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)  # bathc, head, time1, time2
        attn_weights = attn_weights.sum(dim=1) / self.num_heads  # batch, time1, time2

        if torch.any(torch.isnan(attn.detach().data)).item():
            raise ValueError("output nan")

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


class TimeEmbedding(nn.Module):
    """
    修改以后，每次只能接收一个样本

    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Sequential(
            Linear(1, embedding_dim),
            nn.Tanh()
        )

    def forward(self, input, time_stamps):
        """
        Input is expected to be of size [bsz x seqlen].
        timestamps: (sep_len,)
        """
        _, seq_len = input.size()
        weights = self.fc(time_stamps.view(-1, 1))
        return weights.view(1, seq_len, -1)


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = Linear(self.embed_dim, 4 * self.embed_dim)  # The "Add & Norm" part in the paper
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        if x_k is None and x_v is None:
            x, attn = self.self_attn(query=x, key=x, value=x)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, attn = self.self_attn(query=x, key=x_k, value=x_v)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        if torch.any(torch.isnan(x.detach().data)).item():
            raise ValueError("output nan")
        return x, attn

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            # before 为 True
            return self.layer_norms[i](x)  # i 为用第几层的 layer norm, 一共有2层
        else:
            # after 为 True
            return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0):
        super().__init__()
        self.dropout = embed_dropout  # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = TimeEmbedding(embed_dim)

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, t_in, x_in_k=None, t_in_k=None, x_in_v=None, t_in_v=None):
        """
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, 1, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, 1, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, 1, embed_dim)`

            t_in (FloatTensor): embedded input of shape `(src_len, )`
            t_in_k (FloatTensor): embedded input of shape `(src_len, )`
            t_in_v (FloatTensor): embedded input of shape `(src_len, )`


        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0], t_in).transpose(0, 1)  # Add positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None and t_in_k is not None and t_in_v is not None:
            # embed tokens and positions
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0], t_in_k).transpose(0,
                                                                                               1)  # Add positional embedding
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0], t_in_v).transpose(0,
                                                                                               1)  # Add positional embedding
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        # encoder layers
        intermediates = [x]
        attn_list = []
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x, attn = layer(x, x_k, x_v)
            else:
                x, attn = layer(x)
            intermediates.append(x)
            attn_list.append(attn)

        if self.normalize:
            x = self.layer_norm(x)

        if torch.any(torch.isnan(x.detach().data)).item():
            raise ValueError("output nan")

        return x, attn_list

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class MULTModel_1(nn.Module):
    def __init__(self, HP, orig_d_1, d_1):
        super().__init__()
        self.orig_d_1 = orig_d_1
        self.d_1 = d_1
        self.num_heads = HP.num_heads
        self.layers = HP.layers
        self.attn_dropout_1 = HP.attn_dropout_1
        self.relu_dropout = HP.relu_dropout
        self.res_dropout = HP.res_dropout
        self.out_dropout = HP.out_dropout
        self.embed_dropout = HP.embed_dropout

        combined_dim = 1 * self.d_1

        self.proj_1 = nn.Conv1d(self.orig_d_1, self.d_1, kernel_size=1, padding=0, bias=False)

        self.trans = self.get_network(self.layers)

        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)

        output_dim = d_1  # This is actually not a hyperparameter :-)

        self.out_layer = nn.Linear(combined_dim, output_dim)

        self.init_weights()

    def get_network(self, layers=-1):

        embed_dim, attn_dropout = 1 * self.d_1, self.attn_dropout_1

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout)

    def forward(self, x_1, t_1):

        x_1 = x_1.transpose(1, 2)  # 1, n_features, seq_len

        proj_x_1 = x_1 if self.orig_d_1 == self.d_1 else self.proj_1(x_1)
        proj_x_1 = proj_x_1.permute(2, 0, 1)  # seq_len, 1, n_features

        h_1, _ = self.trans(proj_x_1, t_1)  # Dimension (L, N, d_l)

        last_hs = h_1[-1]  # batch=1, dim

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        return output  # output:batch=1, dim

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.uniform_(m.bias.data, 0, 1)
            else:
                for param in m.parameters():
                    nn.init.normal_(param.data)


class MULTModel_2(nn.Module):
    def __init__(self, HP, orig_d_1, d_1, orig_d_2, d_2):
        """
        Construct a MulT model.

        forward 方法每次只能接受一个样本

        l-1, a-2, v-3

        """
        super().__init__()
        self.orig_d_1, self.orig_d_2 = orig_d_1, orig_d_2
        self.d_1, self.d_2 = d_1, d_2
        self.only2 = HP.only2
        self.only1 = HP.only1
        self.num_heads = HP.num_heads
        self.layers = HP.layers
        self.attn_dropout_1 = HP.attn_dropout_1
        self.attn_dropout_2 = HP.attn_dropout_2
        self.relu_dropout = HP.relu_dropout
        self.res_dropout = HP.res_dropout
        self.out_dropout = HP.out_dropout
        self.embed_dropout = HP.embed_dropout

        # combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.only1 + self.only2
        if self.partial_mode == 1:
            combined_dim = 1 * self.d_1  # assuming d_1 == d_2
        else:
            combined_dim = 1 * (self.d_1 + self.d_2)

        output_dim = HP.MulT_output_dim  # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_1 = nn.Conv1d(self.orig_d_1, self.d_1, kernel_size=1, padding=0, bias=False)
        self.proj_2 = nn.Conv1d(self.orig_d_2, self.d_2, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.only1:
            self.trans_1_with_2 = self.get_network(self_type='12')
        if self.only2:
            self.trans_2_with_1 = self.get_network(self_type='21')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1) # 1层的 LSTM
        self.trans_1_mem = self.get_network(self_type='1_mem', layers=3)
        self.trans_2_mem = self.get_network(self_type='2_mem', layers=3)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        self.init_weights()

    def get_network(self, self_type='l', layers=-1):

        if self_type in ['1', '12']:

            embed_dim, attn_dropout = self.d_1, self.attn_dropout_1

        elif self_type in ['2', '21']:

            embed_dim, attn_dropout = self.d_2, self.attn_dropout_2

        elif self_type == '1_mem':  #

            embed_dim, attn_dropout = 1 * self.d_1, self.attn_dropout_1

        elif self_type == '2_mem':

            embed_dim, attn_dropout = 1 * self.d_2, self.attn_dropout_1

        else:

            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout)

    def forward(self, x_1, t_1, x_2, t_2):
        """
        每次只能接受一个样本，因为 t_* 对于不同患者来说是不同的，所以不能完成批量计算
        text, audio, and vision should have dimension [batch_size=1, seq_len, n_features]
        """
        # batch_size, n_features, seq_len
        x_1 = x_1.transpose(1, 2)  # 1, n_features, seq_len
        x_2 = x_2.transpose(1, 2)  # 1, n_features, seq_len

        # Project the textual/visual/audio features
        # Project the textual/visual/audio features
        proj_x_1 = x_1 if self.orig_d_1 == self.d_1 else self.proj_1(x_1)
        proj_x_2 = x_2 if self.orig_d_2 == self.d_2 else self.proj_2(x_2)
        proj_x_2 = proj_x_2.permute(2, 0, 1)  # seq_len, 1, n_features
        proj_x_1 = proj_x_1.permute(2, 0, 1)  # seq_len, 1, n_features

        if self.only1:
            # (2,3) --> 1

            h_1_with_2s, _ = self.trans_1_with_2(proj_x_1, t_1, proj_x_2, t_2, proj_x_2, t_2)  # Dimension (L, N, d_l)
            h_1s = h_1_with_2s
            h_1s, _ = self.trans_1_mem(h_1s, t_1)
            if type(h_1s) == tuple:
                h_1s = h_1s[0]
            last_h_1 = last_hs = h_1s[-1]  # batch=1, dim  # Take the last output for prediction

        if self.only2:
            # (1,3) --> 2
            h_2_with_1s, _ = self.trans_2_with_1(proj_x_2, t_2, proj_x_1, t_1, proj_x_1, t_1)
            h_2s = h_2_with_1s
            h_2s, _ = self.trans_2_mem(h_2s, t_2)
            if type(h_2s) == tuple:
                h_2s = h_2s[0]
            last_h_2 = last_hs = h_2s[-1]

        if self.partial_mode == 2:
            last_hs = torch.cat([last_h_1, last_h_2], dim=1)  # batch=1, 3*dim

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        return output, last_hs  # output:batch=1, dim   # last_hs: batch=1, dim

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.uniform_(m.bias.data, 0, 1)
            else:
                for param in m.parameters():
                    nn.init.normal_(param.data)


class MULTModel_3(nn.Module):
    def __init__(self, HP):
        """
        Construct a MulT model.

        forward 方法每次只能接受一个样本

        l-1, a-2, v-3

        """
        super().__init__()
        self.orig_d_1, self.orig_d_2, self.orig_d_3 = HP.orig_d_1, HP.orig_d_2, HP.orig_d_3
        self.d_1, self.d_2, self.d_3 = HP.d_1, HP.d_2, HP.d_3
        self.only3 = HP.only3
        self.only2 = HP.only2
        self.only1 = HP.only1
        self.num_heads = HP.num_heads
        self.layers = HP.layers
        self.attn_dropout_1 = HP.attn_dropout_1
        self.attn_dropout_2 = HP.attn_dropout_2
        self.attn_dropout_3 = HP.attn_dropout_3
        self.relu_dropout = HP.relu_dropout
        self.res_dropout = HP.res_dropout
        self.out_dropout = HP.out_dropout
        self.embed_dropout = HP.embed_dropout

        # combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.only1 + self.only2 + self.only3
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_1  # assuming d_1 == d_2 == d_3
        else:
            combined_dim = 2 * (self.d_1 + self.d_2 + self.d_3)

        output_dim = HP.MulT_output_dim  # This is the output_dim of the cross-transformer

        # 1. Temporal convolutional layers
        self.proj_1 = nn.Conv1d(self.orig_d_1, self.d_1, kernel_size=1, padding=0, bias=False)
        self.proj_2 = nn.Conv1d(self.orig_d_2, self.d_2, kernel_size=1, padding=0, bias=False)
        self.proj_3 = nn.Conv1d(self.orig_d_3, self.d_3, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.only1:
            self.trans_1_with_2 = self.get_network(self_type='12')
            self.trans_1_with_3 = self.get_network(self_type='13')
        if self.only2:
            self.trans_2_with_1 = self.get_network(self_type='21')
            self.trans_2_with_3 = self.get_network(self_type='23')
        if self.only3:
            self.trans_3_with_1 = self.get_network(self_type='31')
            self.trans_3_with_2 = self.get_network(self_type='32')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1) # 1层的 LSTM
        self.trans_1_mem = self.get_network(self_type='1_mem', layers=3)
        self.trans_2_mem = self.get_network(self_type='2_mem', layers=3)
        self.trans_3_mem = self.get_network(self_type='3_mem', layers=3)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        self.init_weights()

    def get_network(self, self_type='l', layers=-1):

        if self_type in ['1', '12', '13']:

            embed_dim, attn_dropout = self.d_1, self.attn_dropout_1

        elif self_type in ['2', '21', '23']:

            embed_dim, attn_dropout = self.d_2, self.attn_dropout_2

        elif self_type in ['3', '31', '32']:

            embed_dim, attn_dropout = self.d_3, self.attn_dropout_3

        elif self_type == '1_mem':  #

            embed_dim, attn_dropout = 2 * self.d_1, self.attn_dropout_1

        elif self_type == '2_mem':

            embed_dim, attn_dropout = 2 * self.d_2, self.attn_dropout_1

        elif self_type == '3_mem':

            embed_dim, attn_dropout = 2 * self.d_3, self.attn_dropout_1


        else:

            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.uniform_(m.bias.data, 0, 1)
            # elif isinstance(m, nn.Conv1d):
            #     nn.init.xavier_uniform_(m.weight.data)
            else:
                for param in m.parameters():
                    nn.init.normal_(param.data)

    def forward(self, x_1, t_1, x_2, t_2, x_3, t_3):
        """
        每次只能接受一个样本，因为 t_* 对于不同患者来说是不同的，所以不能完成批量计算
        text, audio, and vision should have dimension [batch_size=1, seq_len, n_features]
        """
        # batch_size, n_features, seq_len
        x_1 = x_1.transpose(1, 2)  # 1, n_features, seq_len
        x_2 = x_2.transpose(1, 2)  # 1, n_features, seq_len
        x_3 = x_3.transpose(1, 2)  # 1, n_features, seq_len

        # Project the 1/2/3 features
        proj_x_1 = x_1 if self.orig_d_1 == self.d_1 else self.proj_1(x_1)
        proj_x_2 = x_2 if self.orig_d_2 == self.d_2 else self.proj_2(x_2)
        proj_x_3 = x_3 if self.orig_d_3 == self.d_3 else self.proj_3(x_3)
        proj_x_3 = proj_x_3.permute(2, 0, 1)  # seq_len, 1, n_features
        proj_x_2 = proj_x_2.permute(2, 0, 1)  # seq_len, 1, n_features
        proj_x_1 = proj_x_1.permute(2, 0, 1)  # seq_len, 1, n_features

        if self.only1:
            # (2,3) --> 1

            h_1_with_2s, _ = self.trans_1_with_2(proj_x_1, t_1, proj_x_2, t_2, proj_x_2, t_2)  # Dimension (L, N, d_l)
            h_1_with_3s, _ = self.trans_1_with_3(proj_x_1, t_1, proj_x_3, t_3, proj_x_3, t_3)  # Dimension (L, N, d_l)
            h_1s = torch.cat([h_1_with_2s, h_1_with_3s], dim=2)
            h_1s, _ = self.trans_1_mem(h_1s, t_1)
            if type(h_1s) == tuple:
                h_1s = h_1s[0]
            last_h_1 = last_hs = h_1s[-1]  # batch=1, dim  # Take the last output for prediction

        if self.only2:
            # (1,3) --> 2
            h_2_with_1s, _ = self.trans_2_with_1(proj_x_2, t_2, proj_x_1, t_1, proj_x_1, t_1)
            h_2_with_3s, _ = self.trans_2_with_3(proj_x_2, t_2, proj_x_3, t_3, proj_x_3, t_3)
            h_2s = torch.cat([h_2_with_1s, h_2_with_3s], dim=2)
            h_2s, _ = self.trans_2_mem(h_2s, t_2)
            if type(h_2s) == tuple:
                h_2s = h_2s[0]
            last_h_2 = last_hs = h_2s[-1]

        if self.only3:
            # (1,2) --> 3
            h_3_with_1s, _ = self.trans_3_with_1(proj_x_3, t_3, proj_x_1, t_1, proj_x_1, t_1)
            h_3_with_2s, _ = self.trans_3_with_2(proj_x_3, t_3, proj_x_2, t_2, proj_x_2, t_2)
            h_3s = torch.cat([h_3_with_1s, h_3_with_2s], dim=2)
            h_3s, _ = self.trans_3_mem(h_3s, t_3)
            if type(h_3s) == tuple:
                h_3s = h_3s[0]
            last_h_3 = last_hs = h_3s[-1]

        if self.partial_mode == 3:
            last_hs = torch.cat([last_h_1, last_h_2, last_h_3], dim=1)  # batch=1, 3*dim

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        return output, last_hs  # output:batch=1, dim   # last_hs: batch=1, dim


class MULTModel_3_1(nn.Module):
    def __init__(self, HP, orig_d_1, d_1, orig_d_2, d_2, orig_d_3, d_3):
        """
        Construct a MulT model.

        forward 方法每次只能接受一个样本

        l-1, a-2, v-3

        """
        super().__init__()
        self.orig_d_1, self.orig_d_2, self.orig_d_3 = orig_d_1, orig_d_2, orig_d_3
        self.d_1, self.d_2, self.d_3 = d_1, d_2, d_3
        self.only3 = HP.only3
        self.only2 = HP.only2
        self.only1 = HP.only1
        self.num_heads = HP.num_heads
        self.layers = HP.layers
        self.attn_dropout_1 = HP.attn_dropout_1
        self.attn_dropout_2 = HP.attn_dropout_2
        self.attn_dropout_3 = HP.attn_dropout_3
        self.relu_dropout = HP.relu_dropout
        self.res_dropout = HP.res_dropout
        self.out_dropout = HP.out_dropout
        self.embed_dropout = HP.embed_dropout

        # combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.only1 + self.only2 + self.only3
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_1  # assuming d_1 == d_2 == d_3
        else:
            combined_dim = 2 * (self.d_1 + self.d_2 + self.d_3)

        output_dim = HP.MulT_output_dim  # This is the output_dim of the cross-transformer

        # 1. Temporal convolutional layers
        self.proj_1 = nn.Conv1d(self.orig_d_1, self.d_1, kernel_size=1, padding=0, bias=False)
        self.proj_2 = nn.Conv1d(self.orig_d_2, self.d_2, kernel_size=1, padding=0, bias=False)
        self.proj_3 = nn.Conv1d(self.orig_d_3, self.d_3, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.only1:
            self.trans_1_with_2 = self.get_network(self_type='12')
            self.trans_1_with_3 = self.get_network(self_type='13')
        if self.only2:
            self.trans_2_with_1 = self.get_network(self_type='21')
            self.trans_2_with_3 = self.get_network(self_type='23')
        if self.only3:
            self.trans_3_with_1 = self.get_network(self_type='31')
            self.trans_3_with_2 = self.get_network(self_type='32')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1) # 1层的 LSTM
        self.trans_1_mem = self.get_network(self_type='1_mem', layers=3)
        self.trans_2_mem = self.get_network(self_type='2_mem', layers=3)
        self.trans_3_mem = self.get_network(self_type='3_mem', layers=3)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        self.init_weights()

    def get_network(self, self_type='l', layers=-1):

        if self_type in ['1', '12', '13']:

            embed_dim, attn_dropout = self.d_1, self.attn_dropout_1

        elif self_type in ['2', '21', '23']:

            embed_dim, attn_dropout = self.d_2, self.attn_dropout_2

        elif self_type in ['3', '31', '32']:

            embed_dim, attn_dropout = self.d_3, self.attn_dropout_3

        elif self_type == '1_mem':  #

            embed_dim, attn_dropout = 2 * self.d_1, self.attn_dropout_1

        elif self_type == '2_mem':

            embed_dim, attn_dropout = 2 * self.d_2, self.attn_dropout_1

        elif self_type == '3_mem':

            embed_dim, attn_dropout = 2 * self.d_3, self.attn_dropout_1


        else:

            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.uniform_(m.bias.data, 0, 1)
            # elif isinstance(m, nn.Conv1d):
            #     nn.init.xavier_uniform_(m.weight.data)
            else:
                for param in m.parameters():
                    nn.init.normal_(param.data)

    def forward(self, x_1, t_1, x_2, t_2, x_3, t_3):
        """
        每次只能接受一个样本，因为 t_* 对于不同患者来说是不同的，所以不能完成批量计算
        text, audio, and vision should have dimension [batch_size=1, seq_len, n_features]
        """
        # batch_size, n_features, seq_len
        x_1 = x_1.transpose(1, 2)  # 1, n_features, seq_len
        x_2 = x_2.transpose(1, 2)  # 1, n_features, seq_len
        x_3 = x_3.transpose(1, 2)  # 1, n_features, seq_len

        # Project the 1/2/3 features
        proj_x_1 = x_1 if self.orig_d_1 == self.d_1 else self.proj_1(x_1)
        proj_x_2 = x_2 if self.orig_d_2 == self.d_2 else self.proj_2(x_2)
        proj_x_3 = x_3 if self.orig_d_3 == self.d_3 else self.proj_3(x_3)
        proj_x_3 = proj_x_3.permute(2, 0, 1)  # seq_len, 1, n_features
        proj_x_2 = proj_x_2.permute(2, 0, 1)  # seq_len, 1, n_features
        proj_x_1 = proj_x_1.permute(2, 0, 1)  # seq_len, 1, n_features

        if self.only1:
            # (2,3) --> 1

            h_1_with_2s, _ = self.trans_1_with_2(proj_x_1, t_1, proj_x_2, t_2, proj_x_2, t_2)  # Dimension (L, N, d_l)
            h_1_with_3s, _ = self.trans_1_with_3(proj_x_1, t_1, proj_x_3, t_3, proj_x_3, t_3)  # Dimension (L, N, d_l)
            h_1s = torch.cat([h_1_with_2s, h_1_with_3s], dim=2)
            h_1s, _ = self.trans_1_mem(h_1s, t_1)
            if type(h_1s) == tuple:
                h_1s = h_1s[0]
            last_h_1 = last_hs = h_1s[-1]  # batch=1, dim  # Take the last output for prediction

        if self.only2:
            # (1,3) --> 2
            h_2_with_1s, _ = self.trans_2_with_1(proj_x_2, t_2, proj_x_1, t_1, proj_x_1, t_1)
            h_2_with_3s, _ = self.trans_2_with_3(proj_x_2, t_2, proj_x_3, t_3, proj_x_3, t_3)
            h_2s = torch.cat([h_2_with_1s, h_2_with_3s], dim=2)
            h_2s, _ = self.trans_2_mem(h_2s, t_2)
            if type(h_2s) == tuple:
                h_2s = h_2s[0]
            last_h_2 = last_hs = h_2s[-1]

        if self.only3:
            # (1,2) --> 3
            h_3_with_1s, _ = self.trans_3_with_1(proj_x_3, t_3, proj_x_1, t_1, proj_x_1, t_1)
            h_3_with_2s, _ = self.trans_3_with_2(proj_x_3, t_3, proj_x_2, t_2, proj_x_2, t_2)
            h_3s = torch.cat([h_3_with_1s, h_3_with_2s], dim=2)
            h_3s, _ = self.trans_3_mem(h_3s, t_3)
            if type(h_3s) == tuple:
                h_3s = h_3s[0]
            last_h_3 = last_hs = h_3s[-1]

        if self.partial_mode == 3:
            last_hs = torch.cat([last_h_1, last_h_2, last_h_3], dim=1)  # batch=1, 3*dim

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        return output, last_hs  # output:batch=1, dim   # last_hs: batch=1, dim


class MULTModel_4(nn.Module):
    def __init__(self, HP):
        """
        Construct a MulT model.

        forward 方法每次只能接受一个样本

        """
        super().__init__()
        self.orig_d_1, self.orig_d_2, self.orig_d_3 = HP.orig_d_1, HP.orig_d_2, HP.orig_d_3
        self.orig_d_4 = HP.orig_d_4

        self.d_1, self.d_2, self.d_3, self.d_4 = HP.d_1, HP.d_2, HP.d_3, HP.d_4
        self.only4 = HP.only4
        self.only3 = HP.only3
        self.only2 = HP.only2
        self.only1 = HP.only1
        self.num_heads = HP.num_heads
        self.layers = HP.layers
        self.attn_dropout_1 = HP.attn_dropout_1
        self.attn_dropout_2 = HP.attn_dropout_2
        self.attn_dropout_3 = HP.attn_dropout_3
        self.attn_dropout_4 = HP.attn_dropout_4
        self.relu_dropout = HP.relu_dropout
        self.res_dropout = HP.res_dropout
        self.out_dropout = HP.out_dropout
        self.embed_dropout = HP.embed_dropout

        # combined_dim = self.d_1 + self.d_2 + self.d_3 + self.d_4

        self.partial_mode = self.only1 + self.only2 + self.only3 + self.only4
        if self.partial_mode == 1:
            combined_dim = 3 * self.d_1  # assuming d_1 == d_2 == d_3 == d_4
        else:
            combined_dim = 3 * (self.d_1 + self.d_2 + self.d_3 + self.d_4)

        output_dim = HP.MulT_output_dim  # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_1 = nn.Conv1d(self.orig_d_1, self.d_1, kernel_size=1, padding=0, bias=False)
        self.proj_2 = nn.Conv1d(self.orig_d_2, self.d_2, kernel_size=1, padding=0, bias=False)
        self.proj_3 = nn.Conv1d(self.orig_d_3, self.d_3, kernel_size=1, padding=0, bias=False)
        self.proj_4 = nn.Conv1d(self.orig_d_4, self.d_4, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.only1:
            self.trans_1_with_2 = self.get_network(self_type='12')
            self.trans_1_with_3 = self.get_network(self_type='13')
            self.trans_1_with_4 = self.get_network(self_type='14')
        if self.only2:
            self.trans_2_with_1 = self.get_network(self_type='21')
            self.trans_2_with_3 = self.get_network(self_type='23')
            self.trans_2_with_4 = self.get_network(self_type='24')
        if self.only3:
            self.trans_3_with_1 = self.get_network(self_type='31')
            self.trans_3_with_2 = self.get_network(self_type='32')
            self.trans_3_with_4 = self.get_network(self_type='34')
        if self.only4:
            self.trans_4_with_1 = self.get_network(self_type='41')
            self.trans_4_with_2 = self.get_network(self_type='42')
            self.trans_4_with_3 = self.get_network(self_type='43')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1) # 1层的 LSTM
        self.trans_1_mem = self.get_network(self_type='1_mem', layers=3)
        self.trans_2_mem = self.get_network(self_type='2_mem', layers=3)
        self.trans_3_mem = self.get_network(self_type='3_mem', layers=3)
        self.trans_4_mem = self.get_network(self_type='4_mem', layers=3)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        self.init_weights()

    def get_network(self, self_type='l', layers=-1):

        if self_type in ['1', '12', '13', '14']:

            embed_dim, attn_dropout = self.d_1, self.attn_dropout_1

        elif self_type in ['2', '21', '23', '24']:

            embed_dim, attn_dropout = self.d_2, self.attn_dropout_2

        elif self_type in ['3', '31', '32', '34']:

            embed_dim, attn_dropout = self.d_3, self.attn_dropout_3

        elif self_type in ['4', '41', '42', '43']:

            embed_dim, attn_dropout = self.d_4, self.attn_dropout_4

        elif self_type == '1_mem':  #

            embed_dim, attn_dropout = 3 * self.d_1, self.attn_dropout_1

        elif self_type == '2_mem':

            embed_dim, attn_dropout = 3 * self.d_2, self.attn_dropout_1

        elif self_type == '3_mem':

            embed_dim, attn_dropout = 3 * self.d_3, self.attn_dropout_1

        elif self_type == '4_mem':

            embed_dim, attn_dropout = 3 * self.d_4, self.attn_dropout_1

        else:

            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.uniform_(m.bias.data, 0, 1)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight.data)
            else:
                for param in m.parameters():
                    nn.init.normal_(param.data)

    def forward(self, x_1, t_1, x_2, t_2, x_3, t_3, x_4, t_4):
        """
        每次只能接受一个样本，因为 t_* 对于不同患者来说是不同的，所以不能完成批量计算
        text, audio, and vision should have dimension [batch_size=1, seq_len, n_features]
        """
        # batch_size, n_features, seq_len
        x_1 = x_1.transpose(1, 2)  # 1, n_features, seq_len
        x_2 = x_2.transpose(1, 2)  # 1, n_features, seq_len
        x_3 = x_3.transpose(1, 2)  # 1, n_features, seq_len
        x_4 = x_4.transpose(1, 2)  # 1, n_features, seq_len

        # Project the textual/visual/audio features
        proj_x_1 = x_1 if self.orig_d_1 == self.d_1 else self.proj_1(x_1)
        proj_x_2 = x_2 if self.orig_d_2 == self.d_2 else self.proj_2(x_2)
        proj_x_3 = x_3 if self.orig_d_3 == self.d_3 else self.proj_3(x_3)
        proj_x_4 = x_4 if self.orig_d_4 == self.d_4 else self.proj_4(x_4)
        proj_x_4 = proj_x_4.permute(2, 0, 1)  # seq_len, 1, n_features
        proj_x_3 = proj_x_3.permute(2, 0, 1)  # seq_len, 1, n_features
        proj_x_2 = proj_x_2.permute(2, 0, 1)  # seq_len, 1, n_features
        proj_x_1 = proj_x_1.permute(2, 0, 1)  # seq_len, 1, n_features

        if self.only1:
            # (2,3,4) --> 1

            h_1_with_2s, _ = self.trans_1_with_2(proj_x_1, t_1, proj_x_2, t_2, proj_x_2, t_2)  # Dimension (L, N, d_l)
            h_1_with_3s, _ = self.trans_1_with_3(proj_x_1, t_1, proj_x_3, t_3, proj_x_3, t_3)  # Dimension (L, N, d_l)
            h_1_with_4s, _ = self.trans_1_with_4(proj_x_1, t_1, proj_x_4, t_4, proj_x_4, t_4)  # Dimension (L, N, d_l)
            h_1s = torch.cat([h_1_with_2s, h_1_with_3s, h_1_with_4s], dim=2)
            h_1s, _ = self.trans_1_mem(h_1s, t_1)
            if type(h_1s) == tuple:
                h_1s = h_1s[0]
            last_h_1 = last_hs = h_1s[-1]  # batch=1, dim  # Take the last output for prediction

        if self.only2:
            # (1,3,4) --> 2
            h_2_with_1s, _ = self.trans_2_with_1(proj_x_2, t_2, proj_x_1, t_1, proj_x_1, t_1)
            h_2_with_3s, _ = self.trans_2_with_3(proj_x_2, t_2, proj_x_3, t_3, proj_x_3, t_3)
            h_2_with_4s, _ = self.trans_2_with_3(proj_x_2, t_2, proj_x_4, t_4, proj_x_4, t_4)
            h_2s = torch.cat([h_2_with_1s, h_2_with_3s, h_2_with_4s], dim=2)
            h_2s, _ = self.trans_2_mem(h_2s, t_2)
            if type(h_2s) == tuple:
                h_2s = h_2s[0]
            last_h_2 = last_hs = h_2s[-1]

        if self.only3:
            # (1,2,4) --> 3
            h_3_with_1s, _ = self.trans_3_with_1(proj_x_3, t_3, proj_x_1, t_1, proj_x_1, t_1)
            h_3_with_2s, _ = self.trans_3_with_2(proj_x_3, t_3, proj_x_2, t_2, proj_x_2, t_2)
            h_3_with_4s, _ = self.trans_3_with_4(proj_x_3, t_3, proj_x_4, t_4, proj_x_4, t_4)
            h_3s = torch.cat([h_3_with_1s, h_3_with_2s, h_3_with_4s], dim=2)
            h_3s, _ = self.trans_3_mem(h_3s, t_3)
            if type(h_3s) == tuple:
                h_3s = h_3s[0]
            last_h_3 = last_hs = h_3s[-1]

        if self.only4:
            # (1,2,3) --> 4
            h_4_with_1s, _ = self.trans_3_with_1(proj_x_4, t_4, proj_x_1, t_1, proj_x_1, t_1)
            h_4_with_2s, _ = self.trans_3_with_2(proj_x_4, t_4, proj_x_2, t_2, proj_x_2, t_2)
            h_4_with_3s, _ = self.trans_3_with_4(proj_x_4, t_4, proj_x_3, t_3, proj_x_3, t_3)
            h_4s = torch.cat([h_4_with_1s, h_4_with_2s, h_4_with_3s], dim=2)
            h_4s, _ = self.trans_4_mem(h_4s, t_4)
            if type(h_4s) == tuple:
                h_4s = h_4s[0]
            last_h_4 = last_hs = h_4s[-1]

        if self.partial_mode == 4:
            last_hs = torch.cat([last_h_1, last_h_2, last_h_3, last_h_4], dim=1)  # batch=1, 4*dim

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        if torch.any(torch.isnan(output.detach().data)).item():
            raise ValueError("output nan")
        return output, last_hs  # output:batch=1, dim   # last_hs: batch=1, dim


class Bi_ATT_LSTM(nn.Module):
    def __init__(self, embedding_dim, n_hidden):
        super(Bi_ATT_LSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.init_weights()

    def attention_net(self, lstm_output, final_state):
        # lstm_output : [batch_size=1, n_step, n_hidden * num_directions(=2)], F matrix
        # final_state : [num_layers(=1) * num_directions(=2), batch_size=1, n_hidden]

        # hidden = final_state.view(batch_size=1,-1,1)
        hidden = torch.cat((final_state[0], final_state[1]), dim=1).unsqueeze(2)
        # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        # attn_weights : [batch_size,n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)

        # context: [batch_size, n_hidden * num_directions(=2)]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.uniform_(m.bias.data, 0, 1)
            elif type(m) in [nn.GRU, nn.LSTM, nn.GRUCell, nn.LSTMCell]:
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'bias' in name:
                        nn.init.uniform_(param.data, 0, 1)

    def forward(self, X):
        '''
        :param X: [batch_size=1, seq_len, dim]
        :return:
        '''
        input = X  # input : [batch_size=1, seq_len, embedding_dim]
        input = input.transpose(0, 1)  # input : [seq_len, batch_size=1, embedding_dim]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size=1, n_hidden]
        # output : [seq_len, batch_size, n_hidden * num_directions(=2)]
        output, (final_hidden_state, final_cell_state) = self.lstm(input)
        output = output.transpose(0, 1)  # output : [batch_size, seq_len, n_hidden * num_directions(=2)]

        attn_output, attention = self.attention_net(output, final_hidden_state)
        return attn_output, attention  # attn_output : [batch_size, num_classes], attention : [batch_size, n_step]


class LSTM1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM1, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTMCell(input_size=input_size, \
                                hidden_size=hidden_size)
        self.decay = nn.Sequential(  # 时间衰减函数
            nn.Linear(in_features=1, out_features=self.hidden_size),
            nn.ReLU()
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.uniform_(m.bias.data, 0, 1)
            elif type(m) in [nn.GRU, nn.LSTM, nn.GRUCell, nn.LSTMCell]:
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'bias' in name:
                        nn.init.uniform_(param.data, 0, 1)

    def attention_net(self, lstm_output, final_state):
        # lstm_output : [batch_size=1, n_step, n_hidden * num_directions(=2)], F matrix
        # final_state : [num_layers(=1) * num_directions(=2), batch_size=1, n_hidden]

        hidden = final_state.view(1, -1, 1)
        # hidden = torch.cat((final_state[0], final_state[1]), dim=1).unsqueeze(2)
        # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        # attn_weights : [batch_size,n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)

        # context: [batch_size, n_hidden * num_directions(=2)]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights

    def forward(self, inputs):

        # dt: (time-1, )
        # x: (batch=1, time, dim)

        x, time = inputs
        dev = x.device
        dt = torch.cat([torch.tensor([0.0]).to(dev), time[1:] - time[:-1]], dim=0)

        device = x.device

        h0 = torch.tensor(np.random.normal(0, 0.1, \
                                           (1, self.hidden_size)), dtype=torch.float32).to(
            device)
        c0 = torch.tensor(np.random.normal(0, 0.1, \
                                           (1, self.hidden_size)), dtype=torch.float32).to(
            device)

        hs = []
        for i in range(dt.shape[0]):

            if i > 0:
                gamma = torch.exp(-self.decay(dt[i - 1].reshape(-1, 1)))  # (1) --> (hidden)
                h0 = h0 * gamma  # 时间衰减
            h1, c1 = self.lstm(x[:, i, :], (h0, c0))  # 第一层 GRU
            h0, c0 = h1, c1
            hs.append(h1)

        hs = torch.stack(hs, dim=1)  # batch, time, hidden
        out, _ = self.attention_net(hs, h1.unsqueeze(0))

        return out


class LSTM2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM2, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm1 = nn.LSTMCell(input_size=input_size + 1, \
                                 hidden_size=hidden_size)

        self.lstm2 = nn.LSTMCell(input_size=hidden_size, \
                                 hidden_size=hidden_size)

        self.lstm3 = nn.LSTMCell(input_size=hidden_size, \
                                 hidden_size=hidden_size)

        # self.time_embed = nn.Sequential(
        #     nn.Linear(1, input_size),
        #     nn.Tanh()
        # )

        self.init_weights()

    def attention_net(self, lstm_output, final_state):
        # lstm_output : [batch_size=1, n_step, n_hidden * num_directions(=2)], F matrix
        # final_state : [num_layers(=1) * num_directions(=2), batch_size=1, n_hidden]

        hidden = final_state.view(1, -1, 1)
        # hidden = torch.cat((final_state[0], final_state[1]), dim=1).unsqueeze(2)
        # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        # attn_weights : [batch_size,n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)

        # context: [batch_size, n_hidden * num_directions(=2)]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.uniform_(m.bias.data, 0, 1)
            elif type(m) in [nn.GRU, nn.LSTM, nn.LSTMCell, nn.GRUCell]:
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'bias' in name:
                        nn.init.uniform_(param.data, 0, 1)
            else:
                for param in m.parameters():
                    nn.init.normal_(param.data)

    def forward(self, inputs):
        # time: (time, )
        # x: (batch=1, time, dim)
        x, time = inputs

        device = x.device

        h0_1 = torch.tensor(np.random.normal(0, 0.1, (1, self.hidden_size)), dtype=torch.float32).to(
            device)
        c0_1 = torch.tensor(np.random.normal(0, 0.1, (1, self.hidden_size)), dtype=torch.float32).to(
            device)

        h0_2 = torch.tensor(np.random.normal(0, 0.1, (1, self.hidden_size)), dtype=torch.float32).to(
            device)
        c0_2 = torch.tensor(np.random.normal(0, 0.1, (1, self.hidden_size)), dtype=torch.float32).to(
            device)

        h0_3 = torch.tensor(np.random.normal(0, 0.1, (1, self.hidden_size)), dtype=torch.float32).to(
            device)
        c0_3 = torch.tensor(np.random.normal(0, 0.1, (1, self.hidden_size)), dtype=torch.float32).to(
            device)

        hs = []
        for i in range(time.shape[0]):
            x1 = torch.cat([x[:, i, :], time[i].view(1, -1)], dim=1)
            h1, c1 = self.lstm1(x1, (h0_1, c0_1))  # 第一层 GRU
            h0_1, c0_1 = h1, c1
            h1, c1 = self.lstm2(h1, (h0_2, c0_2))  # 第二层 GRU
            h0_2, c0_2 = h1, c1
            h1, c1 = self.lstm2(h1, (h0_3, c0_3))  # 第三层 GRU
            h0_3, c0_3 = h1, c1
            hs.append(h1)

        hs = torch.stack(hs, dim=1)  # batch, time, hidden
        out, _ = self.attention_net(hs, h1.unsqueeze(0))

        return out


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class HP_DATA:
    size_x_lab = 10
    size_v_lab_c = 10
    dim_x_lab = 10
    dim_v_lab = 10

    size_x_vit = 10
    size_v_vit_c = 10
    dim_x_vit = 10
    dim_v_vit = 10

    size_x_trt = 10
    dim_x_trt = 10

    size_x_state = 10
    dim_x_state = 10


class HP:
    """
    MultModel的超参数配置的类
    """

    orig_d_1 = 20
    orig_d_2 = 20
    orig_d_3 = 30
    orig_d_4 = 10

    d_1 = 30
    d_2 = 30
    d_3 = 30
    d_4 = 30

    only4 = True
    only3 = True
    only2 = True
    only1 = True

    num_heads = 1
    layers = 4

    attn_dropout_1 = 0.0
    attn_dropout_2 = 0.0
    attn_dropout_3 = 0.0
    attn_dropout_4 = 0.0

    relu_dropout = 0.0
    res_dropout = 0.0
    out_dropout = 0.0
    embed_dropout = 0.0

    MulT_output_dim = 30

    final_output_dim = 100


if __name__ == "__main__":
    import torch

    model = LSTM1(5, 10)
    x = torch.randn(1, 3, 5)
    t = torch.tensor([1, 2, 3], dtype=torch.float32)
    out = model([x, t])
    print(out)
