"""TranAD and GDN, using the authors' official model code.

TranAD: model, layers and training objective copied from https://github.com/imperial-qore/TranAD
        (src/models.py, src/dlutils.py, main.py; BSD 3-Clause, (c) 2022 Shreshth Tuli).
GDN:    model and graph layer copied from https://github.com/d-ailin/GDN
        (models/GDN.py, models/graph_layer.py; MIT, (c) 2021 d-ailin).
Only device handling and the fit/score wrapper used by this benchmark were added.
Scores are causal: the score at time t uses only data up to t.
"""
import math, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view as swv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros

SEED = int(os.environ.get("SEED", 0)); torch.set_num_threads(2)
N_TRAIN = 30000


def _windows(X, w):
    Xp = np.concatenate([np.repeat(X[:1], w - 1, 0), X]).astype(np.float32)
    return swv(Xp, w, axis=0).transpose(0, 2, 1)          # (T, w, d)


# ----------------------------- TranAD (official) -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__(); self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term); pe += torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))
    def forward(self, x, pos=0):
        return self.dropout(x + self.pe[pos:pos + x.size(0), :])


class TEncLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward); self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout); self.activation = nn.LeakyReLU(True)
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None):
        src = src + self.dropout1(self.self_attn(src, src, src)[0])
        return src + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(src)))))


class TDecLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward); self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout); self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, tgt_is_causal=None, memory_is_causal=None):
        tgt = tgt + self.dropout1(self.self_attn(tgt, tgt, tgt)[0])
        tgt = tgt + self.dropout2(self.multihead_attn(tgt, memory, memory)[0])
        return tgt + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(tgt)))))


class TranADNet(nn.Module):
    def __init__(self, feats, n_window=10):
        super().__init__()
        self.n_feats = feats; self.n_window = n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, n_window)
        self.transformer_encoder = nn.TransformerEncoder(TEncLayer(2 * feats, feats, 16, 0.1), 1, enable_nested_tensor=False)
        self.transformer_decoder1 = nn.TransformerDecoder(TDecLayer(2 * feats, feats, 16, 0.1), 1)
        self.transformer_decoder2 = nn.TransformerDecoder(TDecLayer(2 * feats, feats, 16, 0.1), 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())
    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2) * math.sqrt(self.n_feats)
        memory = self.transformer_encoder(self.pos_encoder(src))
        return tgt.repeat(1, 1, 2), memory
    def forward(self, src, tgt):
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


class TranAD:
    """Official TranAD; window 10, AdamW + StepLR(5, 0.9), 5 epochs, batch 128, two-phase loss (as in main.py)."""
    name = "TranAD"; W = 10; EPOCHS = 5; LR = 1e-3; BS = 128
    def fit(self, X):
        torch.manual_seed(SEED)
        Wn = _windows(X, self.W); idx = np.random.default_rng(SEED).choice(len(Wn), min(N_TRAIN, len(Wn)), replace=False)
        D = torch.from_numpy(np.ascontiguousarray(Wn[idx]))
        self.net = TranADNet(X.shape[1], self.W)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.LR, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.StepLR(opt, 5, 0.9); l = nn.MSELoss(reduction="none")
        self.net.train()
        for ep in range(1, self.EPOCHS + 1):
            n = ep; perm = torch.randperm(len(D))
            for i in range(0, len(D), self.BS):
                d = D[perm[i:i + self.BS]]; window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, d.shape[0], -1)
                z = self.net(window, elem)
                loss = torch.mean((1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem))
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
        return self

    @torch.no_grad()
    def score(self, X):
        self.net.eval(); Wn = _windows(X, self.W); out = np.empty(len(Wn), np.float32)
        for i in range(0, len(Wn), 4096):
            d = torch.from_numpy(np.ascontiguousarray(Wn[i:i + 4096])); window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, d.shape[0], -1)
            z = self.net(window, elem)[1]
            out[i:i + 4096] = ((z - elem) ** 2)[0].sum(1).numpy()
        return out


# ------------------------------- GDN (official) ------------------------------
def get_batch_edge_index(org_edge_index, batch_num, node_num):
    edge_index = org_edge_index.clone().detach(); edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()
    for i in range(batch_num):
        batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num
    return batch_edge_index.long()


class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True, inter_dim=-1, **kwargs):
        super().__init__(aggr='add', node_dim=0, **kwargs)  # node_dim=0 reproduces the PyG 1.x default the original code relied on
        self.in_channels, self.out_channels, self.heads, self.concat = in_channels, out_channels, heads, concat
        self.negative_slope, self.dropout = negative_slope, dropout; self._alpha = None
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_i = nn.Parameter(torch.Tensor(1, heads, out_channels)); self.att_j = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = nn.Parameter(torch.Tensor(1, heads, out_channels)); self.att_em_j = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.bias = nn.Parameter(torch.Tensor(heads * out_channels if concat else out_channels)) if bias else None
        glorot(self.lin.weight); glorot(self.att_i); glorot(self.att_j); zeros(self.att_em_i); zeros(self.att_em_j); zeros(self.bias)

    def forward(self, x, edge_index, embedding):
        x = self.lin(x); x = (x, x)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x[1].size(self.node_dim))
        out = self.propagate(edge_index, x=x, embedding=embedding, edges=edge_index)
        out = out.view(-1, self.heads * self.out_channels) if self.concat else out.mean(dim=1)
        return out + self.bias if self.bias is not None else out

    def message(self, x_i, x_j, edge_index_i, size_i, embedding, edges):
        x_i = x_i.view(-1, self.heads, self.out_channels); x_j = x_j.view(-1, self.heads, self.out_channels)
        embedding_i = embedding[edge_index_i].unsqueeze(1).repeat(1, self.heads, 1)
        embedding_j = embedding[edges[0]].unsqueeze(1).repeat(1, self.heads, 1)
        key_i = torch.cat((x_i, embedding_i), dim=-1); key_j = torch.cat((x_j, embedding_j), dim=-1)
        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1); cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)
        alpha = ((key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)).view(-1, self.heads, 1)
        alpha = softmax(F.leaky_relu(alpha, self.negative_slope), edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.view(-1, self.heads, 1)


class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num=512):
        super().__init__(); modules = []
        for i in range(layer_num):
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))
            else:
                modules += [nn.Linear(in_num if i == 0 else inter_num, inter_num), nn.BatchNorm1d(inter_num), nn.ReLU()]
        self.mlp = nn.ModuleList(modules)
    def forward(self, x):
        out = x
        for mod in self.mlp:
            out = mod(out.permute(0, 2, 1)).permute(0, 2, 1) if isinstance(mod, nn.BatchNorm1d) else mod(out)
        return out


class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1):
        super().__init__(); self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)
        self.bn = nn.BatchNorm1d(out_channel); self.relu = nn.ReLU()
    def forward(self, x, edge_index, embedding=None):
        return self.relu(self.bn(self.gnn(x, edge_index, embedding)))


class GDNNet(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=20):
        super().__init__()
        self.edge_index_sets = edge_index_sets; embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim); self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)
        self.gnn_layers = nn.ModuleList([GNNLayer(input_dim, dim, inter_dim=dim + embed_dim, heads=1) for _ in edge_index_sets])
        self.topk = topk
        self.out_layer = OutLayer(dim * len(edge_index_sets), node_num, out_layer_num, inter_num=out_layer_inter_dim)
        self.dp = nn.Dropout(0.2); nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))
    def forward(self, data):
        x = data.clone().detach(); batch_num, node_num, all_feature = x.shape; x = x.view(-1, all_feature).contiguous()
        gcn_outs = []
        for i, _ in enumerate(self.edge_index_sets):
            all_embeddings = self.embedding(torch.arange(node_num))
            weights = all_embeddings.detach().clone().view(node_num, -1); all_embeddings = all_embeddings.repeat(batch_num, 1)
            cos_ji_mat = torch.matmul(weights, weights.T) / torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
            topk_indices_ji = torch.topk(cos_ji_mat, self.topk, dim=-1)[1]
            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, self.topk).flatten().unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            batch_gated_edge_index = get_batch_edge_index(torch.cat((gated_j, gated_i), dim=0), batch_num, node_num)
            gcn_outs.append(self.gnn_layers[i](x, batch_gated_edge_index, embedding=all_embeddings))
        x = torch.cat(gcn_outs, dim=1).view(batch_num, node_num, -1)
        out = torch.mul(x, self.embedding(torch.arange(0, node_num)))
        out = F.relu(self.bn_outlayer_in(out.permute(0, 2, 1))).permute(0, 2, 1)
        return self.out_layer(self.dp(out)).view(-1, node_num)


class GDN:
    """Official GDN (forecasting next value from a 5-step window). Hyperparameters follow the authors' SWaT setting
    (slide_win 5, dim 64, topk 15, out_layer_inter_dim 128, lr 1e-3) with 5 epochs on 20,000 sampled windows (CPU budget).
    Anomaly score: per-sensor absolute error normalized by the median and IQR of training errors, max over sensors (as in the paper)."""
    name = "GDN"; W = 5; EPOCHS = 5; BS = 128; NTR = 20000
    def fit(self, X):
        torch.manual_seed(SEED); n = X.shape[1]
        fc = torch.tensor([[j for i in range(n) for j in range(n) if i != j], [i for i in range(n) for j in range(n) if i != j]])
        self.net = GDNNet([fc], n, dim=64, out_layer_inter_dim=128, input_dim=self.W, out_layer_num=1, topk=min(15, n - 1))
        Wn = _windows(X, self.W + 1); idx = np.random.default_rng(SEED).choice(len(Wn) - self.W, min(self.NTR, len(Wn) - self.W), replace=False) + self.W
        D = torch.from_numpy(np.ascontiguousarray(Wn[idx]))           # (N, W+1, d): first W = input, last = target
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=0); self.net.train()
        for _ in range(self.EPOCHS):
            perm = torch.randperm(len(D))
            for i in range(0, len(D), self.BS):
                d = D[perm[i:i + self.BS]]
                pred = self.net(d[:, :-1, :].permute(0, 2, 1).contiguous())
                loss = F.mse_loss(pred, d[:, -1, :]); opt.zero_grad(); loss.backward(); opt.step()
        e = self._err(X[:60000]); self.med = np.median(e, 0); iqr = np.percentile(e, 75, 0) - np.percentile(e, 25, 0)
        self.iqr = np.where(iqr > 1e-6, iqr, 1e-6)
        return self

    @torch.no_grad()
    def _err(self, X):
        self.net.eval(); Wn = _windows(X, self.W + 1); out = np.empty((len(Wn), X.shape[1]), np.float32)
        for i in range(0, len(Wn), 256):                      # small batches: graph messages scale with batch x nodes x top-k
            d = torch.from_numpy(np.ascontiguousarray(Wn[i:i + 256]))
            out[i:i + 256] = (self.net(d[:, :-1, :].permute(0, 2, 1).contiguous()) - d[:, -1, :]).abs().numpy()
        return out

    def score(self, X):
        return ((self._err(X) - self.med) / self.iqr).max(1)


OFFICIAL = [TranAD, GDN]
