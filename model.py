from modeling_utils import BertSelfEncoder, BertCrossEncoder_AttnMap, BertPooler, BertLayerNorm
import torch.nn.functional as F
from transformers import RobertaModel, AutoConfig
import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import math

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True, gate_enabled=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        # initialize weights
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if bias:
            nn.init.zeros_(self.bias)
        # edge-wise gating flag and projection
        self.gate_enabled = gate_enabled
        if self.gate_enabled:
            # project node features before computing pairwise gate logits
            # project to same dimension to compute similarity
            self.gate_proj = nn.Linear(in_features, in_features)
            nn.init.xavier_uniform_(self.gate_proj.weight)
            if self.gate_proj.bias is not None:
                nn.init.zeros_(self.gate_proj.bias)

    def forward(self, text, adj):
        text = text.to(torch.float32)
        hidden = torch.matmul(text, self.weight)
        # optionally compute edge-wise gates (message gating)
        if self.gate_enabled:
            # text: [B, N, D]
            h_proj = self.gate_proj(text)  # [B, N, D]
            # pairwise similarity logits: [B, N, N]
            # use scaled dot-product
            gate_logits = torch.matmul(h_proj, h_proj.transpose(1, 2)) / (self.in_features ** 0.5)
            gate = torch.sigmoid(gate_logits)
            adj2 = adj * gate
        else:
            adj2 = adj
        denom = torch.sum(adj2, dim=2, keepdim=True) + 1
        output = torch.matmul(adj2, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GraphTransformerLayer(nn.Module):
    """Simplified Graph Transformer (Graphormer-style) layer.
    Supports additive + multiplicative edge modulation of attention logits
    and runs multi-head attention + FFN with layernorm residuals.
    """
    def __init__(self, hidden_size, num_heads=12, dropout=0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.all_head_size = self.num_heads * self.head_dim

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(self.all_head_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm1 = BertLayerNorm(hidden_size)
        self.norm2 = BertLayerNorm(hidden_size)

    def _transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_bias=None, attention_mask=None, return_attention=False):
        # hidden_states: [B, N, D]
        mixed_q = self.query(hidden_states)
        mixed_k = self.key(hidden_states)
        mixed_v = self.value(hidden_states)

        query_layer = self._transpose_for_scores(mixed_q)  # [B, heads, N, head_dim]
        key_layer = self._transpose_for_scores(mixed_k)
        value_layer = self._transpose_for_scores(mixed_v)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        if attention_bias is not None:
            # attention_bias shape: [B, num_heads, N, N] or [B, 1, N, N]
            if attention_bias.dim() == 3:
                attention_bias = attention_bias.unsqueeze(1)
            if attention_bias.size(1) == 1 and attention_scores.size(1) != 1:
                attention_bias = attention_bias.expand(-1, attention_scores.size(1), -1, -1)
            attention_bias = attention_bias.to(attention_scores.dtype)
            # Additive + multiplicative PHEB:
            # E_hat = E * sigmoid(B) + B
            attention_scores = attention_scores * torch.sigmoid(attention_bias) + attention_bias

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context = torch.matmul(attention_probs, value_layer)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_context_shape)

        attn_out = self.out(context)
        attn_out = self.proj_dropout(attn_out)
        hidden_states = self.norm1(attn_out + hidden_states)

        ff_out = self.ffn(hidden_states)
        hidden_states = self.norm2(ff_out + hidden_states)
        if return_attention:
            return hidden_states, attention_probs
        return hidden_states

class AHGT(nn.Module):
    def __init__(self, args, img_feat_dim=2048, num_image_regions=49):
        super().__init__()
        self.args = args
        self.img_feat_dim = img_feat_dim
        self.num_image_regions = num_image_regions
        config = AutoConfig.from_pretrained(args.roberta_model_dir)
        self.hidden_dim = config.hidden_size
        self.roberta = RobertaModel.from_pretrained(args.roberta_model_dir)
        self.target_roberta = RobertaModel.from_pretrained(args.roberta_model_dir)
        self.feat_linear = nn.Linear(self.img_feat_dim, self.hidden_dim)
        self.img_self_attn = BertSelfEncoder(config, layer_num=1)
        self.v2t = BertCrossEncoder_AttnMap(config, layer_num=1)
        self.dropout1 = nn.Dropout(0.3)
        self.gather = nn.Linear(self.hidden_dim, 1)
        self.dropout2 = nn.Dropout(0.3)
        self.pred = nn.Linear(self.num_image_regions, 2)
        # pred2 projects gathered per-aspect scores (length = max_seq_length) -> 2
        self.pred2 = nn.Linear(args.max_seq_length, 2)
        self.ce_loss = nn.CrossEntropyLoss()
        self.t2v = BertCrossEncoder_AttnMap(config, layer_num=1)
        #BertCrossEncoder_AttnMap返回两个值 return all_encoder_layers,all_attn_maps
        ##  基于方面的文本
        self.ta2t = BertCrossEncoder_AttnMap(config, layer_num=1)
        self.ta2tv_gcn = BertCrossEncoder_AttnMap(config, layer_num=1)
        self.senti_selfattn = BertSelfEncoder(config, layer_num=1)


        self.first_pooler = BertPooler(config)  # BertPooler 取hidden_states 第一个词 [batch_size,hidden_size];Linear(hidden_size,hidden_size);Tanh()激活
        self.senti_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.senti_detc  = nn.Linear(self.hidden_dim, 3)
        self.senti_detc2 = nn.Linear(self.hidden_dim * 3, 3)
        # enable edge-wise learnable gating if args.edge_gate == 1, default off
        self.use_edge_gate = getattr(args, 'edge_gate', 0) == 1
        self.gc1 = GraphConvolution(768, 768, gate_enabled=self.use_edge_gate)
        self.gc2 = GraphConvolution(768, 768, gate_enabled=self.use_edge_gate)
        # image positional embeddings (2D coords -> hidden_dim)
        # registers as buffer since grid is fixed for 7x7 regions
        grid_size = int(self.num_image_regions ** 0.5)
        coords = []
        for i in range(grid_size):
            for j in range(grid_size):
                # normalized center coordinates in [0,1]
                x = (j + 0.5) / grid_size
                y = (i + 0.5) / grid_size
                coords.append([x, y])
        coords = torch.tensor(coords, dtype=torch.float32)
        # shape: [num_image_regions, 2]
        self.register_buffer('img_grid', coords)
        self.img_pos_proj = nn.Linear(2, self.hidden_dim)
        # structural scalar (degree) projection -> hidden_dim
        self.struct_proj = nn.Linear(1, self.hidden_dim)
        self.cls_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.tanh = nn.Tanh()
        self.add49linear=nn.Linear(args.max_seq_length+self.num_image_regions,args.max_seq_length)
        self.linear49seq = nn.Linear(self.num_image_regions, args.max_seq_length)
        self.s2v=BertCrossEncoder_AttnMap(config, layer_num=1)
        # --- ACGC & PHEB modules ---
        # project cross-attention scalars to edge weights
        self.attn2edge = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # fusion weights for [attn_edge, static_adj]
        self.edge_fusion = nn.Parameter(torch.tensor([1.0, 1.0]))
        self.num_attention_heads = config.num_attention_heads
        # relation-specific MLPs: each relation gets its own projection to per-head bias
        self.relation_edge_projs = nn.ModuleDict({
            'tt': nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_attention_heads)
            ),
            'tv': nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_attention_heads)
            ),
            'vt': nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_attention_heads)
            ),
            'vv': nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_attention_heads)
            ),
        })
        # Graph Transformer layers (used when args.addGCN == 2)
        self.graph_transformer1 = GraphTransformerLayer(self.hidden_dim, num_heads=config.num_attention_heads, dropout=config.attention_probs_dropout_prob)
        self.graph_transformer2 = GraphTransformerLayer(self.hidden_dim, num_heads=config.num_attention_heads, dropout=config.attention_probs_dropout_prob)
        # global scale for edge->attention bias
        self.edge_bias_weight = nn.Parameter(torch.tensor(1.0))
        # S-TopK sparsification: keep top-k neighbors per node in cross-block (0 = disabled)
        self.s_topk = getattr(args, 's_topk', 0)

        # AGRF-style aspect-guided graph readout and gated fusion
        self.agrf_query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.agrf_key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.agrf_struct_gate = nn.Linear(self.hidden_dim + 1, 1)
        self.agrf_fuse = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.agrf_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.agrf_classifier = nn.Linear(self.hidden_dim, 3)
        
        self.init_weight()

    def _build_relation_attention_bias(self, adj_batch, seq):
        batch_size = adj_batch.size(0)
        total_len = adj_batch.size(1)
        attention_bias = torch.zeros(
            (batch_size, self.num_attention_heads, total_len, total_len),
            device=adj_batch.device,
            dtype=adj_batch.dtype,
        )

        tt_block = adj_batch[:, :seq, :seq].unsqueeze(-1)
        tv_block = adj_batch[:, :seq, seq:].unsqueeze(-1)
        vt_block = adj_batch[:, seq:, :seq].unsqueeze(-1)
        vv_block = adj_batch[:, seq:, seq:].unsqueeze(-1)

        tt_bias = self.relation_edge_projs['tt'](tt_block).permute(0, 3, 1, 2)
        tv_bias = self.relation_edge_projs['tv'](tv_block).permute(0, 3, 1, 2)
        vt_bias = self.relation_edge_projs['vt'](vt_block).permute(0, 3, 1, 2)
        vv_bias = self.relation_edge_projs['vv'](vv_block).permute(0, 3, 1, 2)

        attention_bias[:, :, :seq, :seq] = tt_bias
        attention_bias[:, :, :seq, seq:] = tv_bias
        attention_bias[:, :, seq:, :seq] = vt_bias
        attention_bias[:, :, seq:, seq:] = vv_bias
        return self.edge_bias_weight * attention_bias

    def _build_heterogeneous_graph(self, adj_batch, attn_probs, seq, device):
        batch_size = adj_batch.size(0)
        total_len = seq + self.num_image_regions
        hetero_adj = torch.zeros((batch_size, total_len, total_len), device=device, dtype=attn_probs.dtype)
        edge_type = torch.zeros((batch_size, total_len, total_len), device=device, dtype=torch.long)

        text_block = adj_batch[:, :seq, :seq].to(attn_probs.dtype)
        text_block_size = text_block.size(1)
        hetero_adj[:, :text_block_size, :text_block_size] = text_block
        edge_type[:, :text_block_size, :text_block_size] = (text_block > 0).long()

        if adj_batch.size(1) >= seq + self.num_image_regions:
            image_block = adj_batch[:, seq:seq + self.num_image_regions, seq:seq + self.num_image_regions].to(attn_probs.dtype)
        else:
            image_block = torch.eye(self.num_image_regions, device=device, dtype=attn_probs.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        hetero_adj[:, seq:, seq:] = image_block
        edge_type[:, seq:, seq:] = (image_block > 0).long() * 4

        e_cross = torch.sigmoid(self.attn2edge(attn_probs.unsqueeze(-1)).squeeze(-1))
        fused_cross = self.edge_fusion[0] * e_cross + self.edge_fusion[1] * adj_batch[:, :seq, seq:].to(attn_probs.dtype)
        fused_cross = torch.sigmoid(fused_cross)

        if getattr(self, 's_topk', 0) and self.s_topk > 0:
            k = min(self.s_topk, fused_cross.size(-1))
            if k < fused_cross.size(-1):
                _, topk_idx = torch.topk(fused_cross, k, dim=-1)
                sparse_mask = torch.zeros_like(fused_cross)
                sparse_mask.scatter_(-1, topk_idx, 1.0)
                fused_cross = fused_cross * sparse_mask

        hetero_adj[:, :seq, seq:] = fused_cross
        hetero_adj[:, seq:, :seq] = fused_cross.transpose(1, 2)
        edge_type[:, :seq, seq:] = (fused_cross > 0).long() * 2
        edge_type[:, seq:, :seq] = (fused_cross.transpose(1, 2) > 0).long() * 3

        return hetero_adj, edge_type, fused_cross

    def init_weight(self):
        ''' bert init
        '''
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)) and ('roberta' not in name):  # linear/embedding
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, BertLayerNorm) and ('roberta' not in name):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None and ('roberta' not in name):
                module.bias.data.zero_()

    def _masked_mean_pool(self, hidden_states, mask):
        mask = mask.to(hidden_states.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return (hidden_states * mask).sum(dim=1) / denom

    def _aspect_guided_graph_readout(self, graph_states, aspect_state, struct_signal=None):
        aspect_query = self.agrf_query(aspect_state).unsqueeze(1)
        node_key = self.agrf_key(graph_states)
        relation_score = torch.sum(aspect_query * node_key, dim=-1) / math.sqrt(self.hidden_dim)

        if struct_signal is None:
            struct_signal = torch.zeros(graph_states.size(0), graph_states.size(1), 1, device=graph_states.device, dtype=graph_states.dtype)
        elif struct_signal.dim() == 2:
            struct_signal = struct_signal.unsqueeze(-1)
        struct_gate = torch.sigmoid(self.agrf_struct_gate(torch.cat([graph_states, struct_signal.to(graph_states.dtype)], dim=-1))).squeeze(-1)

        beta = torch.softmax(relation_score * struct_gate, dim=-1)
        z_graph = torch.sum(beta.unsqueeze(-1) * graph_states, dim=1)
        return z_graph, beta, struct_gate

    def forward(self, img_id,
                input_ids, input_mask, target_ids,target_mask,img_feat,adj,
                return_intermediates=False):
        # input_ids,input_mask : [N, L]
        #             img_feat : [N, num_image_regions, img_feat_dim]


        # use model parameter device to avoid device mismatch
        device = next(self.parameters()).device
        batch_size, seq = input_ids.size()
        # text feature
        roberta_output = self.roberta(input_ids, input_mask)
        sentence_output = roberta_output.last_hidden_state

        # aspect feature
        target_roberta_output = self.target_roberta(target_ids, target_mask)
        target_output = target_roberta_output.last_hidden_state

        img_feat_real = img_feat.view(-1, self.img_feat_dim, self.num_image_regions).permute(0, 2, 1)
        img_feat_ = self.feat_linear(img_feat_real)  # [N, num_image_regions, 2048] ->[N, num_image_regions, 768]
        # add image positional embedding (2D coords projected to hidden_dim)
        # self.img_grid: [num_image_regions, 2]
        coords = self.img_grid.to(device).unsqueeze(0).expand(batch_size, -1, -1)  # [N, R, 2]
        pos_emb = self.img_pos_proj(coords)  # [N, R, D]
        img_feat_ = img_feat_ + pos_emb
        image_mask = torch.ones((batch_size, self.num_image_regions)).to(device)
        extended_image_mask = image_mask.unsqueeze(1).unsqueeze(2)
        extended_image_mask = extended_image_mask.to(dtype=next(self.parameters()).dtype)
        extended_image_mask = (1.0 - extended_image_mask) * -10000.0

        extended_sent_mask = input_mask.unsqueeze(1).unsqueeze(2)
        extended_sent_mask = extended_sent_mask.to(dtype=next(self.parameters()).dtype)
        extended_sent_mask = (1.0 - extended_sent_mask) * -10000.0

        extended_target_mask = target_mask.unsqueeze(1).unsqueeze(2)
        extended_target_mask = extended_target_mask.to(dtype=next(self.parameters()).dtype)
        extended_target_mask = (1.0 - extended_target_mask) * -10000.0

        target_aware_sentence, _ = self.ta2t(sentence_output,
                                            target_output,
                                            extended_target_mask,
                                            output_all_encoded_layers=False)
        target_aware_sentence = target_aware_sentence[-1]  # [N,l,768]

        target_aware_image, _ = self.t2v(target_aware_sentence,
                                         img_feat_,
                                         extended_image_mask,
                                         output_all_encoded_layers=False)  # image query sentence
        target_aware_image = target_aware_image[-1]  # [N,laspect,768]

        #把文本特征和图像特征在序列维度上拼接起来，通过senti_selfattn自注意力进行编码
        hs_hi_mixed_feature = torch.cat((sentence_output, img_feat_), dim=1)
        hs_hi_mask = torch.cat((input_mask, image_mask), dim=-1).to(device)
        extended_hs_hi_mask = hs_hi_mask.unsqueeze(1).unsqueeze(2)
        extended_hs_hi_mask = extended_hs_hi_mask.to(dtype=next(self.parameters()).dtype)
        extended_hs_hi_mask = (1.0 - extended_hs_hi_mask) * -10000.0
        # 对文本和图像拼接序列做自注意力编码，得到长度为 L+num_image_regions 的联合表示
        hs_hi_mixed_output = self.senti_selfattn(hs_hi_mixed_feature, extended_hs_hi_mask)  # [N, L+49, 768]
        hs_hi_mixed_output = hs_hi_mixed_output[-1]

        gathered_target_aware_image = self.gather(self.dropout1(
            target_aware_image)).squeeze(2)  # [N,la,768]->[N,la,1] ->[N,la]
        rel_pred = self.pred2(self.dropout2(
            gathered_target_aware_image))  # [N,2]

        gate = torch.softmax(rel_pred, dim=-1)[:, 1].unsqueeze(1).\
            expand(batch_size,self.args.max_seq_length).unsqueeze(2).\
            expand(batch_size,self.args.max_seq_length,self.hidden_dim)
        if self.args.addgate == 1:
            gated_target_aware_image = gate * target_aware_image  # [N,l,768]
        else:
            gated_target_aware_image = target_aware_image  # [N,l,768]
        # 现在我们希望在联合序列 (文本 + 图像) 上直接做构图（GCN），但 cross-block（text<->image）由 attention 权重决定
        # 首先确保 adj 是一个 batch 形式的 tensor: [batch_size, total_len, total_len]
        total_len = seq + self.num_image_regions
        # 如果 adj 是 numpy 或形状为 [total_len, total_len]，在此扩展为 batch
        if isinstance(adj, torch.Tensor):
            adj_batch = adj
        else:
            adj_batch = torch.tensor(adj, dtype=torch.float32).to(device)

        if adj_batch.dim() == 2:
            adj_batch = adj_batch.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        elif adj_batch.dim() == 3 and adj_batch.size(0) != batch_size:
            # 如果是单样本 adj，广播到 batch
            adj_batch = adj_batch.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

        # 计算 text <-> image attention（基于 roberta 输出的文本特征和线性映射后的图像特征）
        # 优化：优先使用已实现的 cross-attention 模块返回的 attn_map（self.v2t），
        # 以利用 query/key/value 的线性变换与多头计算；若发生异常则回退到原始点积计算
        # sentence_output: [N, L, H], img_feat_: [N, R, H]
        # 目标 attn_probs: [N, L, R]
        neg_inf = -1e9
        try:
            # self.v2t 返回 (all_encoder_layers, all_attn_maps)
            # NOTE: use target_aware_sentence as query so cross-attention (and resulting adjacency)
            # is conditioned on the aspect (target) -> adjacency becomes aspect-conditioned
            _, all_attn_maps = self.v2t(target_aware_sentence, img_feat_, extended_image_mask, output_all_encoded_layers=False)
            # 取最后一层的 attn_map
            attn_map = all_attn_maps[-1]
            # attn_map 可能的形状通常为 [N, L, R] 或 [N, 1, R] 等，做兼容处理
            attn = attn_map
            # 若 attn 有多余的单维度，尝试 squeeze 掉第一维或第二维
            if attn.dim() > 3:
                # 例如 [N, 1, 1, R] -> squeeze -> [N, 1, R]
                attn = attn.squeeze(1)
            # now attn should be [N, L, R] or [N, 1, R]
            # mask 文本和图像的有效位置（input_mask: [N, L], image_mask: [N, R])
            text_mask_bin = (input_mask > 0).to(dtype=attn.dtype)
            image_mask_bin = (image_mask > 0).to(dtype=attn.dtype)
            mask = text_mask_bin.unsqueeze(2) * image_mask_bin.unsqueeze(1)  # [N, L, R]
            # 如果 attn 为 [N, 1, R]（可能只对整个句子有一个query），扩展到每个文本 token 上
            if attn.size(1) == 1 and attn.size(2) == img_feat_.size(1):
                attn = attn.expand(-1, sentence_output.size(1), -1)
            # 将无效位置设为非常小值，再做 softmax 归一化
            attn = attn * mask + (1.0 - mask) * neg_inf
            attn_probs = torch.softmax(attn, dim=-1)
        except Exception:
            # 回退到原始点积计算（与之前实现保持兼容）
            # Use target_aware_sentence here as well to keep adjacency aspect-conditioned
            attn_scores = torch.matmul(target_aware_sentence, img_feat_.transpose(1, 2))
            text_mask_bin = (input_mask > 0).to(dtype=attn_scores.dtype)
            image_mask_bin = (image_mask > 0).to(dtype=attn_scores.dtype)
            mask = text_mask_bin.unsqueeze(2) * image_mask_bin.unsqueeze(1)  # [N, L, R]
            attn_scores = attn_scores * mask + (1.0 - mask) * neg_inf
            attn_probs = torch.softmax(attn_scores, dim=-1)  # text -> image 权重 [N, L, R]

        # Materialize an explicit heterogeneous graph before graph reasoning.
        adj_batch, edge_type_batch, fused_cross = self._build_heterogeneous_graph(adj_batch, attn_probs, seq, device)

        # 用构造好的 adj_batch 运行 GCN
        # compute simple structural feature (degree) and add to node features to decouple position vs structure
        try:
            deg = adj_batch.sum(dim=-1)  # [B, total_len]
            struct_feat = self.struct_proj(deg.unsqueeze(-1))  # [B, total_len, D]
            # add structural embedding to the joint self-attention output
            hs_hi_mixed_output = hs_hi_mixed_output + struct_feat
        except Exception:
            pass

        intermediates = {}

        if return_intermediates:
            intermediates['joint_pre_graph'] = hs_hi_mixed_output
            intermediates['hetero_edge_type'] = edge_type_batch
            intermediates['hetero_cross_block'] = fused_cross

        # If addGCN == 2, use Graph Transformer with per-head edge bias (PHEB)
        if getattr(self.args, 'addGCN', 1) == 2:
            # build relation-specific per-head bias from the three edge blocks
            attention_bias = self._build_relation_attention_bias(adj_batch, seq)
            pheb_mode = getattr(self.args, 'pheb_mode', 'full')
            if pheb_mode == 'zero':
                attention_bias = None
            elif pheb_mode == 'shared':
                attention_bias = attention_bias.mean(dim=1, keepdim=True)
            # apply GraphTransformer layers
            hs_hi_mixed_output, attention_probs_1 = self.graph_transformer1(
                hs_hi_mixed_output,
                attention_bias=attention_bias,
                attention_mask=extended_hs_hi_mask,
                return_attention=return_intermediates,
            )
            if return_intermediates:
                intermediates['graph_transformer_1'] = hs_hi_mixed_output
                intermediates['graph_attention_1'] = attention_probs_1
            target_aware_image_gcn, attention_probs_2 = self.graph_transformer2(
                hs_hi_mixed_output,
                attention_bias=attention_bias,
                attention_mask=extended_hs_hi_mask,
                return_attention=return_intermediates,
            )
            if return_intermediates:
                intermediates['graph_transformer_2'] = target_aware_image_gcn
                intermediates['graph_attention_2'] = attention_probs_2
        elif getattr(self.args, 'addGCN', 1) == 1:
            hs_hi_mixed_output = F.relu(self.gc1(hs_hi_mixed_output, adj_batch))
            if return_intermediates:
                intermediates['graph_transformer_1'] = hs_hi_mixed_output
            target_aware_image_gcn = F.relu(self.gc2(hs_hi_mixed_output, adj_batch))
        else:
            target_aware_image_gcn = hs_hi_mixed_output

        if return_intermediates:
            intermediates['graph_transformer_2'] = target_aware_image_gcn

        # AGRF-style readout on the full graph before slicing back to text nodes.
        aspect_state = self._masked_mean_pool(target_output, target_mask)
        graph_struct_signal = adj_batch.sum(dim=-1)
        z_graph, graph_beta, graph_struct_gate = self._aspect_guided_graph_readout(
            target_aware_image_gcn,
            aspect_state,
            struct_signal=graph_struct_signal,
        )

        pooled_text = self._masked_mean_pool(target_aware_sentence, input_mask)
        pooled_image = self._masked_mean_pool(gated_target_aware_image, input_mask)
        fusion_input = torch.cat([pooled_text, pooled_image, z_graph], dim=-1)
        fuse_gate = torch.sigmoid(self.agrf_fuse(fusion_input))
        fused_representation = fuse_gate * z_graph + (1.0 - fuse_gate) * (pooled_text + pooled_image)

        # 将联合节点的表示切回到文本部分（前 seq 个位置），以保持后续接口不变
        # seq 为文本长度 (max_seq_len)
        target_aware_image_gcn = target_aware_image_gcn[:, :seq, :]
        # s2 here corresponds to tokens of length `seq` (text part), so use extended_sent_mask
        target_aware_image_gcn_asi, _ = self.ta2tv_gcn(target_aware_sentence,
                               target_aware_image_gcn,
                               extended_sent_mask)
        target_aware_image_gcn_asi = target_aware_image_gcn_asi[-1]  #[N,l,768]

        if return_intermediates:
            intermediates['agrf_graph_beta'] = graph_beta
            intermediates['agrf_graph_struct_gate'] = graph_struct_gate
            intermediates['agrf_fuse_gate'] = fuse_gate
            intermediates['agrf_z_graph'] = z_graph
            intermediates['agrf_fused_representation'] = fused_representation

        # Keep the original branch available as a fallback, but prefer the AGRF path.
        senti_pooled_output = self.agrf_dropout(fused_representation)
        senti_pred = self.agrf_classifier(senti_pooled_output)
        if return_intermediates:
            intermediates['text_gcn_output'] = target_aware_image_gcn
            intermediates['text_aware_image_gcn_asi'] = target_aware_image_gcn_asi
            return senti_pred, intermediates
        return senti_pred



