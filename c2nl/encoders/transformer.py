"""
Implementation of "Attention is All You Need"
"""
from c2nl.inputters import constants
import torch.nn as nn
from c2nl.modules.embeddings import Embeddings
from c2nl.modules.util_class import LayerNorm
from c2nl.modules.multi_head_attn import MultiHeadedAttention
from c2nl.modules.position_ffn import PositionwiseFeedForward
from c2nl.encoders.encoder import EncoderBase
from c2nl.utils.misc import sequence_mask
import torch
import numpy as np
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_dense_batch

class ASTEnc(nn.Module):
    def __init__(self,
                 node_voc_size,
                 pos_voc_size,
                 emb_dims,
                 hid_dims,
                 gnn_layers,
                 GNN=SAGEConv,
                 aggr='add',
                 drop_rate=0.2,
                 **kwargs):
        '''
        AST编码器
        :param node_voc_size: the size of node vocabulary
        :param pos_voc_size:  the size of position vocabulary
        :param emb_dims: the dims of embebddings
        :param hid_dims: the dims of hidden layer
        :param gnn_layers: the number of layers
        :param GNN: GNN module
        :param aggr: the mode of aggregation
        :param drop_rate: the dropout rate
        :param kwargs: others
        '''
        super().__init__()
        # kwargs.setdefault('init_emb','None')
        # kwargs.setdefault('batch','None')   #GraphData.batch to_dense_data用的,放到forward里了
        kwargs.setdefault('pad_idx', 0)  # GraphData.batch to_dense_data用的
        kwargs.setdefault('ast_max_size', None)  # GraphData.batch to_dense_data用的
        # self.node_embedding = Embeddings(emb_dims,
        #                                       node_voc_size,
        #                                       constants.PAD)
        self.emb_dims = emb_dims
        self.pad_idx = 0
        # self.batch=kwargs['batch']
        self.ast_max_size = 400
        # self.node_embedding = nn.Embedding(node_voc_size, emb_dims, padding_idx=self.pad_idx)  # node embedding层
        # nn.init.xavier_uniform_(self.node_embedding.weight[1:,])
        self.pos_embedding = nn.Embedding(pos_voc_size, emb_dims, padding_idx=self.pad_idx)
        nn.init.xavier_uniform_(self.pos_embedding.weight[1:, ])
        self.emb_layer_norm = nn.LayerNorm(emb_dims)
        self.gnn_layers = gnn_layers
        gnn1=GNN(in_channels=emb_dims, out_channels=hid_dims, aggr=aggr)
        gnn2=GNN(in_channels=hid_dims, out_channels=emb_dims, aggr=aggr)
        if gnn_layers==2:
            self.gnns=nn.ModuleList([gnn1,gnn2])
            self.hid_layer_norms=nn.ModuleList([nn.LayerNorm(hid_dims),nn.LayerNorm(emb_dims),])
        if self.gnn_layers>2:
            self.gnns=nn.ModuleList([gnn1]+[GNN(in_channels=hid_dims, out_channels=hid_dims, aggr=aggr)
                                            for _ in range(gnn_layers-2)]+[gnn2])
            self.hid_layer_norms=nn.ModuleList([nn.LayerNorm(hid_dims) for _ in range(gnn_layers-1)]+[nn.LayerNorm(emb_dims)])
        self.relus=nn.ModuleList([nn.Sequential(nn.ReLU(),nn.Dropout(p=drop_rate)) for _ in range(gnn_layers)])
        # self.linear=nn.Linear(hid_dims, emb_dims)
        self.out_layer_norm=nn.LayerNorm(emb_dims)
        # self.gnns = nn.ModuleList(
        #     [GNN(in_channels=emb_dims, out_channels=hid_dims, aggr=aggr) for _ in range(gnn_layers)])
        # self.layer_norms = nn.ModuleList([nn.LayerNorm(emb_dims) for _ in range(gnn_layers)])
        # self.linears = nn.ModuleList([nn.Linear(hid_dims, emb_dims) for _ in range(gnn_layers)])
        self.dropout = nn.Dropout(p=drop_rate)
        # self.layer_norm=nn.LayerNorm(out_dims, elementwise_affine=True)
        self.node_embedding = nn.Embedding(node_voc_size,
                                           emb_dims)
    def forward(self, node_emb, pos, edge, node_batch=None,node_max_len=400):
        '''
        :param node:
        :param pos:
        :param edge:
        :param batch: [batch_node_num,]
        :return:
        '''
        # assert len(node_emb.size()) == 2  # node是一堆节点序号[batch_node_num,emb_dims]
        # assert len(pos.size()) == 1  # pos是一堆节点序号[batch_node_num,]
        # assert len(edge.size()) == 2  # 点是一堆节点序号[2,all_batch_edge_num]
        node_emb = self.node_embedding(node_emb)
        node_emb = node_emb * np.sqrt(self.emb_dims)  # [batch_node_num,emb_dims]
        pos_emb = self.pos_embedding(pos)  # [batch_node_num,emb_dims]
        node_enc = self.dropout(node_emb+pos_emb)  # [batch_node_num,emb_dims]
        node_enc = self.emb_layer_norm(node_enc)  # [batch_node_num,emb_dims]

        for i in range(self.gnn_layers):
            node_enc_=self.gnns[i](x=node_enc,edge_index=edge)   # [batch_node_num,hid_dims]
            node_enc_=self.relus[i](node_enc_)    # [batch_node_num,hid_dims]
            node_enc=self.hid_layer_norms[i](node_enc_.add(node_enc))  # [batch_node_num,hid_dims]
        # node_enc=self.linear(node_enc)
        if node_batch is not None:
            node_enc = to_dense_batch(node_enc,
                                      batch=node_batch,
                                      fill_value=self.pad_idx,
                                      max_num_nodes=node_max_len)[0]  # [batch_ast_num,ast_max_size,emb_dims],必须要用[0]，不然是个tuple
            # node_enc=self.out_layer_norm(node_enc)
        return node_enc  # [batch_node_num,emb_dims] 或者 #[batch_ast_num,ast_max_size,emb_dims]



class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.
    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self,
                 d_model,
                 heads,
                 d_ff,
                 d_k,
                 d_v,
                 dropout,
                 max_relative_positions=0,
                 use_neg_dist=True
                 ):
        super(TransformerEncoderLayer, self).__init__()

        self.attention = MultiHeadedAttention(heads,
                                              d_model,
                                              d_k,
                                              d_v,
                                              dropout=dropout,
                                              max_relative_positions=max_relative_positions,
                                              use_neg_dist=use_neg_dist)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, inputs, mask, adj=None,map=None,src_enc=None):
        """
        Transformer Encoder Layer definition.
        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`
        Returns:
            (`FloatTensor`):
            * outputs `[batch_size x src_len x model_dim]`
        """
        # _inputs = inputs
        # if map is not None and src_enc is not None:
        #      for i, m in enumerate(map):
        #          for ast_idx,src_idx in m.items():
        #             try:
        #                 inputs[i,int(ast_idx)] = src_enc[i,id] + inputs[i,int(ast_idx)]
        #             except:
        #                 pass
        # inputs = self.layer_norm(self.dropout(inputs) + _inputs)
        context, attn_per_head, _ = self.attention(inputs, inputs, inputs,
                                                   mask=mask, attn_type="self",adj=adj)
        out = self.layer_norm(self.dropout(context) + inputs)
        out = self.feed_forward(out)

        return out, attn_per_head


class TransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".
    .. mermaid::
       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O
    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
    Returns:
        (`FloatTensor`, `FloatTensor`):
        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self,
                 num_layers,
                 d_model=512,
                 heads=8,
                 d_ff=2048,
                 d_k=64,
                 d_v=64,
                 dropout=0.2,
                 max_relative_positions=0,
                 use_neg_dist=True
                 ):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        if isinstance(max_relative_positions, int):
            max_relative_positions = [max_relative_positions] * self.num_layers
        assert len(max_relative_positions) == self.num_layers

        assert num_layers % 2 == 0
        self.v_layer = nn.ModuleList(
            [TransformerEncoderLayer(d_model,
                                     heads,
                                     d_ff,
                                     d_k,
                                     d_v,
                                     dropout,
                                     max_relative_positions=max_relative_positions[i],
                                     use_neg_dist=use_neg_dist
                                     )
             for i in range(num_layers// 2)]) # // 2
        self.s_layer = nn.ModuleList(
            [TransformerEncoderLayer(d_model,
                                     heads,
                                     d_ff,
                                     d_k,
                                     d_v,
                                     dropout,
                                     max_relative_positions=max_relative_positions[i],
                                     use_neg_dist=use_neg_dist)
             for i in range(num_layers // 2)])

    def count_parameters(self):
        params = list(self.v_layer.parameters()) + list(self.s_layer.parameters())
        return sum(p.numel() for p in params if p.requires_grad)

    def forward(self, src, lengths=None, adjacency=None,map=None,src_enc=None):
        """
        Args:
            src (`FloatTensor`): `[batch_size x src_len x model_dim]`
            lengths (`LongTensor`): length of each sequence `[batch]`
            adjacency (`LongTensor`): `[batch_size x src_len x src_len]`
        Returns:
            (`FloatTensor`):
            * outputs `[batch_size x src_len x model_dim]`
        """
        self._check_args(src, lengths)
        out = src
        # out1 = src[:,:,:src.size(-1)//2]
        # out2 = src[:,:,src.size(-1)//2:]
        mask = None if lengths is None else \
            ~sequence_mask(lengths, out.shape[1]).unsqueeze(1)
        # mask_m = None if m_lengths is None else \
        #     ~sequence_mask(m_lengths, m.shape[1]).unsqueeze(1)
        adj = None
        if adjacency is not None:
            adj = adjacency[:, :mask.shape[2], :mask.shape[2]]
            # 截取 adjacency 的子张量，使其与 mask 有相同的维度。然后，通过将邻接矩阵转换为布尔型张量，
            # 使用 ~ 操作符取反，再与 mask 取按位与，最终得到一个经过处理的邻接矩阵。
            # 这个处理的目的是将填充位置对应的邻接信息屏蔽掉，以防止在模型计算中产生不必要的影响。
            # adjacency = ~(adjacency[:, :mask.shape[2], :mask.shape[2]].bool() * ~mask)
            adjacency = ~(adjacency[:, :mask.shape[2], :mask.shape[2]] > 0 * ~mask)
        # Run the forward pass of every layer of the tranformer.
        representations = []
        attention_scores = []
        for i in range(self.num_layers // 2): #
            _out, attn_per_head = self.v_layer[i](out, mask, adj=adj,map=map,src_enc=src_enc)
            representations.append(out)
            attention_scores.append(attn_per_head)

            out, attn_per_head = self.s_layer[i](_out, mask, adj=adj,map=map,src_enc=src_enc) #adjacency
            # out = torch.cat((out1,out2),dim=-1)
            # attn_per_head = torch.cat((attn_per_head1,attn_per_head2),dim=-1)
            out = out + _out
            representations.append(out)
            attention_scores.append(attn_per_head)

        return representations, attention_scores
