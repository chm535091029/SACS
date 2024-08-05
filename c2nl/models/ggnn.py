import torch
import torch.nn as nn
import torch.nn.functional as F

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """

    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        A_in = A[:, :, :self.n_node * self.n_edge_types]
        A_out = A[:, :, self.n_node * self.n_edge_types:]

        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output

class GGNN(nn.Module):#GGNN-cfg
    """
    Gated Graph Sequence Neural Networks (GGNN)
    """

    def __init__(self, arg):
        super(GGNN, self).__init__()

        assert (arg.state_dim >= arg.annotation_dim,
                'state_dim must be no less than annotation_dim')

        # self.config = config

        self.device = torch.device(f"cuda:{arg.gpu_id}" if torch.cuda.is_available() else "cpu")

        self.vocab_size = arg.src_vocab_size
        self.annotation_dim = arg.annotation_dim
        self.state_dim = arg.state_dim
        self.n_edge_types = arg.n_edge_types
        self.n_node = arg.n_node
        self.n_steps = arg.n_steps
        self.word_split = arg.word_split
        self.pooling_type = arg.pooling_type
        self.output_type = arg.output_type
        self.batch_size = arg.batch_size
        self.max_word_num = arg.max_word_num

        self.embedding = nn.Embedding(self.vocab_size, self.annotation_dim, padding_idx=0)

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propogation Model
        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

        # Output Model
        self.out_mlp = nn.Sequential(
            nn.Dropout(p=arg.dropout, inplace=False),
            nn.Linear(self.state_dim + self.annotation_dim, self.state_dim),
            nn.Tanh(),
        )

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self.init_xavier_linear(m)

    def init_xavier_linear(self, linear, init_bias=True, gain=1, init_normal_std=1e-4):
        torch.nn.init.xavier_uniform_(linear.weight, gain)
        if init_bias:
            if linear.bias is not None:
                linear.bias.data.normal_(std=init_normal_std)

    def forward(self, annotation, A, graph_node_mask):

        # annotation: [batch_size x n_node x max_word_num_one_node] -> [batch_size x n_node x annotation_dim]
        if self.word_split:
            if self.pooling_type == 'max_pooling':
                annotation = self.embedding(annotation)
                annotation = F.max_pool2d(annotation, kernel_size=(self.max_word_num, 1), stride=1).squeeze(2)
            else:  # 'ave_pooling'
                annotation = self.embedding(annotation)
                annotation = F.avg_pool2d(annotation, kernel_size=(self.max_word_num, 1), stride=1).squeeze(2)

        # annotation: [batch_size x n_node] -> [batch_size x n_node x annotation_dim]
        else:
            annotation = self.embedding(annotation)

        # prop_state: [batch_size x n_node x state_dim]
        padding = torch.zeros(len(annotation), self.n_node, self.state_dim - self.annotation_dim).float().to(
            self.device)
        prop_state = torch.cat((annotation, padding), 2).to(self.device)
        # A: [batch_size x n_node x (n_node * n_edge_types * 2)]

        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            # before in_states: [n_edge_types x batch_size x n_node x state_dim] -> [batch_size x n_edge_types x ...]
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            # after in_states: [batch_size x (n_node * n_edge_types) x state_dim]
            in_states = in_states.reshape(-1, self.n_node * self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.reshape(-1, self.n_node * self.n_edge_types, self.state_dim)

            # prop_state: [batch_size x n_node x state_dim]
            prop_state = self.propogator(in_states, out_states, prop_state, A)

        # when output_type is 'no_reduce'
        if self.output_type == "no_reduce":
            join_state = torch.cat((prop_state, annotation), 2)
            output = self.out_mlp(join_state)
        elif self.output_type == "sum":
            prop_state_cat = torch.cat((prop_state, annotation), 2)

            prop_state_to_sum = self.out_mlp(prop_state_cat)
            output_list = []

            for _i in range(prop_state.size()[0]):
                before_out = torch.masked_select(prop_state_to_sum[_i, :, :].reshape(1, -1, self.state_dim),
                                                 graph_node_mask[_i].bool().reshape(1, -1, 1)).reshape(1, -1,
                                                                                                       self.state_dim)
                out_to_append = torch.tanh(torch.sum(before_out, 1)).reshape(1, self.state_dim)
                output_list.append(out_to_append)
            output = torch.cat(output_list, 0).reshape(1, -1, self.state_dim)

        # output: [batch_size x n_node x state_dim]
        return output