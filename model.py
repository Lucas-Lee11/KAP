import torch
from torch_geometric.nn import DenseGCNConv, DenseGATConv, DenseSAGEConv
from torch_geometric.nn import dense_diff_pool
from torch.nn import BatchNorm1d
import torch.nn.functional as F

from math import ceil



class DenseGNN(torch.nn.Module):
    def __init__(self, conv_type, num_layers, in_channels, hidden_channels, out_channels, skip=False, pred=False, **kwargs):
        super(DenseGNN, self).__init__()

        self.num_layers = num_layers
        self.conv_type = conv_type
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.skip = skip
        self.pred = pred

        if conv_type == 'GCN':
            self.convs.append(DenseGCNConv(in_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(DenseGCNConv(hidden_channels, hidden_channels))
                self.bns.append(BatchNorm1d(hidden_channels))
            # self.convs.append(DenseGCNConv(hidden_channels, out_channels))
            self.convs.append(DenseGCNConv(hidden_channels, hidden_channels))
        elif conv_type == 'SAGE':
            self.convs.append(DenseSAGEConv(in_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(DenseSAGEConv(hidden_channels, hidden_channels))
                self.bns.append(BatchNorm1d(hidden_channels))
            # self.convs.append(DenseSAGEConv(hidden_channels, out_channels))
            self.convs.append(DenseSAGEConv(hidden_channels, hidden_channels))
        elif conv_type == 'MLP':
            self.convs.append(torch.nn.Linear(in_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(torch.nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(BatchNorm1d(hidden_channels))
            # self.convs.append(torch.nn.Linear(hidden_channels, out_channels))
            self.convs.append(torch.nn.Linear(hidden_channels, hidden_channels))
        elif conv_type == 'GAT':
            self.heads = kwargs['heads']
            self.dropout = kwargs['dropout']

            self.convs.append(DenseGATConv(in_channels, hidden_channels, heads=self.heads, dropout=self.dropout))
            self.lins.append(torch.nn.Linear(hidden_channels * self.heads, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))

            for i in range(num_layers - 2):
                self.convs.append(DenseGATConv(hidden_channels, hidden_channels, heads=self.heads, concat=True, dropout=self.dropout))
                self.lins.append(torch.nn.Linear(hidden_channels * self.heads, hidden_channels))
                self.bns.append(BatchNorm1d(hidden_channels))


            # self.convs.append(DenseGATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=self.dropout))
            self.convs.append(DenseGATConv(hidden_channels, hidden_channels, heads=self.heads, concat=False, dropout=self.dropout))
        # elif conv_type == 'GCN-GAT':
        #     self.heads = kwargs['heads']
        #     self.dropout = kwargs['dropout']

        #     # self.convs.append(DenseGCNConv(in_channels, hidden_channels, heads=self.heads, dropout=self.dropout))
        #     # self.bns.append(BatchNorm1d(hidden_channels * self.heads))
        #     self.convs.append(DenseGCNConv(in_channels, hidden_channels))
        #     self.bns.append(BatchNorm1d(hidden_channels))

        #     for i in range(num_layers - 2):
        #         # n_heads = self.heads if i  == 0 else 1
        #         # self.convs.append(DenseGATConv(hidden_channels * n_heads, hidden_channels, heads=1, concat=False, dropout=self.dropout))
        #         # self.bns.append(BatchNorm1d(hidden_channels))

        #         self.convs.append(DenseGATConv(hidden_channels, hidden_channels, heads=self.heads, concat=False, dropout=self.dropout))
        #         self.bns.append(BatchNorm1d(hidden_channels))


        #     # self.convs.append(DenseGATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=self.dropout))
        #     self.convs.append(DenseGATConv(hidden_channels, hidden_channels, heads=self.heads, concat=False, dropout=self.dropout))
        elif conv_type == 'GCN-GAT':
            self.heads = kwargs['heads']
            self.dropout = kwargs['dropout']

            # self.convs.append(DenseGCNConv(in_channels, hidden_channels, heads=self.heads, dropout=self.dropout))
            # self.bns.append(BatchNorm1d(hidden_channels * self.heads))
            self.convs.append(DenseGCNConv(in_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))

            for i in range(num_layers - 2):
                # n_heads = self.heads if i  == 0 else 1
                # self.convs.append(DenseGATConv(hidden_channels * n_heads, hidden_channels, heads=1, concat=False, dropout=self.dropout))
                # self.bns.append(BatchNorm1d(hidden_channels))

                self.convs.append(DenseGATConv(hidden_channels, hidden_channels, heads=self.heads, concat=True, dropout=self.dropout))
                self.lins.append(torch.nn.Linear(hidden_channels * self.heads, hidden_channels))
                self.bns.append(BatchNorm1d(hidden_channels))


            # self.convs.append(DenseGATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=self.dropout))
            self.convs.append(DenseGATConv(hidden_channels, hidden_channels, heads=self.heads, concat=False, dropout=self.dropout))
        else:
            raise ValueError('conv_type must be one of "GCN", "GAT", "SAGE", "MLP"')

        if self.skip:
            self.linear = torch.nn.Linear(hidden_channels * num_layers, out_channels)
        else:
            self.linear = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, a, m=None):
        
        xs = []
        for i in range(self.num_layers):
            # print(i, x.size(), self.convs[i])
            if self.conv_type == 'MLP':
                x = self.convs[i](x)
            else:
                x = self.convs[i](x, a, m)

            if i < self.num_layers - 1:
                
                if self.conv_type == 'GAT':
                    # print(x.shape, self.lins[i])
                    x = F.elu(x)
                    x = self.lins[i](x)
                elif self.conv_type == 'GCN-GAT' and i > 0:
                    x = F.elu(x)
                    x = self.lins[i-1](x)
                

                x = F.relu(x)

                batch_size, n_nodes, n_channels = x.size()
                x = x.view(-1, n_channels)
                x = self.bns[i](x)
                x = x.view(batch_size, n_nodes, n_channels)

            # print('append', x.shape)
            xs.append(x)
                
        if self.skip:
            x = torch.cat(xs, dim=-1)
        
        # print(x.shape, self.linear)
        x = self.linear(x)

        if self.pred:
            x = x.mean(dim=1)

        return x

    


class DenseDiffPool(torch.nn.Module):

    def __init__(self, max_nodes, node_ratio,
                 conv_type, num_layers, in_channels, hidden_channels, out_channels, nodes_list=None, **kwargs):
        super(DenseDiffPool, self).__init__()

        self.max_nodes = max_nodes
        self.node_ratio = node_ratio
        self.conv_type = conv_type
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.skip = kwargs.get('skip', False)
        self.diff_skip = kwargs.get('diff_skip', False)

        self.embeds = torch.nn.ModuleList()
        self.assigns = torch.nn.ModuleList()

        self.bns = torch.nn.ModuleList()

       
        # max_nodes = ceil(max_nodes * node_ratio)
        self.embeds.append(DenseGNN(conv_type, num_layers, in_channels, hidden_channels, hidden_channels, **kwargs))
        self.assigns.append(DenseGNN(conv_type, num_layers, in_channels, hidden_channels, max_nodes, **kwargs))
        # self.bns.append(BatchNorm1d(hidden_channels))
        # print(max_nodes)
        self.num_poolings = 1
        while (max_nodes := max_nodes * node_ratio) >= 1:
            max_nodes = ceil(max_nodes)
            # print(max_nodes)
            # max_nodes = int(max_nodes * node_ratio)
            
            self.embeds.append(DenseGNN(conv_type, num_layers, hidden_channels, hidden_channels, hidden_channels, **kwargs))
            self.assigns.append(DenseGNN(conv_type, num_layers, hidden_channels, hidden_channels, max_nodes, **kwargs))
            # self.ffs.append(DenseGNN(conv_type, num_layers, hidden_channels, hidden_channels, hidden_channels, **kwargs))
            # self.bns.append(BatchNorm1d(hidden_channels))

            self.num_poolings += 1
                    
    
        self.embeds.append(DenseGNN(conv_type, num_layers, hidden_channels, hidden_channels, hidden_channels, **kwargs))
        # self.bns.append(BatchNorm1d(hidden_channels))

        if self.diff_skip:
            self.linear = torch.nn.Linear(hidden_channels * (self.num_poolings), out_channels)
           
        else:
             self.linear = torch.nn.Linear(hidden_channels, out_channels)
            



    def forward(self, x, a, m=None):
        
        xs = []
        lp_loss = 0
        er_loss = 0
        for i in range(self.num_poolings-1):
            # print('layer', i)
            # print('x', x.shape, self.embeds[i], self.assigns[i])
            s = self.assigns[i](x, a, m)
            x = self.embeds[i](x, a, m)
            xs.append(x.mean(dim=1))
            
            x, a, lp, er = dense_diff_pool(x, a, s, m)

            # print(x.shape, a.shape, s.shape)

            lp_loss += lp
            er_loss += er
            m = None

        x = self.embeds[-1](x, a, m)
        x = x.mean(dim=1)
        xs.append(x)

        if self.diff_skip:
            x = torch.cat(xs, dim=1)
        # print(x.shape, self.linear)
        x = self.linear(x)


        return x, lp_loss, er_loss
            
            
        