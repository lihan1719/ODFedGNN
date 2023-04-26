import torch
import torch.nn as nn
from torch_geometric.nn import MetaLayer
from torch_geometric.data import Batch
from torch_scatter import scatter_add
from torch_geometric.utils import dense_to_sparse
import numpy as np


class NodeModel(nn.Module):
    def __init__(self,
                 node_input_size,
                 hidden_size,
                 node_output_size,
                 activation='ReLU',
                 dropout=0.0):
        super(NodeModel, self).__init__()

        self.node_mlp = nn.Sequential(nn.Linear(node_input_size, hidden_size),
                                      getattr(nn, activation)(),
                                      nn.Dropout(p=dropout),
                                      nn.Linear(hidden_size, node_output_size),
                                      getattr(nn, activation)(),
                                      nn.Dropout(p=dropout))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        epsilon = 0.001
        row, col = edge_index

        # Compute the aggregation of geographic neighbors
        geo_sum = scatter_add(edge_attr * x[col],
                              col,
                              dim_size=x.size(0),
                              dim=0)

        geo_denom = scatter_add(edge_attr, col, dim_size=x.size(0), dim=0)
        geo_agg = (x + geo_sum / geo_denom)
        nan_indices = torch.isnan(geo_agg)
        geo_agg = torch.where(nan_indices.to('cuda'),
                              torch.tensor(0.0).to('cuda'), geo_agg)
        # Apply the node MLP to the aggregated geographic neighbor features
        r = self.node_mlp(geo_agg)

        return r


# Initialize and use the GeoNodeModel


# 聚合语义、地理信息
class GeoSemNodeEm(nn.Module):
    def __init__(self,
                 node_input_size,
                 hidden_size,
                 node_output_size,
                 activation,
                 dropout,
                 gn_layer_num=1,
                 *args,
                 **kwargs):
        super().__init__()
        self.geo_net = []
        self.semantic_net = []
        last_node_input_size = node_input_size
        for _ in range(gn_layer_num):
            # 地理信息聚合
            # geo_node_model = NodeModel(last_node_input_size,
            #                            hidden_size,
            #                            node_output_size,
            #                            activation=activation,
            #                            dropout=dropout)
            # 语义信息聚合
            semantic_node_model = NodeModel(last_node_input_size,
                                            hidden_size,
                                            node_output_size,
                                            activation=activation,
                                            dropout=dropout)
            last_node_input_size += node_output_size
            # self.geo_net.append(
            #     MetaLayer(edge_model=None,
            #               node_model=geo_node_model,
            #               global_model=None))
            self.semantic_net.append(
                MetaLayer(edge_model=None,
                          node_model=semantic_node_model,
                          global_model=None))
        self.geo_net = nn.ModuleList(self.geo_net)
        self.semantic_net = nn.ModuleList(self.semantic_net)

        # self.node_out_net = nn.Linear(2 * node_output_size, node_output_size)
        self.node_out_net = nn.Linear(node_output_size, node_output_size)

    def forward(self, data, semantic_data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_attr = edge_attr.expand(-1, x.shape[1], x.shape[2], -1)
        for geo_layer, semantic_layer in zip(self.geo_net, self.semantic_net):
            # updated_x_geo, _, _ = geo_layer(x, edge_index, edge_attr)
            # # 语义信息聚合
            updated_semantic = []
            for i in range(x.shape[1]):
                semantic_edge_index, semantic_edge_attr = dense_to_sparse(
                    semantic_data[i, 0, :, :])
                semantic_edge_attr = semantic_edge_attr.unsqueeze(
                    -1).unsqueeze(-1).to('cuda')
                semantic_edge_attr = semantic_edge_attr.expand(
                    -1, x.shape[2], -1)
                updated_x_semantic, _, _ = semantic_layer(
                    x[:, i, :, :], semantic_edge_index, semantic_edge_attr)
                updated_semantic.append(updated_x_semantic)
            updated_x_semantic = torch.stack(updated_semantic, dim=1)
            x = updated_x_semantic
            # x = torch.cat([updated_x_geo, updated_x_semantic], dim=-1)
        node_out = self.node_out_net(x)
        return node_out
