from torch_cluster import random_walk
from torch_geometric.data import Data , Batch
import torch
import random
from torch_geometric.transforms import SVDFeatureReduction
from torch_geometric.nn import global_add_pool, FAConv,GCNConv, GATConv
import os
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from functools import partial
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
import torch.nn as nn

import torch
from torch_geometric.nn import global_add_pool, FAConv





def preprocess(data, node_feature_dim=256):

    if hasattr(data, 'train_mask'):
        del data.train_mask
    if hasattr(data, 'val_mask'):
        del data.val_mask
    if hasattr(data, 'test_mask'):
        del data.test_mask

    #   
    if data.x.size(-1) > node_feature_dim:
        data = x_svd(data, node_feature_dim)
    elif data.x.size(-1) < node_feature_dim:
        data = x_padding(data, node_feature_dim)
    else:
        pass
    
    return data

def x_svd(data, out_dim):
    
    assert data.x.size(-1) >= out_dim

    reduction = SVDFeatureReduction(out_dim)
    return reduction(data)

def x_padding(data, out_dim):
    
    assert data.x.size(-1) <= out_dim
    
    incremental_dimension = out_dim - data.x.size(-1)
    zero_features = torch.zeros((data.x.size(0), incremental_dimension), dtype=data.x.dtype, device=data.x.device)
    data.x = torch.cat([data.x, zero_features], dim=-1)

    return data


#   
def get_subgraphs(Original_Graph,num_subgraphs=10,walk_length = 15):
    num_nodes = Original_Graph.x.shape[0] #   
    start_nodes = random.sample(range(num_nodes), num_subgraphs)    #   
    start_nodes = torch.tensor(start_nodes, dtype=torch.long)

    walk_list = random_walk(Original_Graph.edge_index[0], Original_Graph.edge_index[1],     # 
                            start=start_nodes,     # 
                            walk_length=walk_length)    #   
    graph_list = [] #   
        
    for walk in walk_list:   # 
        subgraph_nodes = torch.unique(walk) #  
        subgraph_data = Original_Graph.subgraph(subgraph_nodes)   # 
        graph_list.append(subgraph_data)

    return graph_list


def two_graph_into_connected(data_list,connector_1,connector_2):
    two_graph = Batch.from_data_list(data_list) 
    connectors = torch.cat([connector_1.reshape(1,-1),connector_2.reshape(1,-1)], dim=0)
    two_graph.x = torch.cat([two_graph.x, connectors], dim=0)
    two_graph.batch = torch.cat([two_graph.batch, torch.tensor([0,1]).squeeze(0)], dim=0)
    new_index_list = [0,1]
    for node_graph_index in [0,1]: 
        node_indices_corresponding_graph = (two_graph.batch == node_graph_index).nonzero(as_tuple=False).view(-1)  
        new_node_index = node_indices_corresponding_graph[-1]   
        new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), node_indices_corresponding_graph[:-1])
        two_graph.edge_index = torch.cat([two_graph.edge_index, new_edges.t()], dim=1)
        new_edges = torch.cartesian_prod(node_indices_corresponding_graph[:-1], torch.tensor([new_node_index]))
        two_graph.edge_index = torch.cat([two_graph.edge_index, new_edges.t()], dim=1)
    all_added_node_index = [i for i in range(two_graph.num_nodes-len(new_index_list),two_graph.num_nodes)]
    for list_index, new_node_index in enumerate(all_added_node_index):
        other_added_node_index_list = [index for index in all_added_node_index if index != new_node_index]
        new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), torch.tensor(other_added_node_index_list))
        two_graph.edge_index = torch.cat([two_graph.edge_index, new_edges.t()], dim=1)


    x = two_graph.x
    edge_index = two_graph.edge_index
    return Data(x=x,edge_index=edge_index)

    # return two_graph



def induced_graphs(data, smallest_size=10, largest_size=30):

    from torch_geometric.utils import subgraph, k_hop_subgraph
    from torch_geometric.data import Data
    import numpy as np
    induced_graph_list = []
    total_node_num = data.x.size(0) #   
    for index in range(data.x.size(0)): 
        # 
        current_label = data.y[index].item()    #   
        current_hop = 2 #   
        #   
        subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                            edge_index=data.edge_index, relabel_nodes=True)
        
        while len(subset) < smallest_size and current_hop < 5:
            current_hop += 1
            subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                                edge_index=data.edge_index)
            
        if len(subset) < smallest_size:
            need_node_num = smallest_size - len(subset)
            pos_nodes = torch.argwhere(data.y == int(current_label)) 
            candidate_nodes = torch.from_numpy(np.setdiff1d(pos_nodes.numpy(), subset.numpy()))
            candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]
            subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

        if len(subset) > largest_size:
            subset = subset[torch.randperm(subset.shape[0])][0:largest_size - 1]    #   
            subset = torch.unique(torch.cat([torch.LongTensor([index]), torch.flatten(subset)]))    # 
        
    #   


        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)   
    #   
        x = data.x[subset]
        induced_graph = Data(x=x, edge_index=sub_edge_index, y=current_label)   
        induced_graph_list.append(induced_graph)
    
    print('Get{}/{}subgraphs'.format(index,total_node_num))
    return induced_graph_list




def get_posi_nega_graphs(subgraphs, connectors ,num_subgraphs,neg_numbers):



    positives = []
    negatives = []
    for data_name, subgraph_list in subgraphs.items():
        for i in range(num_subgraphs):
            for j in range(i + 1, num_subgraphs):
                graph1 = subgraph_list[i]
                graph2 = subgraph_list[j]
                connected_graph = two_graph_into_connected([graph1, graph2], connectors[data_name], connectors[data_name])
                positives.append([connected_graph, 1])  #
    
    
    # 
    data_names = list(subgraphs.keys())
    k = neg_numbers  # 
    for i in range(len(data_names)):
        for j in range(i + 1, len(data_names)):
            data_name1 = data_names[i]
            data_name2 = data_names[j]
            
            for _ in range(k):
                graph1 = random.choice(subgraphs[data_name1])
                graph2 = random.choice(subgraphs[data_name2])
                connected_graph = two_graph_into_connected([graph1, graph2], connectors[data_name1], connectors[data_name2])
                negatives.append([connected_graph, 0])

    selected_samples = positives + negatives
    random.shuffle(selected_samples)

    return selected_samples



#   
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden):
        super(GCN, self).__init__()
        # 
        self.dropout = 0.2
        self.t1 = torch.nn.Linear(num_features, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.t2 = torch.nn.Linear(hidden, hidden)
        self.global_pool = global_add_pool
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        torch.nn.init.xavier_normal_(self.t2.weight, gain=1.414)



    def forward(self, data):   #  

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch #   

        h = torch.dropout(x, p=self.dropout, train=self.training)
        #   
        h = torch.relu(self.t1(h))  
        h = torch.dropout(h, p=self.dropout, train=self.training)


        h = F.leaky_relu(self.conv1(h, edge_index))
        h = F.leaky_relu(self.conv2(h, edge_index))        

        # 
        h = self.t2(h)


        graph_emb = self.global_pool(h, batch)

    #   
        return graph_emb, h


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    

        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)   


    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class NodeLevelAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        
        #
        self.q_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim)  )
        self.k_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim)    )
        self.v_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim)    )


        # 
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),            
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh()  # 
        )

        
    def forward(self, node_features, connectors):
        num_nodes = node_features.size(0)
        keys = values = connectors
        node_features = self.norm_q(node_features)
        keys = self.norm_kv(keys)
        values = self.norm_kv(values)
        # 
        q = self.q_proj(node_features)  
        k = self.k_proj(keys)           
        v = self.v_proj(values)                   
        # 
        q = q.view(num_nodes, self.num_heads, self.head_dim)  # [N, h, d_h]
        k = k.view(-1, self.num_heads, self.head_dim)  # [M, h, d_h]
        v = v.view(-1, self.num_heads, self.head_dim)  # [M, h, d_h]            
        #  
        attn_scores = torch.einsum('nhd,mhd->nhm', q, k)  # [N, h, M]
        attn_weights = F.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)

        
        # 
        output = torch.einsum('nhm,mhd->nhd', attn_weights, v)  # [N, h, d_h]
        output = output.reshape(num_nodes, self.embed_dim)  # [N, embed_dim]

        
        output = self.out_proj(output) * 1.2
        output = F.dropout(output, p = 0.1, training=self.training) 
        return output + node_features  , attn_weights











class ContrastiveLoss(torch.nn.Module):
    def __init__(self, hidden_dim=128, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        #   
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.temperature = temperature

    def forward(self, zi, zj , num_domains , num_subgraphs ):  
        import torch
        #   
        batch_size = zi.size(0) #   
        x1_abs = zi.norm(dim=1) #   
        x2_abs = zj.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', zi, zj) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / self.temperature)
        #   
            
        

        # 
        num_subgraphs_per_domain = num_subgraphs

        domain_ids = torch.repeat_interleave(
            torch.arange(num_domains, device=zi.device),
            repeats=num_subgraphs_per_domain
        )




        # N = sim_matrix.size(0)

        pos_mask = (domain_ids.unsqueeze(0) == domain_ids.unsqueeze(1))
        pos_mask.fill_diagonal_(False)  #   


        sim_matrix = sim_matrix.clone()
        sim_matrix.fill_diagonal_(0.0)

        numerator = (sim_matrix * pos_mask).sum(dim=1)    # 
        denominator = sim_matrix.sum(dim=1)               # 

        valid = denominator != 0
        loss = -torch.log(numerator[valid] / denominator[valid])
        return loss.mean()


        loss = - torch.log(loss).mean()



        return loss
    


def SVD_node_feat(x,dim_after):
    U, S, _ = torch.linalg.svd(x)  #   
    x = torch.mm(  U[:, :dim_after],    torch.diag(S[:dim_after])  )
    return x




#   
class FAGCN(torch.nn.Module):

    def __init__(self, num_features, hidden, num_conv_layers, 
                 dropout = 0.2, epsilon = 0.1 ,num_head = 4):
        super(FAGCN, self).__init__()
        self.global_pool = global_add_pool
        self.eps = epsilon              #   
        self.layer_num = num_conv_layers    #   
        self.dropout = dropout          #   
        self.hidden_dim = hidden        #   
        #   
        self.layers = torch.nn.ModuleList() 
        for _ in range(self.layer_num):
            self.layers.append(FAConv(hidden, epsilon, dropout))

        self.t1 = torch.nn.Linear(num_features, hidden)
        self.t2 = torch.nn.Linear(hidden, hidden)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        torch.nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    
    def forward(self, data):   #   

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch #   

        h = torch.dropout(x, p=self.dropout, train=self.training)
        h = torch.relu(self.t1(h))  #   
        h = torch.dropout(h, p=self.dropout, train=self.training)
        raw = h #   


        for i in range(self.layer_num): #   
            h = self.layers[i](h, raw, edge_index)  
            
        h = self.t2(h)

        #   
        graph_emb = self.global_pool(h, batch)


        return graph_emb, h






#   GAT
class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden, heads=4):
        super(GAT, self).__init__()
        
        self.dropout = 0.2
        self.t1 = torch.nn.Linear(num_features, hidden)
        

        self.conv1 = GATConv(hidden, hidden, heads=heads, concat=False)
        self.conv2 = GATConv(hidden, hidden, heads=heads, concat=False)
        
        self.t2 = torch.nn.Linear(hidden, hidden)
        self.global_pool = global_add_pool
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        torch.nn.init.xavier_normal_(self.t2.weight, gain=1.414)
        

    def forward(self, data):   

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch #   

        h = torch.dropout(x, p=self.dropout, train=self.training)
        #   
        h = torch.relu(self.t1(h))  
        h = torch.dropout(h, p=self.dropout, train=self.training)

        #   
        h = F.leaky_relu(self.conv1(h, edge_index))
        h = F.leaky_relu(self.conv2(h, edge_index))        

        
        h = self.t2(h)


        graph_emb = self.global_pool(h, batch)


        return graph_emb, h



