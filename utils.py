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


device = torch.device("cuda:0")  # 使用 GPU 0


def preprocess(data, node_feature_dim=256):
    #   比如data是  Cora
    #   删除train_mask等
    if hasattr(data, 'train_mask'):
        del data.train_mask
    if hasattr(data, 'val_mask'):
        del data.val_mask
    if hasattr(data, 'test_mask'):
        del data.test_mask

    #   把节点特征长度  统一到 node_feature_dim    
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


#   在一个原始图中（单图数据集，比如Cora)，随机采样  若干个子图。 返回这些子图构成的列表
def get_subgraphs(Original_Graph,num_subgraphs=10,walk_length = 15):
    num_nodes = Original_Graph.x.shape[0] #   Cora数据集的节点个数 2708
    start_nodes = random.sample(range(num_nodes), num_subgraphs)    #   从所有节点ID中，随机选择num_subgraphs个 节点  作为起点
    start_nodes = torch.tensor(start_nodes, dtype=torch.long)

    walk_list = random_walk(Original_Graph.edge_index[0], Original_Graph.edge_index[1],     # 图中的所有边
                            start=start_nodes,     # 这些节点作为  起点
                            walk_length=walk_length)    #   游走距离，表示一条游走路径上的节点个数
    graph_list = [] #   包括10个 游走序列对应生成的   诱导子图
        
    for walk in walk_list:   # walk是一条 长度为 walk_lenth  的节点路径（游走路径）,路径上可能存在重复节点ID
        subgraph_nodes = torch.unique(walk) #  subgraph_nodes  是这条路径上的  所有节点ID(  不包括重复节点ID，例如有23个节点ID    )
        subgraph_data = Original_Graph.subgraph(subgraph_nodes)   # 用这些节点ID ，从大图中构造一个【诱导子图】，诱导子图中的节点是：上述节点，边是：两端点都是上述节点的边
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



#   下游节点，传入的data是  一个Pyg.Data对象(单图数据集，原始图），比如photo数据集
#   对于data中的每一个节点（比如photo有7650个节点），都生成一个以该节点为中心的诱导子图，返回值就是这些诱导子图
def induced_graphs(data, smallest_size=10, largest_size=30):

    from torch_geometric.utils import subgraph, k_hop_subgraph
    from torch_geometric.data import Data
    import numpy as np
    induced_graph_list = []
    total_node_num = data.x.size(0) #   data的节点总数
    for index in range(data.x.size(0)): 
        # index表示每一个  节点ID
        current_label = data.y[index].item()    #   当前index节点 的 label
        current_hop = 2 #   跳数
        #   k_hop_subgraph是 Pyg提供的寻找k跳子图的函数。返回的subset是 子图中包含的节点ID（整图ID）
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
            subset = subset[torch.randperm(subset.shape[0])][0:largest_size - 1]    #   从subset中随机选择 29个节点ID（整图节点ID）
            subset = torch.unique(torch.cat([torch.LongTensor([index]), torch.flatten(subset)]))    # 把这29个节点ID和  当前节点ID（index）拼到一起
        
    #   经过上面的操作后，subset就是  以index节点为中心的2跳子图  中的 所有节点ID（整图节点ID）

    #   这个subgraph是生成诱导子图的操作，给出subset（一些节点ID），返回一个诱导子图
    #   诱导子图中，节点是：subset这些节点，边是：这些节点之间的边
        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)   #   返回的sub_edge_index就是诱导子图中的边（边上两端点的ID是子图ID）
    #   x是诱导子图中的节点的特征
        x = data.x[subset]
        induced_graph = Data(x=x, edge_index=sub_edge_index, y=current_label)   #   这个induced_graph是就是生成的诱导子图，包含节点特征和边
        induced_graph_list.append(induced_graph)
    
    print('生成了{}/{}张子图数据'.format(index,total_node_num))
    return induced_graph_list




#   生成预训练阶段的正例和负例（使用连接点构造）
def get_posi_nega_graphs(subgraphs, connectors ,num_subgraphs,neg_numbers):


    # 构造正例
    positives = []
    negatives = []
    for data_name, subgraph_list in subgraphs.items():
        for i in range(num_subgraphs):
            for j in range(i + 1, num_subgraphs):
                graph1 = subgraph_list[i]
                graph2 = subgraph_list[j]
                connected_graph = two_graph_into_connected([graph1, graph2], connectors[data_name], connectors[data_name])
                positives.append([connected_graph, 1])  # 1表示同源
    
    
    # 构造负例
    data_names = list(subgraphs.keys())
    k = neg_numbers  # 每对数据集只采样k个负例（可调整）
    for i in range(len(data_names)):
        for j in range(i + 1, len(data_names)):
            data_name1 = data_names[i]
            data_name2 = data_names[j]
            # 随机采样k个负例（避免全部组合）
            for _ in range(k):
                graph1 = random.choice(subgraphs[data_name1])
                graph2 = random.choice(subgraphs[data_name2])
                connected_graph = two_graph_into_connected([graph1, graph2], connectors[data_name1], connectors[data_name2])
                negatives.append([connected_graph, 0])

    selected_samples = positives + negatives
    random.shuffle(selected_samples)

    return selected_samples



#   多层GCN
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden):
        super(GCN, self).__init__()

        
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



    def forward(self, data):   #  如果data是 一个batch，那就包含多个图

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch #   edge_index是 10个图的 所有边，batch用来指明每个节点属于10个图中的哪个图

        h = torch.dropout(x, p=self.dropout, train=self.training)
        #   对初始节点特征（长度num_features）做一个线性变换，变成长度 hidden
        h = torch.relu(self.t1(h))  
        h = torch.dropout(h, p=self.dropout, train=self.training)


        h = F.leaky_relu(self.conv1(h, edge_index))
        h = F.leaky_relu(self.conv2(h, edge_index))        

        # 如果有多个图，那么 h 是多个图的所有节点的特征。  如果有1个图，那h就是 1个图的全部节点特征
        h = self.t2(h)

        #   这就是ReadOut操作， ReadOut得到每个图的 图向量(batch用来区分当前节点属于哪个图)。  
        #   即使只有1个图（batch为None），那就直接把这一个图的全部节点向量sum成1个图向量
        graph_emb = self.global_pool(h, batch)

    #   不论data是单个图还是batch（多个图） ， GCN都返回更新后的节点特征 h  ， 以及ReadOut操作之后的 图向量
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
        
        # 输入投影层：第二版本
        self.q_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim)  )
        self.k_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim)    )
        self.v_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim)    )


        # 输出投影，第二版本
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),            
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh()  # 最终输出限制范围
        )

        
    def forward(self, node_features, connectors):
        num_nodes = node_features.size(0)
        keys = values = connectors
        node_features = self.norm_q(node_features)
        keys = self.norm_kv(keys)
        values = self.norm_kv(values)
        # 1.投影
        q = self.q_proj(node_features)  # [batch中的节点个数, embed_dim]
        k = self.k_proj(keys)           # [预训练连接点个数, embed_dim]
        v = self.v_proj(values)         # [预训练连接点个数, embed_dim]            
        # 2. 拆分为多头
        q = q.view(num_nodes, self.num_heads, self.head_dim)  # [N, h, d_h]
        k = k.view(-1, self.num_heads, self.head_dim)  # [M, h, d_h]
        v = v.view(-1, self.num_heads, self.head_dim)  # [M, h, d_h]            
        # 3. 计算注意力分数。 
        attn_scores = torch.einsum('nhd,mhd->nhm', q, k)  # [N, h, M]
        attn_weights = F.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)

        
        # 4. 加权求和
        output = torch.einsum('nhm,mhd->nhd', attn_weights, v)  # [N, h, d_h]
        output = output.reshape(num_nodes, self.embed_dim)  # [N, embed_dim]

        
        output = self.out_proj(output) * 1.2
        output = F.dropout(output, p = 0.1, training=self.training) 
        return output + node_features  , attn_weights











class ContrastiveLoss(torch.nn.Module):
    def __init__(self, hidden_dim=128, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        #   对比任务头，包含2层 MLP。在GraphCL中，这个head似乎没有用到
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.temperature = temperature

    def forward(self, zi, zj , num_domains , num_subgraphs ):  
        import torch
        #   有10个原始图，zi就是【删除节点】后的10个增强图的图向量，zj就是10个【删除边】后的增强图的图向量
        batch_size = zi.size(0) #   10
        x1_abs = zi.norm(dim=1) #   计算  每个图向量的  Frobenius 范数(即L2范数，就是根号下  每个向量元素的平方的和)
        x2_abs = zj.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', zi, zj) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / self.temperature)
        #   得到的sim_matrix计算了  zi中每个向量  与  zj中每个向量  余弦相似度
            
        

        # num_domains = 源域数量
        num_subgraphs_per_domain = num_subgraphs

        domain_ids = torch.repeat_interleave(
            torch.arange(num_domains), 
            repeats=num_subgraphs_per_domain
        )




        # N = sim_matrix.size(0)

        pos_mask = (domain_ids.unsqueeze(0) == domain_ids.unsqueeze(1))
        pos_mask.fill_diagonal_(False)  #   每个子图和自己，不算正样本。


        sim_matrix = sim_matrix.clone()
        sim_matrix.fill_diagonal_(0.0)

        sim_matrix = sim_matrix.to(device)
        pos_mask = pos_mask.to(device)

        numerator = (sim_matrix * pos_mask).sum(dim=1)    # [N]，  表示的是每个子图的LOSS中的分子
        denominator = sim_matrix.sum(dim=1)               # [N]，  表示的是每个子图的LOSS中的分母

        valid = denominator != 0
        loss = -torch.log(numerator[valid] / denominator[valid])
        return loss.mean()


        loss = - torch.log(loss).mean()



        return loss
    


def SVD_node_feat(x,dim_after):
    U, S, _ = torch.linalg.svd(x)  #   奇异值分解
    x = torch.mm(  U[:, :dim_after],    torch.diag(S[:dim_after])  )
    return x




#   多层FAGCN
class FAGCN(torch.nn.Module):

    def __init__(self, num_features, hidden, num_conv_layers, 
                 dropout = 0.2, epsilon = 0.1 ,num_head = 4):
        super(FAGCN, self).__init__()
        self.global_pool = global_add_pool
        self.eps = epsilon              #   默认epsilon  为 0.1
        self.layer_num = num_conv_layers    #   默认  2 层
        self.dropout = dropout          #   默认dropout 为 0.2
        self.hidden_dim = hidden        #   默认128
        #   self.layers包含  多层的 FAConv
        self.layers = torch.nn.ModuleList() 
        for _ in range(self.layer_num):
            self.layers.append(FAConv(hidden, epsilon, dropout))

        self.t1 = torch.nn.Linear(num_features, hidden)
        self.t2 = torch.nn.Linear(hidden, hidden)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        torch.nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    
    def forward(self, data):   #   这里data是 一个batch（包含若干个图）

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch #   edge_index是 10个图的 所有边，batch用来指明每个节点属于10个图中的哪个图

        h = torch.dropout(x, p=self.dropout, train=self.training)
        h = torch.relu(self.t1(h))  #   对初始节点特征（长度100）做一个线性变换，变成长度128
        h = torch.dropout(h, p=self.dropout, train=self.training)
        raw = h #   raw是初始特征经过一层Linear层变换后的  长度为128的特征（相当于 没经过GNN的初始特征）


        for i in range(self.layer_num): #   这是经过 多层GNN聚合，得到更新后的  节点特征 h 
            h = self.layers[i](h, raw, edge_index)  
            
        h = self.t2(h)

        #   这就是ReadOut操作，得到每个图的  图向量。
        graph_emb = self.global_pool(h, batch)


        return graph_emb, h
