from torch_cluster import random_walk
from torch_geometric.datasets import Planetoid, Amazon ,   FacebookPagePage ,LastFMAsia  ,HeterophilousGraphDataset,WikipediaNetwork,WebKB,Flickr,Coauthor
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data,Batch
import torch
import random
import os
from torch_geometric.transforms import SVDFeatureReduction
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader   #   PyG的 Dataloader
from torchmetrics import MeanMetric, Accuracy, F1Score, AUROC
from tqdm import tqdm
from copy import deepcopy
from utils import *
import numpy as np
import argparse
import time
from datetime import datetime  # 加在文件开头（只需要一次）

now_file = os.path.abspath(__file__)   #   当前文件
current_dir = os.path.dirname(now_file) #   当前文件所在目录
data_dir = os.path.join(os.path.dirname(current_dir) , '_MDGCL_Data_dir')

if torch.cuda.is_available():
    device = torch.device("cuda:0")  
else:
    device = torch.device("cpu")  # 如果没有 GPU，则使用 CPU
#   PyG_Dataset_Dir应该 在 MDGCL_Data_dir之下
PyG_Dataset_Dir = os.path.join(data_dir ,"PyG_Dataset" )




#   预训练
def pretrain(    num_epochs = 100,    
             num_subgraphs = 30,    
             walk_length = 30 ,
             seed = 42 ,
             neg_numbers = 280 ,
             adapt_dataset = 'citeseer',
             model_name = 'GCN',
             node_feats = 100,
             hidden_fts = 128,
             pre_lr = 1e-4,
             pre_wd = 1e-5,
             pre_datasets = ['Cora','PubMed','CiteSeer','Photo','Computers'],
             regular = 0.1,
             regular_si = 0.1,
             regular_3 = 0.1,
             min_sim = -0.5,
             
             ):

    pretrained_model_and_connectors = os.path.join(data_dir,'预训练模型和连接点')  
    if not os.path.exists(pretrained_model_and_connectors):
        os.makedirs(pretrained_model_and_connectors)
    else:
        pass

    pretrain_datasets = '_'.join(pre_datasets)
    # #   已经预训练过的模型，则直接退出。  如果把下面两句注释掉，那就是每次都要重新预训练一遍
    # if os.path.exists(os.path.join(  pretrained_model_and_connectors  , f'Pretrained_{model_name}_of_{pretrain_datasets}.pt')):
    #     return

    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    datasets = {}

    for data_name in pre_datasets:
        if data_name == 'Cora':
            datasets[data_name] = Planetoid(root=PyG_Dataset_Dir, name='Cora')._data
        elif data_name == 'PubMed':
            datasets[data_name] = Planetoid(root=PyG_Dataset_Dir, name='PubMed')._data
        elif data_name == 'CiteSeer':
            datasets[data_name] = Planetoid(root=PyG_Dataset_Dir, name='CiteSeer')._data
        elif data_name == 'Photo':
            datasets[data_name] = Amazon(root=PyG_Dataset_Dir, name='Photo')._data
        elif data_name == 'Computers':
            datasets[data_name] = Amazon(root=PyG_Dataset_Dir, name='Computers')._data
        elif data_name == 'CoauthorPhysics':
            datasets[data_name] = Coauthor(root = PyG_Dataset_Dir , name = 'Physics')._data
        #   接下来是异配图
        elif data_name == 'amazonratings':
            datasets[data_name] = HeterophilousGraphDataset(root=PyG_Dataset_Dir , name = 'amazon_ratings')._data
        elif data_name == 'chameleon':
            datasets[data_name] = WikipediaNetwork(root=PyG_Dataset_Dir, name='chameleon')._data
        elif data_name == 'squirrel':
            datasets[data_name] = WikipediaNetwork(root=PyG_Dataset_Dir, name='squirrel')._data

        elif data_name in ['PROTEINS','ENZYMES','COX2','MUTAG','BZR']:
        #   一个数据集包含多个图
            datasets[data_name] = TUDataset(root=PyG_Dataset_Dir,name = data_name)
        elif data_name == 'Facebook':
            datasets[data_name] = FacebookPagePage(root = os.path.join(PyG_Dataset_Dir,'Facebook') )[0]  
            
        
        

    # 统一节点特征维度。分子图就padding。单图数据集就SVD
    for data_name, data in datasets.items():

        if data_name in ['PROTEINS','ENZYMES','COX2','MUTAG','BZR']:   
            #   图分类数据集,那么 data是一个类似list的东西，包含多个图，需要一个一个进行特征维度对齐
            graph_list = []
            for graph_id in range(data.len()):
                # graph__processed = preprocess(data[graph_id],node_feature_dim = node_feats)
                # graph_list.append(graph__processed)
                # pass
                #   single_mole_graph是  1个  分子图
                single_mole_graph = data[graph_id].clone()
                #   1个分子图  原本的  特征  x
                x = single_mole_graph.x.clone().to(device)
                num_nodes = single_mole_graph.x.size(0)

                target_dim = node_feats
                #   分子图初始的特征维度
                feat_dim = x.size(1)

                pad = torch.zeros(
                    (num_nodes, target_dim - feat_dim),
                    dtype = x.dtype
                        ).to(device)
                x = torch.cat([x, pad], dim=1).to(device)

                single_mole_graph.x = x
                graph_list.append(single_mole_graph)
            #   图分类数据集： 包含多个图的 列表
            datasets[data_name] = graph_list
            pass
        else:

            #   节点分类数据集：只包含一个 整图
            datasets[data_name] = preprocess(data, node_feature_dim = node_feats)


    # 获取每个数据集的子图
    subgraphs = {}
    for data_name, data in datasets.items():
        #   如果是图分类数据集，data本身(graph_list)就相当于多个子图, 子图数量和规定的子图数量不一致
        if data_name in ['PROTEINS','ENZYMES','COX2','MUTAG','BZR']:   
            subgraphs[data_name] = random.sample(data, num_subgraphs)

            pass
        else:
            #   单图数据集，需要采集若干子图。
            subgraphs[data_name] = get_subgraphs(data,num_subgraphs=num_subgraphs,walk_length=walk_length)


    # 构造domain token,每个预训练数据集有一个连接点，用sum获取全局信息(也可以用mean获取全局信息，这样不会因为某个领域节点过多而导致领域token过大)
    connectors = {}
    for data_name, data in datasets.items():    
        if data_name in ['PROTEINS','ENZYMES','COX2','MUTAG','BZR']:
            #   如果是图分类数据集，本身包含多个图。那就把所有图的所有节点特征都加起来
            accum = None
            sum_x_list = []  # 用于存储每个子图的 sum_x
            for subgraph_ in data:
                sum_x = torch.mean(subgraph_.x, dim=0)  # 每个子图的特征按节点求和(或者mean)
                sum_x_list.append(sum_x)

            accum = torch.mean(torch.stack(sum_x_list), dim=0)

            connectors[data_name] = accum
            
            pass
        else:#  单图数据集
            connectors[data_name] = torch.mean(input=data.x, dim=0).to(device)
    

    original_domain_tokens = list(connectors.values())
    original_domain_tokens = torch.stack(original_domain_tokens, dim=0)   # shape: (5, 128)
    original_domain_tokens = original_domain_tokens.to(device)


    num_domains = len(subgraphs)

    from torch_geometric.data import Batch


    all_subgraphs_list = []
    for data_name, sg_list in subgraphs.items():
        all_subgraphs_list.extend(sg_list)
    #   batch_graph包含了每个域产出的所有子图,用batch变量来区分多图

    for a_subgraph in all_subgraphs_list:
        a_subgraph = a_subgraph.to(device)

    batch_graph = Batch.from_data_list(all_subgraphs_list)



    if model_name == 'GCN':
        GNN_Model = GCN(num_features = node_feats  ,  hidden = hidden_fts).to(device)
    elif model_name == 'FAGCN':
        GNN_Model = FAGCN(num_features = node_feats,
                          hidden= hidden_fts , 
                          num_conv_layers= 2 ,
                          
                          dropout = 0.15, 
                          epsilon = 0.2,
                          num_head = 4 #    这个用不到
                          ).to(device)


    #   proj用于把  新的domain_token映射到 50维，这样方便和原来的domain token计算余弦相似度
    proj = nn.Linear(hidden_fts, node_feats, bias=False).to(device)
    # LOSS_fn用于计算  对比学习的LOSS。
    loss_fn = ContrastiveLoss().to(device)
    loss_fn.train() 
    #   Adam优化器
    optimizer_pretrain = torch.optim.Adam(  list(  GNN_Model.parameters() )  + list(proj.parameters() )   ,  
                                          lr = pre_lr,
                                          weight_decay =  pre_wd)


    start_time = time.time()  # 记录预训练开始时间

    for epoch in range(num_epochs):
        GNN_Model.train()
        proj.train()
        total_loss = 0
        # batch_graph表示的是  每个域的  所有子图
        batch_graph = batch_graph.to(device)
        optimizer_pretrain.zero_grad()
        #   所有子图的  表示向量
        graph_embedding, node_embed = GNN_Model(data=batch_graph)
        #   所有子图向量，模长变成1
        graph_embedding / torch.norm(graph_embedding, dim=1, keepdim=True)
        
        # 经过GNN之后的 domain token（表示每个领域的分布）
        domain_tokens = {}
        for data_name, data in datasets.items():    

            if data_name in ['PROTEINS','ENZYMES','COX2','MUTAG','BZR']:
                #   如果是图分类数据集，本身包含多个图。这里的data就是一个list，包含所有分子图的列表
                # 那就把所有图的所有节点特征都加起来
                #   molecules_batch是所有分子图构成的batch
                for _ in data:
                    _ = _.to(device)
                molecules_batch = Batch.from_data_list(data).to(device)
                graph_emb, node_h = GNN_Model(data=molecules_batch)
                domain_tokens[data_name] = torch.mean(node_h,dim=0)
                
                pass
            else:#  单图数据集

                single_graph_embed, node_h = GNN_Model(data=data.to(device))
                domain_tokens[data_name] = torch.mean(node_h,dim=0)

        new_domain_tokens = list(domain_tokens.values())
        new_domain_tokens = torch.stack(new_domain_tokens, dim=0)   # shape: (5, 128)
        new_domain_tokens = proj(new_domain_tokens)
        
        c_0 = original_domain_tokens.mean(dim=0)   # 原来的凸包中心

        dir_loss = 0.0  #   正则项，用于控制扩张方向
        R_out = 0.0 #   正则项，用来保证每个扩张系数都大于等于1

        for i in range(num_domains):

            #   计算正则项 dir_loss
            v1 = original_domain_tokens[i] - c_0
            v2 = new_domain_tokens[i] - c_0
            dir_loss += 1 - F.cosine_similarity(v1, v2, dim=0)           
            #   计算正则项 Rout
            # ori_vec = original_domain_tokens[i] - c_0
            # new_vec = new_domain_tokens[i] - c_0
            # 求出  向量的长度
            ori_norm = torch.norm(v1)
            new_norm = torch.norm(v2)
            # 分数值
            score = 1 - new_norm / (ori_norm )
            # 和 0 取最大值
            R_out += torch.maximum(score, torch.tensor(0.0, device=device))
        

        R3 = 0.0    #   正则项，用来防止  凸包过度扩张，也就是让两个源域之间至少有最低的相似度。
        m = num_domains
        for k in range(m):
            for l in range(k + 1, m):
                uk = new_domain_tokens[k]
                ul = new_domain_tokens[l]

                cos_sim = F.cosine_similarity(uk, ul, dim=0)
                R3 = R3 + torch.clamp(min_sim - cos_sim, min=0.0)


        loss = loss_fn(graph_embedding, graph_embedding, 
                       num_domains = num_domains ,
                       num_subgraphs = num_subgraphs )   + regular*dir_loss  + regular_si * R_out  + regular_3 * R3
        
        loss.backward()
        optimizer_pretrain.step()
        total_loss += loss.item()


        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.7f}")
    
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f"代码运行时间: {elapsed_time:.6f} 秒")    

    #   无条件  覆盖之前保存的模型
    torch.save(GNN_Model,   os.path.join(  pretrained_model_and_connectors  ,
                                           f'Pretrained_{model_name}_of_{pretrain_datasets}.pt')   )
    torch.save(connectors,  os.path.join(  pretrained_model_and_connectors ,
                                           f'Connectors_of_{pretrain_datasets}.pt')    )
    


#   获取【下游】数据集 ， 用于图分类。  
#   如果下游数据集是一个图（Cora），那就把每个节点都构造出诱导子图，附带标签y。然后把这些子图划分成训练、验证、测试
#   如果下游数据集是分子图（PROTEINS），本身就是包含多个图，每个图有一个标签y。然后把这些图划分成训练、验证、测试
def get_downstream_dataset_graph_classification(dataset_dir,
                        data_name,  #   下游数据集名称
                        seed, 
                        few_shot=1,
                        node_feats = 100 ,
                        ):


    import os
    downstream_Graph_classify = os.path.join(dataset_dir,'downstream_Graph_Classify')  # 替换为你的文件夹路径
    if not os.path.exists(downstream_Graph_classify):
        os.makedirs(downstream_Graph_classify)
    else:
        pass


    #   如果之前已经保存了下游数据集，现在直接取出下游数据集
    if os.path.exists(os.path.join(downstream_Graph_classify,  f'{data_name}_seed{seed}_featsize{node_feats}_shot{few_shot}.pt' ) ):
        return torch.load(os.path.join(downstream_Graph_classify, f'{data_name}_seed{seed}_featsize{node_feats}_shot{few_shot}.pt'  )  , weights_only = False)
 
    if data_name == 'Photo':
        data = Amazon(root=PyG_Dataset_Dir, name='Photo')._data
    elif data_name == 'Cora':
        data = Planetoid(root=PyG_Dataset_Dir, name='Cora')._data
    elif data_name == 'Computers':
        data = Amazon(root=PyG_Dataset_Dir, name='Computers')._data
    elif data_name == 'CiteSeer' :
        data = Planetoid(root=PyG_Dataset_Dir, name='CiteSeer')._data
    elif data_name == 'PubMed':
        data = Planetoid(root=PyG_Dataset_Dir, name='PubMed')._data
    elif data_name == 'Facebook' :
        data = FacebookPagePage(root=os.path.join(PyG_Dataset_Dir,'Facebook'))[0]
    elif data_name == 'lastFM':
        data = LastFMAsia(root= os.path.join(PyG_Dataset_Dir,'LastFM'))._data
    elif data_name == 'chameleon':
        data = WikipediaNetwork(root=PyG_Dataset_Dir, name='chameleon')._data
    elif data_name == 'squirrel':
        data = WikipediaNetwork(root=PyG_Dataset_Dir, name='squirrel')._data
    elif data_name == 'roman_empire':
        data = HeterophilousGraphDataset(root=PyG_Dataset_Dir , name = 'roman_empire')._data
    elif data_name == 'cornell':
        data = WebKB(root = PyG_Dataset_Dir, name = 'cornell')._data
    elif data_name == 'amazonratings':
        data = HeterophilousGraphDataset(root=PyG_Dataset_Dir , name = 'amazon_ratings')._data
    elif data_name == 'CoauthorPhysics':
        data = Coauthor( root = PyG_Dataset_Dir , name = 'Physics')._data
    #   分子图 
    elif data_name == 'PROTEINS':
        data = TUDataset(root=PyG_Dataset_Dir,name = 'PROTEINS')
    elif data_name == 'ENZYMES':
        data = TUDataset(root=PyG_Dataset_Dir,name = 'ENZYMES')

    elif data_name == 'arxiv':
        from ogb.nodeproppred import PygNodePropPredDataset

        dataset = PygNodePropPredDataset(root = PyG_Dataset_Dir ,name = 'ogbn-arxiv') 
        #   ognb-arxiv是一个单图数据集，每个节点代表一个论文， 把论文的题目和摘要（文本）  用skip-gram 变成 的向量
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        data = dataset[0] # pyg graph object

    else:
        print("数据集不在可选范围内")



    
    #   做特征维度统一
    if data_name in ['PROTEINS','ENZYMES']:
        #   图分类数据集： 每个数据集包含多个图，要做 Padding
        graph_list = [] #   graph_list用来存储  所有 分子图
        # data包含多个图
        for graph_id in range(data.len()):
            # graph__processed = preprocess(data[graph_id],node_feature_dim = node_feats)
            # graph_list.append(graph__processed)
            # pass        
            #   single_mole_graph是  1个  分子图
            single_mole_graph = data[graph_id].clone()
            #   1个分子图  原本的  特征  x
            x = single_mole_graph.x.clone().to(device)
            num_nodes = single_mole_graph.x.size(0)
            target_dim = node_feats
            #   分子图初始的特征维度
            feat_dim = x.size(1)
            pad = torch.zeros(
                (num_nodes, target_dim - feat_dim),
                dtype = x.dtype
                    ).to(device)
            x = torch.cat([x, pad], dim=1).to(device)
            single_mole_graph.x = x
            graph_list.append(single_mole_graph)

        

        num_classes = torch.unique(data.y).size(0)
        train_dict_list = {key.item():[] for key in torch.unique(data.y)}
        val_test_list = []
        #   target_graph_list 就是下游数据集包括的全部 子图
        target_graph_list = graph_list   #   （每个节点都生成一个以该节点为中心的诱导子图）
        from torch.utils.data import random_split, Subset
        for index, graph in enumerate(target_graph_list):
            i_class = graph.y.item()  
            if( len(train_dict_list[i_class]) >= few_shot):
                val_test_list.append(graph)
            else:
                train_dict_list[i_class].append(index)       
        all_indices = []
        for i_class, indice_list in train_dict_list.items():
            all_indices += indice_list

        train_set = Subset(target_graph_list, all_indices)  #   train_set每个类别 shot个 子图
        val_set, test_set = random_split(val_test_list, [0.1,0.9], 
                                        torch.Generator().manual_seed(seed))
        
        results = [     
        {
            'train': train_set,
            'val': val_set,
            'test': test_set,
        }, 
            num_classes
        ]
        
        
        torch.save(results,     
                os.path.join(     downstream_Graph_classify, 
                             f'{data_name}_seed{seed}_featsize{node_feats}_shot{few_shot}.pt'   )     )
        

        return results 

    #   单个图的数据集（比如Cora）
    else:

        #   特征长度统一
        data = preprocess(data,node_feature_dim = node_feats)
        num_classes = torch.unique(data.y).size(0)
        train_dict_list = {key.item():[] for key in torch.unique(data.y)}
        val_test_list = []
        #   target_graph_list 就是下游数据集包括的全部 子图
        target_graph_list = induced_graphs(data)    #   （每个节点都生成一个以该节点为中心的诱导子图）
        
        from torch.utils.data import random_split, Subset
        for index, graph in enumerate(target_graph_list):
            i_class = graph.y  
            if( len(train_dict_list[i_class]) >= few_shot):
                val_test_list.append(graph)
            else:
                train_dict_list[i_class].append(index)       
        all_indices = []
        for i_class, indice_list in train_dict_list.items():
            all_indices+=indice_list
        train_set = Subset(target_graph_list, all_indices)  #   train_set每个类别 shot个 子图
        val_set, test_set = random_split(val_test_list, [0.1,0.9], 
                                        torch.Generator().manual_seed(seed))
        
        results = [     
        {
            'train': train_set,
            'val': val_set,
            'test': test_set,
        }, 
            num_classes
        ]
        
        
        torch.save(results,     
                os.path.join(downstream_Graph_classify, f'{data_name}_seed{seed}_featsize{node_feats}_shot{few_shot}.pt'  )     )
        

        return results 

        



#   下游微调，使用注意力,做图分类.  这个方法在单图数据集、多图数据集都可以使用
def Adapt_Graph(num_epochs_adapt = 100,  
          learning_rate_adapt = 1e-3,  
          weight_decay_adapt = 1e-5, 
          adapt_data='CiteSeer' , # 下游数据集名称
          pre_datasets = ['PubMed','Photo','Computers'], #  预训练数据集
          adapt_seed = 0,
          model_name = 'GCN',
          node_feats = 100,
          hidden_feats = 128 ,
          few_shot = 1,
          num_head = 2,
          
          ):


    import numpy as np
    random.seed(adapt_seed)
    np.random.seed(adapt_seed)
    torch.manual_seed(adapt_seed)
    torch.cuda.manual_seed(adapt_seed)
    torch.cuda.manual_seed_all(adapt_seed)
        


    import os
    pretrained_model_and_connectors = os.path.join(data_dir,'预训练模型和连接点')  
    if not os.path.exists(pretrained_model_and_connectors):
        #   如果保存下游图分类数据集的文件夹不存在，那么就新建一个
        os.makedirs(pretrained_model_and_connectors)
    else:
        # print(f"文件夹已存在: {downstream_Graph_classify}")
        pass
    pretrain_datasets = '_'.join(pre_datasets)


    # 取出预训练的 GNN模型
    pretrained_model = torch.load(  os.path.join(  pretrained_model_and_connectors  , f'Pretrained_{model_name}_of_{pretrain_datasets}.pt') ,weights_only=False )
    pretrained_model = pretrained_model.to(device)

    #   预训练的所有数据集的连接点 
    pretrained_connectors = torch.load(   os.path.join(  pretrained_model_and_connectors ,
                                           f'Connectors_of_{pretrain_datasets}.pt')    )
    

    connectors = nn.Parameter(torch.stack(list(pretrained_connectors.values())).to(device),
                                requires_grad = False
                              )     #   requires_grad表示有梯度，可以更新这个参数
    



    #   生成下游图分类数据集，包含训练、验证、测试三部分
    datasets_adapt, num_classes = get_downstream_dataset_graph_classification(dataset_dir = data_dir,
                        data_name = adapt_data, seed = adapt_seed ,node_feats = node_feats, few_shot = few_shot) 
    loaders = { k: DataLoader(v, batch_size=100, shuffle=True) for k, v in datasets_adapt.items() }
    Classifier_adapt = MLP(hidden_feats, hidden_feats ,num_classes).to(device)
    Attention = NodeLevelAttention(embed_dim  = node_feats  ,  num_heads = num_head ).to(device)


    optimizer_adapt = torch.optim.Adam(
        [
        # 第一组：GNN和分类器参数
        {'params': list(pretrained_model.parameters()) 
         + list(Classifier_adapt.parameters()) 
         + list(Attention.parameters())  , 
         'lr': learning_rate_adapt },

        # 第二组：connector参数
        {'params': [connectors], 'lr': 5e-3}  
        ], 
        weight_decay = weight_decay_adapt)


    loss_metric = MeanMetric()

    acc_metric = Accuracy(task='multiclass', num_classes=num_classes).to(device)
    #   Macro-F1 ：把每个类别转换成二分类问题，然后有一个二分类F1。把所有类别的二分类F1求平均值，得到 宏F1
    f1_metric = F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device)
    auroc_metric = AUROC(task="multiclass", num_classes=num_classes).to(device)

    best_acc = 0.
    best_model = None

    val_accuracies = []
    for e in range(num_epochs_adapt):  
        pretrained_model.train()
        Classifier_adapt.train()
        Attention.train() 

        loss_metric.reset()
        acc_metric.reset()
        f1_metric.reset()
        auroc_metric.reset()


        #   训练数据集
        pbar = tqdm(loaders['train'], total=len(loaders['train']), ncols=100)
        
        loss_list = []
        #   训练数据集会包含   多个batch。
        for batch in pbar:
            optimizer_adapt.zero_grad()  
            batch = batch.to(device)
            # 特征增强
            batch.x , atten_weights = Attention(batch.x  ,  connectors)
            graph_emb,_ = pretrained_model(batch) 
            pred_score = Classifier_adapt(graph_emb)#   这里的pred是每个图 的 所有类别的预测分数（长度为8的向量，表示8个类别的预测值，可以是负数）
            
            loss = torch.nn.functional.cross_entropy(pred_score, batch.y) #   每个图的预测值向量  先经过softmax得到预测概率（8个类别的概率的和为1），然后取正确类别的概率的 -ln值，这样正确类别的预测概率越大，其他类别的预测概率越小，LOSS值就越小
            loss.backward() #   LOSS对 参数求梯度    
            optimizer_adapt.step()    #   更新参数，更新GNN参数（backbone)  和  下游任务头（2层MLP）

              # 保留4位小数

            loss_metric.update(  round(loss.item(), 4)  )     #   累积
            loss_list.append(  round(loss.item(), 4)  )
            
        pbar.close()

        #   compute()是用之前累积的所有值   求平均
        avg_loss = round(loss_metric.compute().item(),4)#   这一epoch的  训练集loss（所有batch）
        avg = round (  sum(loss_list) / len(loss_list) ,  4)    # 同上


        # 这一轮的  【  验证  】
        pretrained_model.eval()
        Classifier_adapt.eval()
        # attention.eval()                
        pbar = tqdm(loaders['val'], total=len(loaders['val']), ncols=100, desc=f'Epoch {e} Validation, Acc: 0., F1: 0.')
        with torch.no_grad():   #   没有梯度：代表不会更新模型参数
            for batch in pbar:  #   
                batch = batch.to(device)
                batch.x , atten_weights = Attention(  batch.x,  connectors)

                graph_emb,_ = pretrained_model(batch)  
                pred_score = Classifier_adapt(graph_emb)    #  预测分数
                pred_class = pred_score.argmax(dim=-1)#   预测类别   

                      

                #   每次调用update的时候，会把新的【预测值，真实值】加入累积数据当中。因此最后的acc_metirc代表的是整个验证集所有batch的【预测值，真实值】累积
                acc_metric.update(pred_class, batch.y)    
                f1_metric.update(pred_class, batch.y)
                auroc_metric.update(pred_score, batch.y)

                #   update()会累积值，compute()会根据之前所有累积的值来计算最终ACC
                pbar.set_description(f'Epoch {e} Validation Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
            pbar.close()

        val_accuracies.append(acc_metric.compute().item())
        if acc_metric.compute() > best_acc:
            best_acc = acc_metric.compute()
            best_model = deepcopy(pretrained_model)

        


    #   上述有100轮，每一轮都会有一个ACC，我们选取ACC最高的那一轮（表示那一轮的模型预测效果最好，就是在验证集上效果最好），用那一轮的模型来作为最优模型
    model = best_model if best_model is not None else pretrained_model 
    #   TEST:  微调（再训练）的100轮结束，选取验证集上效果最好的一轮，用这一轮的模型来作为最优模型。
    model.eval()
    # attention.eval()
    #   把之前累积的【预测值，真实值】清空
    acc_metric.reset()
    f1_metric.reset()
    auroc_metric.reset()

    attention_list = []
    pbar = tqdm(loaders['test'], total=len(loaders['test']), ncols=100, desc=f'Testing, Acc: 0., F1: 0.')
    with torch.no_grad():
        for batch in pbar:
            
            batch = batch.to(device)
            batch.x , atten_weights = Attention(batch.x  ,  connectors) 
            #   算ACC和 F1，用预测类别
            graph_emb,_ = model(batch)  #    下游图分类，只获取返回的图向量graph_emb，  节点向量不使用
            pred_score = Classifier_adapt(graph_emb)   #   每个类别的预测分数
            pred_class = pred_score.argmax(dim=-1) #   经过argmax预测类别（只有一个）


            acc_metric.update(pred_class, batch.y)    #   预测类别pred  和  真实类别标签y  ，计算出ACC（预测准确率）
            f1_metric.update(pred_class, batch.y)
            #   算AUC，要用预测分数
            auroc_metric.update(pred_score, batch.y) 
            
             #   以下输出的是  目前所有batch累加的  acc，AUC, F1。
            pbar.set_description(f'Testing Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
        


            # ########   分析注意力参数
            # atten_weights = atten_weights.squeeze(1)

            # correct_mask = (pred_class == batch.y)  # [num_graphs, ]，True表示该图预测正确
            # correct_graph_indices = correct_mask.nonzero().squeeze(-1)  # 预测正确的图的id （在batch中的id）

            # # 获取这些正确预测的图的  所有节点ID
            # node_mask = torch.isin(batch.batch, correct_graph_indices)  # [num_nodes, ]，True表示该节点属于预测正确的图
            # correct_node_indices = node_mask.nonzero().squeeze(-1)  # 预测正确的图的所有节点的索引

            # # 提取预测正确节点的 注意力权重
            # correct_atten_weights = atten_weights[correct_node_indices]  # [num_correct_nodes, 5]
        
            # # mean_atten = torch.mean(correct_atten_weights,dim=0)
            # attention_list.append(correct_atten_weights)
            # pass

        # all_batchs_weight = torch.cat(attention_list, dim=0) 
        # mean_atten = torch.mean(all_batchs_weight,dim=0)
        
        pbar.close()




    results = {
        "ACC":acc_metric.compute().item(),
        "AUROC":auroc_metric.compute().item(),
        "F1":f1_metric.compute().item()
    }

    return results


    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "DDGFM")
    parser.add_argument('--num_neg', type=int, default=60)
    parser.add_argument('--repeat_times', type=int, default=5)
    #   预训练阶段的学习率固定了，就在pretrain函数里面。下面是  下游阶段的学习率
    parser.add_argument('--lr_adapt', type=float, default=1e-3)     #   下游的学习率
    parser.add_argument('--wd_adapt', type=float, default=1e-5)

    parser.add_argument('--pretrain_epochs', type=int, default = 250)
    parser.add_argument('--global_num_heads', type=int, default = 2)    #   下游注意力头数
    parser.add_argument('--model_name', type=str, default='FAGCN')


    parser.add_argument('--node_fts', type=int, default=50)#    统一后的初始节点特征维度

    parser.add_argument('--downstream_few_shot', type=int, default= 3 )   #   下游节点分类或者图分类的时候，每个类别有几个 训练样本

################################         需要搜索的超参


    parser.add_argument('--nm_subgs', type=int, default=35)#    每个数据集采集的  子图个数
    parser.add_argument('--walk_len', type=int, default=25) #   每个子图的  游走路径长度
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--regular', type=float, default= 0.45 ) #控制方向的正则项  的系数
    parser.add_argument('--regular_si', type=float, default= 0.85 ) #控制扩张系数的正则项  的系数
    parser.add_argument('--regular_3', type=float, default= 0.85 ) #控制 凸包不要过度扩大，也就是任意两个源域要有最基本的相似度
    parser.add_argument('--min_sim', type=float, default= -0.7 ) #  两个源域的 最小余弦相似度



    parser.add_argument('--pre_dataset', type=str, default='Cora_CiteSeer_PubMed_Computers', help='Use "_" to split')
    parser.add_argument('--adapt_dataset', type=str, default='Photo')

    #   预训练和下游阶段的  seed
    parser.add_argument('--pre_seed', type=int, default=42)
    #   这个表示下游阶段要不要做   图分类
    parser.add_argument('--graph_cls', type = bool, default = True)

    #   把所有  命令行参数都读取进来
    args = parser.parse_args()
    # 处理 pre_dataset 字符串为列表
    pre_dataset_list = args.pre_dataset.strip().split('_')


    #########################   图分类
    if args.graph_cls :
        ACC_List, AUROC_List, F1_List = [], [], []
        for seed in range(1024,1024+5):
            # === 预训练 ===
            if seed==1024 :    #   加这句if，那就是1次预训练，搭配5次微调。 （不加的话，那就是1次预训练+1次微调，做5次）
                pretrain(
                    num_epochs = args.pretrain_epochs,
                    num_subgraphs=args.nm_subgs,
                    walk_length=args.walk_len,
                    seed = seed,
                    neg_numbers=args.num_neg,
                    pre_datasets=pre_dataset_list,
                    model_name=args.model_name,
                    node_feats=args.node_fts,
                    hidden_fts=args.hidden,
                    regular = args.regular,
                    regular_si = args.regular_si,
                    regular_3 = args.regular_3,
                    min_sim = args.min_sim
                )

            # === 下游图分类（微调） ===
            results = Adapt_Graph(
                learning_rate_adapt=args.lr_adapt,
                weight_decay_adapt=args.wd_adapt,
                pre_datasets = pre_dataset_list,
                adapt_data=args.adapt_dataset,
                adapt_seed = seed,
                model_name=args.model_name,
                node_feats=args.node_fts,
                hidden_feats=args.hidden,
                few_shot=args.downstream_few_shot,
                num_head=args.global_num_heads,
            )
            ACC_List.append(results['ACC'])
            AUROC_List.append(results['AUROC'])
            F1_List.append(results['F1'])

        output_path = os.path.join(data_dir, '微调结果【图分类】.txt')
        with open(output_path, 'a+') as f:
            f.write('-------------------------------------------------\n')
            f.write(f'输出时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n') 
            f.write(f'预训练数据:{args.pre_dataset}, 下游数据集： : 【{args.adapt_dataset}】\n')
            f.write(f'ACC :  {np.mean(ACC_List):.4f} ± {np.std(ACC_List):.4f} \n')
            f.write(f'F1 :  {np.mean(F1_List):.4f} ± {np.std(F1_List):.4f} \n')
            f.write(f'AUC :  {np.mean(AUROC_List):.4f} ± {np.std(AUROC_List):.4f} \n')


