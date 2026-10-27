from torch_cluster import random_walk
from torch_geometric.datasets import Planetoid, Amazon ,   FacebookPagePage ,LastFMAsia  ,HeterophilousGraphDataset,WikipediaNetwork,WebKB,Flickr,Coauthor  ,GitHub
from torch_geometric.datasets import TUDataset, WikiCS
from torch_geometric.data import Data,Batch
import torch
import random
import os
from torch_geometric.transforms import SVDFeatureReduction
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader   
from torchmetrics import MeanMetric, Accuracy, F1Score, AUROC
from tqdm import tqdm
from copy import deepcopy
from utils import *
import numpy as np
import argparse
import time
from datetime import datetime 

now_file = os.path.abspath(__file__)   
current_dir = os.path.dirname(now_file) 
data_dir = os.path.join(os.path.dirname(current_dir) , 'DDGFM_Data_dir')


DOWNSTREAM_EPOCH_NUMBER = 150
SEED_NUM = 10
PRETRAIN_EPOCHS = 450


if torch.cuda.is_available():
    device = torch.device("cuda:0")  
else:
    device = torch.device("cpu")  

PyG_Dataset_Dir = os.path.join(data_dir ,"PyG_Dataset" )




def pretrain(    num_epochs = 100,    
             num_subgraphs = 30,    
             walk_length = 30 ,
             seed = 42 ,
             neg_numbers = 280 ,
             adapt_dataset = 'citeseer',
             model_name = 'FAGCN',
             node_feats = 128,
             hidden_fts = 128,
             pre_lr = 1e-4,
             pre_wd = 1e-5,
             pre_datasets = ['Cora','PubMed','CiteSeer','Photo','Computers'],
             regular = 0.1,
             regular_si = 0.1,
             regular_3 = 0.1,  
             min_sim = -0.5,
             
             ):

    pretrained_model_and_connectors = os.path.join(data_dir,'Pretrained_Checkpoint')  
    if not os.path.exists(pretrained_model_and_connectors):
        os.makedirs(pretrained_model_and_connectors)
    else:
        pass

    pretrain_datasets = '_'.join(pre_datasets)
    # #   
    # if os.path.exists(os.path.join(  pretrained_model_and_connectors  , f'Pretrained_{model_name}_of_{pretrain_datasets}.pt')):
    #     return

    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # roman_empire,lastFM

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
        elif data_name == 'CoauthorCS':
            datasets[data_name] = Coauthor(root = PyG_Dataset_Dir , name = 'CS')._data
        # elif data_name == 'GitHub':
        #     datasets[data_name] = GitHub(root = os.path.join(PyG_Dataset_Dir,'GitHub'))._data
        #   
        elif data_name == 'amazonratings':
            datasets[data_name] = HeterophilousGraphDataset(root=PyG_Dataset_Dir , name = 'amazon_ratings')._data
        elif data_name == 'romanempire':
            datasets[data_name] = HeterophilousGraphDataset(root=PyG_Dataset_Dir , name = 'Roman-empire')._data 
        elif data_name == 'Questions':
            datasets[data_name] = HeterophilousGraphDataset(root=PyG_Dataset_Dir , name = 'Questions')._data 
           
        elif data_name == 'chameleon':
            datasets[data_name] = WikipediaNetwork(root=PyG_Dataset_Dir, name='chameleon')._data
        elif data_name == 'squirrel':
            datasets[data_name] = WikipediaNetwork(root=PyG_Dataset_Dir, name='squirrel')._data
        elif data_name == 'cornell':
            datasets[data_name] = WebKB(root = PyG_Dataset_Dir, name = 'cornell')._data

        elif data_name in ['PROTEINS','ENZYMES','COX2','MUTAG','BZR']:
        #   
            datasets[data_name] = TUDataset(root=PyG_Dataset_Dir,name = data_name)
        elif data_name == 'Facebook':
            datasets[data_name] = FacebookPagePage(root = os.path.join(PyG_Dataset_Dir,'Facebook') )[0]  

        elif data_name == 'lastFM':
            datasets[data_name]  = LastFMAsia(   root = os.path.join(PyG_Dataset_Dir,'LastFM')  )._data
        
        elif data_name == "WikiCS":
            datasets[data_name] = WikiCS(  root = os.path.join(PyG_Dataset_Dir,'WikiCS')  )._data


    
    for data_name, data in datasets.items():

        if data_name in ['PROTEINS','ENZYMES','COX2','MUTAG','BZR']:   
            #   
            graph_list = []
            for graph_id in range(data.len()):
                # graph__processed = preprocess(data[graph_id],node_feature_dim = node_feats)
                # graph_list.append(graph__processed)
                # pass
                #   
                single_mole_graph = data[graph_id].clone()
                #   
                x = single_mole_graph.x.clone().to(device)
                num_nodes = single_mole_graph.x.size(0)

                target_dim = node_feats
                #   
                feat_dim = x.size(1)

                pad = torch.zeros(
                    (num_nodes, target_dim - feat_dim),
                    dtype = x.dtype
                        ).to(device)
                x = torch.cat([x, pad], dim=1).to(device)

                single_mole_graph.x = x
                graph_list.append(single_mole_graph)
            #  
            datasets[data_name] = graph_list
            pass
        else:

            #   
            datasets[data_name] = preprocess(data, node_feature_dim = node_feats)


    
    subgraphs = {}
    for data_name, data in datasets.items():
        #   
        if data_name in ['PROTEINS','ENZYMES','COX2','MUTAG','BZR']:   
            subgraphs[data_name] = random.sample(data, num_subgraphs)

            pass
        else:
            #   
            subgraphs[data_name] = get_subgraphs(data,num_subgraphs=num_subgraphs,walk_length=walk_length)




    connectors = {}
    for data_name, data in datasets.items():    
        if data_name in ['PROTEINS','ENZYMES','COX2','MUTAG','BZR']:
            # 
            accum = None
            sum_x_list = []  # 
            for subgraph_ in data:
                sum_x = torch.mean(subgraph_.x, dim=0)  # 
                sum_x_list.append(sum_x)
            accum = torch.mean(torch.stack(sum_x_list), dim=0)
            connectors[data_name] = accum
            pass
        else:#  
            x = data.x.to(device)
            edge_index, _ = add_self_loops(
                data.edge_index.to(device),
                num_nodes=data.num_nodes,
            )
            row, col = edge_index

            
            degree = torch.bincount(row, minlength=data.num_nodes).to(dtype=x.dtype)
            degree_inv_sqrt = degree.pow(-0.5)
            degree_inv_sqrt.masked_fill_(torch.isinf(degree_inv_sqrt), 0.0)
            norm = degree_inv_sqrt[row] * degree_inv_sqrt[col]

            for _ in range(2):
                propagated_x = torch.zeros_like(x)
                propagated_x.index_add_(
                    0,
                    row,
                    x[col] * norm.unsqueeze(-1),
                )
                x = propagated_x

            connectors[data_name] = torch.mean(x, dim=0)
    original_domain_tokens = list(connectors.values())
    original_domain_tokens = torch.stack(original_domain_tokens, dim=0)  
    original_domain_tokens = original_domain_tokens.to(device)

    

    num_domains = len(subgraphs)

    from torch_geometric.data import Batch


    all_subgraphs_list = []
    for data_name, sg_list in subgraphs.items():
        all_subgraphs_list.extend(sg_list)
    
    for a_subgraph in all_subgraphs_list:
        a_subgraph = a_subgraph.to(device)

    #   
    batch_graph = Batch.from_data_list(all_subgraphs_list)



    if model_name == 'GCN':
        GNN_Model = GCN(num_features = node_feats  ,  hidden = hidden_fts).to(device)
        
    elif model_name == 'FAGCN':
        GNN_Model = FAGCN(num_features = node_feats,
                          hidden= hidden_fts , 
                          num_conv_layers= 2 ,
                          
                          dropout = 0.15, 
                          epsilon = 0.2,
                          num_head = 4 #    
                          ).to(device)
        
    elif model_name == 'GAT':
        GNN_Model = GAT(num_features = node_feats  ,  hidden = hidden_fts).to(device)



    encoder_layer = nn.TransformerEncoderLayer(
        d_model= hidden_fts,
        dim_feedforward = 256,
        nhead= 2,
        batch_first=True,   # 
        dropout=0.0         # 
    )

    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = 6)
    transformer_encoder = transformer_encoder.to(device)



    total_params = sum(p.numel() for p in transformer_encoder.parameters() if p.requires_grad)


    params_in_mb = total_params * 4 / (1024 ** 2)



    loss_fn = ContrastiveLoss().to(device)
    loss_fn.train() 


    optimizer_pretrain = torch.optim.Adam(  list(  GNN_Model.parameters() )  + list(transformer_encoder.parameters()),  
                                          lr = pre_lr,
                                          weight_decay =  pre_wd)


    start_time = time.time()  # 

    for epoch in range(num_epochs):
        GNN_Model.train()
        # 
        transformer_encoder.train()
        total_loss = 0
        # 
        batch_graph = batch_graph.to(device)
        optimizer_pretrain.zero_grad()

        #  
        graph_embedding, node_embed = GNN_Model(data=batch_graph)
        #  
        graph_embedding = ( graph_embedding / torch.norm(graph_embedding, dim=1, keepdim=True) )
        

        # 
        input_tokens = graph_embedding.unsqueeze(0)  

        # 
        output_tokens = transformer_encoder(input_tokens)  # [1, total_subgraphs, node_feats]
        # 
        output_tokens = output_tokens.squeeze(0)  # [total_subgraphs, node_feats]
        

        
        # 
        domain_tokens = {}
        for data_name, data in datasets.items():    

            if data_name in ['PROTEINS','ENZYMES','COX2','MUTAG','BZR']:

                for _ in data:
                    _ = _.to(device)
                molecules_batch = Batch.from_data_list(data).to(device)
                graph_emb, node_h = GNN_Model(data=molecules_batch)
                domain_tokens[data_name] = torch.mean(node_h,dim=0)
                
                pass
            else:#  
                #   
                single_graph_embed, node_h = GNN_Model(data=data.to(device))
                domain_tokens[data_name] = torch.mean(node_h,dim=0)

        new_domain_tokens = list(domain_tokens.values())
        new_domain_tokens = torch.stack(new_domain_tokens, dim=0)   # 


        
        c_0 = original_domain_tokens.mean(dim=0)   # 

        dir_loss = 0.0  #   
        R_out = 0.0 #   

        for i in range(num_domains):

            #   
            v1 = original_domain_tokens[i] - c_0
            v2 = new_domain_tokens[i] - c_0
            dir_loss += 1 - F.cosine_similarity(v1, v2, dim=0) 



            ori_norm = torch.norm(v1)
            new_norm = torch.norm(v2)

            score = 1 - new_norm / (ori_norm )
            # 
            R_out += torch.maximum(score, torch.tensor(0.0, device=device))
        



        loss = loss_fn(output_tokens, output_tokens, 
                       num_domains = num_domains ,
                       num_subgraphs = num_subgraphs )   + regular*dir_loss  + regular_si * R_out  
        
        loss.backward()
        optimizer_pretrain.step()
        total_loss += loss.item()


        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.7f}")


        # 
        GNN_Model.eval()
        if epoch == (num_epochs - 1) :
            # 
            domain_tokens = {}
            for data_name, data in datasets.items():    

                if data_name in ['PROTEINS','ENZYMES','COX2','MUTAG','BZR']:

                    for _ in data:
                        _ = _.to(device)
                    molecules_batch = Batch.from_data_list(data).to(device)
                    graph_emb, node_h = GNN_Model(data=molecules_batch)
                    domain_tokens[data_name] = torch.mean(node_h,dim=0)
                    
                    pass
                else:#  

                    single_graph_embed, node_h = GNN_Model(data=data.to(device))
                    domain_tokens[data_name] = torch.mean(node_h,dim=0)
            #  
            final_GNN_encoded_domain_tokens = list(domain_tokens.values())
            final_GNN_encoded_domain_tokens = torch.stack(final_GNN_encoded_domain_tokens, dim=0)  



    
    end_time = time.time()  # 
    elapsed_time = end_time - start_time  # 
    print(f"代码运行时间: {elapsed_time:.6f} 秒")    

    #  
    torch.save(GNN_Model,   os.path.join(  pretrained_model_and_connectors  ,
                                           f'Pretrained_{model_name}_of_{pretrain_datasets}.pt')   )
 
    torch.save(transformer_encoder,   os.path.join(  pretrained_model_and_connectors  ,
                                           f'Pretrained_TFencoder_of_{pretrain_datasets}.pt')   )
    torch.save(final_GNN_encoded_domain_tokens, os.path.join(  pretrained_model_and_connectors  ,
                                        f'final_tokens_of_{pretrain_datasets}.pt') )




def get_downstream_dataset_graph_classification(dataset_dir,
                        data_name,  
                        seed, 
                        few_shot=1,
                        node_feats = 128 ,
                        ):


    import os
    downstream_Graph_classify = os.path.join(dataset_dir,'downstream_Graph_Classify')  
    if not os.path.exists(downstream_Graph_classify):
        os.makedirs(downstream_Graph_classify)
    else:
        pass


    #   
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
    elif data_name == 'GitHub' :
        data = GitHub(root=os.path.join(PyG_Dataset_Dir,'GitHub'))[0]
    elif data_name == 'lastFM':
        data = LastFMAsia(root= os.path.join(PyG_Dataset_Dir,'LastFM'))._data
    elif data_name == 'chameleon':
        data = WikipediaNetwork(root=PyG_Dataset_Dir, name='chameleon')._data
    elif data_name == 'squirrel':
        data = WikipediaNetwork(root=PyG_Dataset_Dir, name='squirrel')._data
    elif data_name == 'roman_empire':
        data = HeterophilousGraphDataset(root=PyG_Dataset_Dir , name = 'roman_empire')._data

    elif data_name == 'Questions':
        data = HeterophilousGraphDataset(root=PyG_Dataset_Dir , name = 'Questions')._data 

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
    elif data_name == 'COX2':
        data = TUDataset(root=PyG_Dataset_Dir,name = 'COX2')
    elif data_name == 'BZR':
        data = TUDataset(root=PyG_Dataset_Dir,name = 'BZR')

    elif data_name == 'arxiv':
        from ogb.nodeproppred import PygNodePropPredDataset

        dataset = PygNodePropPredDataset(root = PyG_Dataset_Dir ,name = 'ogbn-arxiv') 
        #   
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        data = dataset[0] # pyg graph object



    else:
        print("Not allowed Dataset")



    
    #   
    if data_name in ['PROTEINS','ENZYMES','COX2','BZR']:
        #   
        graph_list = [] #   
        # 
        for graph_id in range(data.len()):

            
            single_mole_graph = data[graph_id].clone()
            
            x = single_mole_graph.x.clone().to(device)
            num_nodes = single_mole_graph.x.size(0)
            target_dim = node_feats
            
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
        
        target_graph_list = graph_list   
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

        train_set = Subset(target_graph_list, all_indices)  
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

    
    else:

        #
        data = preprocess(data,node_feature_dim = node_feats)
        num_classes = torch.unique(data.y).size(0)
        train_dict_list = {key.item():[] for key in torch.unique(data.y)}
        val_test_list = []
        #   
        target_graph_list = induced_graphs(data)    #   
        
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
        train_set = Subset(target_graph_list, all_indices)  #   
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

        



#   
def Adapt_Graph(num_epochs_adapt = DOWNSTREAM_EPOCH_NUMBER,     
          learning_rate_adapt = 1e-3,  
          weight_decay_adapt = 1e-5, 
          adapt_data='CiteSeer' , 
          pre_datasets = ['PubMed','Photo','Computers'], 
          adapt_seed = 0,
          model_name = 'GCN',
          node_feats = 128,
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
    pretrained_model_and_connectors = os.path.join(data_dir,'Pretrained_Checkpoint')  
    if not os.path.exists(pretrained_model_and_connectors):
        
        os.makedirs(pretrained_model_and_connectors)
    else:
        
        pass
    pretrain_datasets = '_'.join(pre_datasets)




    
    transformer_encoder =  torch.load(  os.path.join(  pretrained_model_and_connectors  , f'Pretrained_TFencoder_of_{pretrain_datasets}.pt') , 
                                      weights_only=False )
    transformer_encoder = transformer_encoder.to(device)





    
    GNN_Model = torch.load(  os.path.join(  pretrained_model_and_connectors  , f'Pretrained_{model_name}_of_{pretrain_datasets}.pt') ,
                           weights_only=False )
    GNN_Model = GNN_Model.to(device)
    

    
    final_tokens = torch.load( os.path.join(  pretrained_model_and_connectors  ,
                                        f'final_tokens_of_{pretrain_datasets}.pt') )
    final_tokens = final_tokens.to(device)


    
    datasets_adapt, num_classes = get_downstream_dataset_graph_classification(dataset_dir = data_dir,
                        data_name = adapt_data, seed = adapt_seed ,node_feats = node_feats, few_shot = few_shot) 
    loaders = { k: DataLoader(v, batch_size=100, shuffle=True) for k, v in datasets_adapt.items() }
    
    Classifier_adapt = MLP(hidden_feats, hidden_feats ,num_classes).to(device)


    optimizer_adapt = torch.optim.Adam(
        [
        # 
        {'params':list(GNN_Model.parameters())  + list(transformer_encoder.parameters())  + list(Classifier_adapt.parameters()) 
         , 
         'lr': learning_rate_adapt },

        ], 
        weight_decay = weight_decay_adapt)


    loss_metric = MeanMetric()

    acc_metric = Accuracy(task='multiclass', num_classes=num_classes).to(device)
    #   
    f1_metric = F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device)
    auroc_metric = AUROC(task="multiclass", num_classes=num_classes).to(device)

    best_acc = 0.
    best_model = None

    val_accuracies = []
    for e in range(num_epochs_adapt):  
        GNN_Model.train()
        Classifier_adapt.train()
        transformer_encoder.train()

        loss_metric.reset()
        acc_metric.reset()
        f1_metric.reset()
        auroc_metric.reset()


        #   
        pbar = tqdm(loaders['train'], total=len(loaders['train']), ncols=100)
        
        loss_list = []
        #   
        for batch in pbar:
            optimizer_adapt.zero_grad()  
            batch = batch.to(device)


            graph_emb,_ = GNN_Model(batch)  
        

            graph_nums = graph_emb.size(0)

            cls_tokens = graph_emb.unsqueeze(1)  # (graph_nums , 1, 128)



            domain_tokens_batch = final_tokens.expand(graph_nums, -1, -1) 

            seq = torch.cat([cls_tokens, domain_tokens_batch], dim=1)  # 
            # 
            output = transformer_encoder(seq)  # 
            # 
            cls_output = output[:, 0, :]  # (batch, 128)
            # 
            pred_score = Classifier_adapt(cls_output)


            loss = torch.nn.functional.cross_entropy(pred_score, batch.y) #  
            loss.backward() #    
            optimizer_adapt.step()    #  

              # 

            loss_metric.update(  round(loss.item(), 4)  )     #   
            loss_list.append(  round(loss.item(), 4)  )
            
        pbar.close()




        transformer_encoder.eval()
        Classifier_adapt.eval()
        GNN_Model.eval()
                     
        pbar = tqdm(loaders['val'], total=len(loaders['val']), ncols=100, desc=f'Epoch {e} Validation, Acc: 0., F1: 0.')
        with torch.no_grad():   #   
            for batch in pbar:  #   

                batch = batch.to(device)

                graph_emb,_ = GNN_Model(batch)  
            
                #   
                graph_nums = graph_emb.size(0)
                # 
                cls_tokens = graph_emb.unsqueeze(1)  # (graph_nums , 1, 128)

                # 
                domain_tokens_batch = final_tokens.expand(graph_nums, -1, -1)  # 
                # 
                seq = torch.cat([cls_tokens, domain_tokens_batch], dim=1)  # 
                # 
                output = transformer_encoder(seq)  # 
                # 
                cls_output = output[:, 0, :]  # (batch, 128)
                # 
                pred_score = Classifier_adapt(cls_output)




                pred_class = pred_score.argmax(dim=-1)#   

                      

                #   
                acc_metric.update(pred_class, batch.y)    
                f1_metric.update(pred_class, batch.y)
                auroc_metric.update(pred_score, batch.y)

                
                pbar.set_description(f'Epoch {e} Validation Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
            pbar.close()

        val_accuracies.append(acc_metric.compute().item())

        if acc_metric.compute() > best_acc:
            best_acc = acc_metric.compute()

            best_TFencoder_model = deepcopy(transformer_encoder)
            best_GNN_model = deepcopy(GNN_Model)
            best_MLP = deepcopy(Classifier_adapt)

        


    #   
    TF_encoder_model = best_TFencoder_model if best_TFencoder_model is not None else transformer_encoder 
    GNN__model = best_GNN_model if best_GNN_model is not None else GNN_Model
    MLP__model = best_MLP if best_MLP is not None else Classifier_adapt



    #   
    TF_encoder_model.eval()
    GNN__model.eval()
    MLP__model.eval()

    #   
    acc_metric.reset()
    f1_metric.reset()
    auroc_metric.reset()

    attention_list = []
    pbar = tqdm(loaders['test'], total=len(loaders['test']), ncols=100, desc=f'Testing, Acc: 0., F1: 0.')
    with torch.no_grad():
        for batch in pbar:
            
            batch = batch.to(device)


            graph_emb,_ = GNN__model(batch)  
            
            graph_nums = graph_emb.size(0)
            
            cls_tokens = graph_emb.unsqueeze(1)  # 

            
            domain_tokens_batch = final_tokens.expand(graph_nums, -1, -1)  # 
            # 
            seq = torch.cat([cls_tokens, domain_tokens_batch], dim=1)  # 
            # 
            output = TF_encoder_model(seq)  # 
            # 
            cls_output = output[:, 0, :]  # (batch, 128)
            # 
            pred_score = MLP__model(cls_output)

            pred_class = pred_score.argmax(dim=-1)#  
            
            acc_metric.update(pred_class, batch.y)    #   
            f1_metric.update(pred_class, batch.y)
            #   
            auroc_metric.update(pred_score, batch.y) 
            
             #   
            pbar.set_description(f'Testing Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
        

        
        pbar.close()




    results = {
        "ACC":acc_metric.compute().item(),
        "AUROC":auroc_metric.compute().item(),
        "F1":f1_metric.compute().item()
    }

    return results



    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "DDGFM")
    parser.add_argument('--num_neg', type=int, default=60)  # 
    parser.add_argument('--repeat_times', type=int, default=5) # 
    #   
    parser.add_argument('--lr_adapt', type=float, default=1e-3)     #  
    parser.add_argument('--wd_adapt', type=float, default=1e-5)
    parser.add_argument('--pretrain_epochs', type=int, default = PRETRAIN_EPOCHS)   # 
    parser.add_argument('--global_num_heads', type=int, default = 2)    # 
    parser.add_argument('--model_name', type=str, default='FAGCN')  #     
    parser.add_argument('--node_fts', type=int, default= 128)#   
    parser.add_argument('--hidden', type=int, default=128)  #  
    parser.add_argument('--downstream_few_shot', type=int, default= 1 )   #   
    #   
    parser.add_argument('--pre_seed', type=int, default=42)
    #   
    parser.add_argument('--graph_cls', type = bool, default = True)
    #   
    parser.add_argument('--node_cls', type = bool, default = False)  # 


    parser.add_argument('--regular_3', type=float, default= 0.0 ) 

    
    parser.add_argument('--min_sim', type=float, default= -0.75 ) # 

    parser.add_argument('--nm_subgs', type=int, default= 50)#    
    parser.add_argument('--walk_len', type=int, default= 20) #   
    parser.add_argument('--regular', type=float, default= 0.95 ) #
    parser.add_argument('--regular_si', type=float, default= 0.1 ) #
    
    

    parser.add_argument('--pre_dataset', type=str, default='Cora_CiteSeer', help='Use "_" to split')
    
    parser.add_argument('--adapt_dataset', type=str, default='Cora')




    args = parser.parse_args()

    pre_dataset_list = args.pre_dataset.strip().split('_')


    ######
    if args.graph_cls :
        ACC_List, AUROC_List, F1_List = [], [], []
        for seed in range(1024,1024 + SEED_NUM): #  
              # 

              # 
            
            if seed==1024 :    #   
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



            # 
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

        output_path = os.path.join(data_dir, 'fine-tuning results.txt')
        with open(output_path, 'a+') as f:
            f.write('-----------------------DDGFM--------------------------\n')
            f.write(f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n') 
            f.write(f'Pretrain:{args.pre_dataset}, Downstream: 【{args.adapt_dataset}】\n')
            f.write(f'ACC :  {np.mean(ACC_List):.4f} ± {np.std(ACC_List):.4f} \n')
            # f.write(f'F1 :  {np.mean(F1_List):.4f} ± {np.std(F1_List):.4f} \n')
            # f.write(f'AUC :  {np.mean(AUROC_List):.4f} ± {np.std(AUROC_List):.4f} \n')

