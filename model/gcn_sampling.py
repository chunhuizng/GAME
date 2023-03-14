import grb.utils as utils
from grb.dataset import Dataset
from grb.utils.normalize import GCNAdjNorm
from grb.utils.sample import cluster_sampling
from grb.model.torch.gcn import GCN
from grb.train.trainer import Trainer

# load ogbn-arxiv dataset
dataset = Dataset(name='ogbn-arxiv')

# get the adjacency matrix of the dataset
adj = dataset.adj

# normalize the adjacency matrix using GCN normalization
norm = GCNAdjNorm()
norm_adj = norm(adj)

# get the training/valid/test mask from the dataset
train_mask = dataset.train_mask
valid_mask = dataset.valid_mask
test_mask = dataset.test_mask

# create a GCN model
model = GCN(in_features=dataset.num_features,
            hidden_features=[64],
            out_features=dataset.num_classes)

# create a trainer with cross-entropy loss and Adam optimizer
trainer = Trainer(model=model,
                  loss='cross_entropy',
                  optimizer='Adam')

# set the training epochs and early stopping settings
trainer.set_optimizer({'lr': 1e-2, 'weight_decay': 5e-4})
trainer.set_batch_size(512)
trainer.set_train_params({'n_epochs': 200,
                          'early_stop': True,
                          'early_stop_patience': 20,
                          'early_stop_epsilon': 1e-5})
trainer.set_val_data(norm_adj, dataset.labels, valid_mask)


# define a sampling function that samples a subgraph with a given number of nodes
# using K-hop clustering sampling
def clustering_subgraph_sampling(adj, mask, num_clusters, k):
    # create a subgraph mask with the given number of nodes
    subgraph_mask = cluster_sampling(mask, num_clusters, adj)
    subgraph_adj = utils.get_subgraph(adj, subgraph_mask)
    subgraph_labels = dataset.labels[subgraph_mask]

    # create a adjacency matrix with K-hop neighbors
    subgraph_adj_k = utils.get_k_hop_adj(subgraph_adj, k)

    # normalize the adjacency matrix using GCN normalization
    norm = GCNAdjNorm()
    norm_subgraph_adj_k = norm(subgraph_adj_k)

    return norm_subgraph_adj_k, subgraph_labels, subgraph_mask


# sample a subgraph with 10k nodes using K-hop clustering sampling
subgraph_adj, subgraph_labels, subgraph_mask = clustering_subgraph_sampling(norm_adj, train_mask, num_clusters=10000,
                                                                            k=2)

# train a GCN on the subgraph
trainer.fit(subgraph_adj, subgraph_labels, subgraph_mask)