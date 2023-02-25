import os
import torch
import grb.utils as utils
from grb.dataset import Dataset
from grb.model.dgl import GAT
from grb.model.torch import GCN
from grb.utils.normalize import GCNAdjNorm
from grb.attack.injection import FGSM
from grb.defense import AdvTrainer
from model.gnn import GAT_moe
from model.utils import AdvTrainer_diversity
import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset_name', type=str, default='grb-cora')
    argparser.add_argument('--train_mode', type=str, default='inductive')
    argparser.add_argument('--model_name', type=str, default='GAT')
    argparser.add_argument('--lr', type=float, default=0.01)
    argparser.add_argument('--feat_norm', type=str, default='arctan')
    argparser.add_argument('--save_dir', type=str)#, default="./saved_models/{}/{}_at".format(dataset_name, model_name))
    argparser.add_argument('--save_name', type=str, default="model.pt")
    argparser.add_argument('--device', type=str, default="cuda:0")

    argparser.add_argument('--n_epoch', type=int, default=1000)
    argparser.add_argument('--n_inject_max', type=int, default=19)
    argparser.add_argument('--n_edge_max', type=int, default=20)
    argparser.add_argument('--epsilon', type=float, default=0.01)
    argparser.add_argument('--n_epoch_attack', type=int, default=10)
    argparser.add_argument('--early_stop', type=bool, default=True)

    argparser.add_argument('--verbose', type=bool, default=False)
    argparser.add_argument('--eval_every', type=int, default=1)
    argparser.add_argument('--save_after', type=int, default=0)
    argparser.add_argument('--early_stop_patience', type=int, default=500)
    argparser.add_argument('--lr_scheduler', type=bool, default=False)

    argparser.add_argument('--hidden_features', type=int, default=64)
    argparser.add_argument('--n_layers', type=int, default=3)
    argparser.add_argument('--n_heads', type=int, default=4)
    argparser.add_argument('--feat_dropout', type=float, default=0.75)
    argparser.add_argument('--attn_dropout', type=float, default=0.75)
    argparser.add_argument('--dropout', type=float, default=0.6)
    argparser.add_argument('--adj_norm_func', type=str, default=None)
    argparser.add_argument('--layer_norm', type=bool, default=False)
    argparser.add_argument('--residual', type=bool, default=False)
    args = argparser.parse_args()

    args.save_dir = "./saved_models/{}/{}_at".format(args.dataset_name, args.model_name)
    dataset = Dataset(name=args.dataset_name,
                      data_dir=args.data_dir,
                      mode='full',
                      feat_norm=args.feat_norm)
    adj = dataset.adj
    features = dataset.features
    labels = dataset.labels
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    test_mask = dataset.test_mask

    # define model
    model = GAT_moe(in_features=dataset.num_features,  # GAT(in_features=dataset.num_features,
                    out_features=dataset.num_classes,
                    hidden_features=args.hidden_features,
                    n_layers=args.n_layers,
                    n_heads=args.n_heads,
                    adj_norm_func=args.adj_norm_func,
                    layer_norm=args.layer_norm,
                    residual=args.residual,
                    feat_dropout=args.feat_dropout,
                    attn_dropout=args.attn_dropout,
                    dropout=args.dropout)
    # model = GAT(in_features=dataset.num_features,
    #             out_features=dataset.num_classes,
    #             hidden_features=64,
    #             n_layers=3,
    #             n_heads=4,
    #             adj_norm_func=None,
    #             layer_norm=False,
    #             residual=False,
    #             feat_dropout=0.6,
    #             attn_dropout=0.6,
    #             dropout=0.5)
    print("Number of parameters: {}.".format(utils.get_num_params(model)))
    print(model)
    # print(model.state_dict())
    for pname, pweight in model.named_parameters():
        print(pname, pweight.shape)

    attack = FGSM(epsilon=args.epsilon,
                  n_epoch=args.n_epoch_attack,
                  n_inject_max=args.n_inject_max,
                  n_edge_max=args.n_edge_max,
                  feat_lim_min=features.min(),
                  feat_lim_max=features.max(),
                  early_stop=args.early_stop,
                  device=args.device,
                  verbose=args.verbose)
    trainer = AdvTrainer(dataset=dataset,
    # trainer = AdvTrainer_diversity(dataset=dataset,
                                   attack=attack,
                                   optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
                                   loss=torch.nn.functional.cross_entropy,
                                   lr_scheduler=args.lr_scheduler,
                                   early_stop=args.early_stop,
                                   early_stop_patience=args.early_stop_patience,
                                   device=args.device,)

    trainer.train(model=model,
                  n_epoch=args.n_epoch,
                  eval_every=args.eval_every,
                  save_after=args.save_after,
                  save_dir=args.save_dir,
                  save_name=args.save_name,
                  train_mode=args.train_mode,
                  verbose=args.verbose)

    # by trainer
    test_score = trainer.evaluate(model, dataset.test_mask)
    print("model_name: {}, test score: {:.4f}".format(args.model_name, test_score))
    print("args: ", args)
