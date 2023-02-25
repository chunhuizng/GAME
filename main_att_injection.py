import os
import torch
import grb.utils as utils
from grb.dataset import Dataset
from grb.model.dgl import GAT
from grb.model.torch import GCN
from grb.utils.normalize import GCNAdjNorm
from grb.attack.injection import FGSM, PGD, RAND, SPEIT, TDGIA
from grb.defense import AdvTrainer
from model.gnn import GAT_moe, gcn_bn, GCN_bn_moe, GCN_ln_moe
from model.utils import AdvTrainer_diversity, AdvTrainer_ncontrast
import argparse
from grb.trainer.trainer import Trainer
import numpy as np
from model.diver_tools import Ncontrast, get_feature_dis, get_A_r

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset_name', type=str, default='grb-cora')
    argparser.add_argument('--train_mode', type=str, default='inductive', choices=['inductive', 'transductive'])
    argparser.add_argument('--model_name', type=str, default='GCN_bn_moe', choices=['GCN', 'gcn_bn', 'GAT', 'GAT_moe', 'GCN_bn_moe', 'GCN_ln_moe'])
    argparser.add_argument('--lr', type=float, default=0.0075)
    argparser.add_argument('--feat_norm', type=str, default='arctan')
    argparser.add_argument('--save_dir', type=str)
    argparser.add_argument('--save_name', type=str, default="model.pt")
    argparser.add_argument('--device', type=str, default="cuda:0")

    # define surrogate model
    argparser.add_argument('--model_sur_name', type=str, default='GCN', choices=['GCN', 'gcn_bn', 'GAT', 'GAT_moe', 'GCN_bn_moe', 'GCN_ln_moe'], help='surrogate model name')
    argparser.add_argument('--n_epoch_sur', type=int, default=2000, help='number of epochs to train surrogate model')
    argparser.add_argument('--lr_sur', type=float, default=0.01, help='learning rate of surrogate model')
    argparser.add_argument('--use_pretrain_model_sur', type=bool, default=True, help='whether to use pre-trained surrogate model')
    argparser.add_argument('--pretrain_model_sur_path', type=str, default='saved_models/grb-cora/GCN_sur_model.pt', help='path to pre-trained surrogate model')

    argparser.add_argument('--n_epoch', type=int, default=8000)
    argparser.add_argument('--n_inject_max', type=int, default=19)
    argparser.add_argument('--n_edge_max', type=int, default=20)
    argparser.add_argument('--epsilon', type=float, default=0.01)
    argparser.add_argument('--n_epoch_attack', type=int, default=10)
    argparser.add_argument('--early_stop', type=bool, default=True)

    argparser.add_argument('--verbose', type=bool, default=False)
    argparser.add_argument('--eval_every', type=int, default=1)
    argparser.add_argument('--save_after', type=int, default=0)
    argparser.add_argument('--early_stop_patience', type=int, default=500)
    argparser.add_argument('--lr_scheduler', type=bool, default=True, choices=[True, False])

    argparser.add_argument('--hidden_features', type=int, default=96, choices=[64, 96, 128])
    argparser.add_argument('--n_layers', type=int, default=3)
    argparser.add_argument('--n_heads', type=int, default=4)
    argparser.add_argument('--feat_dropout', type=float, default=0.5)
    argparser.add_argument('--attn_dropout', type=float, default=0.5)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--adj_norm_func', type=str, default='GCNAdjNorm')
    argparser.add_argument('--layer_norm', type=bool, default=True, choices=[True, False])
    argparser.add_argument('--residual', type=bool, default=False)
    argparser.add_argument('--attack_method', type=str, default='FGSM', choices=['FGSM', 'PGD', 'RAND', 'SPEIT', 'TDGIA'])

    argparser.add_argument('--num_experts', type=int, default=3, help='number of experts in GAT_moe')
    argparser.add_argument('--noisy_gating', type=bool, default=True, help='whether to use noisy gating')
    argparser.add_argument('--k', type=int, default=1, help='number of experts to use')
    argparser.add_argument('--use_diversity_loss', type=bool, default=False, help="whether to use diversity loss")
    argparser.add_argument('--use_ncontrast', type=bool, default=False, help='whether to use ncontrast')
    argparser.add_argument('--tau_ncontrast', type=float, default=1.0, help='temperature for ncontrast')
    argparser.add_argument('--delta_ncontrast', type=int, default=2, help='to compute order-th power of adj')
    argparser.add_argument('--alpha_ncontrast', type=float, default=2.0, help='To control the ratio of Ncontrast loss')
    args = argparser.parse_args()

    # 1. Load dataset
    args.save_dir = "./saved_models/{}/{}_at".format(args.dataset_name, args.model_name)
    dataset = Dataset(name=args.dataset_name,
                      data_dir=args.data_dir,
                      mode='full',
                      feat_norm=args.feat_norm)
    print(args)
    adj = dataset.adj

    # Acoo = Acsr.tocoo()
    # print('Acoo',Acoo)
    #
    # Apt = torch.sparse.LongTensor(torch.LongTensor([Acoo.row.tolist(), Acoo.col.tolist()]),
    #                               torch.LongTensor(Acoo.data.astype(np.int32)))
    Acoo = adj.tocoo()
    _adj = torch.sparse.LongTensor(torch.LongTensor([Acoo.row.tolist(), Acoo.col.tolist()]),
                                  torch.LongTensor(Acoo.data.astype(np.int32)))
    # convert to torch sparse tensor
    adj_label = get_A_r(_adj, args.delta_ncontrast)

    features = dataset.features
    labels = dataset.labels
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    test_mask = dataset.test_mask

    # 2. Define model
    if args.model_name == 'gcn_bn':
        model = gcn_bn(in_features=num_features,
                    hidden_features=args.hidden_features,
                    out_features=num_classes,
                    n_layers=args.n_layers,
                    dropout=args.dropout,
                    adj_norm_func=GCNAdjNorm,
                    layer_norm=True,
                    residual=args.residual)
    elif args.model_name == 'GAT':
        model = GAT(in_features=num_features,
                    hidden_features=args.hidden_features,
                    out_features=num_classes,
                    n_layers=args.n_layers,
                    n_heads=args.n_heads,
                    feat_dropout=args.feat_dropout,
                    attn_dropout=args.attn_dropout,
                    dropout=args.dropout,
                    adj_norm_func=args.adj_norm_func,
                    layer_norm=args.layer_norm,
                    residual=args.residual)
    elif args.model_name == 'GAT_moe':
        model = GAT_moe(in_features=num_features,
                    hidden_features=args.hidden_features,
                    out_features=num_classes,
                    n_layers=args.n_layers,
                    n_heads=args.n_heads,
                    feat_dropout=args.feat_dropout,
                    attn_dropout=args.attn_dropout,
                    dropout=args.dropout,
                    adj_norm_func=None,
                    layer_norm=args.layer_norm,
                    residual=args.residual,
                    num_experts=args.num_experts, noisy_gating=args.noisy_gating, k=args.k,
                    )
    elif args.model_name == 'GCN_bn_moe':
        model = GCN_bn_moe(in_features=num_features,
                    hidden_features=args.hidden_features,
                    out_features=num_classes,
                    n_layers=args.n_layers,
                    dropout=args.dropout,
                    adj_norm_func=GCNAdjNorm,
                    layer_norm=True,
                    residual=args.residual,
                    num_experts=args.num_experts, noisy_gating=args.noisy_gating, k=args.k,)
    elif args.model_name == 'GCN':
        model = GCN(in_features=num_features,
                 out_features=num_classes,
                 hidden_features=64,
                 n_layers=3,
                 adj_norm_func=GCNAdjNorm,
                 layer_norm=False,
                 residual=False,
                 dropout=0.5)
    elif args.model_name == 'GCN_ln_moe':
        model = GCN_ln_moe(in_features=num_features,
                    hidden_features=args.hidden_features,
                    out_features=num_classes,
                    n_layers=args.n_layers,
                    dropout=args.dropout,
                    adj_norm_func=GCNAdjNorm,
                    layer_norm=True,
                    residual=args.residual,
                    num_experts=args.num_experts, noisy_gating=args.noisy_gating, k=args.k,)
    else:
        raise NotImplementedError("Model {} not implemented.".format(args.model_name))
    print("Number of parameters: {}.".format(utils.get_num_params(model)))
    print(model)

    # 3. Define model_sur
    if args.model_sur_name == 'GCN':
        model_sur = GCN(in_features=num_features,
                 out_features=num_classes,
                 hidden_features=64,
                 n_layers=2,
                 adj_norm_func=GCNAdjNorm,
                 layer_norm=False,
                 residual=False,
                 dropout=0.5)
    elif args.model_sur_name == 'GAT':
        model_sur = GAT(in_features=num_features,
                    hidden_features=args.hidden_features,
                    out_features=num_classes,
                    n_layers=args.n_layers,
                    n_heads=args.n_heads,
                    feat_dropout=args.feat_dropout,
                    attn_dropout=args.attn_dropout,
                    dropout=args.dropout,
                    adj_norm_func=args.adj_norm_func,
                    layer_norm=args.layer_norm,
                    residual=args.residual)
    elif args.model_sur_name == 'GAT_moe':
        model_sur = GAT_moe(in_features=num_features,
                    hidden_features=args.hidden_features,
                    out_features=num_classes,
                    n_layers=args.n_layers,
                    n_heads=args.n_heads,
                    feat_dropout=args.feat_dropout,
                    attn_dropout=args.attn_dropout,
                    dropout=args.dropout,
                    adj_norm_func=args.adj_norm_func,
                    layer_norm=args.layer_norm,
                    residual=args.residual,
                    num_experts=args.num_experts, noisy_gating=args.noisy_gating, k=args.k,
                    )
    elif args.model_sur_name == 'GCN_bn_moe':
        model_sur = GCN_bn_moe(in_features=num_features,
                    hidden_features=args.hidden_features,
                    out_features=num_classes,
                    n_layers=args.n_layers,
                    dropout=args.dropout,
                    # adj_norm_func=args.adj_norm_func,
                    layer_norm=True,
                    residual=args.residual,
                    num_experts=args.num_experts, noisy_gating=args.noisy_gating, k=args.k,)
    else:
        raise NotImplementedError("Model {} not implemented.".format(args.model_name))
    if args.use_pretrain_model_sur:
        model_sur.load_state_dict(torch.load(args.pretrain_model_sur_path))
        print("load pretrain model_sur from {}".format(args.pretrain_model_sur_path))
    else:
        trainer_sur = Trainer(dataset=dataset,
                          optimizer=torch.optim.Adam(model_sur.parameters(), lr=args.lr_sur),
                          loss=torch.nn.functional.cross_entropy,
                          lr_scheduler=False,
                          early_stop=True,
                          early_stop_patience=500,
                          feat_norm=None,
                          device=args.device,)
        trainer_sur.train(model_sur, args.n_epoch_sur,
                  eval_every=1,
                  save_after=0,
                  save_dir=args.save_dir,
                  save_name=args.model_sur_name+'_sur_'+args.save_name,
                  train_mode=args.train_mode,
                  verbose=False)
        # save model
        torch.save(model_sur.state_dict(), args.save_dir+args.model_sur_name+'_sur_'+args.save_name)
        print("Model_sur saved in {}".format(args.save_dir+args.model_sur_name+'_sur_'+args.save_name))

    # 4. Define attack
    if args.attack_method == 'FGSM':
        attack = FGSM(epsilon=args.epsilon,
                      n_epoch=args.n_epoch_attack,
                      n_inject_max=args.n_inject_max,
                      n_edge_max=args.n_edge_max,
                      feat_lim_min=features.min(),
                      feat_lim_max=features.max(),
                      early_stop=args.early_stop,
                      device=args.device,
                      verbose=args.verbose)
    elif args.attack_method == 'PGD':
        attack = PGD(epsilon=args.epsilon,
                     n_epoch=args.n_epoch_attack,
                     n_inject_max=args.n_inject_max,
                     n_edge_max=args.n_edge_max,
                     feat_lim_min=features.min(),
                     feat_lim_max=features.max(),
                     early_stop=args.early_stop,
                     device=args.device,
                     verbose=args.verbose)
    elif args.attack_method == 'RAND':
        attack = RAND(n_inject_max=args.n_inject_max,
                      n_edge_max=args.n_edge_max,
                      feat_lim_min=features.min(),
                      feat_lim_max=features.max(),
                      device=args.device,
                      verbose=args.verbose)
    elif args.attack_method == 'SPEIT':
        attack = SPEIT(lr=args.lr,
                       n_epoch=args.n_epoch_attack,
                       n_inject_max=args.n_inject_max,
                       n_edge_max=args.n_edge_max,
                       feat_lim_min=features.min(),
                       feat_lim_max=features.max(),
                       early_stop=args.early_stop,
                       device=args.device,
                       verbose=args.verbose)
    elif args.attack_method == 'TDGIA':
        attack = TDGIA(lr=args.lr,
                       n_epoch=args.n_epoch_attack,
                       n_inject_max=args.n_inject_max,
                       n_edge_max=args.n_edge_max,
                       feat_lim_min=features.min(),
                       feat_lim_max=features.max(),
                       early_stop=args.early_stop,
                       device=args.device,
                       verbose=args.verbose)
    else:
        raise NotImplementedError("Attack {} not implemented.".format(args.attack_method))

    # 5. Do attack
    adj_attack, features_attack = attack.attack(model=model_sur,
                                                adj=adj,
                                                features=features,
                                                target_mask=test_mask,
                                                adj_norm_func=model_sur.adj_norm_func)
    # 6. Do adversarial training
    if args.use_diversity_loss:
        print("Use diversity loss")
        trainer = AdvTrainer_diversity(dataset=dataset,
                             attack=attack,
                             optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
                             loss=torch.nn.functional.cross_entropy,
                             lr_scheduler=args.lr_scheduler,
                             early_stop=args.early_stop,
                             early_stop_patience=args.early_stop_patience,
                             device=args.device, )
    if args.use_ncontrast:
        print("Use ncontrast loss")
        trainer = AdvTrainer_ncontrast(dataset=dataset,
                             attack=attack,
                             optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
                             loss=torch.nn.functional.cross_entropy,
                             lr_scheduler=args.lr_scheduler,
                             early_stop=args.early_stop,
                             early_stop_patience=args.early_stop_patience,
                             device=args.device,
                                       adj_label=adj_label,
                                       alpha_ncontrast=args.alpha_ncontrast,
                                       tau_ncontrast=args.tau_ncontrast,
                                       delta_ncontrast=args.delta_ncontrast)
    else:
        print("Use normal loss")
        trainer = AdvTrainer(dataset=dataset,
                             attack=attack,
                             optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
                             loss=torch.nn.functional.cross_entropy,
                             lr_scheduler=args.lr_scheduler,
                             early_stop=args.early_stop,
                             early_stop_patience=args.early_stop_patience,
                             device=args.device, )


    trainer.train(model=model,
                  n_epoch=args.n_epoch,
                  eval_every=args.eval_every,
                  save_after=args.save_after,
                  save_dir=args.save_dir,
                  save_name=args.save_name,
                  train_mode=args.train_mode,
                  verbose=args.verbose)

    # 7. Evaluate
    features_attacked = torch.cat([features.to(args.device), features_attack])
    test_score = utils.evaluate(model,
                                features=features_attacked,
                                adj=adj_attack,
                                labels=dataset.labels,
                                adj_norm_func=model.adj_norm_func,
                                mask=dataset.test_mask,
                                device=args.device)

    print("model_name: {}, test score: {:.4f}".format(args.model_name, test_score))
    print("args: ", args)
