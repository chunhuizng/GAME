from grb.evaluator import metric
import grb.utils as utils
from tqdm.auto import tqdm
import os
import time
from grb.defense import AdvTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
def get_A_r(adj, r):
    adj_label = adj.to_dense()
    if r == 1:
        adj_label = adj_label
    elif r == 2:
        adj_label = adj_label@adj_label
    elif r == 3:
        adj_label = adj_label@adj_label@adj_label
    elif r == 4:
        adj_label = adj_label@adj_label@adj_label@adj_label
    return adj_label
def Ncontrast(x_dis, adj_label, tau = 1.0):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp( tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).cuda()
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis


class mixture_bn(nn.Module):
    def __init__(self, num_features):
        super(mixture_bn, self).__init__()
        self.num_features = num_features
        self.clean_bn = nn.BatchNorm1d(num_features)
        self.adv_bn = nn.BatchNorm1d(num_features)

    def forward(self, x, adv_x):
        clean_x = self.clean_bn(x)
        adv_x = self.adv_bn(adv_x)
        return clean_x, adv_x

def cosine_loss(hids, eps=1e-8):
    hids_n = hids.norm(dim=1).unsqueeze(dim=1)
    hids_norm = hids / torch.max(hids_n, eps * torch.ones_like(hids_n))
    sim_matrix = torch.einsum('ij,jk->ik', hids_norm, hids_norm.transpose(0,1))
    loss_cos = sim_matrix.mean()
    return loss_cos

def Loss_cosine_weight(h_emb, eps=1e-8):
    # adopted from def Loss_cosine_attn(h_emb, eps=1e-8) in diversity transformer
    # h_emb (B, Tokens, dims * heads)
    # normalize
    target_h_emb = h_emb
    # hshape = target_h_emb.shape
    # target_h_emb = target_h_emb.reshape(hshape[0], hshape[1], -1)
    a_n = target_h_emb.norm(dim=1).unsqueeze(1)
    a_norm = target_h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

    # patch-wise absolute value of cosine similarity
    sim_matrix = torch.einsum('abc,acd->abd', a_norm, a_norm.transpose(0,1))
    loss_cos = sim_matrix.mean() # also add diagnoal elements
    return loss_cos


def dominant_eigenvalue(A, dev):
    N, _ = A.size()
    x = torch.rand(N, 1, device=dev)

    Ax = (A @ x)
    AAx = (A @ Ax)

    return AAx.permute(1, 0) @ Ax / (Ax.permute(1, 0) @ Ax)

def get_singular_values(A, dev):
    ATA = A.permute(1, 0) @ A
    N, _ = ATA.size()
    largest = dominant_eigenvalue(ATA, dev)
    I = torch.eye(N, device=dev)
    I = I * largest
    tmp = dominant_eigenvalue(ATA - I, dev)
    return tmp + largest, largest

def Loss_condition_orth_weight(W):
    W = W.permute(1, 0) # (in, out)
    smallest, largest = get_singular_values(W, W.device)
    return torch.mean((largest - smallest)**2)


class AdvTrainer_diversity(AdvTrainer):
    def __init__(self,
                 dataset,
                 optimizer,
                 loss,
                 feat_norm=None,
                 attack=None,
                 attack_mode="injection",
                 lr_scheduler=None,
                 lr_patience=100,
                 lr_factor=0.75,
                 lr_min=1e-5,
                 early_stop=None,
                 early_stop_patience=100,
                 early_stop_epsilon=1e-5,
                 eval_metric=metric.eval_acc,
                 device='cpu'):
        super(AdvTrainer_diversity, self).__init__(dataset=dataset, optimizer=optimizer, loss=loss, feat_norm=feat_norm,
                                             attack=attack, attack_mode=attack_mode, lr_scheduler=lr_scheduler,
                                             lr_patience=lr_patience, lr_factor=lr_factor, lr_min=lr_min, early_stop=early_stop,
                                             early_stop_patience=early_stop_patience, early_stop_epsilon=early_stop_epsilon,
                                             eval_metric=eval_metric, device=device)
    def train(self,
                  model,
                  n_epoch,
                  save_dir=None,
                  save_name=None,
                  eval_every=10,
                  save_after=0,
                  train_mode="trasductive",
                  verbose=True):
            model.to(self.device)
            model.train()

            if save_dir is None:
                cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                save_dir = "./tmp_{}".format(cur_time)
            else:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

            if save_name is None:
                save_name = "checkpoint.pt"
            else:
                if save_name.split(".")[-1] != "pt":
                    save_name = save_name + ".pt"

            train_score_list = []
            val_score_list = []
            best_val_score = 0.0
            features = self.features
            train_mask = self.train_mask
            val_mask = self.val_mask
            labels = self.labels

            if train_mode == "inductive":
                # Inductive setting
                train_val_mask = torch.logical_or(train_mask, val_mask)
                train_val_index = torch.where(train_val_mask)[0]
                train_index, val_index = torch.where(train_mask)[0], torch.where(val_mask)[0]
                train_index_induc, val_index_induc = utils.get_index_induc(train_index, val_index)
                train_mask_induc = torch.zeros(len(train_val_index), dtype=bool)
                train_mask_induc[train_index_induc] = True
                val_mask_induc = torch.zeros(len(train_val_index), dtype=bool)
                val_mask_induc[val_index_induc] = True

                features_train = features[train_mask]
                features_val = features[train_val_mask]
                adj_train = utils.adj_preprocess(self.adj,
                                                 adj_norm_func=model.adj_norm_func,
                                                 mask=self.train_mask,
                                                 model_type=model.model_type,
                                                 device=self.device)
                adj_val = utils.adj_preprocess(self.adj,
                                               adj_norm_func=model.adj_norm_func,
                                               mask=train_val_mask,
                                               model_type=model.model_type,
                                               device=self.device)
                num_train = torch.sum(train_mask).item()
                epoch_bar = tqdm(range(n_epoch))
                for epoch in epoch_bar:
                    logits = model(features_train, adj_train)[:num_train]
                    if self.loss == F.nll_loss:
                        out = F.log_softmax(logits, 1)
                        train_loss = self.loss(out, labels[train_mask])
                        logits_val = model(features_val, adj_val)[:]
                        out_val = F.log_softmax(logits_val, 1)
                        val_loss = self.loss(out_val[val_mask_induc], labels[val_mask])
                    elif self.loss == F.cross_entropy:
                        out = logits
                        train_loss = self.loss(out, labels[train_mask])
                        logits_val = model(features_val, adj_val)
                        out_val = logits_val
                        val_loss = self.loss(out_val[val_mask_induc], labels[val_mask])
                    elif self.loss == F.binary_cross_entropy:
                        out = F.sigmoid(logits)
                        train_loss = self.loss(out, labels[train_mask].float())
                        logits_val = model(features_val, adj_val)
                        out_val = F.sigmoid(logits_val)
                        val_loss = self.loss(out_val[val_mask_induc], labels[val_mask].float())
                    elif self.loss == F.binary_cross_entropy_with_logits:
                        out = logits
                        train_loss = self.loss(out, labels[train_mask].float())
                        logits_val = model(features_val, adj_val)
                        out_val = F.sigmoid(logits_val)
                        val_loss = self.loss(out_val[val_mask_induc], labels[val_mask].float())

                    self.optimizer.zero_grad()
                    div_loss = cosine_loss(features_train, eps=1e-8)
                    print('div_loss:', div_loss.item(), 'train_loss:', train_loss.item())
                    train_loss = train_loss + div_loss * 1

                    train_loss.backward()
                    self.optimizer.step()

                    if self.attack is not None:
                        if self.attack_mode == "injection":
                            adj_attack, features_attack = self.attack.attack(model=model,
                                                                             adj=self.adj[train_mask][:, train_mask],
                                                                             features=features[train_mask],
                                                                             target_mask=torch.ones(num_train,
                                                                                                    dtype=bool),
                                                                             adj_norm_func=model.adj_norm_func)
                            adj_train = utils.adj_preprocess(adj=adj_attack,
                                                             adj_norm_func=model.adj_norm_func,
                                                             model_type=model.model_type,
                                                             device=self.device)
                            features_train = torch.cat([features[train_mask], features_attack])
                        else:
                            adj_attack, features_attack = self.attack.attack(model=model,
                                                                             adj=self.adj[train_mask][:, train_mask],
                                                                             features=features[train_mask],
                                                                             index_target=torch.range(0,
                                                                                                      num_train - 1).multinomial(
                                                                                 int(num_train * 0.01)))
                            adj_train = utils.adj_preprocess(adj=adj_attack,
                                                             adj_norm_func=model.adj_norm_func,
                                                             model_type=model.model_type,
                                                             device=self.device)
                            features_train = features_attack

                    if self.lr_scheduler:
                        self.lr_scheduler.step(val_loss)
                    if self.early_stop:
                        self.early_stop(val_loss)
                        if self.early_stop.stop:
                            print("Training: early stopped.")
                            utils.save_model(model, save_dir, "final_" + save_name)
                            return

                    if epoch % eval_every == 0:
                        train_score = self.eval_metric(out, labels[train_mask], mask=None)
                        val_score = self.eval_metric(out_val, labels[train_val_mask], mask=val_mask_induc)
                        train_score_list.append(train_score)
                        val_score_list.append(val_score)
                        if val_score > best_val_score:
                            best_val_score = val_score
                            if epoch > save_after:
                                epoch_bar.set_description(
                                    "Training: Epoch {:05d} | Best validation score: {:.4f}".format(epoch,
                                                                                                    best_val_score))
                                utils.save_model(model, save_dir, save_name, verbose=verbose)
                        epoch_bar.set_description(
                            'Training: Epoch {:05d} | Train loss {:.4f} | Train score {:.4f} '
                            '| Val loss {:.4f} | Val score {:.4f}'.format(
                                epoch, train_loss, train_score, val_loss, val_score))
            else:
                # Transductive setting
                adj_train = utils.adj_preprocess(self.adj,
                                                 adj_norm_func=model.adj_norm_func,
                                                 mask=None,
                                                 model_type=model.model_type,
                                                 device=self.device)
                features_train = features
                epoch_bar = tqdm(range(n_epoch))
                for epoch in epoch_bar:
                    logits = model(features_train, adj_train)[:self.num_nodes]
                    if self.loss == F.nll_loss:
                        out = F.log_softmax(logits, 1)
                        train_loss = self.loss(out[train_mask], labels[train_mask])
                        val_loss = self.loss(out[val_mask], labels[val_mask])
                    elif self.loss == F.cross_entropy:
                        out = logits
                        train_loss = self.loss(out[train_mask], labels[train_mask])
                        val_loss = self.loss(out[val_mask], labels[val_mask])
                    elif self.loss == F.binary_cross_entropy:
                        out = F.sigmoid(logits)
                        train_loss = self.loss(out[train_mask], labels[train_mask].float())
                        val_loss = self.loss(out[val_mask], labels[val_mask].float())
                    elif self.loss == F.binary_cross_entropy_with_logits:
                        out = logits
                        train_loss = self.loss(out[train_mask], labels[train_mask].float())
                        val_loss = self.loss(out[val_mask], labels[val_mask].float())

                    self.optimizer.zero_grad()
                    div_loss = cosine_loss(features_train, eps=1e-8)
                    print('div_loss:', div_loss.item(), 'train_loss:', train_loss.item())
                    train_loss = train_loss + div_loss * 1
                    train_loss.backward()
                    self.optimizer.step()

                    if self.attack is not None:
                        adj_attack, features_attack = self.attack.attack(model=model,
                                                                         adj=self.adj,
                                                                         features=self.features,
                                                                         target_mask=val_mask,
                                                                         adj_norm_func=model.adj_norm_func)
                        adj_train = utils.adj_preprocess(adj=adj_attack,
                                                         adj_norm_func=model.adj_norm_func,
                                                         model_type=model.model_type,
                                                         device=self.device)
                        features_train = torch.cat([features, features_attack])

                    if self.lr_scheduler:
                        self.lr_scheduler.step(val_loss)
                    if self.early_stop:
                        self.early_stop(val_loss)
                        if self.early_stop.stop:
                            print("Training: early stopped.")
                            utils.save_model(model, save_dir, "final_" + save_name, verbose=verbose)
                            return

                    if epoch % eval_every == 0:
                        train_score = self.eval_metric(out, labels, train_mask)
                        val_score = self.eval_metric(out, labels, val_mask)
                        train_score_list.append(train_score)
                        val_score_list.append(val_score)
                        if val_score > best_val_score:
                            best_val_score = val_score
                            if epoch > save_after:
                                epoch_bar.set_description(
                                    "Training: Epoch {:05d} | Best validation score: {:.4f}".format(epoch,
                                                                                                    best_val_score))
                                utils.save_model(model, save_dir, save_name, verbose=verbose)

                        epoch_bar.set_description(
                            'Training: Epoch {:05d} | Train loss {:.4f} | Train score {:.4f} '
                            '| Val loss {:.4f} | Val score {:.4f}'.format(
                                epoch, train_loss, train_score, val_loss, val_score))

            utils.save_model(model, save_dir, "final_" + save_name)


class AdvTrainer_ncontrast(AdvTrainer):
    def __init__(self,
                 dataset,
                 optimizer,
                 loss,
                 feat_norm=None,
                 attack=None,
                 attack_mode="injection",
                 lr_scheduler=None,
                 lr_patience=100,
                 lr_factor=0.75,
                 lr_min=1e-5,
                 early_stop=None,
                 early_stop_patience=100,
                 early_stop_epsilon=1e-5,
                 eval_metric=metric.eval_acc,
                 device='cpu',
                 adj_label=None, alpha_ncontrast=2, tau_ncontrast=1.0, delta_ncontrast=2,
                 ):
        super(AdvTrainer_ncontrast, self).__init__(dataset=dataset, optimizer=optimizer, loss=loss, feat_norm=feat_norm,
                                             attack=attack, attack_mode=attack_mode, lr_scheduler=lr_scheduler,
                                             lr_patience=lr_patience, lr_factor=lr_factor, lr_min=lr_min, early_stop=early_stop,
                                             early_stop_patience=early_stop_patience, early_stop_epsilon=early_stop_epsilon,
                                             eval_metric=eval_metric, device=device)
        self.adj_label = adj_label.to(device)
        self.alpha_ncontrast = alpha_ncontrast
        self.tau_ncontrast = tau_ncontrast
    def train(self,
                  model,
                  n_epoch,
                  save_dir=None,
                  save_name=None,
                  eval_every=10,
                  save_after=0,
                  train_mode="trasductive",
                  verbose=True):
            model.to(self.device)
            model.train()

            if save_dir is None:
                cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                save_dir = "./tmp_{}".format(cur_time)
            else:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

            if save_name is None:
                save_name = "checkpoint.pt"
            else:
                if save_name.split(".")[-1] != "pt":
                    save_name = save_name + ".pt"

            train_score_list = []
            val_score_list = []
            best_val_score = 0.0
            features = self.features
            train_mask = self.train_mask
            val_mask = self.val_mask
            labels = self.labels

            if train_mode == "inductive":
                # Inductive setting
                train_val_mask = torch.logical_or(train_mask, val_mask)
                train_val_index = torch.where(train_val_mask)[0]
                train_index, val_index = torch.where(train_mask)[0], torch.where(val_mask)[0]
                train_index_induc, val_index_induc = utils.get_index_induc(train_index, val_index)
                train_mask_induc = torch.zeros(len(train_val_index), dtype=bool)
                train_mask_induc[train_index_induc] = True
                val_mask_induc = torch.zeros(len(train_val_index), dtype=bool)
                val_mask_induc[val_index_induc] = True

                features_train = features[train_mask]
                features_val = features[train_val_mask]
                adj_train = utils.adj_preprocess(self.adj,
                                                 adj_norm_func=model.adj_norm_func,
                                                 mask=self.train_mask,
                                                 model_type=model.model_type,
                                                 device=self.device)
                adj_val = utils.adj_preprocess(self.adj,
                                               adj_norm_func=model.adj_norm_func,
                                               mask=train_val_mask,
                                               model_type=model.model_type,
                                               device=self.device)
                num_train = torch.sum(train_mask).item()
                epoch_bar = tqdm(range(n_epoch))
                for epoch in epoch_bar:
                    logits = model(features_train, adj_train)[:num_train]
                    if self.loss == F.nll_loss:
                        out = F.log_softmax(logits, 1)
                        train_loss = self.loss(out, labels[train_mask])
                        logits_val = model(features_val, adj_val)[:]
                        out_val = F.log_softmax(logits_val, 1)
                        val_loss = self.loss(out_val[val_mask_induc], labels[val_mask])
                    elif self.loss == F.cross_entropy:
                        out = logits
                        train_loss = self.loss(out, labels[train_mask])
                        logits_val = model(features_val, adj_val)
                        out_val = logits_val
                        val_loss = self.loss(out_val[val_mask_induc], labels[val_mask])
                    elif self.loss == F.binary_cross_entropy:
                        out = F.sigmoid(logits)
                        train_loss = self.loss(out, labels[train_mask].float())
                        logits_val = model(features_val, adj_val)
                        out_val = F.sigmoid(logits_val)
                        val_loss = self.loss(out_val[val_mask_induc], labels[val_mask].float())
                    elif self.loss == F.binary_cross_entropy_with_logits:
                        out = logits
                        train_loss = self.loss(out, labels[train_mask].float())
                        logits_val = model(features_val, adj_val)
                        out_val = F.sigmoid(logits_val)
                        val_loss = self.loss(out_val[val_mask_induc], labels[val_mask].float())

                    self.optimizer.zero_grad()
                    x_dis = get_feature_dis(features_train)
                    div_loss = Ncontrast(x_dis, self.adj_label, tau=self.tau_ncontrast)
                    train_loss = train_loss + div_loss * self.alpha_ncontrast

                    train_loss.backward()
                    self.optimizer.step()

                    if self.attack is not None:
                        if self.attack_mode == "injection":
                            adj_attack, features_attack = self.attack.attack(model=model,
                                                                             adj=self.adj[train_mask][:, train_mask],
                                                                             features=features[train_mask],
                                                                             target_mask=torch.ones(num_train,
                                                                                                    dtype=bool),
                                                                             adj_norm_func=model.adj_norm_func)
                            adj_train = utils.adj_preprocess(adj=adj_attack,
                                                             adj_norm_func=model.adj_norm_func,
                                                             model_type=model.model_type,
                                                             device=self.device)
                            features_train = torch.cat([features[train_mask], features_attack])
                        else:
                            adj_attack, features_attack = self.attack.attack(model=model,
                                                                             adj=self.adj[train_mask][:, train_mask],
                                                                             features=features[train_mask],
                                                                             index_target=torch.range(0,
                                                                                                      num_train - 1).multinomial(
                                                                                 int(num_train * 0.01)))
                            adj_train = utils.adj_preprocess(adj=adj_attack,
                                                             adj_norm_func=model.adj_norm_func,
                                                             model_type=model.model_type,
                                                             device=self.device)
                            features_train = features_attack

                    if self.lr_scheduler:
                        self.lr_scheduler.step(val_loss)
                    if self.early_stop:
                        self.early_stop(val_loss)
                        if self.early_stop.stop:
                            print("Training: early stopped.")
                            utils.save_model(model, save_dir, "final_" + save_name)
                            return

                    if epoch % eval_every == 0:
                        train_score = self.eval_metric(out, labels[train_mask], mask=None)
                        val_score = self.eval_metric(out_val, labels[train_val_mask], mask=val_mask_induc)
                        train_score_list.append(train_score)
                        val_score_list.append(val_score)
                        if val_score > best_val_score:
                            best_val_score = val_score
                            if epoch > save_after:
                                epoch_bar.set_description(
                                    "Training: Epoch {:05d} | Best validation score: {:.4f}".format(epoch,
                                                                                                    best_val_score))
                                utils.save_model(model, save_dir, save_name, verbose=verbose)
                        epoch_bar.set_description(
                            'Training: Epoch {:05d} | Train loss {:.4f} | Train score {:.4f} '
                            '| Val loss {:.4f} | Val score {:.4f}'.format(
                                epoch, train_loss, train_score, val_loss, val_score))
            else:
                # Transductive setting
                adj_train = utils.adj_preprocess(self.adj,
                                                 adj_norm_func=model.adj_norm_func,
                                                 mask=None,
                                                 model_type=model.model_type,
                                                 device=self.device)
                features_train = features
                epoch_bar = tqdm(range(n_epoch))
                for epoch in epoch_bar:
                    logits = model(features_train, adj_train)[:self.num_nodes]
                    if self.loss == F.nll_loss:
                        out = F.log_softmax(logits, 1)
                        train_loss = self.loss(out[train_mask], labels[train_mask])
                        val_loss = self.loss(out[val_mask], labels[val_mask])
                    elif self.loss == F.cross_entropy:
                        out = logits
                        train_loss = self.loss(out[train_mask], labels[train_mask])
                        val_loss = self.loss(out[val_mask], labels[val_mask])
                    elif self.loss == F.binary_cross_entropy:
                        out = F.sigmoid(logits)
                        train_loss = self.loss(out[train_mask], labels[train_mask].float())
                        val_loss = self.loss(out[val_mask], labels[val_mask].float())
                    elif self.loss == F.binary_cross_entropy_with_logits:
                        out = logits
                        train_loss = self.loss(out[train_mask], labels[train_mask].float())
                        val_loss = self.loss(out[val_mask], labels[val_mask].float())

                    self.optimizer.zero_grad()
                    x_dis = get_feature_dis(features_train)
                    div_loss = Ncontrast(x_dis, self.adj_label, tau=self.tau_ncontrast)
                    train_loss = train_loss + div_loss * self.alpha_ncontrast
                    # div_loss = cosine_loss(features_train, eps=1e-8)
                    # print('div_loss:', div_loss.item(), 'train_loss:', train_loss.item())
                    # train_loss = train_loss + div_loss * self.
                    train_loss.backward()
                    self.optimizer.step()

                    if self.attack is not None:
                        adj_attack, features_attack = self.attack.attack(model=model,
                                                                         adj=self.adj,
                                                                         features=self.features,
                                                                         target_mask=val_mask,
                                                                         adj_norm_func=model.adj_norm_func)
                        adj_train = utils.adj_preprocess(adj=adj_attack,
                                                         adj_norm_func=model.adj_norm_func,
                                                         model_type=model.model_type,
                                                         device=self.device)
                        features_train = torch.cat([features, features_attack])

                    if self.lr_scheduler:
                        self.lr_scheduler.step(val_loss)
                    if self.early_stop:
                        self.early_stop(val_loss)
                        if self.early_stop.stop:
                            print("Training: early stopped.")
                            utils.save_model(model, save_dir, "final_" + save_name, verbose=verbose)
                            return

                    if epoch % eval_every == 0:
                        train_score = self.eval_metric(out, labels, train_mask)
                        val_score = self.eval_metric(out, labels, val_mask)
                        train_score_list.append(train_score)
                        val_score_list.append(val_score)
                        if val_score > best_val_score:
                            best_val_score = val_score
                            if epoch > save_after:
                                epoch_bar.set_description(
                                    "Training: Epoch {:05d} | Best validation score: {:.4f}".format(epoch,
                                                                                                    best_val_score))
                                utils.save_model(model, save_dir, save_name, verbose=verbose)

                        epoch_bar.set_description(
                            'Training: Epoch {:05d} | Train loss {:.4f} | Train score {:.4f} '
                            '| Val loss {:.4f} | Val score {:.4f}'.format(
                                epoch, train_loss, train_score, val_loss, val_score))

            utils.save_model(model, save_dir, "final_" + save_name)