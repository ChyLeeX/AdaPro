# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(f"{os.getcwd()}/scMM-master/src")

import argparse
import datetime
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp
import numpy as np
import torch
from torch import optim
from torch.utils.data import Subset
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite, mmread
import models
import objectives
from utils import Timer, save_vars, EarlyStopping
from datasets import RNA_Dataset


def pipeline(**kwargs):
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    runId = datetime.datetime.now().isoformat()
    experiment_dir = Path(f'{args.base_dir}/experiments/' + args.experiment)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runPath = mkdtemp(prefix=runId, dir=str(experiment_dir))
    
    
    ### load reference data
    base_dir = args.base_dir
    train_rna_path = f"{base_dir}/train_RNA-seq"
    train_modal_path = f"{base_dir}/train_CITE-seq"
    
    train_r_dataset = RNA_Dataset(train_rna_path)   
    args.r_dim = train_r_dataset.data.shape[1] 
    
    train_modal_dataset = RNA_Dataset(train_modal_path)   
    args.p_dim = train_modal_dataset.data.shape[1]
    
    
    ### load query data
    test_rna_path = f"{base_dir}/test_RNA-seq"
    test_modal_path = f"{base_dir}/test_CITE-seq"
    
    test_r_dataset = RNA_Dataset(test_rna_path)   
    args.r_dim = test_r_dataset.data.shape[1] 
    
    test_modal_dataset = RNA_Dataset(test_modal_path)
    args.p_dim = test_modal_dataset.data.shape[1]
    
    
    ### randomly select 20% of reference cells as validation set
    train_num_cell = train_r_dataset.data.shape[0]
    t_size = np.round(train_num_cell * 0.80).astype('int')   
    t_id = np.random.choice(a=train_num_cell, size=t_size, replace=False)  
    s_id = np.delete(range(train_num_cell), t_id)   
    torch.save(t_id, runPath + '/t_id.rar')
    torch.save(s_id, runPath + '/s_id.rar')
    
    train_dataset = [Subset(train_r_dataset, t_id), Subset(train_modal_dataset, t_id)]
    valid_dataset = [Subset(train_r_dataset, s_id), Subset(train_modal_dataset, s_id)]
    
    t_id = pd.DataFrame(t_id)
    t_id.to_csv('{}/t_id.csv'.format(runPath))
    s_id = pd.DataFrame(s_id)
    s_id.to_csv('{}/s_id.csv'.format(runPath))
    
    
    ### initialize model
    modelC = getattr(models, 'VAE_{}'.format(args.model))
    model = modelC(args).to(device)
    torch.save(args, runPath+'/args.rar')
    
    train_loader = model.getDataLoaders(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, device=device)  
    test_loader = model.getDataLoaders(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, device=device)  
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, amsgrad=True)
    objective = getattr(objectives, args.obj) 
    s_objective = getattr(objectives, args.obj)
    
    
    def train(epoch, agg, W):
        model.train()
        b_loss = 0
        for i, dataT in enumerate(train_loader):
            beta = (epoch - 1) / W  if epoch <= W else 1
            if dataT[0].size()[0] == 1:
                continue
            data = [d.to(device) for d in dataT] #multimodal
            optimizer.zero_grad()
            loss = -objective(model, data, beta)
            loss.backward()
            optimizer.step()
            b_loss += loss.item()
            if args.print_freq > 0 and i % args.print_freq == 0:
                print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / args.batch_size))
        agg['train_loss'].append(b_loss / len(train_loader.dataset))
        print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, agg['train_loss'][-1]))
        return b_loss
    
    def test(epoch, agg, W):
        model.eval()
        b_loss = 0
        with torch.no_grad():
            for i, dataT in enumerate(test_loader):
                beta = (epoch - 1) / W  if epoch <= W else 1
                if dataT[0].size()[0] == 1:
                    continue
                data = [d.to(device) for d in dataT]
                loss = -s_objective(model, data, beta)
                b_loss += loss.item()
        agg['test_loss'].append(b_loss / len(test_loader.dataset))
        print('====>             Test loss: {:.4f}'.format(agg['test_loss'][-1]))
        
    
    ### start training
    with Timer('MM-VAE') as t:
        agg = defaultdict(list)
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=10, verbose=True) 
        W = args.deterministic_warmup
        start_early_stop = W
        for epoch in range(1, args.epochs + 1):
            b_loss = train(epoch, agg, W)
            if torch.isnan(torch.tensor([b_loss])):
                break          
            test(epoch, agg, W)
            save_vars(agg, runPath + '/losses.rar')
            if epoch > start_early_stop: 
                early_stopping(agg['test_loss'][-1], model, runPath)
            if early_stopping.early_stop:
                print('Early stopping')
                break
            
            
    def predict(dataloader, train_test, runPath):
        model.eval()
        with torch.no_grad():
            uni, cross = [], []
            for i, dataT in enumerate(dataloader):
                data = [d.to(device) for d in dataT]
                recons_mat = model.reconstruct_sample(data)
                for e, recons_list in enumerate(recons_mat):
                    for d, recon in enumerate(recons_list):
                        if e == d:
                            recon = recon.squeeze(0).cpu().detach().numpy()
                            recon = pd.DataFrame(recon)
                            recon = sp.csr_matrix(recon)
                            if i == 0:
                                uni.append(recon)
                            else:
                                uni[e] = sp.vstack((uni[e], recon), format='csr')
                        if e != d:
                            recon = recon.squeeze(0).cpu().detach().numpy()
                            recon = pd.DataFrame(recon)
                            recon = sp.csr_matrix(recon)
                            if i == 0:
                                cross.append(recon)
                            else:
                                cross[e] = sp.vstack((cross[e], recon), format='csr')
            
            mmwrite('{}/pred_{}_r_r.mtx'.format(runPath, train_test), uni[0])
            mmwrite('{}/pred_{}_p_p.mtx'.format(runPath, train_test), uni[1])
            mmwrite('{}/pred_{}_r_p.mtx'.format(runPath, train_test), cross[0])
            mmwrite('{}/pred_{}_p_r.mtx'.format(runPath, train_test), cross[1])
            
    def get_latent(dataloader, train_test, runPath):
        model.eval()
        with torch.no_grad():
            if args.model == 'rna_atac':
                modal = ['rna', 'atac'] 
            elif args.model == 'rna_protein':
                modal = ['rna', 'protein']
            pred = []
            for i, dataT in enumerate(dataloader):
                data = [d.to(device) for d in dataT]
                lats = model.latents(data, sampling=False)
                if i == 0:
                    pred = lats
                else:
                    for m,lat in enumerate(lats):
                        pred[m] = torch.cat([pred[m], lat], dim=0) 
                        
            for m,lat in enumerate(pred):
                lat = lat.cpu().detach().numpy()
                lat = pd.DataFrame(lat)
                lat.to_csv('{}/lat_{}_{}.csv'.format(runPath, train_test, modal[m]))
            mean_lats = sum(pred)/len(pred)
            mean_lats = mean_lats.cpu().detach().numpy()
            mean_lats = pd.DataFrame(mean_lats)
            mean_lats.to_csv('{}/lat_{}_mean.csv'.format(runPath,train_test))
  
            
    test_id = np.array(range(test_r_dataset.data.shape[0]))
    test_dataset = [Subset(test_r_dataset, test_id), Subset(test_modal_dataset, test_id)]
    
    train_loader = model.getDataLoaders(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, device=device)
    test_loader = model.getDataLoaders(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, device=device)
    
    
    get_latent(train_loader, 'train', runPath)
    get_latent(test_loader, 'test', runPath)
    predict(test_loader, 'test', runPath)   # perform prediction on the query
    model.traverse(runPath, device)
    
    predicted_protein = mmread(f"{runPath}/pred_test_r_p.mtx")
    predicted_protein_expression = pd.DataFrame(predicted_protein.todense(), index=test_modal_dataset.barcode, columns=test_modal_dataset.genes)
    predicted_protein_expression.to_csv(f"{args.base_dir}/test_predicted_protein_expression (scMM).csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scMM Hyperparameters')
    parser.add_argument('--experiment', type=str, default='208_M2-208_M1', metavar='E', help='experiment name')
    parser.add_argument('--model', type=str, default='rna_protein', metavar='M', help='model name (default: mnist_svhn)')
    parser.add_argument('--obj', type=str, default='m_elbo_naive_warmup', metavar='O', help='objective to use (default: elbo)')
    parser.add_argument('--llik_scaling', type=float, default=1., help='likelihood scaling for cub images/svhn modality when running in multimodal setting, set as 0 to use default value')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size for data (default: 256)')
    parser.add_argument('--epochs', type=int, default=50, metavar='E', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='L', help='learning rate (default: 1e-3)')                   
    parser.add_argument('--latent_dim', type=int, default=10, metavar='L', help='latent dimensionality (default: 20)')
    parser.add_argument('--num_hidden_layers', type=int, default=2, metavar='H', help='number of hidden layers in enc and dec (default: 1)')
    parser.add_argument('--r_hidden_dim', type=int, default=100, help='number of hidden units in enc/dec for gene')
    parser.add_argument('--p_hidden_dim', type=int, default=20, help='number of hidden units in enc/dec for protein/peak')
    parser.add_argument('--pre_trained', type=str, default="", help='path to pre-trained model (train from scratch if empty)')
    parser.add_argument('--learn_prior', action='store_true', default=True, help='learn model prior parameters')
    parser.add_argument('--analytics', action='store_true', default=True, help='disable plotting analytics')
    parser.add_argument('--print_freq', type=int, default=0, metavar='f', help='frequency with which to print stats (default: 0)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disable CUDA use')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--base_dir', type=str, default="")
    parser.add_argument('--r_dim', type=int, default=1)
    parser.add_argument('--p_dim', type=int, default=1)
    parser.add_argument('--deterministic_warmup', type=int, default=25, metavar='W', help='deterministic warmup')
    
    args = parser.parse_args()
    
    pipeline(**vars(args))










