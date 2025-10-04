# -*- coding: utf-8 -*-
import argparse
import logging
import os
import pandas as pd
from dance import logger
from dance.modules.multi_modality.predict_modality.babel import BabelWrapper
from dance.utils import set_seed
import scanpy as sc
import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix
import mudata
from dance.data import Data


def pipeline(**kwargs):
    dataset_dir = kwargs["dataset_dir"]   # dataset path
    base_dir = kwargs["base_dir"]   # output path
    os.makedirs(base_dir, exist_ok=True)
        
    ### load data
    train_data = ad.read(f"{dataset_dir}/SLN208_M2.h5ad")
    test_data = ad.read(f"{dataset_dir}/SLN208_M1.h5ad")
    print(f"Loading train data from: {dataset_dir}/SLN208_M2.h5ad")
    print(f"Loading test data from: {dataset_dir}/SLN208_M1.h5ad")
    
    
    ### select common genes
    common_genes = np.intersect1d(train_data.var.index, test_data.var.index)
    train_data = train_data[:, common_genes]
    test_data = test_data[:, common_genes]
    
    
    ### select common proteins
    common_proteins = np.intersect1d(np.array(train_data.uns["protein_names"]), np.array(test_data.uns["protein_names"]))
    if isinstance(train_data.obsm["protein_expression"], pd.DataFrame):
        train_protein_expression = train_data.obsm["protein_expression"]
        train_protein_expression = train_protein_expression.loc[:, common_proteins]
        train_data.obsm["protein_expression"] = train_protein_expression
        train_data.uns["protein_names"] = np.array(train_protein_expression.columns)
    else:
        train_protein_expression = pd.DataFrame(train_data.obsm["protein_expression"].todense(), index=train_data.obs.index, columns=train_data.uns["protein_names"])
        train_protein_expression = train_protein_expression.loc[:, common_proteins]
        train_data.obsm["protein_expression"] = csr_matrix(train_protein_expression.values)
        train_data.uns["protein_names"] = np.array(train_protein_expression.columns)

    if isinstance(test_data.obsm["protein_expression"], pd.DataFrame):
        test_protein_expression = test_data.obsm["protein_expression"]
        test_protein_expression = test_protein_expression.loc[:, common_proteins]
        test_data.obsm["protein_expression"] = test_protein_expression
        test_data.uns["protein_names"] = np.array(test_protein_expression.columns)
    else:
        test_protein_expression = pd.DataFrame(test_data.obsm["protein_expression"].todense(), index=test_data.obs.index, columns=test_data.uns["protein_names"])
        test_protein_expression = test_protein_expression.loc[:, common_proteins]
        test_data.obsm["protein_expression"] = csr_matrix(test_protein_expression.values)
        test_data.uns["protein_names"] = np.array(test_protein_expression.columns)
    
    
    ### normalize raw gene expression
    sc.pp.normalize_total(train_data)
    sc.pp.normalize_total(test_data)
    sc.pp.log1p(train_data)
    sc.pp.log1p(test_data)
    sc.pp.scale(train_data)
    sc.pp.scale(test_data)
    
    X_flat = train_data.X.flatten()
    lower, upper = np.percentile(X_flat, [0.5, 99.5])
    train_data.X = np.clip(train_data.X, lower, upper)

    X_flat = test_data.X.flatten()
    lower, upper = np.percentile(X_flat, [0.5, 99.5])
    test_data.X = np.clip(test_data.X, lower, upper)
    
    train_rna = ad.AnnData(X=train_data.X, obs=train_data.obs, var=train_data.var)
    test_rna = ad.AnnData(X=test_data.X, obs=test_data.obs, var=test_data.var)
    train_rna.X = csr_matrix(train_rna.X)
    test_rna.X = csr_matrix(test_rna.X)
    
    
    ### normalize raw protein expression
    gmean_train = ((train_protein_expression.applymap(np.log1p).sum(axis=1)) / train_protein_expression.shape[1]).apply(np.exp)
    gmean_test = ((test_protein_expression.applymap(np.log1p).sum(axis=1)) / test_protein_expression.shape[1]).apply(np.exp)
    
    train_protein_clr = (train_protein_expression.div(gmean_train, axis=0)).applymap(np.log1p)
    test_protein_clr = (test_protein_expression.div(gmean_test, axis=0)).applymap(np.log1p)
    
    protein_name = pd.DataFrame(train_data.uns["protein_names"], index=train_data.uns["protein_names"], columns=["protein_names"])
    train_protein = ad.AnnData(X=train_protein_clr, var=protein_name, obs=train_data.obs)
    test_protein = ad.AnnData(X=test_protein_clr, var=protein_name, obs=test_data.obs)
    train_protein.X = csr_matrix(train_protein.X)
    test_protein.X = csr_matrix(test_protein.X)
    
    
    modalities = [train_rna, train_protein, test_rna, test_protein]
    mod1 = ad.concat((modalities[0], modalities[2]))   
    mod2 = ad.concat((modalities[1], modalities[3]))
    mod1.var_names_make_unique()
    mod2.var_names_make_unique()
    
    mdata = mudata.MuData({"mod1": mod1, "mod2": mod2})
    train_size = modalities[0].shape[0]
    data = Data(mdata, train_size=train_size)
    data.set_config(feature_mod="mod1", label_mod="mod2")
    
    
    kwargs["outdir"] = os.path.abspath(kwargs["outdir"])
    os.makedirs(kwargs["model_folder"], exist_ok=True)
    os.makedirs(kwargs["outdir"], exist_ok=True)

    fh = logging.FileHandler(f"{kwargs['outdir']}/training_{kwargs['subtask']}_{kwargs['seed']}.log", "w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    for arg in kwargs:
        logger.info(f"Parameter {arg}: {kwargs[arg]}")
    
    
    x_train, y_train = data.get_train_data(return_type="torch")
    x_test, y_test = data.get_test_data(return_type="torch")
    x_train, y_train, x_test, y_test = x_train.float(), y_train.float(), x_test.float(), y_test.float()
    
    set_seed(kwargs["seed"])
    args_namespace = argparse.Namespace(**kwargs)
    model = BabelWrapper(args_namespace, dim_in=x_train.shape[1], dim_out=y_train.shape[1])
    model.fit(x_train, y_train, val_ratio=0.15)
    
    predicted_protein_expression = pd.DataFrame(model.predict(x_test).cpu().detach().numpy(), index=test_protein.obs.index, columns=test_protein.var.index)
    predicted_protein_expression.to_csv(f"{base_dir}/test_predicted_protein_expression (Babel).csv")
    print(f"Prediction saved to {base_dir}/test_predicted_protein_expression (Babel).csv")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--subtask", default="openproblems_bmmc_cite_phase2_rna")
    parser.add_argument("-device", "--device", default="cuda")
    parser.add_argument("-cpu", "--cpus", default=1, type=int)
    parser.add_argument("-seed", "--seed", default=1, type=int)
    parser.add_argument("--runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("-m", "--model_folder", default=None)
    parser.add_argument("--outdir", "-o", default=None, help="Directory to output to")
    parser.add_argument("--lossweight", type=float, default=1., help="Relative loss weight")
    parser.add_argument("--lr", "-l", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batchsize", "-b", type=int, default=512, help="Batch size")
    parser.add_argument("--hidden", type=int, default=16, help="Hidden dimensions")
    parser.add_argument("--earlystop", type=int, default=20, help="Early stopping after N epochs")
    parser.add_argument("--naive", "-n", action="store_true", help="Use a naive model instead of lego model")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("-bd", "--base_dir", default="")
    parser.add_argument("-dd", "--dataset_dir", default="")
    args = parser.parse_args()
    args.resume = True
    
    if args.model_folder is None:
        args.model_folder = os.path.join(args.base_dir, "model")
    if args.outdir is None:
        args.outdir = os.path.join(args.base_dir, "logs")
    
    pipeline(**vars(args))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
