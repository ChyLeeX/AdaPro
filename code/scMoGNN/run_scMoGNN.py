# -*- coding: utf-8 -*-
import os
import argparse
from argparse import Namespace
import mudata
import numpy as np
import torch
from dance.data import Data
from dance.modules.multi_modality.predict_modality.scmogcn import ScMoGCNWrapper
from dance.transforms.graph import ScMoGNNGraph
from dance.utils import set_seed
import pandas as pd
from dance import logger
import anndata as ad
from scipy.io import mmread
from scipy.sparse import csr_matrix


def pipeline(inductive=False, verbose=2, logger=None, **kwargs):
    base_dir = kwargs["base_dir"]   # output path
    dataset_dir = kwargs["dataset_dir"]   # dataset path
    
    kwargs["log_folder"] = f"{base_dir}/logs"
    kwargs["model_folder"] = f"{base_dir}/models"
    kwargs["result_folder"] = f"{base_dir}/results"
    
    PREFIX = kwargs["prefix"]
    os.makedirs(kwargs["log_folder"], exist_ok=True)
    os.makedirs(kwargs["model_folder"], exist_ok=True)
    os.makedirs(kwargs["result_folder"], exist_ok=True)
    
    
    ### load data
    train_data = ad.read(f"{dataset_dir}/SLN208_M2.h5ad")   
    test_data = ad.read(f"{dataset_dir}/SLN208_M1.h5ad")
    normalized_train_gene_expression = mmread(f"{base_dir}/normalized_gene_expression_train.mtx").tocsr()
    normalized_test_gene_expression = mmread(f"{base_dir}/normalized_gene_expression_test.mtx").tocsr()
    print(f"Loading train data from: {base_dir}/SLN208_M2.h5ad")
    print(f"Loading test data from: {base_dir}/SLN208_M1.h5ad")
    
    
    train_rna = ad.AnnData(X=normalized_train_gene_expression, obs=train_data.obs, var=train_data.var)
    test_rna = ad.AnnData(X=normalized_test_gene_expression, obs=test_data.obs, var=test_data.var)

    
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
    
    
    ### normalize raw protein expression
    gmean_train = ((train_protein_expression.applymap(np.log1p).sum(axis=1)) / train_protein_expression.shape[1]).apply(np.exp)
    gmean_test = ((test_protein_expression.applymap(np.log1p).sum(axis=1)) / test_protein_expression.shape[1]).apply(np.exp)
    
    train_protein_clr = (train_protein_expression.div(gmean_train, axis=0)).applymap(np.log1p)
    test_protein_clr = (test_protein_expression.div(gmean_test, axis=0)).applymap(np.log1p)


    train_protein = ad.AnnData(X=train_protein_clr)
    test_protein = ad.AnnData(X=test_protein_clr)
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
    
    
    ### construct the graph
    data = ScMoGNNGraph(inductive, kwargs["cell_init"], kwargs["pathway"], kwargs["subtask"], kwargs["pathway_weight"], kwargs["pathway_threshold"], kwargs["pathway_path"])(data)
    
    
    ### randomly select 15% of reference cells as validation set
    idx = np.random.permutation(modalities[0].shape[0])
    split = {"train": idx[:-int(len(idx) * 0.15)], "valid": idx[-int(len(idx) * 0.15):]}
    kwargs["FEATURE_SIZE"] = modalities[0].shape[1]
    kwargs["TRAIN_SIZE"] = modalities[0].shape[0]
    kwargs["OUTPUT_SIZE"] = modalities[1].shape[1]
    kwargs["CELL_SIZE"] = modalities[0].shape[0] + modalities[2].shape[0]
    
    
    gtest = g = data.data.uns["g"]
    
    
    _, y_train = data.get_train_data(return_type="torch")
    _, y_test = data.get_test_data(return_type="torch")
    
    
    ### initialize model
    set_seed(args.seed)
    model = ScMoGCNWrapper(Namespace(**kwargs))
    
    logger = open(f"{kwargs['log_folder']}/{PREFIX}.log", "w")
    logger.write(str(kwargs) + "\n")
    
    ### start training
    model.fit(g, y_train, split, not inductive, verbose, y_test, logger)
    
    ### perform prediction on the query
    predicted_protein_expression = pd.DataFrame(model.predict(g, np.arange(kwargs["TRAIN_SIZE"], kwargs["CELL_SIZE"]), device="cpu").detach().numpy(), 
                                                columns=modalities[3].var.index, index=modalities[3].obs.index)
    predicted_protein_expression.to_csv(f"{base_dir}/test_predicted_protein_expression (scMoGNN).csv")
    print(f"Prediction saved to {base_dir}/test_predicted_protein_expression (scMoGNN).csv")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-prefix", "--prefix", default="208_M2-208_M1")
    parser.add_argument("-t", "--subtask", default="openproblems_bmmc_cite_phase2_rna")
    parser.add_argument("-pww", "--pathway_weight", default="cos", choices=["cos", "one", "pearson"])   
    parser.add_argument("-pwth", "--pathway_threshold", type=float, default=0.1)   
    parser.add_argument("-bd", "--base_dir", default="")
    parser.add_argument("-dd", "--dataset_dir", default="")
    parser.add_argument("-e", "--epoch", type=int, default=15000)
    parser.add_argument("-nbf", "--no_batch_features", action="store_true", default=True)
    parser.add_argument("-npw", "--pathway", action="store_true", default=False)   
    parser.add_argument("-res", "--residual", default="res_cat", choices=["none", "res_add", "res_cat"])
    parser.add_argument("-inres", "--initial_residual", action="store_true", default=True)   
    parser.add_argument("-pwagg", "--pathway_aggregation", default="alpha", choices=["sum", "attention", "two_gate", "one_gate", "alpha", "cat"])
    parser.add_argument("-pwalpha", "--pathway_alpha", type=float, default=0.25)
    parser.add_argument("-nrc", "--no_readout_concatenate", action="store_true")
    parser.add_argument("-bs", "--batch_size", default=1000, type=int)
    parser.add_argument("-nm", "--normalization", default="group", choices=["batch", "layer", "group", "none"])
    parser.add_argument("-ac", "--activation", default="gelu", choices=["leaky_relu", "relu", "prelu", "gelu"])
    parser.add_argument("-em", "--embedding_layers", default=1, type=int, choices=[1, 2, 3])
    parser.add_argument("-ro", "--readout_layers", default=1, type=int, choices=[1, 2])
    parser.add_argument("-conv", "--conv_layers", default=4, type=int, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("-agg", "--agg_function", default="mean", choices=["gcn", "mean"])
    parser.add_argument("-device", "--device", default="cuda")
    parser.add_argument("-sb", "--save_best", action="store_true", default=True)   
    parser.add_argument("-sf", "--save_final", action="store_true")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-2)
    parser.add_argument("-lrd", "--lr_decay", type=float, default=0.99)
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-4)   
    parser.add_argument("-hid", "--hidden_size", type=int, default=256)   
    parser.add_argument("-edd", "--edge_dropout", type=float, default=0.4)   
    parser.add_argument("-mdd", "--model_dropout", type=float, default=0.3)   
    parser.add_argument("-es", "--early_stopping", type=int, default=200)
    parser.add_argument("-c", "--cpu", type=int, default=1)
    parser.add_argument("-or", "--output_relu", default="none", choices=["relu", "leaky_relu", "none"])
    parser.add_argument("-i", "--inductive", action="store_true")
    parser.add_argument("-sa", "--subpath_activation", action="store_true")
    parser.add_argument("-ci", "--cell_init", default="none", choices=["none", "svd"])
    parser.add_argument("-bas", "--batch_seperation", action="store_true")
    parser.add_argument("-pwpath", "--pathway_path", default="")
    parser.add_argument("-seed", "--seed", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("-ws", "--weighted_sum", action="store_true", default=True)   
    parser.add_argument("-samp", "--sampling", action="store_true")
    parser.add_argument("-ns", "--node_sampling_rate", type=float, default=0.5)
    parser.add_argument("-prep", "--preprocessing", default="none", choices=["none", "feature_selection", "svd"])
    parser.add_argument("-lm", "--low_memory", type=bool, default=False)
    
    args = parser.parse_args()
    
    torch.set_num_threads(args.cpu)

    pipeline(**vars(args))
