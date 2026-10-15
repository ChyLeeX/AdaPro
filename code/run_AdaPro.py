# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 14:47:47 2026

@author: LCY
"""
import anndata
import pandas as pd
import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
from pathlib import Path
import os


### Set path
working_dir = Path(os.getcwd())
base_dir = f"{working_dir.parents[0]}/outputs/batch variation/208_M2-208_M1"


### Load base results
base_methods = ["Babel", "cTPnet", "Dengkw", "Liger", "moETM", "sciPENN", "scMM", "scMOG", "scMoGNN", "scVAEIT", "Seurat", "TotalVI"]
base_results = {}
for name in base_methods:
    if name == "scMOG":
        results = anndata.read(f"{base_dir}/{name}/RNA_protein_adata.h5ad")
        results = pd.DataFrame(results.X.toarray(), index=results.obs.index, columns=results.var.index)
        base_results[name] = results
#    
    elif name == "Liger":
        results = pd.read_csv(f"{base_dir}/{name}/test_predicted_protein_expression ({name}).csv", index_col=0).T
        base_results[name] = results
#       
    elif name == "Seurat":
        results = pd.read_table(f"{base_dir}/{name}/test_predicted_protein_expression ({name}).txt", index_col=0, sep=" ").T
        base_results[name] = results
#       
    else:
        results = pd.read_csv(f"{base_dir}/{name}/test_predicted_protein_expression ({name}).csv", index_col=0)
        base_results[name] = results


### Recover the protein names in cTPnet, Liger, and Seurat
cTPnet_protein_names_convert = pd.read_csv(f"{base_dir}/cTPnet/cTPnet_protein_names_convert.csv", index_col=0)
cTPnet_protein_names_convert.set_index("convert_name", inplace=True)
Liger_protein_names_convert = pd.read_csv(f"{base_dir}/Liger/Liger_protein_names_convert.csv", index_col=0)
Liger_protein_names_convert.set_index("convert_name", inplace=True)
Seurat_protein_names_convert = pd.read_csv(f"{base_dir}/Seurat/Seurat_protein_names_convert.csv", index_col=0)
Seurat_protein_names_convert.set_index("convert_name", inplace=True)

base_results["cTPnet"].columns = cTPnet_protein_names_convert.loc[base_results["cTPnet"].columns, "original_name"].values
base_results["Liger"].columns = Liger_protein_names_convert.loc[base_results["Liger"].columns, "original_name"].values
base_results["Seurat"].columns = Seurat_protein_names_convert.loc[base_results["Seurat"].columns, "original_name"].values


### Align rows and columns of all base results to the first one as reference
base_df = base_results[base_methods[0]]
base_index = base_df.index
base_columns = base_df.columns
for key, df in base_results.items():
    base_results[key] = df.reindex(index=base_index, columns=base_columns)


### Verify that rows and columns have the same order across base results
for key, df in base_results.items():
    same_index = (df.index.equals(base_index))
    same_columns = (df.columns.equals(base_columns))
    print(f"{key}: index aligned? {same_index}, columns aligned? {same_columns}")


### Convert log-scale base results back to the original scale by exponentiation
need_exp = ["Babel", "cTPnet", "Dengkw", "sciPENN", "scMOG", "scMoGNN", "scVAEIT", "Seurat"]
for key, df in base_results.items():
    if key in need_exp:
        base_results[key] = np.exp(df)
#   
    else:
        base_results[key] = df.copy()
   
    
### Normalize base results
def log_normalize(data, target_sum=10000):
    adata = anndata.AnnData(X=data)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var.index)

def Z_score(data):
    adata = anndata.AnnData(X=data)
    sc.pp.scale(adata)
    return pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var.index)

base_results_ln = {k: log_normalize(v.copy()) for k, v in base_results.items()}
base_results_zs = {k: Z_score(v.copy()) for k, v in base_results_ln.items()}


### Define a function to compute cosine distance
def cosine_distance_sum(H1: torch.Tensor, H2: torch.Tensor) -> torch.Tensor:
    assert H1.shape == H2.shape, "H1 and H2 must have the same shape"
    N, P = H1.shape
#
    row_norm_H1 = H1.norm(dim=1)
    row_norm_H2 = H2.norm(dim=1)
    row_valid = (row_norm_H1 > 0) & (row_norm_H2 > 0)
#
    A = F.normalize(H1[row_valid], dim=1)
    B = F.normalize(H2[row_valid], dim=1)
    row_cos_sim = (A * B).sum(dim=1)
    row_dist = torch.full((N,), 2.0, device=H1.device)
    row_dist[row_valid] = 1 - row_cos_sim
#
    H1_T = H1.T
    H2_T = H2.T
    col_norm_H1 = H1_T.norm(dim=1)
    col_norm_H2 = H2_T.norm(dim=1)
    col_valid = (col_norm_H1 > 0) & (col_norm_H2 > 0)
#
    A_T = F.normalize(H1_T[col_valid], dim=1)
    B_T = F.normalize(H2_T[col_valid], dim=1)
    col_cos_sim = (A_T * B_T).sum(dim=1)
    col_dist = torch.full((P,), 2.0, device=H1.device)
    col_dist[col_valid] = 1 - col_cos_sim
#
    return row_dist + col_dist


### Calculate the naive average of the base results
base_results_zs = list(base_results_zs.values())
M = len(base_results_zs)
mean_res = sum(base_results_zs) / len(base_results_zs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H0 = torch.tensor(mean_res.values, dtype=torch.float32, device=device)
H_list = [torch.tensor(h.values, dtype=torch.float32, device=device) for h in base_results_zs]
alpha = torch.zeros(M, requires_grad=True, device=device)
optimizer = torch.optim.Adam([alpha], lr=0.1)


### Empirically set the values of lambda
cos_dist_init = [cosine_distance_sum(H0, Hm).item() for Hm in H_list]
lamda = np.median(cos_dist_init)


### Start optimization
loss_list = []
cos_dist_list = []
tol = 1e-5
patience = 10
stable_steps = 0
for step in range(500):
    optimizer.zero_grad()
#
    weights = F.softmax(alpha, dim=0)  # (M,)
#   
    with torch.no_grad():
        H_ens = sum(w * H for w, H in zip(weights, H_list))  # (N, D)
#
    cos_dist = 0.0
    for w, Hm in zip(weights, H_list):
        dist = cosine_distance_sum(H_ens, Hm)
        cos_dist += w * dist
#   
    loss = cos_dist + lamda * (weights * torch.log(weights + 1e-8)).sum()
#
    loss.backward()
    optimizer.step()
#
    with torch.no_grad():
        updated_weights = F.softmax(alpha, dim=0)
        weight_change = torch.norm(updated_weights - weights.detach(), p=2).item()
#
    stable_steps = stable_steps + 1 if weight_change < tol else 0
#
    loss_list.append(loss.item())
    cos_dist_list.append(cos_dist.item())
#
    if stable_steps >= patience:
        break
#
    print(f"Step {step}: Loss = {loss.item():.4f}, Weights = {weights.detach().cpu().numpy()}")


### Save the loss during optimization
loss_list = np.array(loss_list)
cos_dist_list = np.array(cos_dist_list)
np.savetxt(f"{base_dir}/AdaPro_total_loss.csv", loss_list, delimiter='\t')
np.savetxt(f"{base_dir}/AdaPro_cosine_loss.csv", cos_dist_list, delimiter='\t')


### Save the consensus weights and predictions
AdaPro_weights = F.softmax(alpha, dim=0).detach().cpu().numpy()
AdaPro_weights_df = pd.DataFrame(AdaPro_weights, index=base_methods, columns=["AdaPro_weight"])
AdaPro_weights_df.to_csv(f"{base_dir}/AdaPro_weights.csv")

AdaPro_prediction = pd.DataFrame(sum(w * H for w, H in zip(AdaPro_weights, H_list)).detach().cpu().numpy(), index=mean_res.index, columns=mean_res.columns)
AdaPro_prediction.to_csv(f"{base_dir}/test_predicted_protein_expression (AdaPro).csv")






































