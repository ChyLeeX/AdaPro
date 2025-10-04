library(rliger)
library(Seurat)
library(plyr)
library(Matrix)
library(glue)
library(fs)
library(anndata)
library(reticulate)

use_condaenv("rliger", required = TRUE)


working_dir <- here::here()
parent_dir <- fs::path_dir(fs::path_dir(working_dir))
dataset_dir <- fs::path(parent_dir, "datasets/CITE-SLN-Gayoso")   # dataset path
base_dir <- fs::path(parent_dir, "outputs/batch variation/208_M2-208_M1/Liger")   # output path
dir.create(base_dir, recursive = TRUE, showWarnings = FALSE)


### load data
train_data <- read_h5ad(glue("{dataset_dir}/SLN208_M2.h5ad"))
test_data <- read_h5ad(glue("{dataset_dir}/SLN208_M1.h5ad"))


if (inherits(train_data$X, "sparseMatrix")) {
  train_rna <- t(train_data$X) 
} else {
  train_rna <- t(as.matrix(train_data$X)) 
}

train_adt <- t(train_data$obsm$protein_expression)
row.names(train_adt) = train_data$uns$protein_names
colnames(train_adt) = row.names(train_data$obs)


if (inherits(test_data$X, "sparseMatrix")) {
  test_rna <- t(test_data$X) 
} else {
  test_rna <- t(as.matrix(test_data$X)) 
}

test_adt <- t(test_data$obsm$protein_expression)
row.names(test_adt) = test_data$uns$protein_names
colnames(test_adt) = row.names(test_data$obs)


### select common genes
common_genes = intersect(row.names(train_rna), row.names(test_rna))
train_rna = train_rna[common_genes, ]
test_rna = test_rna[common_genes, ]

### select common proteins
common_proteins = intersect(row.names(train_adt), row.names(test_adt))
train_adt = train_adt[common_proteins, ]
test_adt = test_adt[common_proteins, ]


train_seurat <- CreateSeuratObject(counts = train_rna, assay = "RNA")
train_seurat[["ADT"]] <- CreateAssayObject(counts = train_adt)
if (any(grepl("_", row.names(train_adt)))) {
    protein_names_convert <- data.frame(original_name = row.names(train_adt), convert_name = row.names(train_seurat[["ADT"]]))   # Seurat replaces '_' with '-' in protein names; save mapping for later analysis
    write.csv(protein_names_convert, glue("{base_dir}/Liger_protein_names_convert.csv"))
}

test_seurat <- CreateSeuratObject(counts = test_rna, assay = "RNA")
test_seurat[["ADT"]] <- CreateAssayObject(counts = test_adt)


train_rna <- train_seurat[["RNA"]]@counts   
test_rna <- test_seurat[["RNA"]]@counts   

train_adt <- train_seurat[["ADT"]]@counts


Ligerex.leaveout <- createLiger(list(train = train_rna, test = test_rna))   
Ligerex.leaveout <- normalize(Ligerex.leaveout)   
Ligerex.leaveout <- selectGenes(Ligerex.leaveout)   
Ligerex.leaveout <- scaleNotCenter(Ligerex.leaveout)   

Ligerex.leaveout <- optimizeALS(Ligerex.leaveout, k = 30)   
Ligerex.leaveout <- quantile_norm(Ligerex.leaveout)   


Ligerex.leaveout@norm.data$train <- rbind(train_rna, train_adt)  
Ligerex.leaveout@raw.data$train <- rbind(train_rna, train_adt)

Imputation <- imputeKNN(Ligerex.leaveout, reference = 'train', queries = list('test'), norm = FALSE, scale = FALSE, weight=FALSE, knn_k = 20) 
Result = as.matrix(Imputation@raw.data$test)[unlist(row.names(train_adt)), ]
write.table(Result, file = glue("{base_dir}/test_predicted_protein_expression (Liger).csv"), sep=',', quote=F, col.names = TRUE)


