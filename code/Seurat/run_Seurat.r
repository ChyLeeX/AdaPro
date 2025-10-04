library(Seurat)
library(glue)
library(fs)
library(anndata)
library(reticulate)

use_condaenv("seurat", required = TRUE)


working_dir <- here::here()
parent_dir <- fs::path_dir(fs::path_dir(working_dir))
dataset_dir <- fs::path(parent_dir, "datasets/CITE-SLN-Gayoso")   # dataset path
base_dir <- fs::path(parent_dir, "outputs/batch variation/208_M2-208_M1/Seurat")   # output path
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
  write.csv(protein_names_convert, glue("{base_dir}/Seurat_protein_names_convert.csv"))
}

test_seurat <- CreateSeuratObject(counts = test_rna, assay = "RNA")
test_seurat[["ADT"]] <- CreateAssayObject(counts = test_adt)


### normalize raw gene and protein expression
train_seurat <- NormalizeData(train_seurat)
test_seurat <- NormalizeData(test_seurat)

hvgs = SelectIntegrationFeatures(list(train_seurat, test_seurat), nfeatures = 2000, fvf.nfeatures = 2000) 

train_seurat <- NormalizeData(train_seurat, normalization.method = "CLR", margin = 2, assay = "ADT")
test_seurat <- NormalizeData(test_seurat, normalization.method = "CLR", margin = 2, assay = "ADT")


### mask the protein expression in the query
test_seurat_protein <- test_seurat[["ADT"]]
test_seurat[["ADT"]] <- NULL


### identify anchors
transfer_anchors <- FindTransferAnchors(reference = train_seurat, query = test_seurat, reduction = "cca", dims = 1:50, features = hvgs, 
                                        k.anchor = 5, k.filter = 200, k.score = 30, max.features = 200)
saveRDS(transfer_anchors, glue("{base_dir}/Seurat_anchors_cca.rds"))


### perform prediction on the query
refdata <- GetAssayData(object = train_seurat, assay = 'ADT', slot = 'data')
test_protein_imputation <- TransferData(anchorset = transfer_anchors, refdata = refdata, query = test_seurat, dims = 1:50, k.weight = 50, 
                                        weight.reduction = "pca")
test_protein_prediction <- as.data.frame(test_protein_imputation@assays[["id"]]@data)
write.table(test_protein_prediction, glue("{base_dir}/test_predicted_protein_expression (Seurat).txt"))










