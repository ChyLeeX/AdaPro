library(Seurat)
library(sctransform)
library(glue)
library(fs)
library(Matrix)
library(anndata)
library(reticulate)


use_condaenv("scVAEIT", required = TRUE)


working_dir <- here::here()
parent_dir <- fs::path_dir(fs::path_dir(working_dir))
dataset_dir <- fs::path(parent_dir, "datasets/CITE-SLN-Gayoso")   # dataset path
base_dir <- fs::path(parent_dir, "outputs/batch variation/208_M2-208_M1/scVAEIT")   # output path
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


train_seurat <- CreateSeuratObject(counts = train_rna, assay = "RNA")
train_seurat[["ADT"]] <- CreateAssayObject(counts = train_adt)

test_seurat <- CreateSeuratObject(counts = test_rna, assay = "RNA")
test_seurat[["ADT"]] <- CreateAssayObject(counts = test_adt)


### select highly variable genes in reference data
train_seurat <- SCTransform(train_seurat, variable.features.n=5000, verbose = TRUE)
HVGs <- VariableFeatures(train_seurat)
train_gene_expression = Matrix::t(train_seurat[["RNA"]]@counts[HVGs, ])
test_gene_expression = Matrix::t(test_seurat[["RNA"]]@counts)


Matrix::writeMM(train_gene_expression, glue("{base_dir}/train_gene_expression.mtx"))
Matrix::writeMM(test_gene_expression, glue("{base_dir}/test_gene_expression.mtx"))
write.table(colnames(train_gene_expression), file = glue("{base_dir}/train_HVGs.txt"), row.names = FALSE, col.names = FALSE)
write.table(colnames(test_gene_expression), file = glue("{base_dir}/test_gene_names.txt"), row.names = FALSE, col.names = FALSE)
write.table(rownames(train_gene_expression), file = glue("{base_dir}/train_cell_names.txt"), row.names = FALSE, col.names = FALSE)
write.table(rownames(test_gene_expression), file = glue("{base_dir}/test_cell_names.txt"), row.names = FALSE, col.names = FALSE)


