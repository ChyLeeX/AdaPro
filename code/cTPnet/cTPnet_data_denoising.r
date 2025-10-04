library(anndata)
library(SAVERX)
library(Matrix)
library(here)
library(fs)
library(reticulate)

use_condaenv("SAVERX", required = TRUE)


working_dir <- here::here()
parent_dir <- fs::path_dir(fs::path_dir(working_dir))
dataset_dir <- fs::path(parent_dir, "datasets/CITE-SLN-Gayoso")   # dataset path


### load data
data <- read_h5ad(file.path(dataset_dir, "SLN208_M1.h5ad"))
if (inherits(data$X, "dgCMatrix")) {
  rna_expression <- t(data$X)
} else {
  rna_expression <- t(as(as.matrix(data$X), "dgCMatrix"))
}


### run SAVER-X
file <- saverx(rna_expression)
denoised.data <- readRDS(file)
denoised.data <- denoised.data$estimate
write.table(denoised.data, file.path(dataset_dir, "SLN208_M1_denoised_gene_expression_data.txt"))



