library(scran)
library(SingleCellExperiment)
library(Matrix)
library(glue)
library(here)
library(fs)


working_dir <- here::here()
parent_dir <- fs::path_dir(fs::path_dir(working_dir))
base_dir <- fs::path(parent_dir, "outputs/batch variation/208_M2-208_M1/Dengkw")   # output path


### load reference data
train_gene_expression <- t(as(readMM(glue("{base_dir}/gene_expression_train.mtx")), "CsparseMatrix"))
train_meta <- read.table(glue("{base_dir}/cell_metadata_train.txt"), sep = ",", header = TRUE, row.names = 1)
train_var <- read.table(glue("{base_dir}/var_metadata_train.txt"), sep = ",", header = TRUE, row.names = 1)

row.names(train_gene_expression) = row.names(train_var)
colnames(train_gene_expression) = row.names(train_meta)


### normalize reference raw gene expression
sce_train <- SingleCellExperiment(list(counts = train_gene_expression))
clusters_train <- train_meta$groups
sum_fac_train <- computeSumFactors(sce_train, clusters = clusters_train, min.mean = 0.1)
size_fac_train <- sizeFactors(sum_fac_train)

train_gene_expression_normalized <- t(train_gene_expression) / size_fac_train
train_gene_expression_normalized <- log(train_gene_expression_normalized + 1)
train_gene_expression_normalized <- as(train_gene_expression_normalized, "dgCMatrix")


### load query data
test_gene_expression <- t(as(readMM(glue("{base_dir}/gene_expression_test.mtx")), "CsparseMatrix"))
test_meta <- read.table(glue("{base_dir}/cell_metadata_test.txt"), sep = ",", header = TRUE, row.names = 1)
test_var <- read.table(glue("{base_dir}/var_metadata_test.txt"), sep = ",", header = TRUE, row.names = 1)

row.names(test_gene_expression) = row.names(test_var)
colnames(test_gene_expression) = row.names(test_meta)


### normalize query raw gene expression
sce_test <- SingleCellExperiment(list(counts = test_gene_expression))
clusters_test <- test_meta$groups
sum_fac_test <- computeSumFactors(sce_test, clusters = clusters_test, min.mean = 0.1)
size_fac_test <- sizeFactors(sum_fac_test)

test_gene_expression_normalized <- t(test_gene_expression) / size_fac_test
test_gene_expression_normalized <- log(test_gene_expression_normalized + 1)
test_gene_expression_normalized <- as(test_gene_expression_normalized, "dgCMatrix")


writeMM(train_gene_expression_normalized, glue("{base_dir}/normalized_gene_expression_train.mtx"))
writeMM(test_gene_expression_normalized, glue("{base_dir}/normalized_gene_expression_test.mtx"))













