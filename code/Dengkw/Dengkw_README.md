## Installation
Dengkw requires some basic Python packages (`anndata`, `numpy`, `scikit-learn`, `scipy`, `pandas` and `scanpy`) and R packages (`scran`, `SingleCellExperiment` and `Matrix`). These dependencies can be installed via `pip/conda` for Python and `BiocManager` or `install.packages` for R.  

After installation, we provide an example of running Dengkw using the 208-M2 → 208-M1 experiment in the CITE-SLN-Gayoso dataset. 

## Cluster gene expression data
First, please run `Dengkw_cluster.ipynb` to generate the clustering results needed for gene expression data preprocessing.

## Preprocess data
Then, please run `Dengkw_preprocess.r` to normalize the gene expression data.

## Run model
You can run Dengkw following the detailed instructions provided in `run_Dengkw.ipynb`.


