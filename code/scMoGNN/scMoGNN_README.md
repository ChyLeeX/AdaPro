## Installation
scMoGNN is implemented on top of the dance Python package. Please refer to its [official GitHub repository](https://github.com/OmicsML/dance) for detailed installation instructions.

After installing dance, we provide an example of running Babel using the 208-M2 → 208-M1 experiment in the CITE-SLN-Gayoso dataset. 


## Cluster gene expression data
The preprocessing of gene expression in scMoGNN follows the same procedure as in Dengkw. For completeness, the corresponding code is still provided. First, please run `scMoGNN_cluster.ipynb` to generate the clustering results needed for gene expression data preprocessing.

## Preprocess data
Then, please run `scMoGNN_preprocess.r` to normalize the gene expression data.

## Run model
scMoGNN can be run from the command line. Before running, please make sure to set your working directory to `<your_path>/codes/scMoGNN`. To train the model, use the following command:
```bash
python run_scMoGNN.py --base_dir "<your_path>/outputs/batch variation/208_M2-208_M1/scMoGNN" --dataset_dir "<your_path>/demo/datasets/CITE-SLN-Gayoso" --pathway_path "<your_path>/codes/scMoGNN/h.all.v7.4"
```
Here, `base_dir` is the folder where all outputs, such as models, logs, and predictions, will be saved, `dataset_dir` points to the location of your raw dataset files and `pathway_path` is the folder containing the prior pathway database, which includes two files: `h.all.v7.4.entrez.gmt` and `h.all.v7.4.symbols.gmt`. Note that the prior pathway database provided by the original authors is available only for human data. Therefore, it can be used only when running human-related datasets, in which case you should add --`pathway` in the command line.


