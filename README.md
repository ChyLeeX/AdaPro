# Extending proteomic accessibility to transcriptome-only single-cell data through adaptive consensus learning
## Overview
Here, we present AdaPro, an adaptive consensus learning framework for extending proteomic accessibility to transcriptome-only scRNA-seq datasets. By integrating heterogeneous method outputs into a unified prediction and dynamically optimizing method-specific weights, AdaPro strengthens reliable predictors, dampens unstable ones, and provides interpretable indicators of accuracy. Across benchmarks encompassing limited references, dropout noise, batch variation, and biological heterogeneity, AdaPro consistently outperforms existing methods. Applied to real scRNA-seq datasets, AdaPro faithfully recovers canonical protein markers, preserves inter-cell-type organization, and resolves intra-cell-type diversity, underscoring its biological utility. Collectively, these results establish AdaPro as a principled and generalizable framework that facilitates proteomic accessibility for transcriptome-only datasets and lays a foundation for large-scale integrative single-cell multi-omics analyses, accelerating biological discovery across diverse contexts.

![image](Overview.png 'pipeline')

## Datasets
All datasets used in this study can be accessed via the original links provided in the Data availability section of the manuscript. In addition, the curated datasets have been uploaded to a public repository in Zenodo with a DOI assignment (DOI: <https://doi.org/10.5281/zenodo.17251049>). If you find these datasets useful for your research, please consider citing:
```bibtex
@article{li2025extending,
  title={Extending proteomic accessibility to transcriptome-only single-cell data through adaptive consensus learning},
  author={Li Chen-Yang, Hong Yong-Jia, Li Bo and Zhang Xiao-Fei},
  url={https://doi.org/10.5281/zenodo.17251049},
  year={2025},
}
```
## Installation
Running AdaPro involves multiple base methods, which may have conflicting package dependencies. To avoid potential issues during installation, each base method is executed in its own isolated environment. Installation instructions for each base method can be found in their respective `README` files. 

For integrating the base results, AdaPro only requires a few common Python packages, such as `anndata`, `scanpy`, and a version of `PyTorch` compatible with your CUDA setup.  

## Implementation
After downloading the datasets and code, please organize them according to the following directory structure:
```
root directory
│
├── datasets
│   ├── CITE-BMMC-Stuart
│   ├── CITE-SLN-Gayoso
│   ├── ...
│   └── your own dataset
│
├── code
│   ├── Babel
│   ├── cTPnet
│   ├── ...
│   └── TotalVI
│
└── outputs
```
The `code` folder contains detailed tutorials and documentation for 12 base methods. The implementation pipeline of each method is illustrated using the 208-M2 → 208-M1 experiment in the CITE-SLN-Gayoso dataset as an example. Unless otherwise specified, the working directory for each program should be set to the `root directory/code/method`. The `outputs` folder is used to store all experimental results.

After running the base methods independently and obtaining their respective results, AdaPro integrates these outputs into a consensus prediction. Detailed step-by-step guidance is provided in the notebook `run_AdaPro.ipynb`.

## Contact
For any questions or clarifications regarding the content or usage of this repository, please contact Mr. Chen-Yang Li (<lichy@mails.ccnu.edu.cn>) or Prof. Xiao-Fei Zhang (<zhangxf@ccnu.edu.cn>).
