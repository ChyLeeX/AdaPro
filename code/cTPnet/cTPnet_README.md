## Installation
For instructions on installing the cTPnet package, please refer to its [official GitHub repository](https://github.com/zhouzilu/ctpnetpy). cTPnet relies on SAVER-X to denoise raw gene expression data. For installing SAVER-X, please consult its [official GitHub repository](https://github.com/jingshuw/SAVERX).

After installation, we provide an example of running cTPnet using the 208-M2 → 208-M1 experiment in the CITE-SLN-Gayoso dataset. 

## Denoise gene expression data
Before training the model, cTPnet applies SAVER-X to denoise the raw gene expression data. For detailed instructions, please refer to `cTPnet_data_denoising.r`.

## Run model
You can run cTPnet following the detailed instructions provided in `run_cTPnet.ipynb`.

