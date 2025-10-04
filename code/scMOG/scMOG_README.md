## Installation
For instructions on installing the scMOG package, please refer to its [official GitHub repository](https://github.com/GaoLabXDU/scMOG).

After installation, we provide an example of running scMOG using the 208-M2 → 208-M1 experiment in the CITE-SLN-Gayoso dataset. 

## Prepare data
To meet the input requirements of scMOG, gene expression and protein expression data should be converted into separate objects and saved accordingly. Detailed steps are provided in `scMOG_prepare_data.ipynb`.

## Run model
scMOG can be run from the command line. Before running, please make sure to set your working directory to `<your_path>/codes/scMOG/scMOG_code/bin`. To train the model on reference data, use the following command:
```bash
python train_protein.py --outdir "<your_path>/outputs/batch variation/208_M2-208_M1/scMOG" --dataset_path "<your_path>/outputs/batch variation/208_M2-208_M1/scMOG"
```
Here, `outdir` is the folder where training results, such as models and logs, will be saved, and `dataset_dir` points to the location of the converted input data. 

To perform prediction on query data, please use the following command:
```bash
python predict_protein.py --outdir "<your_path>/outputs/batch variation/208_M2-208_M1/scMOG" --dataset_path "<your_path>/outputs/batch variation/208_M2-208_M1/scMOG"
```
Here, `outdir` is the folder where predictions will be saved, and `dataset_dir` points to the location of the training results and converted input data.
