## Installation
For instructions on installing the scMM package, please refer to its [official GitHub repository](https://github.com/kodaim1115/scMM).

After installation, we provide an example of running scMM using the 208-M2 → 208-M1 experiment in the CITE-SLN-Gayoso dataset. 

## Prepare data
To meet the input requirements of scMM, gene expression and protein expression data should be converted into separate objects and saved accordingly. Detailed steps are provided in `scMM_prepare_data.ipynb`.

## Run model
scMM can be run from the command line. Before running, please make sure to set your working directory to `<your_path>/codes/scMM`. To train the model, use the following command:
```bash
python run_scMM.py --base_dir "<your_path>/outputs/batch variation/208_M2-208_M1/scMM"
```
Here, `base_dir` is the folder where the converted input data are stored, and all outputs such as models, logs, and predictions will also be saved.