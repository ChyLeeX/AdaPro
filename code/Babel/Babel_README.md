## Installation
Babel is implemented on top of the dance Python package. Please refer to its [official GitHub repository](https://github.com/OmicsML/dance) for detailed installation instructions.

After installing dance, we provide an example of running Babel using the 208-M2 → 208-M1 experiment in the CITE-SLN-Gayoso dataset. 

## Run model
Babel can be run from the command line. Before running, please make sure to set your working directory to `<your_path>/codes/Babel`. To train the model, use the following command:
```bash
python run_Babel.py --base_dir "<your_path>/outputs/batch variation/208_M2-208_M1/Babel" --dataset_dir "<your_path>/datasets/CITE-SLN-Gayoso"
```
Here, `base_dir` is the folder where all outputs, such as models, logs, and predictions, will be saved, and `dataset_dir` points to the location of your raw dataset files.

