# Multi Forecasting
Neuroplasticity-inspired method for multi-tasking in demand forecasting scenarios on vast datasets.

## Initial info

This is a code repository for the paper **Dynamic ANNs for multi-task demand forecasting**. 

We tested our methods on datasets:

- [Demand-Forecasting](https://www.kaggle.com/datasets/aswathrao/demand-forecasting)
- [Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only)
- [Product Demand Forecast](https://www.kaggle.com/code/flavioakplaka/product-demand-forecast)

and are happy with the results. Below, you'll find information on how to use our methods out-of-the-box with **Demand-Forecasting** as well as how to set up your own experiments on the data you would like to try.

---

## Requirements

We prepared the code repository using **Python 3.10** and any higher version should also work just fine. You can install all the requirements via `env.yml` for conda environment or `reqs.txt` if you'd like to use pip. If you'd hoverver encounter any trouble setting up your environment, installing theese should get you going and you'll probably be fine with default Python environment:

- lightning==2.5.1 (torch lightning)
- torch==2.7.0
- torchmetrics==1.7.1
- numpy==2.2.6
- scikit-learn==1.6.1
- matplotlib==3.10.3
- tensorboard
- seaborn

## Using the repository

Having the dataset and requirements done, you should be able to run some experiments. Here's how.

### Prepare onehot embeddings for Cat2Vec

Every dataset listed has some categorical variables. We want to milk as much of the information as we can from those variables, and since they come in various forms, first we encode them as one-hot vectors (ordinal encoding would also be fine, after dataloader update).

To do so, use `one_hot_encoding.py` from the root directory. In the file, update the path to your dataset `.csv` file and paths where you'd like to save your embedders. We use `OneHotEncoder` from `sklearn.preprocessing`. Right now we encode only 2 columns of variables, as Demand-Forecasting dataset has only two of them -- you can however add as much as you need, as long as you'll update the ANN model architecture for the experiments. 

### Update the config file

The config files are located in `cfgs/` directory. By default, without running the code with additional parameters, `default.py` will be used. In the config file, you'll probably like to change:

- `DEVICE` - for using CUDA or GPU
- `DATA_PATH` - to point where your dataset is located
- `ONEHOT_EMBEDDERS` - to point where you saved your one-hot embedders
- `LOG_DIR` - where you'd like to save the experiment results
- `LAG` - for number of past timesteps you want to consider for the forecast
- `COLUMNS` - you can add additional columns of the data as your input vector (e.g. in Demand-Forecasting dataset you can select additional `[4, 5, 6, 7]` columns)
- `YEARS` - select year of data for pre-training, meta-phase (where dynamics happen) and model evaluation
- `MODEL` - set the name of the model you want to train (now there is only `MLP_base` in `src/models.py`)
- `EMBEDDING_SIZE` - set the length of Cat2Vec vector for categorical variables embedding
- `EPOCHS_PRE`, `EPOCHS_META` - number of epochs for pre-training and meta-phase
- `LR_PRE`, `LR_META` - learning rates for the phases
- `WEIGHT_DECAY`, `BATCH_SIZE` - weight decay for optimizer and batch size
- `SIM` - set the similarity metric for similar task recognition (can be *RMSE* or *EUC*, in this setup they will work the same)
- `TEST_FN`, `LOSS_FN` - set loss functions for training loss and error measurement (can be *RMSELoss*, *MSE*, *MSLE* or *RMSLE*)
- `SCHEDULER_PRE`, `SCHEDULER_META` - set scheduler parameters for *Reduce on Plateau* scheduler for each of training phases

You can add your own config files in `cfgs/` directory and it will work fine as long as you'll keep the initial file structure.

### Run the experiments

When all previous steps are done, you can simply run the experiments using `python3 main.py` in the main directory. If you prepared your own config file, run the code with `python3 main.py #name_of_your_config#` with just name, no extension as the parameter.

The results will be logged to `log_dir` with a date and hour of the initial run. In the log, you'll find `metadata.yml` file containing the aggregated results of all trained tasks and subfolder for every single task trained -- there will also be graphs showing how the training went, as well as `results.yml` file for single task. In `checkpoints/` directory, best and last models will be saved.

### Check the results

When the training is done, use `results.py` from main directory to check the results of your runs. If more than one experiment was performed, you'll get mean, max and min error of the experiments along with their STDs.

## Cite us

You are encouraged to cite this repository with automatic GitHub citing tool in **About** section.

This code repository is linked to our paper **Dynamic ANNs for multi-task demand forecasting**, and we would be happy if you'd cite it as well. We'll add bibtex info on the citation as soon as the paper is published.
