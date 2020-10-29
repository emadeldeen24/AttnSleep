# AttnSleep
#### AttnSleep: An Attention-based Deep Learning Approach for Sleep Stage Classification with Single-Channel EEG 

The architecture of the model:
![AttnSleep Architecture](imgs/AttnSleep.png)


## Requirmenets:
- Intall jq package (for linux)
- Python3.x
- Pytorch
- Numpy
- Sklearn
- Pandas
- openpyxl
- mne

## Prepare datasets
We used two datasets in this study: 
- [Sleep-EDF datasets](https://physionet.org/content/sleep-edfx/1.0.0/)
- [SHHS dataset](https://sleepdata.org/datasets/shhs)


After downloading the datasets, you can prepare them as follows:
```
cd prepare_datasets
python prepare_physionet.py --data_dir /path/to/PSG/files --output_dir edf_20_npz --select_ch EEG Fpz-Cz
python prepare_shhs.py --data_dir /path/to/EDF/files --ann_dir /path/to/Annotation/files --output_dir shhs_npz --select_ch EEG C4-A1
```

## Training AttnSleep 
For updating the training parameters, you have to update the `config.json` file.
In this file, you can update:
- The experiment name (Recommended to update this for different experiments)
- The number of GPUs.
- Batch size
- Number of folds (as we use K-fold cross validation)
- Optimizer type along with its parameters.
- the loss function. (to update this you have to include the new loss function in the [loss.py](./model/loss.py) file).
- the metrics you want to see while training (also to add more metrics, update the [metrics.py](./model/metric.py) file).
- The number of epochs
- The save directory (where the results of experiment will be saved)
- The save_period which show the interval of saving the checkpoints and best model.
- verbosity of log (for less logs use 0, for all logs use 2, 1 in between)


To perform the standard K-fold cross validation, specify the number of folds in `config.json` and run the following:
```
chmod +x batch_train.sh
./batch_train.sh 0 /path/to/npz/files
```
where the first argument represents the GPU id (If you want to use CPU, pass an ID > NUM_OF_GPUS, so if no GPUs, use 1 for example, if you have 2 GPUs and want to use CPU, pass 3 for example)

If you want to train only one specific fold, use this command:
```
python train_Kfold_CV.py --device 0 --fold_id 10 --np_data_dir /path/to/npz/files
```
## Results
The log file of each fold is found in the fold directory inside the save_dir.   
The final classification report is found the directory of the last fold (for example if K=5, you will find it inside fold_4 directory), because it sums up all the previous folds results to calculate the metrics.

