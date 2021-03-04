### Model description:
Uno, is a convolutional network for predicting tumor dose response across multiple data sources:  CCLE, CTRP, gCSI, GDSC, NCI60 single drug response, and ALMANAC drug pair response.

### Setup:
To setup the python environment needed to train and run this model, first make sure you install [conda](https://docs.conda.io/en/latest/) package manager, clone this repository, then create the environment as shown below.

```bash
   conda env create -f environment.yml -n UNO
   conda activate UNO
   ```

To download the processed data needed to train and test the model, and the trained model files, you should create an account first on the Model and Data Clearinghouse [MoDac](modac.cancer.gov). The training and test scripts will prompt you to enter your MoDac credentials.

### Training:
To train the model from scratch, the script [uno_baseline_keras2.py](uno_baseline_keras2.py) does the following:
* Reads the model configuration parameters from [uno_default_model.txt](uno_default_model.txt)
* Downloads the training data and splits it to training/validation sets
* Creates and trains the keras model
* Saves the best trained model based on the validation accuracy

Uno can be trained with a subset of dose response data sources. Here is an example of training with all 6 sources: CCLE, CTRP, gCSI, GDSC, NCI60 single drug response, ALMANAC drug pair response.

```
python uno_baseline_keras2.py --train_sources all --cache cache/all --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True --cp True --batch_size 256 --timeout -1 
Using TensorFlow backend.
Params: {'train_sources': ['all'], 'test_sources': ['train'], 'cell_types': None, 'cell_features': ['rnaseq'], 'drug_features': ['descriptors', 'fingerprints'], 'dense': [1000, 1000, 1000], 'dense_feature_layers': [1000, 1000, 1000], 'activation': 'relu', 'loss': 'mse', 'optimizer': 'adam', 'scaling': 'std', 'dropout': 0, 'epochs': 10, 'batch_size': 256, 'val_split': 0.2, 'cv': 1, 'max_val_loss': 1.0, 'learning_rate': None, 'base_lr': None, 'residual': False, 'reduce_lr': False, 'warmup_lr': False, 'batch_normalization': False, 'feature_subsample': 0, 'rng_seed': 2018, 'save_path': 'save/uno', 'no_gen': False, 'verbose': None, 'gpus': [0], 'timeout': -1, 'logfile': None, 'train_bool': True, 'experiment_id': 'EXP000', 'run_id': 'RUN000', 'shuffle': False, 'profiling': False, 'agg_dose': None, 'by_cell': None, 'by_drug': None, 'cell_subset_path': '', 'drug_subset_path': '', 'drug_median_response_min': -1, 'drug_median_response_max': 1, 'no_feature_source': True, 'no_response_source': True, 'dense_cell_feature_layers': None, 'dense_drug_feature_layers': None, 'use_landmark_genes': True, 'use_filtered_genes': False, 'feature_subset_path': '', 'cell_feature_subset_path': '', 'drug_feature_subset_path': '', 'preprocess_rnaseq': 'source_scale', 'es': False, 'cp': True, 'tb': False, 'tb_prefix': 'tb', 'partition_by': None, 'cache': 'cache/all', 'single': False, 'export_csv': None, 'export_data': None, 'use_exported_data': None, 'growth_bins': 0, 'initial_weights': None, 'save_weights': None, 'data_type': <class 'numpy.float32'>, 'output_dir': '/gpfs/gsfs12/users/lup2/NCI-DOE-Collab-Pilot1-Tumor_Classifier/Benchmarks/Pilot1/Uno_1/Output/EXP000/RUN000'}
Cache parameter file does not exist: cache/ALL.params.json
Loading data from scratch ...
Loaded 27769716 single drug dose response measurements
Loaded 3686475 drug pair dose response measurements
Combined dose response data contains sources: ['CCLE' 'CTRP' 'gCSI' 'GDSC' 'NCI60' 'SCL' 'SCLC' 'ALMANAC.FG'
 'ALMANAC.FF' 'ALMANAC.1A']
Summary of combined dose response by source:
              Growth  Sample  Drug1  Drug2
Source                                    
ALMANAC.1A    208605      60    102    102
ALMANAC.FF   2062098      60     92     71
ALMANAC.FG   1415772      60    100     29
CCLE           93251     504     24      0
CTRP         6171005     887    544      0
GDSC         1894212    1075    249      0
NCI60       18862308      59  52671      0
SCL           301336      65    445      0
SCLC          389510      70    526      0
gCSI           58094     409     16      0
Combined raw dose response data has 3070 unique samples and 53520 unique drugs
Limiting drugs to those with response min <= 1, max >= -1, span >= 0, median_min <= -1, median_max >= 1 ...
Selected 47005 drugs from 53520
Loaded combined RNAseq data: (15198, 943)
Loaded combined dragon7 drug descriptors: (53507, 5271)
Loaded combined dragon7 drug fingerprints: (53507, 2049)
Filtering drug response data...
  2375 molecular samples with feature and response data
  46837 selected drugs with feature and response data
Summary of filtered dose response by source:
        Growth  Sample  Drug1  Drug2
Source                              
CCLE     80213     474     22      0
gCSI     50822     357     16      0
Grouped response data by drug_pair: 32 groups
Input features shapes:
  dose1: (1,)
  dose2: (1,)
  cell.rnaseq: (942,)
  drug1.descriptors: (5270,)
  drug1.fingerprints: (2048,)
  drug2.descriptors: (5270,)
  drug2.fingerprints: (2048,)
Total input dimensions: 15580
Saved data to cache: cache/all.pkl
Combined model:
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input.cell.rnaseq (InputLayer)  (None, 942)          0                                            
__________________________________________________________________________________________________
input.drug1.descriptors (InputL (None, 5270)         0                                            
__________________________________________________________________________________________________
input.drug1.fingerprints (Input (None, 2048)         0                                            
__________________________________________________________________________________________________
input.drug2.descriptors (InputL (None, 5270)         0                                            
__________________________________________________________________________________________________
input.drug2.fingerprints (Input (None, 2048)         0                                            
__________________________________________________________________________________________________
input.dose1 (InputLayer)        (None, 1)            0                                            
__________________________________________________________________________________________________
input.dose2 (InputLayer)        (None, 1)            0                                            
__________________________________________________________________________________________________
cell.rnaseq (Model)             (None, 1000)         2945000     input.cell.rnaseq[0][0]          
__________________________________________________________________________________________________
drug.descriptors (Model)        (None, 1000)         7273000     input.drug1.descriptors[0][0]    
                                                                 input.drug2.descriptors[0][0]    
__________________________________________________________________________________________________
drug.fingerprints (Model)       (None, 1000)         4051000     input.drug1.fingerprints[0][0]   
                                                                 input.drug2.fingerprints[0][0]   
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 5002)         0           input.dose1[0][0]                
                                                                 input.dose2[0][0]                
                                                                 cell.rnaseq[1][0]                
                                                                 drug.descriptors[1][0]           
                                                                 drug.fingerprints[1][0]          
                                                                 drug.descriptors[2][0]           
                                                                 drug.fingerprints[2][0]          
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 1000)         5003000     concatenate_1[0][0]              
__________________________________________________________________________________________________
dense_11 (Dense)                (None, 1000)         1001000     dense_10[0][0]                   
__________________________________________________________________________________________________
dense_12 (Dense)                (None, 1000)         1001000     dense_11[0][0]                   
__________________________________________________________________________________________________
dense_13 (Dense)                (None, 1)            1001        dense_12[0][0]                   
==================================================================================================
Total params: 21,275,001
Trainable params: 21,275,001
Non-trainable params: 0
__________________________________________________________________________________________________
Between random pairs in y_val:
  mse: 0.6071
  mae: 0.5459
  r2: -1.0006
  corr: -0.0003
Data points per epoch: train = 20158208, val = 5144576, test = 0
Steps per epoch: train = 78743, val = 20096, test = 0
Epoch 1/10
78743/78743 [==============================] - 28572s 363ms/step - loss: 0.1730 - mae: 0.2771 - r2: -0.4261 - val_loss: 0.1611 - val_mae: 0.2615 - val_r2: -0.0986
Current time ....28572.014
Epoch 2/10
78743/78743 [==============================] - 28343s 360ms/step - loss: 0.1313 - mae: 0.2390 - r2: -0.0332 - val_loss: 0.1508 - val_mae: 0.2539 - val_r2: -0.1457
Current time ....56916.216
Epoch 3/10
78743/78743 [==============================] - 28214s 358ms/step - loss: 0.1211 - mae: 0.2278 - r2: 0.0822 - val_loss: 0.1540 - val_mae: 0.2491 - val_r2: -0.0944
Current time ....85130.823
Epoch 4/10
78743/78743 [==============================] - 28181s 358ms/step - loss: 0.1132 - mae: 0.2195 - r2: 0.1719 - val_loss: 0.1467 - val_mae: 0.2488 - val_r2: -0.0814
Current time ....113311.712
Epoch 5/10
78743/78743 [==============================] - 28177s 358ms/step - loss: 0.1085 - mae: 0.2139 - r2: 0.2240 - val_loss: 0.1432 - val_mae: 0.2416 - val_r2: 0.0158
Current time ....141493.836
Epoch 6/10
78743/78743 [==============================] - 28165s 358ms/step - loss: 0.1053 - mae: 0.2095 - r2: 0.2596 - val_loss: 0.1449 - val_mae: 0.2389 - val_r2: 0.0597
Current time ....169658.979
Epoch 7/10
78743/78743 [==============================] - 28214s 358ms/step - loss: 0.1018 - mae: 0.2070 - r2: 0.2963 - val_loss: 0.1479 - val_mae: 0.2474 - val_r2: -0.2203
Current time ....197873.198
Epoch 8/10
78743/78743 [==============================] - 28382s 360ms/step - loss: 0.0991 - mae: 0.2041 - r2: 0.3220 - val_loss: 0.1421 - val_mae: 0.2346 - val_r2: 0.0360
Current time ....226255.583
Epoch 9/10
78743/78743 [==============================] - 28360s 360ms/step - loss: 0.0961 - mae: 0.2007 - r2: 0.3488 - val_loss: 0.1470 - val_mae: 0.2384 - val_r2: 0.0094
Current time ....254616.103
Epoch 10/10
78743/78743 [==============================] - 28093s 357ms/step - loss: 0.0951 - mae: 0.1991 - r2: 0.3574 - val_loss: 0.1412 - val_mae: 0.2385 - val_r2: 0.0136
Comparing y_true and y_pred:
  mse: 0.1412
  mae: 0.2385
  r2: 0.5346
  corr: 0.7323
```

Training Uno on all data sources is slow. The `--train_sources` parameter can be used to test the code with a smaller set of training data. An example command line is the following.
```
uno_baseline_keras2.py --train_sources CCLE --cache cache/CCLE --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True
```

A faster example is given in the `uno_by_drug_example.txt` configuration file. This example focuses on a single drug (paclitaxel) and trains at 15s/epoch on a single P100.
```
uno_baseline_keras2.py --config_file uno_by_drug_example.txt
```

### Inference: 
To test the trained model in inference, the script [uno_infer2.py](uno_infer2.py) does the following:
* Downloads the trained model
* Downloads the processed test dataset with the corresponding labels
* Performs inference on the test dataset
* Reports the accuracy of the model on the test dataset

```
python uno_baseline_keras2.py --train_sources CCLE --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True --test_sources gCSI  
...
Testing on data from gCSI (31648)
  mse: 0.1656
  mae: 0.2882
  r2: 0.6206
  corr: 0.8022
```
