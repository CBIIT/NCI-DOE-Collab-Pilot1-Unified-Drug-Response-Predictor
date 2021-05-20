# NCI-DOE-Collab-Pilot1-Unified-Drug-Response-Predictor

### Description
The Pilot 1 Unified Drug Response Predictor benchmark, also called Uno, shows how to train and use a neural network model to predict tumor dose response across multiple data sources.

### User Community
Primary: Cancer biology data modeling</br>
Secondary: Machine Learning; Bioinformatics; Computational Biology

### Usability
To use the untrained model, users must be familiar with processing and feature extraction of molecular drug data, gene expression, and training of neural networks. The input to the model is preprocessed data. Users should have extended experience with preprocessing this data. There is not information how the processed data is generated from original public sources.

### Uniqueness
The community can use a neural network and multiple machine learning techniques to predict drug response. The general rule is that classical methods like random forests would perform better for small datasets, while neural network approaches like Uno would perform better for relatively larger datasets. The baseline for comparison can be: mean response, linear regression, or random forest regression.

### Components
The following components are in the Model and Data Clearinghouse (MoDaC):
* The [Unified Drug Response Predictor (Uno)](https://modac.cancer.gov/searchTab?dme_data_id=NCI-DME-MS01-7654821) asset contains the untrained model and trained model:
  * The model topology file is uno.model.json. 
  * The trained model is defined by combining the untrained model and model weights.
  * The trained model weights are used in inference uno.model.h5.
* The [Pilot 1 Cancer Drug Response Prediction Dataset](https://modac.cancer.gov/searchTab?dme_data_id=NCI-DME-MS01-8088592) asset contains the processed training and test data. 

### Technical Details
Refer to this [README](./Pilot1/Uno/README.md).
