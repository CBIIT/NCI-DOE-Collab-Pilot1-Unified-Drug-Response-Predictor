# NCI-DOE-Collab-Pilot1-Unified-Drug-Response-Predictor

### Description:
The Unified Drug Response Predictor (UNO) shows how to train and use a neural network model to predict tumor dose response across multiple data sources.

### User Community:	
Primary: Cancer biology data modeling</br>
Secondary: Machine Learning; Bioinformatics; Computational Biology

### Usability:	
To use the untrained model, users have to be familiar with processing and feature extraction of molecular drug data, gene expression, and training of neural networks. The input to the model is a preprocessed data. Users should have extended experience with preprocessing this data. There is not information how the processed data is generated from original public sources.

### Uniqueness:	
Using neural network to predict drug response can be carried using multiple machine learning techniques. The general rule is that classical methods like random forests would perform better for small size datasets, while neural network approaches like UNO would perform better for relatively larger size data. The baseline for comparison can be: mean response, linear regression or random forest regression.

### Components:	

Untrained model: 
* Untrained neural network model is defined in [uno.model.json](https://modac.cancer.gov/searchTab?dme_data_id=).

Data:
* Processed training and test data in [MoDaC](https://modac.cancer.gov/searchTab?dme_data_id=).

Trained Model:
* Trained model is defined by combining the untrained model + model weights.
* Trained model weights are used in inference [uno.model.h5](https://modac.cancer.gov/searchTab?dme_data_id=)

### Technical Details:
Please refer to this [README](./Pilot1/Uno/README.md)
