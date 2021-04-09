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
The following components are in the [Unified Drug Response Predictor] dataset in the Model and Data Clearinghouse (MoDaC): &#x1F534;_**(Link TBD)**_
* Untrained model: 
  * The untrained neural network model is defined in uno.model.json. &#x1F534;_**(Question: In the README.txt file, this filename is identified as "model topology file". Would it help the user if we use consistent terminology for this file across GitHub and MoDaC? Such as "topology file for the untrained neural network model"?)**_ The name "Model topology file" may be okay, becuause topolopy can mean "untrained" and "Model topology file" can be used for non neural network model in future if need be.
* Data:
  * The processed training and test data in MoDaC. &#x1F534;_**(Question: Would it be helpful to specify filenames here, too?)**_ Yes, but there are too many data files (close to 100 files) for this model. That's why we are not going to put data files toghether with model files for Uno, Combo, and P1B3. Instead, we will create a separated collection/asset for those data files.
* Trained model:
  * The trained model is defined by combining the untrained model and model weights.
  * The trained model weights are used in inference uno.model.h5.

### Technical Details
Refer to this [README](./Pilot1/Uno/README.md).
