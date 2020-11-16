# GenAutoNN_Reserve_Forecast
Predicting the reserve requirement (Total Demand - (Solar Supply + Wind Supply))MWh for a energy supply system enables us to have integration of renewable energy sources. By reducing predictive a more accurate deficit of energy we can rely more on renewables. We aim to explore and experiment with model based on deep state space models by Syama Sundar Rangapuram et al. 

## Requirements
We use [GluonTS](https://ts.gluon.ai/) based on MXNet software package to implement our baseline work. To ensure you have the correct version use
```
pip install -r requirements.txt
```
 
## File Structures
1. data - has the data files in .csv format that we used for the project. It also has dataloaders that ingest the data into the model.
2. models - has the models that are used in experimentation.
3. notebooks - has the notebooks where we visualize the raw data.
4. scripts - has all the files that can be useful to scrape and download the data from the CAISO, EIA webpage.
5. utils - has configuration and parser files.
6. saved_models - has trained models

## Usage 
### Download and Pre-process the data 
We found two main sources for the energy requirements and supply data for California. Here we have approach to download data from both EIA and CAISO. Use data_prep_caiso.ipynb to prepare the data from CAISO and use data_prep_eia.ipynb to download the data from EIA. 

### Training the Model 
The model can be trained either by running the code in train_dssm.ipynb or train.py module. To run the code from command line use 
```
python train.py num_epochs 100 
```
The model parameters will be saved in saved_model

### Evaluating the Model
To run evaluations use test.ipynb or test.py. This generates the prediction in the based on the trained model and also gives evaluation results for evaluation metrics like MSE.
