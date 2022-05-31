import pandas as pd
import numpy as np
import os
import geoio
from utils import process_ethiopia, process_malawi, process_nigeria, generate_download_locations, download_images, load_country, load_data
from functions import model, feature_extraction, regression, plot

#Processing LSMS dataset and adding nightlight information
BASE_DIR = '..'
NIGHTLIGHTS_DIRS = [os.path.join(BASE_DIR, 'data/nightlights/viirs_2015_00N060W.tif'),
                    os.path.join(BASE_DIR, 'data/nightlights/viirs_2015_75N060W.tif')]

COUNTRIES_DIR = os.path.join(BASE_DIR, 'data', 'countries')
COUNTRIES_DIR = os.path.join(BASE_DIR, 'data', 'countries')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CNN_TRAIN_IMAGE_DIR = os.path.join(BASE_DIR, 'data', 'cnn_images')

df_mw = process_malawi()
df_eth = process_ethiopia()
df_ng = process_nigeria()

tifs = [geoio.GeoImage(ndir) for ndir in NIGHTLIGHTS_DIRS]

#Downloading Images
df_mw = pd.read_csv(os.path.join(COUNTRIES_DIR, 'malawi_2016', 'processed/clusters.csv'))
df_eth = pd.read_csv(os.path.join(COUNTRIES_DIR, 'ethiopia_2015', 'processed/clusters.csv'))
df_ng = pd.read_csv(os.path.join(COUNTRIES_DIR, 'nigeria_2015', 'processed/clusters.csv'))

df_mw_download = generate_download_locations(df_mw)
df_eth_download = generate_download_locations(df_eth)
df_ng_download = generate_download_locations(df_ng)

#Model Training for VGG
vgg_model = model('vgg')
alex_model = model('alex')

trainingLabels, trainingImages, testingLabels, testingImages = load_data()

initial_epochs = 10
vgg_history = vgg_model.fit(trainingImages,
                    epochs=initial_epochs,
                    validation_data=testingImages)

alex_history = alex_model.fit(trainingImages,
                    epochs=initial_epochs,
                    validation_data=testingImages)

#Feature Extraction
feature_extraction('vgg', trainingLabels,vgg_model)
feature_extraction('alex', trainingLabels,alex_model)

#Regression
malawi_data_vgg = load_country('malawi', 'vgg')
nigeria_data_vgg = load_country('nigeria', 'vgg')
ethiopia_data_vgg = load_country('ethiopia', 'vgg')

regression(malawi_data_vgg)
regression(nigeria_data_vgg)
regression(ethiopia_data_vgg)

malawi_data_alex = load_country('malawi', 'alex')
nigeria_data_alex = load_country('nigeria', 'alex')
ethiopia_data_alex = load_country('ethiopia', 'alex')

regression(malawi_data_alex)
regression(nigeria_data_alex)
regression(ethiopia_data_alex)

