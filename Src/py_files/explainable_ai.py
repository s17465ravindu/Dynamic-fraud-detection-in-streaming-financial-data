import pandas as pd
import numpy as np
import tensorflow as tf
from lime.lime_tabular import LimeTabularExplainer
import streamlit.components.v1 as components
import matplotlib.pyplot

def predict_proba(data):
    proba = model3.predict(data)
    return np.hstack((1 - proba, proba))

X_train = pd.read_csv(r'Data\X_train.csv')
model3 = tf.keras.models.load_model(r'CNN_output\best_model_us_data.h5')

explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['Genuine', 'Fraud'], verbose=True, mode='classification')

def get_explanation(sample):
    sample = sample.reshape(-1)
    explanation = explainer.explain_instance(sample, predict_proba, num_features=5)
    #return explanation.show_in_notebook(show_table=True).as_pyplot_figure()
    components.html(explanation.as_html(), height=200,width = 1100, scrolling = False)
    return explanation.as_pyplot_figure()
    #return explanation








