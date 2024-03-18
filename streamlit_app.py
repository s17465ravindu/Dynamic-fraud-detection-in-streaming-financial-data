import streamlit as st
import json
import numpy as np
from kafka import KafkaConsumer
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy import Table, Column, Integer, Float, String, MetaData
from Src.py_files import explainable_ai
import streamlit.components.v1 as components
import base64

st.set_page_config(layout="wide" , initial_sidebar_state='expanded')
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    
    return base64.b64encode(data).decode()


model = keras.models.load_model(r'D:\Uni Docs\DSC4996\Dynamic_fraud_detection_system\CNN_output\best_model_us_data.h5')
scaler = StandardScaler()

features_to_extract = ['Time','V1', 'V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']

engine = create_engine('postgresql://postgres:1234@localhost:6000/dsc4996')

meta = MetaData()

transactions = Table(
   'transactions', meta, 
    Column('id', Integer, primary_key = True), 
    Column('Time', Float), 
    Column('V1', Float),
    Column('V2', Float),
    Column('V3', Float),
    Column('V4', Float),
    Column('V5', Float),
    Column('V6', Float),
    Column('V7', Float),
    Column('V8', Float),
    Column('V9', Float),
    Column('V10', Float),
    Column('V11', Float),
    Column('V12', Float),
    Column('V13', Float),
    Column('V14', Float),
    Column('V15', Float),
    Column('V16', Float),
    Column('V17', Float),
    Column('V18', Float),
    Column('V19', Float),
    Column('V20', Float),
    Column('V21', Float),
    Column('V22', Float),
    Column('V23', Float),
    Column('V24', Float),
    Column('V25', Float),
    Column('V26', Float),
    Column('V27', Float),
    Column('V28', Float),
    Column('Amount', Float),
    Column('prediction', String),
)

meta.create_all(engine)

consumer = KafkaConsumer(
    'quick-start',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda x: x.decode('utf-8')
)

exp_trans = {}
transaction_counter = 0

all_transactions = pd.DataFrame()

st.header('Real-Time Transaction Dashboard',divider='rainbow')

c1,c2 = st.columns((7,3))
c2.markdown('### Explanation')
with c1:
    st.markdown('### Transaction Details')
    table_placeholder = st.empty()

    try:
        for transaction in consumer:
            transaction_list = json.loads(transaction.value)
            for transaction_dict in transaction_list:

                transaction_data = [float(transaction_dict[feature]) for feature in features_to_extract]

                transaction_data = np.array(transaction_data).reshape(1, len(features_to_extract),1)
                
                df = pd.DataFrame([transaction_dict])

                prediction = model.predict(transaction_data)

                if prediction > 0.5:
                    df['prediction'] = 'Fraud'
                else:
                    df['prediction'] = 'Legit'

                df.to_sql('transactions', engine, if_exists='append', index=False)

                df = df[['V4','V7', 'V12', 'V14','V15', 'V17', 'Amount', 'prediction']]
                all_transactions = pd.concat([all_transactions, df], ignore_index=True)

                # Display only the last 10 transactions
                all_transactions = all_transactions.tail(10)

                # Apply color to 'prediction' column
                df_styled = all_transactions.style.apply(lambda x: ["background: red" if v == "Fraud" else "background: green" for v in x], subset=['prediction'])


                # Display the actual table
                table_placeholder.table(df_styled)

                if df['prediction'].values[0] == 'Fraud':
                    with c2:
                        c2.markdown(f'Transaction Id: {transaction_counter}')
                        explainable_ai.get_explanation(transaction_data)
                    
                transaction_counter += 1


    except Exception as e:
        print(f"Error: {e}")

    finally:
        consumer.close()



