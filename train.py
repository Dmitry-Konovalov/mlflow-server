from io import BytesIO 
import os 

import pandas as pd
import boto3
from dotenv import load_dotenv

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split


def read_df_from_s3(bucket='datasets', file_name='winequality-red.csv'):
    s3 = boto3.resource(
        's3',
        endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL') 
    )

    obj = s3.Object(bucket, file_name)

    with BytesIO(obj.get()['Body'].read()) as bio:
        df = pd.read_csv(bio)

    return df 

def train(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['quality'], axis=1),
                                                    df['quality'], test_size=0.30,
                                                    random_state=42)
        
    model_1 = RandomForestClassifier().fit(X_train, y_train)
    model_1_pred = model_1.predict(X_test)

    balanced_accuracy = balanced_accuracy_score(y_test, model_1_pred)
    recall = recall_score(y_test, model_1_pred, average='weighted')
    precision = precision_score(y_test, model_1_pred, average='weighted')
    f1 = f1_score(y_test, model_1_pred, average='weighted')
    f2 = r2_score(y_test, model_1_pred)
    print('Balanced accuracy: ', balanced_accuracy)
    print('Recall: ', recall)
    print('Precision: ', precision)
    print('F1-score: ', f1)
    print("r2:", f2)  

if __name__ == '__main__':
    load_dotenv()

    df = read_df_from_s3()
    print(df.head())

    train(df)