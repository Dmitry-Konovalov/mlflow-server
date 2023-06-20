from io import BytesIO 
import os 

import pandas as pd
import boto3
from dotenv import load_dotenv

def read_df_from_s3(bucket='datasets', file_name='winequality-red.csv'):
    s3 = boto3.resource(
        's3',
        endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL') 
    )

    obj = s3.Object(bucket, file_name)

    with BytesIO(obj.get()['Body'].read()) as bio:
        df = pd.read_csv(bio)

    return df 


if __name__ == '__main__':
    load_dotenv()
    
    df = read_df_from_s3()
    print(df.head())
