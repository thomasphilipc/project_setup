from flask import Flask,request,render_template,redirect,url_for
import numpy as np
import pandas as pd
import os
from werkzeug.utils import secure_filename
import joblib
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline
import requests
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.logger import logging
from src.exception import CustomException
from src.logger import logging
import sys

def load_csv(file_path=None, file_url=None):
    if file_path:
        return pd.read_csv(file_path)
    elif file_url:
        response = requests.get(file_url)
        return pd.read_csv(StringIO(response.text))
    return None


application=Flask(__name__)

app=application

## Route for a home page


# this is the catchall landing page
@app.route('/')
def index():
    return render_template('index.html') 


# this is the url to get precition based on some custom data
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    
@app.route('/train',methods=['GET','POST'])
def get_csv_data():
    return render_template('train.html')



@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    file_url = request.form.get('csvurl')
    #
    upload_data_path: str=os.path.join('uploads')
    #os.makedirs(os.path.dirname(upload_data_path),exist_ok=True)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_data_path, filename)
        file.save(filepath)
        data = load_csv(file_path=filepath)
    elif file_url:
        data = load_csv(file_url=file_url)
    else:
        return "No file or URL provided", 400

    upload_data_path: str=os.path.join('uploads')
    filename='raw_data.csv'
    filepath = os.path.join(upload_data_path, filename)
    # Save data for further processing
    data.to_csv(filepath, index=False)

    logging.info("The uploaded file and a file called  raw_data.csv has been saved and uploaded in uploads folder")

    # Redirect to column selection page
    return redirect(url_for('delete_columns'))

@app.route('/delete_columns')
def delete_columns():
    upload_data_path: str=os.path.join('uploads')
    filename='raw_data.csv'
    filepath = os.path.join(upload_data_path, filename)
    data = pd.read_csv(filepath)
    columns = data.columns.tolist()
    
    logging.info("The columns names are shown to confirm for deletion")

    return render_template('delete_columns.html', columns=columns)

@app.route('/process_columns', methods=['POST'])
def process_columns():
    upload_data_path: str=os.path.join('uploads')
    filename='raw_data.csv'
    filepath = os.path.join(upload_data_path, filename)
    data = pd.read_csv(filepath)
    drop_columns = request.form.getlist('drop_columns')
    #modify_columns = request.form.getlist('modify_columns')
    #new_columns = request.form.get('new_columns').split('\n')

    # Drop columns
    data.drop(columns=drop_columns, inplace=True)

    upload_data_path: str=os.path.join('uploads')
    filename='processed_data.csv'
    filepath = os.path.join(upload_data_path, filename)

    data.to_csv(filepath, index=False,encoding='utf-8')
    columns = data.columns.tolist()

    logging.info("The selected columns are deleted and now the target column needs to be selected")

    # Proceed to data transformation
    return render_template('select_columns.html', columns=columns)

@app.route('/process_target_column', methods=['POST','GET'])
def process_target_column():
    target_column = request.form.getlist('target_column')
    #modify_columns = request.form.getlist('modify_columns')
    #new_columns = request.form.get('new_columns').split('\n')

    # Drop columns
    target_column_name=target_column[0]
  
    logging.info(f"The target column is {target_column_name}")



    train_pipeline=TrainPipeline(target_column_name=target_column_name)
    print("Before training pass the target column")
    try:
        results=train_pipeline.train_model()
        logging.info(f"got a result of {results}")
    except Exception as e:
        raise CustomException(e,sys)
    print("after Prediction")
    return render_template('result.html',results=results)



@app.route('/train_model')
def train_model():
 

    return f"Model Training Complete"



if __name__=="__main__":
    app.run(host="0.0.0.0")        