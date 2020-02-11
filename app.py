# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 00:39:59 2020

@author: sagar
"""

from flask import Flask,render_template,url_for,request, send_file
import pickle

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def encode_df(df):
    ordinal_enc_dict = {}

    from sklearn.preprocessing import OrdinalEncoder

    for col_name in df:
        # Create ordinal encoder for the column
        ordinal_enc_dict[col_name] = OrdinalEncoder()

        # Select the non-null values in the column
        col = df[col_name]
        col_not_null = col[col.notnull()]
        reshaped_vals = col_not_null.values.reshape(-1, 1)

        # Encode the non-null values of the column
        encoded_vals = ordinal_enc_dict[col_name].fit_transform(reshaped_vals)

        # Replace the column with ordinal values
        df.loc[col.notnull(), col_name] = np.squeeze(encoded_vals)
    
    return ordinal_enc_dict

# load the model from disk
filename = 'model.pkl'
clf = pickle.load(open(filename, 'rb'))
imputer = pickle.load(open('imputer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        test = pd.read_csv('Data/test.csv')
        test_df = test.copy()
        test.drop(["S.No", "change_request","problem_ID" ], axis=1, inplace=True)
        test.drop(["ID", "ID_caller", "opened_by", 'opened_time', "Created_by", "created_at", "updated_by", "updated_at"], axis=1, inplace=True)
        
        imp_test = imputer.fit_transform(test)
        imp_test = pd.DataFrame(imp_test, columns = test.columns)
        
        encode_df(imp_test)
    
        my_prediction = clf.predict(imp_test.values)
        my_prediction = my_prediction.tolist()
        my = my_prediction[0:10]
        data = { 'ID' : test_df.ID, 'Prediction' : my_prediction}
        Pred_df = pd.DataFrame(data= data)
        test_shape = Pred_df.shape
        my = Pred_df['ID'].head(3)
        n = len(my_prediction)
    
        Pred_df.to_csv('downloadFile.csv', index=False)
    
        #return render_template('result.html',prediction = my, col_num = n, na_num = test_shape)

    return send_file('downloadFile.csv',
                     mimetype='text/csv',
                     attachment_filename='downloadFile.csv',
                     as_attachment=True)


if __name__ == '__main__':
	app.run(debug=True)