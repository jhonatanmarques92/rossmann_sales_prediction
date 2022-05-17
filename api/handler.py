from flask import Flask, request, Response
import pandas as pd
from rossmann.Rossmann import Rossmann
import joblib

model = joblib.load('/mnt/c/users/jhonatan/desktop/comunidade_ds/repos/ds_em_producao/rossmann_sales_prediction/modelo/xgb_model_tunned.joblib') 

app = Flask(__name__)

@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
    test_json = request.get_json()
    
    if test_json: # Caso tenha dados
        if isinstance(test_json, dict): # Dados únicos
            test_raw = pd.DataFrame(test_json, index=[0])
        else: # Multiplos dados
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
    else:
        return Response('{}', status=200, mimetype='application/json')

    # Instanciar Rossmann
    pipeline = Rossmann()

    # Data cleaning
    df1 = pipeline.data_cleaning(test_raw)
    
    # Data engineering
    df2 = pipeline.feature_engineering(df1)
    
    # Data preparation
    df3 = pipeline.data_preparation(df2)
    
    # Prediction
    data_prediciton = pipeline.get_prediction(model, test_raw, df3)

    return data_prediciton

if __name__ == '__main__':
    app.run(debug=True)