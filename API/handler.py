import joblib
import pandas as pd

from flask            import Flask, request, Response
from airbnb.Airbnb    import Airbnb

# loading model
model = joblib.load( '../model/full_rf_model_compressed.joblib' )

app = Flask( __name__ )

@app.route( '/airbnb/predict', methods=['POST'] )
 
def airbnb_predict():
    test_json = request.get_json()
    
    if test_json: # there is data
        if isinstance( test_json, dict ): # unique example
            test_raw = pd.DataFrame( test_json, index=[0] )
            
        else: # multiple example
            test_raw = pd.DataFrame( test_json, columns = test_json[0].keys() )
            
        # Instantiate Airbnb class
        pipeline = Airbnb()
        
        # data cleaning
        df1 = pipeline.data_cleaning( test_raw )
        
        # feature engineering
        df3 = pipeline.feature_engineering( df1 )
        
        # data preparation
        df6 = pipeline.data_preparation( df3 )
        
        # get prediction
        df_response = pipeline.get_prediction( model, test_raw, df6 )
        
        return df_response
    
    else:
        return Response( '{}', status=200, mimetype='application/json' )

if __name__ == '__main__':
    app.run( debug=True )
