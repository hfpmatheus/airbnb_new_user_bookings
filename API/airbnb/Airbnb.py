import pickle
import pandas as pd
import numpy  as np

class Airbnb( object ):
    def __init__( self ):

        state = 1
        
    def data_cleaning( self, df1 ):
        
        # 1.0. DATA DESCRIPTION

        df1 = df1.drop( columns=['date_first_booking'] )

        ### 1.3.1. NA Fulfill

        # age - median age
        df_aux = df1[ (df1['age'] > 18) & (df1['age'] < 100) ]
        df1['age'] = df1['age'].fillna( df_aux['age'].mean() )

        # first affiliate tracked - drop
        #df1 = df1.dropna()

        ### 1.4.1. Change types

        # date account created - datetime
        df1['date_account_created'] = pd.to_datetime( df1['date_account_created'] )

        # timestamp first active - datetime
        df1['timestamp_first_active'] = pd.to_datetime( df1['timestamp_first_active'], format='%Y%m%d%H%M%S' )

        # age - int
        df1['age'] = df1['age'].astype( 'int64' )
        
        return df1
    
    def feature_engineering( self, df3 ):
        
        # days timestamp first active to date account created
        df3['date_first_active'] = pd.to_datetime( df3['timestamp_first_active'].dt.strftime('%Y-%m-%d') )
        df3['timestamp_first_active_to_date_account_created'] = df3['date_account_created'] - df3['date_first_active']
        df3['timestamp_first_active_to_date_account_created'] = df3['timestamp_first_active_to_date_account_created'].apply( lambda x: x.days )

        # # ================ first active ===================

        # year
        df3['year_date_first_active'] = df3['date_first_active'].dt.year

        # month
        df3['month_date_first_active'] = df3['date_first_active'].dt.month

        # week
        df3['week_date_first_active'] = df3['date_first_active'].dt.week

        # day
        df3['day_date_first_active'] = df3['date_first_active'].dt.day

        # day_of_week
        df3['day_of_week_date_first_active'] = df3['date_first_active'].dt.dayofweek

        # ============ date account created ===============

        # year
        df3['year_date_account_created'] = df3['date_account_created'].dt.year

        # month
        df3['month_date_account_created'] = df3['date_account_created'].dt.month

        # week
        df3['week_date_account_created'] = df3['date_account_created'].dt.week

        # day
        df3['day_date_account_created'] = df3['date_account_created'].dt.day

        # day_of_week
        df3['day_of_week_date_account_created'] = df3['date_account_created'].dt.dayofweek
        
        return df3
    
    def data_preparation( self, df6 ):
        
        # 6.0. DATA PREPARATION
        ## 6.2. Encoding
        # gender  
        df6 = pd.get_dummies( columns=['gender'], data=df6)

        # signup_method   
        df6 = pd.get_dummies( columns=['signup_method'], data=df6 )

        # language                
        df6 = df6.drop( columns=['language'] )

        # affiliate_provider     
        df6 = df6.drop( columns=['affiliate_provider'] )

        # signup_app  
        df6 = df6.drop( columns=['signup_app'] )

        # first_device_type 
        df6 = df6.drop( columns=['first_device_type'] )

        # first_browser
        df6 = df6.drop( columns=['first_browser'] )

        ## 6.3. Transformation

        # month_date_account_created
        df6['month_date_account_created_sin'] = df6['month_date_account_created'].apply( lambda x: np.sin( x * (2*np.pi/12 ) ) )
        df6['month_date_account_created_cos'] = df6['month_date_account_created'].apply( lambda x: np.cos( x * (2*np.pi/12 ) ) )

        # week_date_account_created
        df6['week_date_account_created_sin'] = df6['week_date_account_created'].apply( lambda x: np.sin( x * (2*np.pi/52 ) ) )
        df6['week_date_account_created_cos'] = df6['week_date_account_created'].apply( lambda x: np.cos( x * (2*np.pi/52 ) ) )

        # day_date_account_created
        df6['day_date_account_created_sin'] = df6['day_date_account_created'].apply( lambda x: np.sin( x * (2*np.pi/30 ) ) )
        df6['day_date_account_created_cos'] = df6['day_date_account_created'].apply( lambda x: np.cos( x * (2*np.pi/30 ) ) )

        # day_of_week_date_account_created
        df6['day_of_week_date_account_created_sin'] = df6['day_of_week_date_account_created'].apply( lambda x: np.sin( x * (2*np.pi/7 ) ) )
        df6['day_of_week_date_account_created_cos'] = df6['day_of_week_date_account_created'].apply( lambda x: np.cos( x * (2*np.pi/7 ) ) )

        # month_date_first_active
        df6['month_date_first_active_sin'] = df6['month_date_first_active'].apply( lambda x: np.sin( x * (2*np.pi/12 ) ) )
        df6['month_date_first_active_cos'] = df6['month_date_first_active'].apply( lambda x: np.cos( x * (2*np.pi/12 ) ) )

        # week_date_first_active
        df6['week_date_first_active_sin'] = df6['week_date_first_active'].apply( lambda x: np.sin( x * (2*np.pi/52 ) ) )
        df6['week_date_first_active_cos'] = df6['week_date_first_active'].apply( lambda x: np.cos( x * (2*np.pi/52 ) ) )

        # day_date_first_active
        df6['day_date_first_active_sin'] = df6['day_date_first_active'].apply( lambda x: np.sin( x * (2*np.pi/30 ) ) )
        df6['day_date_first_active_cos'] = df6['day_date_first_active'].apply( lambda x: np.cos( x * (2*np.pi/30 ) ) )

        # day_of_week_date_first_active
        df6['day_of_week_date_first_active_sin'] = df6['day_of_week_date_first_active'].apply( lambda x: np.sin( x * (2*np.pi/7 ) ) )
        df6['day_of_week_date_first_active_cos'] = df6['day_of_week_date_first_active'].apply( lambda x: np.cos( x * (2*np.pi/7 ) ) )

        cols_selected = [ 
            'age',
            'signup_flow',
            'month_date_first_active',
            'week_date_first_active',
            'day_date_first_active',
            'day_of_week_date_first_active',
            'year_date_account_created',
            'week_date_account_created',
            'day_date_account_created',
            'day_of_week_date_account_created',
            'gender_-unknown-',
            'gender_FEMALE',
            'gender_MALE',
            'signup_method_basic',
            'signup_method_facebook',
            'week_date_account_created_sin',
            'week_date_account_created_cos',
            'day_date_account_created_sin',
            'day_date_account_created_cos',
            'day_of_week_date_account_created_sin',
            'week_date_first_active_sin',
            'week_date_first_active_cos',
            'day_date_first_active_sin',
            'day_date_first_active_cos',
            'day_of_week_date_first_active_sin',
            'day_of_week_date_first_active_cos' ]


        return df6[ cols_selected ]
    
    def get_prediction( self, model, original_data, prepared_data ):
        
        # model prediction
        pred = model.predict( prepared_data )
        
        # join prediction to original data
        original_data.dropna( axis=0, subset=['first_affiliate_tracked'] )
        original_data['prediction'] = pred
        
        return original_data.to_json( orient='records', date_format='iso' )
