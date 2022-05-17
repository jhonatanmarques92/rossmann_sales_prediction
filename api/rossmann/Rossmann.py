import pandas as pd
import joblib
import numpy as np
import inflection
import math
import datetime
import calendar

class Rossmann(object):
    def __init__(self):
        self_home_path = '/mnt/c/users/jhonatan/desktop/comunidade_ds/repos/ds_em_producao/rossmann_sales_prediction/'

        self.encoding_competition_distance = joblib.load(self_home_path + 'modelo/rs_competition_distance.joblib')
        self.encoding_competition_time_month = joblib.load(self_home_path + 'modelo/rs_competition_time_month.joblib')
        self.encoding_promo_time_week = joblib.load(self_home_path + 'modelo/mms_promo_time_week.joblib')
        self.encoding_year = joblib.load(self_home_path + 'modelo/mms_year.joblib')
        self.encoding_story_type = joblib.load(self_home_path + 'modelo/le_sotry_type.joblib')
    
    def data_cleaning(self, df1):

        old_name = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo','StateHoliday', 
        'SchoolHoliday', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 
        'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

        snakecase = lambda x: inflection.underscore(x)
        new_name = list(map(snakecase, old_name))

        # Renomear colunas
        df1.columns = new_name
        
        ## 2.2 - Tipo dos dados

        df1['date'] = pd.to_datetime(df1['date'])

        ## 2.4 - Corrigindo os dados nulos

        # competition_distance
        df1['competition_distance'].fillna(200000, inplace=True)

        #competition_open_since_month
        df1['competition_open_since_month'].fillna(df1['date'].dt.month, inplace=True)

        #competition_open_since_year
        df1['competition_open_since_year'].fillna(df1['date'].dt.year, inplace=True)

        #promo2_since_week
        df1['promo2_since_week'].fillna(df1['date'].dt.week, inplace=True)

        #promo2_since_year
        df1['promo2_since_year'].fillna(df1['date'].dt.year, inplace=True)

        #promo_interval
        df1['promo_interval'].fillna(0, inplace=True)

        month_map = dict(enumerate(calendar.month_abbr))

        df1['month_map'] = df1['date'].dt.month.map(month_map)
        
        df1['is_promo'] = df1[['month_map', 'promo_interval']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)

        ## 2.5 - Alterar o tipo dos dados

        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)
        
        return df1
    
    def feature_engineering(self, df2):
    
        # year
        df2['year'] = df2['date'].dt.year

        # month
        df2['month'] = df2['date'].dt.month

        # day
        df2['day'] = df2['date'].dt.day

        # week of year
        df2['week_of_year'] = df2['date'].dt.weekofyear

        # year week
        df2['year_week'] = df2['year'].astype(str) + '-' + df2['week_of_year'].apply(lambda x: '{0:0>2}'.format(x)).astype(str)

        # competition since
        df2['competition_since'] = df2.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'], month=x['competition_open_since_month'], day=1), axis=1)

        # competition_time_month
        df2['competition_time_month'] = ((df2['date'] - df2['competition_since'])/30).apply(lambda x: x.days).astype(int)

        #promo since
        df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)
        df2['promo_since'] = df2['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))

        df2['promo_time_week'] = ((df2['date'] - df2['promo_since'])/7).apply(lambda x: x.days).astype(int)

        #assortment
        df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')

        #state holiday
        df2['state_holiday'] = df2['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')

        ## 4.1 - Filtragem das linhas

        df2 = df2[df2['open'] != 0]

        ## 4.2 - Seleção das colunas

        df2.drop(['open', 'promo_interval', 'month_map'], axis=1, inplace=True)
        
        return df2
    
    def data_preparation(self, df5):

        # competition distance
        df5['competition_distance'] = self.encoding_competition_distance.transform(df5[['competition_distance']].values)

        # competition time month
        df5['competition_time_month'] = self.encoding_competition_time_month.transform(df5[['competition_time_month']].values)
        
        # promo time week
        df5['promo_time_week'] = self.encoding_promo_time_week.transform(df5[['promo_time_week']].values)
        
        # year
        df5['year'] = self.encoding_year.transform(df5[['year']].values)

        ## 6.3 - Transformação

        ### 6.3.1 - Encoding


        # story_type - Label Encoding
        df5['store_type'] = self.encoding_story_type.transform(df5['store_type'])
        
        # assortment - Ordinal Encoding
        assortment = {'basic':1, 'extra':2, 'extended':3}
        df5['assortment'] = df5['assortment'].map(assortment)

        # state_holiday - One Hot Encoding
        df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'])
    
        
        ### 6.3.3 - Transformação de natureza

        # day of week
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x * (2 * np.pi / 7)))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x * (2 * np.pi / 7)))

        # month
        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x * (2 * np.pi / 12)))
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x * (2 * np.pi / 12)))

        # day
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x * (2 * np.pi / 30)))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x * (2 * np.pi / 30)))

        # week of year
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x * (2 * np.pi / 52)))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x * (2 * np.pi / 52)))
        
        cols_selected = ['store','store_type','assortment','competition_distance','competition_open_since_month',
'competition_open_since_year','promo2','promo2_since_week','promo2_since_year','promo','competition_time_month',
'promo_time_week','day_of_week_sin','day_of_week_cos','month_sin','month_cos','day_sin',
'day_cos','week_of_year_sin','week_of_year_cos']
        
        return df5[cols_selected]
    
    def get_prediction(self, model, original_data, test_data):
        # predição
        pred = model.predict(test_data)
        
        # Unir os dados de predição na original
        original_data['prediction'] = pred ** 3
        
        return original_data.to_json(orient='records', date_format='iso')