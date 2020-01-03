import pandas as pd
import numpy as np

train = pd.read_csv(r'data/train.csv')
c_info = pd.read_csv(r'data/center_info.csv')
m_info = pd.read_csv(r'data/meal_info.csv')

train['shift'] = 0.0
train['orders_cum2'] = 0.0

# create shift and cummulative mean for every meal
for meal in train.meal_id.unique():
    df = train[train.meal_id==meal]
    for center in train.center_id.unique():
        print(meal)
        df2 = df[df.center_id==center]
        shft = df2['num_orders'] - df2['num_orders'].shift() + np.random.normal(scale=1, size=df2['num_orders'].shape)
        cum_mean2 = df2['num_orders'].expanding(2).mean() + np.random.normal(scale=1, size=df2['num_orders'].shape)
        df2['shift'] = shft.fillna(0) 
        df2['orders_cum2'] = cum_mean2.fillna(0)
        train['shift'] = train['shift'].add(df2['shift'].shift().fillna(0), fill_value=0)
        train['orders_cum2'] = train['orders_cum2'].add(df2['orders_cum2'].shift().fillna(0), fill_value=0)
        
ratio = train.checkout_price / train.base_price
train['price_ratio'] = ratio

train['price_dummy'] = 0
train.loc[train.checkout_price < train.base_price, 'price_dummy'] = 1
        
#test = pd.read_csv(r'data/test.csv')
#ratiot = test.checkout_price / test.base_price
#test['price_ratio'] = ratiot
#test['price_dummy'] = 0
#test.loc[test.checkout_price < test.base_price, 'price_dummy'] = 1      
        
        
        
        
        
        
        
        
        
        
        
        
        
        