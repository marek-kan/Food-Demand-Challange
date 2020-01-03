import pandas as pd
import xgboost
import pickle
from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import r2_score, mean_squared_log_error

train = pd.read_csv(r'data/train_meal_scale1.csv')
#train['price_diff'] = train.checkout_price - train.base_price

dummies = pd.get_dummies(train.center_id)
train = pd.concat([train, dummies], axis = 1)
res = {}

for meal in train.meal_id.unique():
    df = train[(train.meal_id==meal)]
    df.drop(['meal_id', 'center_id', 'week'],axis=1,inplace=True)

    x = df.drop(['num_orders'], axis=1)
    features = list(x.columns)
    
    y = df.num_orders
    
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    
    xgb = xgboost.XGBRegressor(
            eta = 0.3,
            min_child_weight=3,
            max_depth=4,
            reg_lambda=5,
            n_jobs=-1)
    xgb.fit(x, y)
    
   
    temp = {f'{meal}': 
                {'model': xgb,
                 'scaler': scaler,
                 'features': features}
            }
    res.update(temp)

pickle.dump(res, open(r'models/regressor/final_common.pkl', 'wb'))