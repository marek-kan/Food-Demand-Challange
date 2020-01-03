import pandas as pd
import numpy as np
import gc
import tensorflow as tf
from . import predict

test = pd.read_csv(r'data/test_synth.csv')
test.drop(['price_ratio','price_dummy'],axis=1, inplace=True)

train = pd.read_csv(r'data/train_scale0_synth.csv')
train.drop(['price_ratio','price_dummy'],axis=1, inplace=True)

test['shift'] = 0
test['orders_cum2'] = 0
test['num_orders'] = 0

sub = pd.read_csv(r'data/sample_submission.csv')
sub.drop(['num_orders'],axis=1, inplace=True)

reg_model = pd.read_pickle(r'models/regressor/reg_wo_synth.pkl')
comm_model = pd.read_pickle(r'models/regressor/common_reg_wo_synth.pkl')
nn_scalers = pd.read_pickle(r'models/nn/scalers.pkl')

callback = 0
errors = []
i = 0

for center in test.center_id.unique():
    progress = i/77
    print('*******************************'+'\n'+
          f'{center}'+'\n'+
          '*******************************'+'\n'+
          f'progress = {progress}%')
    try:
        df = predict(train, test, center, reg_model, comm_model, nn_scalers)
        test['num_orders'] = test['num_orders'].add(df['prediction'], fill_value=0)
        
        del(df)
        gc.collect()
        
    except Exception as e:
        errors.append(f'error, center: {center}, {e}')
        
    i += 1
    gc.collect()
    
sub = pd.merge(sub, test, on='id')
sub = sub[['id','num_orders']]
sub.to_csv(r'submissions/submission.csv', index=False)

log = pd.DataFrame(errors)
log.to_csv('pred_log.txt')