import pandas as pd
import numpy as np
import gc
import tensorflow as tf

def predict(train, test, center, reg_model, comm_model, nn_scalers):
    if 'num_orders' in test.columns.unique():
        df = test[test.center_id == center].drop(['num_orders'], axis=1)
    else:
        df = test[test.center_id == center]
        test['num_orders'] = 0
        
    df.drop(['id', 'center_id'], axis=1, inplace=True)
    
    # I used previosly learned models for tests instead of configs because
    # i cannt pickle DNN into dictonary.    
    meals_learned = reg_model[str(center)]['meals']
    meals_df = df.meal_id.unique()
    missing_meals = list(set(meals_learned).difference(set(meals_df)))
    
    if len(missing_meals)!=0:
        for i in range(len(missing_meals)):
            df.loc[999999999+i] = [0] * len(df.columns.unique())
            df['meal_id'].loc[999999999+i]= missing_meals[i]
        callback = 1
        
    if len(list(set(meals_df).difference(set(meals_learned)))):
        callback = 2
    
    if callback==2:
        
        diff = list(set(meals_df).difference(set(meals_learned)))
        data = pd.DataFrame()
        for d in diff:
            df_train = train[train.meal_id==int(d)]
            x = df[df.meal_id==int(d)]
            x.drop(['week', 'meal_id'], axis=1, inplace=True)
            
#            loop for prediction and creation of shifted and cummulative mean columns            
            for it in range(len(x)):
                if it==0:
                    idx = x.index[it]
                    shft = df_train.num_orders.iloc[-1] - df_train.num_orders.iloc[-2]
                    c_m = sum(df_train.num_orders)/len(df_train)
                    x['shift'].iloc[0] = shft
                    x['orders_cum2'].iloc[0] = c_m
                    scaler = comm_model[str(d)]['scaler']
                    X = scaler.transform(np.asarray(x.iloc[0]).reshape(1, -1))
                    prediction = (abs(comm_model[str(d)]['model'].predict(X))[0])
                    temp = pd.DataFrame({'prediction': prediction, 'index': idx}, index=[it])
                    data = data.append(temp)
                elif it==1:
                    idx = x.index[it]
                    shft = data.prediction.iloc[0] - df_train.num_orders.iloc[-1]
                    c_m = (sum(df_train.num_orders)+sum(data.prediction))/(len(df_train)+it)
                    x['shift'].iloc[1] = shft
                    x['orders_cum2'].iloc[1] = c_m
                    scaler = comm_model[str(d)]['scaler']
                    X = scaler.transform(np.asarray(x.iloc[1]).reshape(1, -1))
                    prediction = (abs(comm_model[str(d)]['model'].predict(X))[0])
                    temp = pd.DataFrame({'prediction': prediction, 'index': idx}, index=[it])
                    data = data.append(temp)
                else:
                    idx = x.index[it]
                    shft = data.prediction.iloc[it-1] - data.prediction.iloc[it-2]
                    c_m = (sum(df_train.num_orders)+sum(data.prediction))/(len(df_train)+it)
                    x['shift'].iloc[it] = shft
                    x['orders_cum2'].iloc[it] = c_m
                    scaler = comm_model[str(d)]['scaler']
                    X = scaler.transform(np.asarray(x.iloc[it]).reshape(1, -1))
                    prediction = (abs(comm_model[str(d)]['model'].predict(X))[0])
                    temp = pd.DataFrame({'prediction': prediction, 'index': idx}, index=[it])
                    data = data.append(temp)
                    
        callback = 0
#        get rid of additional meals for normall regressor
        data.index = data['index']
        df.drop(index=data['index'],axis=0, inplace=True)
        
        dummies = pd.get_dummies(df.meal_id)
        df = pd.concat([df, dummies], axis = 1)
        del(dummies)
        
        df['prediction'] = 0
#        load DNN model
        scaler = nn_scalers[str(center)]
        nn = tf.keras.models.load_model(r'\models\nn\{}.h5'.format(center))
        
        for meal in df.meal_id.unique():
            df_train = train[(train.center_id == center)&(train.meal_id==meal)]
            df2 = df[df.meal_id==meal]
            
        
            x = df2.copy()
            x.drop(['meal_id', 'prediction', 'week'], axis=1, inplace=True)
            
#            loop for prediction and creation of shifted and cummulative mean columns  
            for it in range(len(x)):
                if it==0:
                    shft = df_train.num_orders.iloc[-1] - df_train.num_orders.iloc[-2]
                    c_m = sum(df_train.num_orders)/len(df_train)
                    x['shift'].iloc[0] = shft
                    x['orders_cum2'].iloc[0] = c_m
                    X = scaler.transform(np.asarray(x.iloc[0]).reshape(1, -1))
                    df2['prediction'].iloc[0] = abs(nn.predict(X, use_multiprocessing=False))
                elif it==1:
                    shft = df2.prediction.iloc[0] - df_train.num_orders.iloc[-1]
                    c_m = (sum(df_train.num_orders)+sum(df2.prediction))/(len(df_train)+it)
                    x['shift'].iloc[1] = shft
                    x['orders_cum2'].iloc[1] = c_m
                    X = scaler.transform(np.asarray(x.iloc[1]).reshape(1, -1))
                    df2['prediction'].iloc[1] = abs(nn.predict(X, use_multiprocessing=False))
                else:
                    shft = df2.prediction.iloc[it-1] - df2.prediction.iloc[it-2]
                    c_m = (sum(df_train.num_orders)+sum(df2.prediction))/(len(df_train)+it)
                    x['shift'].iloc[it] = shft
                    x['orders_cum2'].iloc[it] = c_m
                    X = scaler.transform(np.asarray(x.iloc[it]).reshape(1, -1))
                    df2['prediction'].iloc[it] = abs(nn.predict(X, use_multiprocessing=False))
            df['prediction'] = df.prediction.add(df2.prediction, fill_value=0)

#        combine predictions from common regressor and DNN regressor
        test['num_orders'] = test['num_orders'].combine(df['prediction'], max)
        test['num_orders'] = test['num_orders'].combine(data['prediction'], max)
    else:
#        normall situation
        dummies = pd.get_dummies(df.meal_id)
        df = pd.concat([df, dummies], axis = 1)
        del(dummies)
        df['prediction'] = 0
        
        scaler = nn_scalers[str(center)]
        nn = tf.keras.models.load_model(r'models\nn\{}.h5'.format(center))
        
        for meal in df.meal_id.unique():
            df2 = df[df.meal_id==meal]
            df_train = train[(train.center_id == center)&(train.meal_id==meal)]
            x = df2.copy()
            x.drop(['meal_id', 'prediction', 'week'], axis=1, inplace=True)
            
            for it in range(len(x)):
                if meal in missing_meals:
                    continue
                if it==0:
                    shft = df_train.num_orders.iloc[-1] - df_train.num_orders.iloc[-2]
                    c_m = sum(df_train.num_orders)/len(df_train)
                    x['shift'].iloc[0] = shft
                    x['orders_cum2'].iloc[0] = c_m
                    X = scaler.transform(np.asarray(x.iloc[0]).reshape(1, -1))
                    df2['prediction'].iloc[0] = abs(nn.predict(X, use_multiprocessing=False))
                elif it==1:
                    shft = df2.prediction.iloc[0] - df_train.num_orders.iloc[-1] 
                    c_m = (sum(df_train.num_orders)+sum(df2.prediction))/(len(df_train)+it)
                    x['shift'].iloc[1] = shft
                    x['orders_cum2'].iloc[1] = c_m
                    X = scaler.transform(np.asarray(x.iloc[1]).reshape(1, -1))
                    df2['prediction'].iloc[1] = abs(nn.predict(X, use_multiprocessing=False))
                else:
                    shft = df2.prediction.iloc[it-1] - df2.prediction.iloc[it-2]
                    c_m = (sum(df_train.num_orders)+sum(df2.prediction))/(len(df_train)+it)
                    x['shift'].iloc[it] = shft
                    x['orders_cum2'].iloc[it] = c_m
                    X = scaler.transform(np.asarray(x.iloc[it]).reshape(1, -1))
                    df2['prediction'].iloc[it] = abs(nn.predict(X, use_multiprocessing=False))
            df['prediction'] = df.prediction.add(df2.prediction, fill_value=0)
    if callback==1:
#        get rid of meals which are not contained in test set
        df.drop(df.tail(len(missing_meals)).index,inplace=True)
        callback = 0
    del(nn)
    gc.collect()   
    return df


