import numpy as np
import pandas as pd
import tensorflow as tf
submission = pd.read_csv("dataset/sample_submission.csv")
resumes = list(set(submission['resume_seq']))
print(len(resumes))
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, Input, LSTM
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split, KFold
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras.applications import EfficientNetB4, EfficientNetB2, EfficientNetB7, VGG16, MobileNet, ResNet50
# from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation, Dropout, UpSampling2D, Conv2D

# db = pd.read_pickle('list_10000.pkl')
# print(db.head())
# print(type(db), db.shape)
# data_size = len(db)

# train_value = db[:int(data_size*0.8)]

# x_train = train_value.iloc[:,2:-1].astype('float').to_numpy()
# y_train = train_value.iloc[:,-1].astype('float').to_numpy()

# test_value = db[int(data_size*0.8):]

# x_test = test_value.iloc[:,2:-1].astype('float').to_numpy()
# y_test = test_value.iloc[:,-1].astype('float').to_numpy()

# x_pred = db.iloc[:,2:-1].astype('float').to_numpy()

# def RMSE(y_test, y_predict): 
#     return np.sqrt(mean_squared_error(y_test, y_predict)) 

# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  train_size=0.9, random_state = 77, shuffle=True) 
# # x_train = x_train.reshape(38833, 38, 28, 3)
# print(x_train.shape, x_val.shape, x_test.shape) # (3124915, 42) (347213, 42) (177408, 42)

# inputs = Input(shape=(x_train.shape[1]),name='input')
# x = Dense(1024,activation='relu')(inputs)
# x = Dropout(0.2)(x)
# x = Dense(256,activation='relu')(x)
# x = Dropout(0.2)(x)
# x = Dense(64,activation='relu')(x)
# x = Dense(16,activation='relu')(x)
# outputs = Dense(1)(x)

# model = Model(inputs=inputs, outputs=outputs)
# model.summary()


# # es= EarlyStopping(monitor='val_loss', patience=10)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
# # cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
# cp = ModelCheckpoint('../data/h5/effiB2__dense_1.hdf5', monitor='val_loss', save_best_only=True, verbose=1,mode='auto')
# model.compile(loss='mse', optimizer='adam', metrics='mae')
# model.fit(x_train, y_train, epochs=1, batch_size=512, validation_data=(x_val,y_val), callbacks=[reduce_lr,cp])

# # 4. 평가, 예측

# loss, mae = model.evaluate(x_test, y_test, batch_size=512)
# y_predict = model.predict(x_test)

# # RMSE 
# print("RMSE : ", RMSE(y_test, y_predict))

# # R2 만드는 법
# r2 = r2_score(y_test, y_predict)
# print("R2 : ", r2)

# y_pred = model.predict(x_pred)
# db['result'] = y_pred

# one_df = db[db['result']==1]

# submission = pd.read_csv("dataset/sample_submission.csv")
# resumes = list(set(submission['resume_seq']))

# g = []
# for res in resumes:
#     orr = one_df[one_df['resume_seq']==res]
#     print(res)
#     if len(orr) == 0:
#         orr = db[db['resume_seq']==res]
#         print(orr)
#     for m in range(5):
#         g = g.append([res,orr[m]])

# df_g = pd.DataFrame(g, columns=["resume_seq","recruitment_seq"])
# df_g.to_excel("summit.xlsx")