

from tensorflow import keras
#from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector
from tensorflow.keras import models 
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, precision_recall_fscore_support, cohen_kappa_score, confusion_matrix

def craft_real_samples(ps_gdf, defo_df, vv_arr_ts, vh_arr_ts, defo_threshold_points_list, defo_anomaly_max_epochs, SM_class_name_dict, defo_class_dict, CLASS_CROSSING=True, MAX_TRAIN_SAMPLES = 500, num_features=1, MIN_TRAIN_SAMPLES=100):
    #get number of classes
    SM_class_ids = list(SM_class_name_dict.keys())
    defo_class_ids = list(defo_class_dict.keys())
    #print(defo_class_ids)
    total_classes = len(defo_class_ids)
    #assign SM classes to points
    
    train_x_all = np.empty((0,defo_df.shape[1],num_features))
    intersect_indices_dict = {}
    train_y_all = np.empty((0,1))
    train_classes = total_classes
    if CLASS_CROSSING:
        #print(ps_gdf)
        #PS_X, PS_Y = ps_gdf.iloc[:,[4,3]].T.values
        #ps_gdf['Class'] = rforest_classified[(PS_Y,PS_X)]
        total_classes = len(SM_class_ids) * len(defo_class_ids)
        new_classes = np.array(SM_class_ids)[np.newaxis].T *10 + np.repeat(np.array(defo_class_ids)[np.newaxis], len(SM_class_ids), axis=0) 
        #print(new_classes)
        new_classes = new_classes.flatten()
        train_classes = new_classes.size
        print('train_classes', train_classes)
    
    
    count=0
    for defo_class in defo_class_ids:
        print('defo_class:', defo_class_dict[defo_class])
        defo_indices = np.array(defo_threshold_points_list[defo_class-1]) #because defo class dict starts from 1
        
        if CLASS_CROSSING:
            for sm_class in SM_class_ids:
                
                sm_indices = ps_gdf.loc[ps_gdf['Class']==sm_class].index.values
                print('SM_class:', SM_class_name_dict[sm_class])
                intersect_indices = np.intersect1d(defo_indices, sm_indices)
                intersect_indices_dict[count] = intersect_indices
                print('Num samples',intersect_indices.size)
                samples = defo_df.iloc[intersect_indices].values[...,np.newaxis][:MAX_TRAIN_SAMPLES]
                print(samples.shape)
                if samples.shape[0]>MIN_TRAIN_SAMPLES:
                    train_x_all = np.vstack((train_x_all, samples))
                    train_y_all = np.append(train_y_all, np.repeat(count, samples.shape[0]))
                    count+=1
        else:
            samples = defo_df.iloc[defo_threshold_points_list[defo_class-1]].values[...,np.newaxis][:MAX_TRAIN_SAMPLES]
            
            train_x_all = np.vstack((train_x_all, samples))
            #train_x_all = np.vstack((train_x_all, np.dstack((defo_df.values[linear_pts], vv_arr_ts[linear_pts], vh_arr_ts[linear_pts]))))
            #print('count', count)
            train_y_all = np.append(train_y_all, np.repeat(count, samples.shape[0]))
            count+=1
    #else:
        
    ##add linear samples : LABEL: 0 
    #train_x_all = np.vstack((train_x_all, defo_df.values[linear_pts]))
    ##train_x_all = np.vstack((train_x_all, np.dstack((defo_df.values[linear_pts], vv_arr_ts[linear_pts], vh_arr_ts[linear_pts]))))
    #train_y_all = np.append(train_y_all, np.repeat(0, linear_pts.size))
    
    ##add sin samples : LABEL: 1 
    #train_x_all = np.vstack((train_x_all, defo_df.values[brp_pts]))
    ##train_x_all = np.vstack((train_x_all, np.dstack((defo_df.values[linear_pts], vv_arr_ts[linear_pts], vh_arr_ts[linear_pts]))))
    #train_y_all = np.append(train_y_all, np.repeat(1, linear_pts.size))
    
    ##add hea samples : LABEL: 2
    #train_x_all = np.vstack((train_x_all, defo_df.values[heaviside_pts]))
    ##train_x_all = np.vstack((train_x_all, np.dstack((defo_df.values[heaviside_pts], vv_arr_ts[heaviside_pts], vh_arr_ts[heaviside_pts]))))
    #train_y_all = np.append(train_y_all, np.repeat(2, heaviside_pts.size))
    
    ##add brp samples : LABEL: 3 
    #train_x_all = np.vstack((train_x_all, defo_df.values[brp_pts]))
    ##train_x_all = np.vstack((train_x_all, np.dstack((defo_df.values[brp_pts], vv_arr_ts[brp_pts], vh_arr_ts[brp_pts]))))
    #train_y_all = np.append(train_y_all, np.repeat(3, brp_pts.size))
    
    plt.hist(train_y_all, label ='training sample distribution', bins=2*count, color='k')
    #plt.hist(test_y, 'test sample distribution')
    #plt.legend()
    plt.show()
    
    train_y_all = keras.utils.to_categorical(train_y_all, train_classes)
    
    print('train_x_all', train_x_all.shape)
    print('train_y_all', train_y_all.shape)
    
    
    
    
    return (train_x_all, train_y_all, intersect_indices_dict)

    
    
def divide_train_test_samples(data_x, data_y, TRAIN_RATIO=0.67):
    
    #data_x, data_y = real_data_preprocessing(def_series,dates,total_flags, features_axis = 0, NUM_POINTS_PLOT=500)
    num_features = data_x.shape[-1]
    num_classes = data_y.shape[-1]
    import random
    train_size = int(data_x.shape[0] * TRAIN_RATIO)
    #train_x_pos = (np.random.random(train_size)*data_x.shape[0]).astype(np.int)#sampling with replacement
    train_x_pos = random.sample(list(np.arange(data_x.shape[0])), train_size)
    #print(train_x_pos.size)
    test_x_pos = np.delete(np.arange(data_x.shape[0]), train_x_pos)
    
    #print(len(train_x_pos),test_x_pos.shape)
    #sys.exit()
    
    train_x = data_x[train_x_pos]
    test_x = data_x[test_x_pos]
    train_y = data_y[train_x_pos]#, np.delete(data_y, train_x_pos, axis=0)
    test_y = data_y[test_x_pos]
    
    print(np.unique(np.where(train_y==1)[1], return_counts=True))
    print(np.unique(np.where(test_y==1)[1], return_counts=True))
    
    #plt.hist(np.argmax(train_y, axis=1), label='training sample distribution', bins = 4)
    #plt.hist(np.argmax(test_y, axis=1), label='test sample distribution',bins = 4)
    #plt.xticks([0,1,2,3], ['Linear', 'Sinusoidal', 'Heaviside', 'Breakpoint'])
    #plt.xlabel('Deformation Classes')
    #plt.ylabel('Frequency')
    #plt.legend()
    #plt.show()
    
    return train_x, train_y, test_x, test_y



def LSTM_class_1(train_x, train_y, test_x, test_y, LOOK_BACK=40, LSTM_NEURONS = 100, LSTM_EPOCHS = 10, batch_size = 10, B_temp=0):
    from tensorflow.keras.callbacks import ModelCheckpoint
    import time
    print('Shapes \ntrain_x: {}\n, train_y: {}\n, test_x: {}\n, test_y: {}\n'.format(train_x.shape, train_y.shape, test_x.shape, test_y.shape))
    model1, _ = define_model(len_ts = train_x.shape[1],
    hidden_neurons = LSTM_NEURONS,
    num_classes=train_y.shape[1])
    model1.summary()
    
    start = time.time()
    hist1 = model1.fit(train_x, train_y,  epochs=LSTM_EPOCHS, batch_size=batch_size, validation_split=0.1, verbose = 1)#,callbacks=[ModelCheckpoint(filepath="weights{epoch:03d}.hdf5")])#validation_data=(X_test, y_test),
    end = time.time()
    print("Time took {:3.1f} min".format((end-start)/60))
    '''
    labels = ["loss","val_loss"]
    for lab in labels:
        plt.plot(hist1.history[lab],label=lab + " model1")
    plt.yscale("log")
    plt.legend()
    plt.show()
    '''
    score = model1.evaluate(test_x, test_y, verbose=0)
    print('Model evaluation score', score)
    
    test_predict = model1.predict(test_x)
    y_true = np.argmax(test_y, axis=1)
    y_pred = np.argmax(test_predict, axis=1)
    
    print(test_predict)
    print('cohen_kappa_score', cohen_kappa_score(y_true, y_pred))
    print(confusion_matrix(np.argmax(test_y, axis=1), np.argmax(test_predict, axis=1)))
    print(classification_report(y_true, y_pred))
    
    #get_weights
    for layer in model1.layers:
        print(str(layer))
        if "LSTM" in str(layer):
            weightLSTM = layer.get_weights()
    warr,uarr, barr = weightLSTM
    print('weight shapes', warr.shape,uarr.shape,barr.shape)
    
    c_tm1 = np.array([0]*LSTM_NEURONS).reshape(1,LSTM_NEURONS)
    h_tm1 = np.array([0]*LSTM_NEURONS).reshape(1,LSTM_NEURONS)
    
    #xs  = np.array([0.003,0.002,1])
    #xs = train_x
    '''
    for x_t, y_t, y_t_pred, pred_prob in zip(test_x, test_y, y_pred, test_predict):
        #x_t = xs[i].reshape(1,1)
        #x_t = train_x[0]
        print('x_t.shape', x_t.shape)
        h_tm1,c_tm1 = LSTMlayer(weightLSTM,x_t,h_tm1,c_tm1)
        #print("h3={}".format(h_tm1))
        #print("c3={}".format(c_tm1))
        
        print("h3 shape={}".format(h_tm1.shape))
        print("c3 shape={}".format(c_tm1.shape))
        print("Class", y_t)
        print('Predicted class', y_t_pred)
        print('Predicted prob', pred_prob)
        
        plt.subplot(221)
        plt.plot(B_temp, x_t)
        plt.subplot(223)
        plt.imshow(h_tm1.T, cmap='jet_r')
        plt.colorbar()
        plt.subplot(224)
        plt.imshow(c_tm1.T, cmap='jet_r')
        plt.colorbar()
        plt.show()
    '''
    return y_pred, test_predict, cohen_kappa_score(y_true, y_pred), score, model1, classification_report(y_true, y_pred)

if __name__=='_main_':
    pass
