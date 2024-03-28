def define_model(len_ts,
                 hidden_neurons = 1,
                 nfeature=1,
                 batch_size=None,
                 stateful=False,
                 num_classes=3,
                 bidirectional=False):
    from tensorflow.keras.regularizers import l2
    inp = layers.Input(batch_shape= (batch_size, len_ts, nfeature),
                       name="input")  

    bi_rnn = layers.Bidirectional(layers.LSTM(hidden_neurons, 
                    return_sequences=True,
                    stateful=stateful,
                    name="bi_lstm",))(inp)
                    #kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01))
    
    dro_out_ly = layers.Dropout(rate=0.2)(bi_rnn)
    rnn = layers.LSTM(hidden_neurons, 
                    return_sequences=False,
                    stateful=stateful,
                    name="RNN",)(dro_out_ly)
                    #kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)

    #dens = layers.TimeDistributed(layers.Dense(in_out_neurons,name="dense"))(rnn)
    dens = layers.Dense(num_classes, name="dense", activation='softmax')(rnn)
    model = models.Model(inputs=[inp],outputs=[dens])
    
    #model.compile(loss="mean_squared_error",
                  #sample_weight_mode="temporal",
                  #optimizer="rmsprop")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return(model,(inp,rnn,dens))
