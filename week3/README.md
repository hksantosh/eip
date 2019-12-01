# HK Santosh - EIP 4 - Week 3

# Final Validation accuracy for Base Network
Accuracy on test data is: 82.32

# My model definition
mymodel = Sequential()  
mymodel.add(SeparableConv2D(32, 3, padding='same', activation='relu', input_shape=(32, 32, 3)))  # *Output Size: 32x32x32, Receptive Field: 3x3*  
mymodel.add(BatchNormalization())  
mymodel.add(Dropout(0.06))  

mymodel.add(SeparableConv2D(64, 3, activation='relu')) # *Output Size: 30x30x64, Receptive Field: 5x5*  
mymodel.add(BatchNormalization())  
mymodel.add(Dropout(0.06))  
mymodel.add(MaxPooling2D(pool_size=(2, 2))) # *Output Size: 15x15x64, Receptive Field: 6x6*  

mymodel.add(SeparableConv2D(128, 3, activation='relu')) # *Output Size: 13x13x128, Receptive Field: 10x10*  
mymodel.add(BatchNormalization())  
mymodel.add(Dropout(0.03))  

mymodel.add(SeparableConv2D(256, 3, activation='relu')) # *Output Size: 11x11x256, Receptive Field: 14x14*  
mymodel.add(BatchNormalization())  
mymodel.add(Dropout(0.03))  

mymodel.add(SeparableConv2D(10, 3, activation='relu')) # *Output Size: 9x9x10, Receptive Field: 18x18*  
mymodel.add(GlobalAveragePooling2D())  
mymodel.add(Activation('softmax'))  

mymodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  
mymodel.summary()  


datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)  

mymodel_info = mymodel.fit_generator(datagen.flow(train_features, train_labels, batch_size = 64), steps_per_epoch=train_features.shape[0]/64,
                                 epochs = 50, validation_data = (test_features, test_labels), verbose=1)  


# 50 epoch logs of my model
Epoch 1/50  
782/781 [==============================] - 103s 132ms/step - loss: 1.5075 - accuracy: 0.4578 - val_loss: 1.2667 - val_accuracy: 0.5453  
Epoch 2/50  
782/781 [==============================] - 103s 131ms/step - loss: 1.1268 - accuracy: 0.6046 - val_loss: 1.0452 - val_accuracy: 0.6384  
Epoch 3/50  
782/781 [==============================] - 103s 132ms/step - loss: 0.9848 - accuracy: 0.6566 - val_loss: 0.9296 - val_accuracy: 0.6812  
Epoch 4/50  
782/781 [==============================] - 103s 132ms/step - loss: 0.8998 - accuracy: 0.6872 - val_loss: 0.9799 - val_accuracy: 0.6646  
Epoch 5/50  
782/781 [==============================] - 103s 131ms/step - loss: 0.8390 - accuracy: 0.7100 - val_loss: 0.8422 - val_accuracy: 0.7125  
Epoch 6/50  
782/781 [==============================] - 102s 130ms/step - loss: 0.7922 - accuracy: 0.7229 - val_loss: 0.8673 - val_accuracy: 0.7082  
Epoch 7/50  
782/781 [==============================] - 102s 131ms/step - loss: 0.7557 - accuracy: 0.7366 - val_loss: 0.7433 - val_accuracy: 0.7456  
Epoch 8/50  
782/781 [==============================] - 103s 131ms/step - loss: 0.7271 - accuracy: 0.7474 - val_loss: 0.7259 - val_accuracy: 0.7533  
Epoch 9/50  
782/781 [==============================] - 102s 130ms/step - loss: 0.7027 - accuracy: 0.7559 - val_loss: 0.7135 - val_accuracy: 0.7548  
Epoch 10/50  
782/781 [==============================] - 102s 130ms/step - loss: 0.6827 - accuracy: 0.7627 - val_loss: 0.7343 - val_accuracy: 0.7539  
Epoch 11/50  
782/781 [==============================] - 102s 131ms/step - loss: 0.6586 - accuracy: 0.7723 - val_loss: 0.7134 - val_accuracy: 0.7583  
Epoch 12/50  
782/781 [==============================] - 102s 131ms/step - loss: 0.6430 - accuracy: 0.7768 - val_loss: 0.7007 - val_accuracy: 0.7633  
Epoch 13/50  
782/781 [==============================] - 102s 130ms/step - loss: 0.6266 - accuracy: 0.7851 - val_loss: 0.7206 - val_accuracy: 0.7569  
Epoch 14/50  
782/781 [==============================] - 102s 131ms/step - loss: 0.6131 - accuracy: 0.7895 - val_loss: 0.6553 - val_accuracy: 0.7764  
Epoch 15/50  
782/781 [==============================] - 102s 130ms/step - loss: 0.5993 - accuracy: 0.7925 - val_loss: 0.6582 - val_accuracy: 0.7778  
Epoch 16/50  
782/781 [==============================] - 102s 130ms/step - loss: 0.5910 - accuracy: 0.7958 - val_loss: 0.6316 - val_accuracy: 0.7852  
Epoch 17/50  
782/781 [==============================] - 102s 131ms/step - loss: 0.5807 - accuracy: 0.7987 - val_loss: 0.6310 - val_accuracy: 0.7880  
Epoch 18/50  
782/781 [==============================] - 102s 130ms/step - loss: 0.5695 - accuracy: 0.8030 - val_loss: 0.6285 - val_accuracy: 0.7933  
Epoch 19/50  
782/781 [==============================] - 102s 131ms/step - loss: 0.5621 - accuracy: 0.8052 - val_loss: 0.6747 - val_accuracy: 0.7828  
Epoch 20/50  
782/781 [==============================] - 102s 131ms/step - loss: 0.5526 - accuracy: 0.8096 - val_loss: 0.5941 - val_accuracy: 0.8014  
Epoch 21/50  
782/781 [==============================] - 102s 131ms/step - loss: 0.5463 - accuracy: 0.8117 - val_loss: 0.6337 - val_accuracy: 0.7883  
Epoch 22/50  
782/781 [==============================] - 103s 132ms/step - loss: 0.5356 - accuracy: 0.8165 - val_loss: 0.6049 - val_accuracy: 0.7966  
Epoch 23/50  
782/781 [==============================] - 103s 132ms/step - loss: 0.5302 - accuracy: 0.8171 - val_loss: 0.6412 - val_accuracy: 0.7866  
Epoch 24/50  
782/781 [==============================] - 103s 132ms/step - loss: 0.5211 - accuracy: 0.8208 - val_loss: 0.6289 - val_accuracy: 0.7877  
Epoch 25/50  
782/781 [==============================] - 103s 132ms/step - loss: 0.5170 - accuracy: 0.8194 - val_loss: 0.6144 - val_accuracy: 0.7983  
Epoch 26/50  
782/781 [==============================] - 104s 132ms/step - loss: 0.5099 - accuracy: 0.8236 - val_loss: 0.5874 - val_accuracy: 0.8062  
Epoch 27/50  
782/781 [==============================] - 103s 132ms/step - loss: 0.5008 - accuracy: 0.8277 - val_loss: 0.6145 - val_accuracy: 0.7978  
Epoch 28/50  
782/781 [==============================] - 103s 132ms/step - loss: 0.4990 - accuracy: 0.8261 - val_loss: 0.6488 - val_accuracy: 0.7942  
Epoch 29/50  
782/781 [==============================] - 104s 132ms/step - loss: 0.4907 - accuracy: 0.8304 - val_loss: 0.5912 - val_accuracy: 0.8039  
Epoch 30/50  
782/781 [==============================] - 103s 132ms/step - loss: 0.4910 - accuracy: 0.8296 - val_loss: 0.5890 - val_accuracy: 0.8035  
Epoch 31/50  
782/781 [==============================] - 103s 132ms/step - loss: 0.4812 - accuracy: 0.8321 - val_loss: 0.6123 - val_accuracy: 0.8024  
Epoch 32/50  
782/781 [==============================] - 103s 132ms/step - loss: 0.4819 - accuracy: 0.8323 - val_loss: 0.5549 - val_accuracy: 0.8156  
Epoch 33/50  
782/781 [==============================] - 103s 132ms/step - loss: 0.4743 - accuracy: 0.8333 - val_loss: 0.5787 - val_accuracy: 0.8090  
Epoch 34/50  
782/781 [==============================] - 103s 132ms/step - loss: 0.4764 - accuracy: 0.8348 - val_loss: 0.5826 - val_accuracy: 0.8058  
Epoch 35/50  
782/781 [==============================] - 103s 131ms/step - loss: 0.4691 - accuracy: 0.8375 - val_loss: 0.5648 - val_accuracy: 0.8166  
Epoch 36/50  
782/781 [==============================] - 102s 131ms/step - loss: 0.4671 - accuracy: 0.8378 - val_loss: 0.5905 - val_accuracy: 0.8102  
Epoch 37/50  
782/781 [==============================] - 103s 131ms/step - loss: 0.4588 - accuracy: 0.8405 - val_loss: 0.6144 - val_accuracy: 0.7972  
Epoch 38/50  
782/781 [==============================] - 103s 132ms/step - loss: 0.4575 - accuracy: 0.8408 - val_loss: 0.5549 - val_accuracy: 0.8108  
Epoch 39/50  
782/781 [==============================] - 103s 132ms/step - loss: 0.4529 - accuracy: 0.8390 - val_loss: 0.5718 - val_accuracy: 0.8133  
Epoch 40/50  
782/781 [==============================] - 103s 132ms/step - loss: 0.4529 - accuracy: 0.8407 - val_loss: 0.5528 - val_accuracy: 0.8149  
Epoch 41/50  
782/781 [==============================] - 103s 132ms/step - loss: 0.4470 - accuracy: 0.8440 - val_loss: 0.5965 - val_accuracy: 0.8079  
Epoch 42/50  
782/781 [==============================] - 104s 133ms/step - loss: 0.4478 - accuracy: 0.8456 - val_loss: 0.5689 - val_accuracy: 0.8153  
Epoch 43/50  
782/781 [==============================] - 104s 133ms/step - loss: 0.4457 - accuracy: 0.8447 - val_loss: 0.5721 - val_accuracy: 0.8103  
Epoch 44/50  
782/781 [==============================] - 104s 132ms/step - loss: 0.4346 - accuracy: 0.8494 - val_loss: 0.5729 - val_accuracy: 0.8098  
Epoch 45/50  
782/781 [==============================] - 104s 133ms/step - loss: 0.4362 - accuracy: 0.8467 - val_loss: 0.5702 - val_accuracy: 0.8162  
Epoch 46/50  
782/781 [==============================] - 104s 133ms/step - loss: 0.4268 - accuracy: 0.8508 - val_loss: 0.6448 - val_accuracy: 0.7966  
Epoch 47/50  
782/781 [==============================] - 103s 132ms/step - loss: 0.4270 - accuracy: 0.8505 - val_loss: 0.5940 - val_accuracy: 0.8105  
Epoch 48/50  
782/781 [==============================] - 103s 132ms/step - loss: 0.4321 - accuracy: 0.8493 - val_loss: 0.5804 - val_accuracy: 0.8130  
**Epoch 49/50**  
782/781 [==============================] - 103s 132ms/step - loss: 0.4245 - accuracy: 0.8514 - val_loss: 0.5490 - **val_accuracy: 0.8242**  
Epoch 50/50  
782/781 [==============================] - 103s 132ms/step - loss: 0.4245 - accuracy: 0.8524 - val_loss: 0.5717 - val_accuracy: 0.8156  
mymodel took 5143.45 seconds to train  