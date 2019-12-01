# HK Santosh - EIP 4 - Week 3

# Final Validation accuracy for Base Network
Accuracy on test data is: 82.52
Accuracy on test data (with tf2) is: 82.44

# My model definition
mymodel = Sequential()
mymodel.add(SeparableConv2D(32, 3, padding='same', activation='relu', input_shape=(32, 32, 3)))  # Output Size: 32x32x32, Receptive Field: 3x3
mymodel.add(BatchNormalization())
mymodel.add(Dropout(0.1))

mymodel.add(SeparableConv2D(64, 3, activation='relu')) # Output Size: 30x30x64, Receptive Field: 5x5
mymodel.add(BatchNormalization())
mymodel.add(Dropout(0.1))

mymodel.add(SeparableConv2D(128, 3, activation='relu')) # Output Size: 28x28x128, Receptive Field: 7x7
mymodel.add(BatchNormalization())
mymodel.add(Dropout(0.05))

mymodel.add(SeparableConv2D(256, 3, activation='relu')) # Output Size: 26x26x256, Receptive Field: 9x9
mymodel.add(BatchNormalization())
mymodel.add(Dropout(0.05))

mymodel.add(SeparableConv2D(10, 3, activation='relu')) # Output Size: 24x24x10, Receptive Field: 11x11
mymodel.add(GlobalAveragePooling2D())
mymodel.add(Activation('softmax'))

mymodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
mymodel.summary()

datagen = ImageDataGenerator(zoom_range=0.0, horizontal_flip=True)

mymodel_info = mymodel.fit_generator(datagen.flow(train_features, train_labels, batch_size = 64), steps_per_epoch=train_features.shape[0]/64,
                                 epochs = 50, validation_data = (test_features, test_labels), verbose=1)

# 50 epoch logs of my model
Epoch 1/50
782/781 [==============================] - 109s 140ms/step - loss: 1.5481 - accuracy: 0.4421 - val_loss: 1.4075 - val_accuracy: 0.4997
Epoch 2/50
782/781 [==============================] - 105s 134ms/step - loss: 1.2174 - accuracy: 0.5713 - val_loss: 1.1495 - val_accuracy: 0.5905
Epoch 3/50
782/781 [==============================] - 105s 134ms/step - loss: 1.0611 - accuracy: 0.6267 - val_loss: 1.0138 - val_accuracy: 0.6438
Epoch 4/50
782/781 [==============================] - 105s 134ms/step - loss: 0.9588 - accuracy: 0.6627 - val_loss: 0.9617 - val_accuracy: 0.6628
Epoch 5/50
782/781 [==============================] - 104s 133ms/step - loss: 0.8937 - accuracy: 0.6880 - val_loss: 0.9178 - val_accuracy: 0.6764
Epoch 6/50
782/781 [==============================] - 105s 134ms/step - loss: 0.8450 - accuracy: 0.7069 - val_loss: 0.8366 - val_accuracy: 0.7081
Epoch 7/50
782/781 [==============================] - 106s 135ms/step - loss: 0.8019 - accuracy: 0.7197 - val_loss: 0.8327 - val_accuracy: 0.7127
Epoch 8/50
782/781 [==============================] - 105s 134ms/step - loss: 0.7646 - accuracy: 0.7343 - val_loss: 0.7787 - val_accuracy: 0.7331
Epoch 9/50
782/781 [==============================] - 104s 133ms/step - loss: 0.7365 - accuracy: 0.7442 - val_loss: 0.7879 - val_accuracy: 0.7301
Epoch 10/50
782/781 [==============================] - 104s 133ms/step - loss: 0.7153 - accuracy: 0.7525 - val_loss: 0.7681 - val_accuracy: 0.7374
Epoch 11/50
782/781 [==============================] - 104s 133ms/step - loss: 0.6943 - accuracy: 0.7596 - val_loss: 0.7406 - val_accuracy: 0.7473
Epoch 12/50
782/781 [==============================] - 104s 133ms/step - loss: 0.6679 - accuracy: 0.7680 - val_loss: 0.7852 - val_accuracy: 0.7385
Epoch 13/50
782/781 [==============================] - 105s 134ms/step - loss: 0.6541 - accuracy: 0.7724 - val_loss: 0.7279 - val_accuracy: 0.7518
Epoch 14/50
782/781 [==============================] - 104s 133ms/step - loss: 0.6399 - accuracy: 0.7767 - val_loss: 0.7291 - val_accuracy: 0.7570
Epoch 15/50
782/781 [==============================] - 104s 134ms/step - loss: 0.6262 - accuracy: 0.7845 - val_loss: 0.6756 - val_accuracy: 0.7694
Epoch 16/50
782/781 [==============================] - 105s 134ms/step - loss: 0.6114 - accuracy: 0.7880 - val_loss: 0.6783 - val_accuracy: 0.7669
Epoch 17/50
782/781 [==============================] - 104s 133ms/step - loss: 0.5989 - accuracy: 0.7923 - val_loss: 0.7105 - val_accuracy: 0.7591
Epoch 18/50
782/781 [==============================] - 104s 133ms/step - loss: 0.5852 - accuracy: 0.7965 - val_loss: 0.7368 - val_accuracy: 0.7535
Epoch 19/50
782/781 [==============================] - 104s 134ms/step - loss: 0.5795 - accuracy: 0.7994 - val_loss: 0.6425 - val_accuracy: 0.7843
Epoch 20/50
782/781 [==============================] - 104s 134ms/step - loss: 0.5654 - accuracy: 0.8047 - val_loss: 0.7073 - val_accuracy: 0.7651
Epoch 21/50
782/781 [==============================] - 105s 134ms/step - loss: 0.5554 - accuracy: 0.8056 - val_loss: 0.6542 - val_accuracy: 0.7831
Epoch 22/50
782/781 [==============================] - 105s 134ms/step - loss: 0.5476 - accuracy: 0.8082 - val_loss: 0.6295 - val_accuracy: 0.7888
Epoch 23/50
782/781 [==============================] - 105s 134ms/step - loss: 0.5397 - accuracy: 0.8121 - val_loss: 0.6700 - val_accuracy: 0.7791
Epoch 24/50
782/781 [==============================] - 104s 133ms/step - loss: 0.5341 - accuracy: 0.8149 - val_loss: 0.6299 - val_accuracy: 0.7904
Epoch 25/50
782/781 [==============================] - 104s 134ms/step - loss: 0.5251 - accuracy: 0.8165 - val_loss: 0.6395 - val_accuracy: 0.7881
Epoch 26/50
782/781 [==============================] - 104s 133ms/step - loss: 0.5174 - accuracy: 0.8207 - val_loss: 0.6353 - val_accuracy: 0.7877
Epoch 27/50
782/781 [==============================] - 104s 133ms/step - loss: 0.5146 - accuracy: 0.8210 - val_loss: 0.5926 - val_accuracy: 0.8034
Epoch 28/50
782/781 [==============================] - 105s 134ms/step - loss: 0.5049 - accuracy: 0.8237 - val_loss: 0.6097 - val_accuracy: 0.8025
Epoch 29/50
782/781 [==============================] - 104s 132ms/step - loss: 0.4977 - accuracy: 0.8265 - val_loss: 0.6595 - val_accuracy: 0.7852
Epoch 30/50
782/781 [==============================] - 104s 133ms/step - loss: 0.4955 - accuracy: 0.8267 - val_loss: 0.6216 - val_accuracy: 0.7926
Epoch 31/50
782/781 [==============================] - 105s 134ms/step - loss: 0.4848 - accuracy: 0.8300 - val_loss: 0.6161 - val_accuracy: 0.7963
Epoch 32/50
782/781 [==============================] - 105s 134ms/step - loss: 0.4856 - accuracy: 0.8314 - val_loss: 0.6360 - val_accuracy: 0.7894
Epoch 33/50
782/781 [==============================] - 105s 134ms/step - loss: 0.4778 - accuracy: 0.8335 - val_loss: 0.6244 - val_accuracy: 0.7898
Epoch 34/50
782/781 [==============================] - 105s 134ms/step - loss: 0.4735 - accuracy: 0.8330 - val_loss: 0.6129 - val_accuracy: 0.8012
Epoch 35/50
782/781 [==============================] - 104s 133ms/step - loss: 0.4709 - accuracy: 0.8345 - val_loss: 0.6038 - val_accuracy: 0.8049
Epoch 36/50
782/781 [==============================] - 104s 133ms/step - loss: 0.4644 - accuracy: 0.8396 - val_loss: 0.6222 - val_accuracy: 0.7998
Epoch 37/50
782/781 [==============================] - 104s 134ms/step - loss: 0.4624 - accuracy: 0.8398 - val_loss: 0.6258 - val_accuracy: 0.7919
Epoch 38/50
782/781 [==============================] - 104s 133ms/step - loss: 0.4575 - accuracy: 0.8398 - val_loss: 0.6652 - val_accuracy: 0.7827
Epoch 39/50
782/781 [==============================] - 104s 133ms/step - loss: 0.4511 - accuracy: 0.8416 - val_loss: 0.5956 - val_accuracy: 0.8033
Epoch 40/50
782/781 [==============================] - 104s 133ms/step - loss: 0.4493 - accuracy: 0.8446 - val_loss: 0.5714 - val_accuracy: 0.8100
Epoch 41/50
782/781 [==============================] - 104s 133ms/step - loss: 0.4445 - accuracy: 0.8446 - val_loss: 0.5836 - val_accuracy: 0.8085
Epoch 42/50
782/781 [==============================] - 104s 133ms/step - loss: 0.4388 - accuracy: 0.8460 - val_loss: 0.6534 - val_accuracy: 0.7878
Epoch 43/50
782/781 [==============================] - 104s 133ms/step - loss: 0.4310 - accuracy: 0.8504 - val_loss: 0.5993 - val_accuracy: 0.8035
Epoch 44/50
782/781 [==============================] - 105s 134ms/step - loss: 0.4297 - accuracy: 0.8498 - val_loss: 0.6027 - val_accuracy: 0.8073
Epoch 45/50
782/781 [==============================] - 104s 133ms/step - loss: 0.4258 - accuracy: 0.8504 - val_loss: 0.6028 - val_accuracy: 0.8072
Epoch 46/50
782/781 [==============================] - 105s 134ms/step - loss: 0.4223 - accuracy: 0.8519 - val_loss: 0.5745 - val_accuracy: 0.8107
Epoch 47/50
782/781 [==============================] - 104s 132ms/step - loss: 0.4222 - accuracy: 0.8523 - val_loss: 0.5935 - val_accuracy: 0.8030
Epoch 48/50
782/781 [==============================] - 105s 134ms/step - loss: 0.4172 - accuracy: 0.8553 - val_loss: 0.5961 - val_accuracy: 0.8119
**Epoch 49/50**
782/781 [==============================] - 105s 134ms/step - loss: 0.4144 - accuracy: 0.8550 - val_loss: 0.5676 - **val_accuracy: 0.8182**
Epoch 50/50
782/781 [==============================] - 104s 133ms/step - loss: 0.4130 - accuracy: 0.8552 - val_loss: 0.6298 - val_accuracy: 0.7976
mymodel took 5223.61 seconds to train