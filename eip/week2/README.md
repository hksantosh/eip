# HK Santosh - EIP 4 - Week 2

# Strategy
* Reduced batch size and learning rate to achieve higher accuracy in lesser epochs
* Reduced convolution layers to reduce the parameters

# Training Logs
Train on 60000 samples, validate on 10000 samples

Epoch 00001: LearningRateScheduler reducing learning rate to 0.0015.
Epoch 1/20
59648/60000 [============================>.] - ETA: 0s - loss: 0.4967 - accuracy: 0.8777
Epoch 00001: val_accuracy improved from -inf to 0.98370, saving model to wk2_ninth.h5
60000/60000 [==============================] - 8s 135us/sample - loss: 0.4957 - accuracy: 0.8779 - val_loss: 0.1044 - val_accuracy: 0.9837

Epoch 00002: LearningRateScheduler reducing learning rate to 0.0011372252.
Epoch 2/20
59776/60000 [============================>.] - ETA: 0s - loss: 0.2612 - accuracy: 0.9336
Epoch 00002: val_accuracy improved from 0.98370 to 0.98610, saving model to wk2_ninth.h5
60000/60000 [==============================] - 7s 109us/sample - loss: 0.2609 - accuracy: 0.9337 - val_loss: 0.0772 - val_accuracy: 0.9861

Epoch 00003: LearningRateScheduler reducing learning rate to 0.0009157509.
Epoch 3/20
59712/60000 [============================>.] - ETA: 0s - loss: 0.2192 - accuracy: 0.9404
Epoch 00003: val_accuracy improved from 0.98610 to 0.98900, saving model to wk2_ninth.h5
60000/60000 [==============================] - 7s 110us/sample - loss: 0.2190 - accuracy: 0.9404 - val_loss: 0.0533 - val_accuracy: 0.9890

Epoch 00004: LearningRateScheduler reducing learning rate to 0.0007664793.
Epoch 4/20
59712/60000 [============================>.] - ETA: 0s - loss: 0.1862 - accuracy: 0.9453
Epoch 00004: val_accuracy improved from 0.98900 to 0.99010, saving model to wk2_ninth.h5
60000/60000 [==============================] - 7s 112us/sample - loss: 0.1862 - accuracy: 0.9453 - val_loss: 0.0452 - val_accuracy: 0.9901

Epoch 00005: LearningRateScheduler reducing learning rate to 0.000659051.
Epoch 5/20
59648/60000 [============================>.] - ETA: 0s - loss: 0.1662 - accuracy: 0.9478
Epoch 00005: val_accuracy improved from 0.99010 to 0.99160, saving model to wk2_ninth.h5
60000/60000 [==============================] - 7s 112us/sample - loss: 0.1663 - accuracy: 0.9478 - val_loss: 0.0366 - val_accuracy: 0.9916

Epoch 00006: LearningRateScheduler reducing learning rate to 0.0005780347.
Epoch 6/20
59776/60000 [============================>.] - ETA: 0s - loss: 0.1557 - accuracy: 0.9504
Epoch 00006: val_accuracy improved from 0.99160 to 0.99170, saving model to wk2_ninth.h5
60000/60000 [==============================] - 7s 109us/sample - loss: 0.1556 - accuracy: 0.9505 - val_loss: 0.0302 - val_accuracy: 0.9917

Epoch 00007: LearningRateScheduler reducing learning rate to 0.0005147563.
Epoch 7/20
59584/60000 [============================>.] - ETA: 0s - loss: 0.1440 - accuracy: 0.9504
Epoch 00007: val_accuracy did not improve from 0.99170
60000/60000 [==============================] - 6s 108us/sample - loss: 0.1439 - accuracy: 0.9504 - val_loss: 0.0338 - val_accuracy: 0.9911

Epoch 00008: LearningRateScheduler reducing learning rate to 0.0004639654.
Epoch 8/20
59648/60000 [============================>.] - ETA: 0s - loss: 0.1383 - accuracy: 0.9510
Epoch 00008: val_accuracy improved from 0.99170 to 0.99180, saving model to wk2_ninth.h5
60000/60000 [==============================] - 7s 109us/sample - loss: 0.1386 - accuracy: 0.9510 - val_loss: 0.0319 - val_accuracy: 0.9918

Epoch 00009: LearningRateScheduler reducing learning rate to 0.0004222973.
Epoch 9/20
59904/60000 [============================>.] - ETA: 0s - loss: 0.1326 - accuracy: 0.9518
Epoch 00009: val_accuracy improved from 0.99180 to 0.99330, saving model to wk2_ninth.h5
60000/60000 [==============================] - 7s 109us/sample - loss: 0.1326 - accuracy: 0.9518 - val_loss: 0.0269 - val_accuracy: 0.9933

Epoch 00010: LearningRateScheduler reducing learning rate to 0.0003874968.
Epoch 10/20
59968/60000 [============================>.] - ETA: 0s - loss: 0.1240 - accuracy: 0.9533
Epoch 00010: val_accuracy did not improve from 0.99330
60000/60000 [==============================] - 7s 108us/sample - loss: 0.1242 - accuracy: 0.9533 - val_loss: 0.0269 - val_accuracy: 0.9920

Epoch 00011: LearningRateScheduler reducing learning rate to 0.0003579952.
Epoch 11/20
59968/60000 [============================>.] - ETA: 0s - loss: 0.1223 - accuracy: 0.9545
Epoch 00011: val_accuracy did not improve from 0.99330
60000/60000 [==============================] - 6s 108us/sample - loss: 0.1225 - accuracy: 0.9545 - val_loss: 0.0241 - val_accuracy: 0.9926

Epoch 00012: LearningRateScheduler reducing learning rate to 0.000332668.
Epoch 12/20
59520/60000 [============================>.] - ETA: 0s - loss: 0.1153 - accuracy: 0.9547
Epoch 00012: val_accuracy did not improve from 0.99330
60000/60000 [==============================] - 6s 108us/sample - loss: 0.1153 - accuracy: 0.9547 - val_loss: 0.0252 - val_accuracy: 0.9932

Epoch 00013: LearningRateScheduler reducing learning rate to 0.0003106877.
Epoch 13/20
59840/60000 [============================>.] - ETA: 0s - loss: 0.1184 - accuracy: 0.9533
Epoch 00013: val_accuracy improved from 0.99330 to 0.99340, saving model to wk2_ninth.h5
60000/60000 [==============================] - 7s 109us/sample - loss: 0.1184 - accuracy: 0.9532 - val_loss: 0.0255 - val_accuracy: 0.9934

Epoch 00014: LearningRateScheduler reducing learning rate to 0.0002914319.
Epoch 14/20
59968/60000 [============================>.] - ETA: 0s - loss: 0.1156 - accuracy: 0.9544
Epoch 00014: val_accuracy improved from 0.99340 to 0.99350, saving model to wk2_ninth.h5
60000/60000 [==============================] - 7s 109us/sample - loss: 0.1156 - accuracy: 0.9544 - val_loss: 0.0276 - val_accuracy: 0.9935

Epoch 00015: LearningRateScheduler reducing learning rate to 0.0002744237.
Epoch 15/20
59840/60000 [============================>.] - ETA: 0s - loss: 0.1120 - accuracy: 0.9548
Epoch 00015: val_accuracy did not improve from 0.99350
60000/60000 [==============================] - 6s 108us/sample - loss: 0.1121 - accuracy: 0.9548 - val_loss: 0.0255 - val_accuracy: 0.9931

Epoch 00016: LearningRateScheduler reducing learning rate to 0.0002592913.
Epoch 16/20
59712/60000 [============================>.] - ETA: 0s - loss: 0.1073 - accuracy: 0.9566
Epoch 00016: val_accuracy did not improve from 0.99350
60000/60000 [==============================] - 7s 109us/sample - loss: 0.1072 - accuracy: 0.9566 - val_loss: 0.0248 - val_accuracy: 0.9929

**Epoch 00017**: LearningRateScheduler reducing learning rate to 0.0002457405.
Epoch 17/20
59776/60000 [============================>.] - ETA: 0s - loss: 0.1110 - accuracy: 0.9536
Epoch 00017: val_accuracy improved from 0.99350 to 0.99370, saving model to wk2_ninth.h5
60000/60000 [==============================] - 7s 110us/sample - loss: 0.1110 - accuracy: 0.9536 - val_loss: 0.0234 - **val_accuracy: 0.9937**

Epoch 00018: LearningRateScheduler reducing learning rate to 0.0002335357.
Epoch 18/20
59712/60000 [============================>.] - ETA: 0s - loss: 0.1078 - accuracy: 0.9560
Epoch 00018: val_accuracy did not improve from 0.99370
60000/60000 [==============================] - 7s 108us/sample - loss: 0.1081 - accuracy: 0.9559 - val_loss: 0.0236 - val_accuracy: 0.9936

Epoch 00019: LearningRateScheduler reducing learning rate to 0.0002224859.
Epoch 19/20
59584/60000 [============================>.] - ETA: 0s - loss: 0.1066 - accuracy: 0.9556
Epoch 00019: val_accuracy did not improve from 0.99370
60000/60000 [==============================] - 6s 108us/sample - loss: 0.1066 - accuracy: 0.9556 - val_loss: 0.0233 - val_accuracy: 0.9936

Epoch 00020: LearningRateScheduler reducing learning rate to 0.0002124345.
Epoch 20/20
59840/60000 [============================>.] - ETA: 0s - loss: 0.1073 - accuracy: 0.9554
Epoch 00020: val_accuracy did not improve from 0.99370
60000/60000 [==============================] - 6s 108us/sample - loss: 0.1073 - accuracy: 0.9554 - val_loss: 0.0238 - val_accuracy: 0.9933
<tensorflow.python.keras.callbacks.History at 0x7f70440a9e48>

# Results/Score on test data
[0.023434858139790595, 0.9937]