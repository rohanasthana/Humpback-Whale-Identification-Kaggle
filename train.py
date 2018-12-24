import prepare_data
from keras import layers
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model
import gc
import keras.backend as K
from keras.models import Sequential
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow



def prepareModel(y):
	model = Sequential()

	model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (100, 100, 3)))

	model.add(BatchNormalization(axis = 3, name = 'bn0'))
	model.add(Activation('relu'))

	model.add(MaxPooling2D((2, 2), name='max_pool'))
	model.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))
	model.add(Activation('relu'))
	model.add(AveragePooling2D((3, 3), name='avg_pool'))

	model.add(Flatten())
	model.add(Dense(500, activation="relu", name='rl'))
	model.add(Dropout(0.8))
	model.add(Dense(y, activation='softmax', name='sm'))

	model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
	return model





y_train=prepare_data.prepareLabels()
model=prepareModel(y_train[0,:].size)
X_train=prepare_data.prepareImages()
X_train /= 255

history=model.fit(X_train,y_train,epochs=100,batch_size=100,verbose=1)
model.save('model.h5')
gc.collect()
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

