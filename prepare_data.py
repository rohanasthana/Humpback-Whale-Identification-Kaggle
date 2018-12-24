from keras.preprocessing import image
import pandas as pd
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
def prepareImages():
	imgdata=[]
	df=pd.read_csv('train.csv')
	imagearray=df['Image'].values
	print(imagearray)
	path="train/"
	for i in imagearray:
		imagepath=path+i
		im=image.load_img(imagepath,target_size=(100,100,3))
		x1=image.img_to_array(im)
		x1=preprocess_input(x1)
		print(imagepath)
		imgdata.append(x1)
	image_data=np.array(imgdata)
	print(image_data.shape)
	return image_data

def prepareLabels():
	df=pd.read_csv('train.csv')
	values=df['Id'].values
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(values)
	onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
	return onehot_encoded



