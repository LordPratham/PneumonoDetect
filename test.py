from keras_preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np
model=load_model('E:/our_modelsss.h5') 
img=image.load_img('C:/Users/mkuma/OneDrive/Desktop/ai project/val/NORMAL/NORMAL2-IM-1427-0001.jpeg',target_size=(224,224))
imagee=image.img_to_array(img) 
imagee=np.expand_dims(imagee, axis=0)
img_data=preprocess_input(imagee)
prediction=model.predict(img_data)
if prediction[0][0]>prediction[0][1]: 
	print('Person is safe.')
else:
	print('Person7 is affected with Pneumonia.')
print(f'Predictions: {prediction}')
