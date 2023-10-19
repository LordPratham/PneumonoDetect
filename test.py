from keras_preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np
model = load_model('./our_model.h5')
img = image.load_img(
    '/Users/pratham/Desktop/ai project/chest_xray/val/NORMAL/NORMAL2-IM-1431-0001.jpeg', target_size=(224, 224))
imagee = image.img_to_array(img)
imagee = np.expand_dims(imagee, axis=0)
imgData = preprocess_input(imagee)
prediction = model.predict(imgData)
if prediction[0][0] > prediction[0][1]:
    print('Person is safe.')
else:
    print('Person is affected with Pneumonia.')
print(f'Predictions: {prediction}')
