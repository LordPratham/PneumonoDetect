import keras.preprocessing.image
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense
from glob import glob
import matplotlib.pyplot as plt
import tensorflow

vgg16Model = VGG16(input_shape=[224, 224, 3],
                   weights='imagenet', include_top=False)

trainingDataset = 'chest_xray/train'
testingDataset = 'chest_xray/test'
for eachLayer in vgg16Model.layers:
    eachLayer.trainable = False


classes = glob('chest_xray/train/*')
flattenLayer = Flatten()(vgg16Model.output)
predict = Dense(len(classes), activation='softmax')(flattenLayer)
finalModel = Model(inputs=vgg16Model.input, outputs=predict)

finalModel.summary()

finalModel.compile(
    optimizer='adam',
    metrics=['accuracy'],
    loss='categorical_crossentropy'
)
trainingDataGenerator = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)
testingDatasetGenerator = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255)
testingSet = trainingDataGenerator.flow_from_directory(
    'chest_xray/train',
    target_size=(224, 224),
    batch_size=4,
    class_mode='categorical')


testSet = testingDatasetGenerator.flow_from_directory(
    'chest_xray/test',
    target_size=(224, 224),
    batch_size=4,
    class_mode='categorical')


fitModel = finalModel.fit(
    testingSet,
    validation_data=testSet,
    epochs=5,
    steps_per_epoch=len(testingSet),
    validation_steps=len(testSet)
)
plt.plot(fitModel.history['loss'], label='training loss')
plt.plot(fitModel.history['val_loss'], label='validation loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')
plt.plot(fitModel.history['accuracy'], label='training accuracy')
plt.plot(fitModel.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
finalModel.save('./our_model.h5')
