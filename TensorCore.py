import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import model_from_json
import json


class TensorCoreClass:
    instance = None

    class TensorCore:
        trainedModelPath = 'TrainedModel'
        loaded_model = None
        model = Sequential()
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


        def Train(self):
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
            x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
            input_shape = (28, 28, 1)
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255

            model = Sequential()
            model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(128, activation=tf.nn.relu))
            model.add(Dropout(0.2))
            model.add(Dense(10, activation=tf.nn.sigmoid))

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            model.fit(x=x_train, y=y_train, epochs=1)

            #model.evaluate(x_test, y_test)

            model_json = model.to_json()
            with open(self.trainedModelPath + ".json", "w") as json_file:
                json.dump(model_json, json_file)
            model.save_weights(self.trainedModelPath + ".h5")


        def LoadTrainedModel(self):
            print(self.trainedModelPath)
            with open(self.trainedModelPath + ".json", 'r') as f:
                model_json = json.load(f)
            self.loaded_model = model_from_json(model_json)
            self.loaded_model.load_weights(self.trainedModelPath + ".h5")
            print("Model Loaded")


        def PredictNumber(self,img):
            pred = self.loaded_model.predict(img)
            print(pred.argmax())

    def __new__(cls):
        if TensorCoreClass.instance is None:
            TensorCoreClass.instance = TensorCoreClass.TensorCore()
        return TensorCoreClass.instance


singleInstance = TensorCoreClass()
