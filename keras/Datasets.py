from keras.datasets import mnist
from keras.utils import to_categorical


class MNISTDataset():

    def __init__(self):
        self.image_shape = (28, 28, 1)
        self.num_classes = 10
    
    def get_batch(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train, X_test = [self.preprocess(d) for d in [X_train, X_test]]
        y_train, y_test = [self.preprocess(d, label_data=True) for d in [y_train, y_test]]

        return X_train, y_train, X_test, y_test

    def preprocess(self, data, label_data=False):
        if label_data:
            data = to_categorical(data, self.num_classes)
        else:
            data = data.astype('float32')
            data /= 255
            shape = (data.shape[0],) + self.image_shape
            data = data.reshape(shape)
        
        return data
