from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import Adam

from Datasets import MNISTDataset
from Trainer import Trainer

def lenet_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(
        20, kernel_size=5, padding='same',
        input_shape=input_shape, activation='relu'
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(50, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


if __name__ == '__main__':
    dataset = MNISTDataset()

    model = lenet_model(dataset.image_shape, dataset.num_classes)

    X_train, y_train, X_test, y_test = dataset.get_batch()
    trainer = Trainer(model, loss='categorical_crossentropy', optimizer=Adam(), log_dir='logdir_lenet')
    trainer.train(
        X_train, y_train, batch_size=128, epochs=12, validation_split=0.2
    )

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])