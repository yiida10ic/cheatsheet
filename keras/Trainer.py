import os
from keras.callbacks import TensorBoard


class Trainer():

    def __init__(self, model, loss, optimizer, log_dir):
        self._target = model
        self._target.compile(
            loss=loss, optimizer=optimizer, metrics=['accuracy']
        )
        self.verbose = 1
        self.log_dir = os.path.join(os.path.dirname(__file__), log_dir)
    
    def train(self, X_train, y_train, batch_size, epochs, validation_split):
        if os.path.exists(self.log_dir):
            import shutil
            shutil.rmtree(self.log_dir)
        os.mkdir(self.log_dir)

        self._target.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[TensorBoard(log_dir=self.log_dir)],
            verbose=self.verbose
        )