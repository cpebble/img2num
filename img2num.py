from keras.models import load_model
import keras
import numpy as np


class imgRecognizer():
    def __init__(self, filename, outputs, *args, **kwargs):
        self.filename = filename
        self.outputs = [x for x in range(10)]  # labels for neural output
        self.debug = kwargs["debug"] if "debug" in kwargs else False
        # Try to load the model
        self.model = load_model(filename, compile=False)

        if self.debug:
            print(f"Model loaded with input shape: {self.model.input_shape}")
            self.model.summary()

    def guess(self, imgData):
        predict = self.model.predict(imgData, verbose=1)

        highest_index = np.argmax(predict)
        prediction = self.outputs[highest_index]

        return prediction

    def compile(self):
        self.model.compile(
            optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        self.model.load_weights(self.filename)
        if self.debug:
            print("Model successfully compiled")
        self.model._make_predict_function()
