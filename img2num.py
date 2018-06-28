from keras.models import load_model
import keras


class imgRecognizer():
    def __init__(self, filename, outputs, *args, **kwargs):
        self.filename = filename
        self.outputs = [x for x in range(10)]  # labels for neural output

        # Try to load the model
        self.model = load_model(filename, compile=False)

        print(f"Model loaded ")
        print(self.model.input_shape)

    def guess(self, imgData):
        #assert imgData.shape == self.model.input_shape, f'{imgData.shape}, \t {self.model.input_shape}'

        predict = self.model.predict(imgData, verbose=1)
        print(predict)
        return predict

    def compile(self):
        self.model.compile(
            optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        self.model.load_weights(self.filename)
        print("Model successfully compiled")
