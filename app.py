from flask import Flask, request, Response, send_file
import numpy as np
import cv2
import img2num as i2

app = Flask(__name__)

imgr = i2.imgRecognizer("mnist.h5", [], debug=True)
imgr.compile()


@app.route("/image", methods=["POST"])
def recognize():
    r = request
    filestr = request.files['image'].read()
    # convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    if img.shape != (28, 28):
        return Response("Err, please upload in 28x28 grayscale", 408)
    imgData = img.astype(np.float32) / 255
    imgData = imgData.reshape((1, 28, 28, 1))
    pred = imgr.guess(imgData)
    return Response(str(pred), 200)
