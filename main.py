import img2num as i2
import numpy as np
import cv2
imgr = i2.imgRecognizer("mnist.h5", [])
image = "input.png"
imageArr = cv2.imread(image, 0)
imageArr = imageArr.reshape((1, 28, 28, 1))
print(imageArr.shape)

# cv2.imshow("img", imageArr)
# cv2.waitKey(0)
# lets compile
imgr.compile()

# Preproccess data
imgData = imageArr.astype(np.float32) / 255

g = imgr.guess(imgData)
for i in range(10):
    print(g[0][i])
