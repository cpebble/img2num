import img2num as i2
import numpy as np
import cv2
import sys
imgr = i2.imgRecognizer("mnist.h5", [])
image = sys.argv[1] if len(sys.argv) > 1 else "input.png"
imageArr = cv2.imread(image, 0)
print(f"image {image} read")
imageArr = imageArr.reshape((1, 28, 28, 1))
# print(imageArr.shape)

# cv2.imshow("img", imageArr)
# cv2.waitKey(0)
# lets compile
imgr.compile()

# Preproccess data
imgData = imageArr.astype(np.float32) / 255

g = imgr.guess(imgData)
print(g)
