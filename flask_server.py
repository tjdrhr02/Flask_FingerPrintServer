from __future__ import division
from flask import Flask, request, send_file
from math import dist

import math
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No image file uploaded', 400

    image = request.files['image']
    if image.filename == '':
        return 'No image file selected', 400

    # save original image
    save_path = 'original_image.jpg'
    image.save(save_path)

    ###
    # model
    protoFile = "hand/pose_deploy.prototxt"
    weightsFile = "hand/pose_iter_102000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    # get original image
    orig_img = cv2.imread(save_path)
    imgWidth = orig_img.shape[1]
    imgHeight = orig_img.shape[0]
    imgRatio = imgWidth/imgHeight

    #
    copy_img = np.copy(orig_img)

    # input image dimensions for the network
    inHeight = 368
    inWidth = int(((imgRatio * inHeight) * 8) // 8)
    inpBlob = cv2.dnn.blobFromImage(orig_img, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()

    # detect
    points = []

    for i in range(22):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (imgWidth, imgHeight))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(point[0]), int(point[1])))

    # fingerprint points
    # x : topFpoints[i][0], y : topFpoints[i][1]
    topFPoints = [points[4], points[8], points[12], points[16], points[20]]
    downFPoints = [points[3], points[7], points[11], points[15], points[19]]

    # draw oval on every fingerprint
    for i in range(5):
        topX = topFPoints[i][0]
        topY = topFPoints[i][1]

        downX = downFPoints[i][0]
        downY = downFPoints[i][1]

        centerPoint = (int((topX + downX) / 2), int((topY + downY) / 2))

        distance = dist((topX, topY), (downX, downY))
        axesLength = (int((distance) / 2), int(distance / 4))

        angle = int(math.atan2(topY - downY, topX - downX) * 180 / math.pi)

        cv2.ellipse(copy_img, centerPoint, axesLength, angle, 0, 360, (0, 0, 0), -1)

    bit_xor = cv2.bitwise_xor(copy_img, orig_img)
    # cv2.imshow('Output-bit_xor', bit_xor)

    blurred = cv2.medianBlur(bit_xor, 9)
    # cv2.imshow('Output-blur', blurred)

    bit_or = cv2.bitwise_or(copy_img, blurred)
    # cv2.imshow('Output-bit_or', bit_or)
    cv2.imwrite('blurred_image.jpg', bit_or)

    cv2.waitKey(0)

    ###
    return 'Image uploaded successfully'

@app.route('/image', methods=['GET'])
def send_image():
    # 이미지 파일 경로
    image_path = 'blurred_image.jpg'

    return send_file(image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run()