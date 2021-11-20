# USAGE
# python main.py --input "image dir" --source "image dir"

# import the necessary packages
import numpy as np
import cv2 as cv
import argparse
import imutils
import sys

# Construct an argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input image containing Aruco Tag")
ap.add_argument("-s", "--source", required=True, help="path to input source image that will be put on input")
args = vars(ap.parse_args())

# Load the input image from disk resize and grab the spatial dimension
input_image = cv.imread(args["input"])
input_image = imutils.resize(input_image, width=600)
(IMH, IMW) = input_image.shape[:2]

# load the source image from disk
source_image = cv.imread(args["source"])

# Load the Aruco dictionary, grab the Aruco Parameter, and detect the markers
arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_ARUCO_ORIGINAL)
arucoParams = cv.aruco.DetectorParameters_create()
(corners, ids, rejected) = cv.aruco.detectMarkers(input_image, arucoDict, parameters=arucoParams)
print("[INFO]-----> Printing the Corners and IDs of the detected Aruco Markers")
print(f"Corners {corners}\nIDs {ids}")

# If we have not found four markers in the input image then we cannot apply our AR
if len(corners) != 4:
    print("[INFO]----->could not find four corners.....exiting")
    sys.exit(0)

# Otherwise, we've found four corners Aruco Markers, so we can continue
# by flattening the Aruco IDs list and initializing our list of reference points
print("[INFO]----->constructing AR visualization")
ids = ids.flatten()
print(f"[INFO]----->After Ids has been flattened\n IDs {ids}")
refPts = []

# loop oer the IDS of the Aruco markers in top-left, top-right, bottom-right, and bottom-left
for i in (923, 1001, 241, 1007):
    # grab the index of the corner with the current ID and append the corner to our list of reference points
    j = np.squeeze(np.where(ids == i))
    corner = np.squeeze(corners[j])
    refPts.append(corner)
print(f"[INFO]----->Printing the reference points of the input image\nReference Points {refPts}")

# unpack our Aruco reference points and use the the reference the points to define the desitination transform
# matrix, making sure that the points are specified in tl, tr, br, bl order
(refPtTL, refPtTR, refPtBR, refPtBL) = refPts
dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
dstMat = np.array(dstMat)
print(f"[INFO]----->The destination matrix\ndstMat {dstMat}")

# Grab the spatial dimensions of the source mage and define the transform matrix for the source image
# in tl, tr, br, bl order
(srcH, srcW) = source_image.shape[:2]
srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])

# Compute the homography matrix and then warp the source image to the destination based on the homography
(H, _) = cv.findHomography(srcMat, dstMat)
warped = cv.warpPerspective(source_image, H, (IMW, IMH))
print(f"[INFO]----->Homography matrix\nH {H}")

# construct a mask of the source image now that the perspective warp has taken place
# (we'll need this mask to copy the source image into the destination)
mask = np.zeros((IMH, IMW), dtype="uint8")
cv.fillConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255), cv.LINE_AA)

# Create a 3 channel version of the mask by stacking it depth-wise,
# such that we can copy the warped source image into the input
maskScaled = mask.copy() / 255.0
maskScaled = np.dstack([maskScaled] * 3)

# copy the wraped source image into the input image by 1) multiplying the masked image and masked together,
# 2) multiplying the the original input with the maskand 3) adding the results multiplication together.
warpedMultiplied = cv.multiply(warped.astype(float), maskScaled)
imagemultiply = cv.multiply(input_image.astype(float), 1.0 - maskScaled)
output = cv.add(warpedMultiplied, imagemultiply)
output = output.astype("uint8")

# show the input image, source image, output of our augented reality
cv.imshow("Input", input_image)
cv.imshow("Source", source_image)
cv.imshow("Opencv AR output", output)

cv.waitKey(0)