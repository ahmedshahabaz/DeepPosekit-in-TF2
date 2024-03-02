<HDF5 dataset "annotated": shape (100, 32), type "|b1">
- True/False values for each keypoint of each image.

[[False False False ... False False False]
 [False False False ... False False False]
 [False False False ... False False False]
 ...
 [False False False ... False False False]
 [False  True False ... False False False]
 [False False False ... False False False]]
 

<HDF5 dataset "annotations": shape (100, 32, 2), type "<f8">
- 100 images. 32 keypoints for each images. each keypoint has 2 (x,y) coords.


<HDF5 dataset "images": shape (100, 192, 192, 1), type "|u1">
- 100 images in the dataset each of size 192x192xchannels. channels = 1. so grayscale image.


<HDF5 dataset "skeleton": shape (32, 2), type "<i4">
- Defines the parent, swap for each keypoint

[[-1 -1] keypoint 0: no parent, no swap
 [ 0  2] keypoint 1: parent 0, swap 2
 [ 0  1] keypoint 2: parent 0, swap 1
 [ 0 -1]
 [ 3 -1]
 [ 4 -1]
 [-1 18]
 [ 6 19]
 [ 7 20]
 [ 8 21]
 [-1 22]
 [10 23]
 [11 24]
 [12 25]
 [-1 26]
 [14 27]
 [15 28]
 [16 29]
 [-1  6]
 [18  7]
 [19  8]
 [20  9]
 [-1 10]
 [22 12]
 [23 -1]
 [24 13]
 [-1 14]
 [26 15]
 [27 16]
 [28 17]
 [ 3 31]
 [ 3 30]]

