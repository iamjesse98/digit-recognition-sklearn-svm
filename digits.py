from sklearn import datasets
from sklearn.svm import SVC
from scipy import misc
import numpy as np

digits = datasets.load_digits()
# print(digits)

features = digits.data
labels = digits.target

# print(features, labels)

clf = SVC(gamma = 0.001)
clf.fit(features, labels)

# print(features.shape)
# print(clf.predict([features[-1]]))

img = misc.imread('test/test.jpg')
img = misc.imresize(img, (8, 8))
img = img.astype(digits.images.dtype)
# print(img.dtype)
img = misc.bytescale(img, high=16, low=0)

x_test = []

for eachRow in img:
	for eachPixel in eachRow:
		x_test.append(sum(eachPixel)/3.0)

print(clf.predict([x_test]))