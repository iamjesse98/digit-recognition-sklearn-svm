from sklearn import datasets
from sklearn.svm import SVC
from scipy import misc
import numpy as np

def digit_recognise(img_src):
	digits = datasets.load_digits()

	features = digits.data
	labels = digits.target

	clf = SVC(gamma = 0.001)
	clf.fit(features, labels)

	img = misc.imread(img_src)
	img = misc.imresize(img, (8, 8))
	img = img.astype(digits.images.dtype)
	img = misc.bytescale(img, high=16, low=0)

	x_test = []

	for eachRow in img:
		for eachPixel in eachRow:
			x_test.append(sum(eachPixel)/3.0)

	return clf.predict([x_test])[0]