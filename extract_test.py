#Class Labels - Cat - 0, Dog -1
import math
import cv2
import numpy as np
from sklearn.cluster import KMeans


sift = cv2.SIFT()

'''
img = cv2.imread('train/train/cat.0.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
kp = sift.detect(gray,None)
tkp, td = sift.compute(gray, kp)
points = []
for k in tkp:
	tuples = (int(math.ceil(k.pt[0])),int(math.ceil(k.pt[1])))
	points.append(tuples)
'''

test_features = {}
points = []
max_records = 1000#12499
for idx in range(1,max_records):
	print "processing "+str(idx)+".jpg\n"
	img = cv2.imread('test1/test1/'+str(idx)+'.jpg')
	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	kp = sift.detect(gray,None)
	tkp, td = sift.compute(gray, kp)
	temp_points = []
	for k in tkp:
		tuples = (int(math.ceil(k.pt[0])),int(math.ceil(k.pt[1])))
		points.append(tuples)
		temp_points.append(tuples)
	test_features[idx] = temp_points

kmeans = KMeans()
kmeans = kmeans.fit(points)
params = kmeans.get_params()
n_clusters = params["n_clusters"]

overall_feats = []
count = 1
for data in test_features:
	print "Record-->"+str(count)
	clusters = kmeans.predict(test_features[data])
	print clusters
	feats = []
	for i in range(0,n_clusters):
		feats.append(0)
	for num in clusters:
		feats[num] = feats[num]+1
	overall_feats.append(feats)
	count = count+1
	

feature_file = open("test_features_1000","wb")
for feats in overall_feats:
	val = ""
	for vals in feats:
		val += (str(vals)+",")
	val = val[0:len(val)-1]
	feature_file.write(val+"\n")
'''
arr =  kmeans.predict(points)
params = kmeans.get_params()
print params
'''