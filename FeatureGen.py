#Class Labels - Cat - 0, Dog -1
import math,json
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

class FeatureGen:
	max_cat_records = 12499
	max_dog_records = 12499
	kmeans = None
	n_clusters = 0
	sift = None
	def __init__(self):
		#self.kmeans = KMeans(precompute_distances=False)
		self.kmeans = MiniBatchKMeans()
		self.sift = cv2.SIFT()
		self.n_clusters = self.kmeans.get_params()["n_clusters"]
		print self.n_clusters
	def prepareTrainTestTuples(self):
		file = open("train_tuples","w")
		for idx in range(0,self.max_cat_records):
			print "processing cat."+str(idx)+".jpg\n"
			img = cv2.imread('train/train/cat.'+str(idx)+'.jpg')
			gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			kp = self.sift.detect(gray,None)
			tkp, td = self.sift.compute(gray, kp)
			temp_points = []
			for k in tkp:
				tuples = (int(math.ceil(k.pt[0])),int(math.ceil(k.pt[1])))
				temp_points.append(tuples)
			file.write(json.dumps(temp_points)+"\n")
		for idx in range(0,self.max_dog_records):
			print "processing dog."+str(idx)+".jpg\n"
			img = cv2.imread('train/train/dog.'+str(idx)+'.jpg')
			gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			kp = self.sift.detect(gray,None)
			tkp, td = self.sift.compute(gray, kp)
			temp_points = []
			for k in tkp:
				tuples = (int(math.ceil(k.pt[0])),int(math.ceil(k.pt[1])))
				temp_points.append(tuples)
			file.write(json.dumps(temp_points)+"\n")
		file.close()
		'''
		file = open("test_tuples","w")
		for idx in range(1,12501):
			print "processing "+str(idx)+".jpg\n"
			img = cv2.imread('test1/test1/'+str(idx)+'.jpg')
			gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			kp = self.sift.detect(gray,None)
			tkp, td = self.sift.compute(gray, kp)
			temp_points = []
			for k in tkp:
				tuples = (int(math.ceil(k.pt[0])),int(math.ceil(k.pt[1])))
				temp_points.append(tuples)
			file.write(json.dumps(temp_points)+
		'''
	
	def getSIFTTrainFeatures(self):
		print "refined version"
		points = []
		loaded_feats = []
		file = open("train_tuples","r")
		lines = file.readlines()
		count=1
		for line in lines:
			if count >=10000 and count <=12500:
				print "continue"+str(count)
				count = count+1
				continue
			elif count == 24000:
				break
			print count
			feat_vals = json.loads(line)
			loaded_feats.append(feat_vals)
			for feat in feat_vals:
				points.append(feat)
			count = count+1
		print count
		self.kmeans = self.kmeans.fit(points)
		overall_feats = []
		count = 1
		for feat in loaded_feats:
			print "Record-->"+str(count)
			clusters = self.kmeans.predict(feat)
			print clusters
			feats = []
			for i in range(0,self.n_clusters):
				feats.append(0)
			if count <10000:
			#if count<self.max_cat_records:
				feats.append(0)
			else:
				feats.append(1)
			for num in clusters:
				feats[num] = feats[num]+1
			overall_feats.append(feats)
			count = count+1
		return overall_feats
	'''
	def getSIFTTrainFeatures(self):
		print "Enter train"
		points = []
		loaded_feats = []
		file = open("train_tuples","r")
		lines = file.readlines()
		for line in lines:
			print line
			feat_vals = json.loads(line)
			loaded_feats.append(feat_vals)
			for feat in feat_vals:
				points.append(feat)
		self.kmeans = self.kmeans.fit(points)
		overall_feats = []
		count = 1
		for feat in loaded_feats:
			print "Record-->"+str(count)
			clusters = self.kmeans.predict(feat)
			print clusters
			feats = []
			for i in range(0,self.n_clusters):
				feats.append(0)
			if count<self.max_cat_records:
				feats.append(0)
			else:
				feats.append(1)
			for num in clusters:
				feats[num] = feats[num]+1
			overall_feats.append(feats)
			count = count+1
		return overall_feats
	'''
	
	def getSIFTTestFeatures(self):
		features = {}
		file = open("test_tuples","r")
		lines = file.readlines()
		loaded_feats = []
		for line in lines:
			print line
			feat_vals = json.loads(line)
			loaded_feats.append(feat_vals)
		overall_feats = []
		count = 1
		for feat in loaded_feats:
			print "Record-->"+str(count)
			feats = []
			if len(feat)>0:
				clusters = self.kmeans.predict(feat)
				print clusters
				for i in range(0,self.n_clusters):
					feats.append(0)
				for num in clusters:
					feats[num] = feats[num]+1
			else:
				for i in range(0,self.n_clusters):
					feats.append(0)
			overall_feats.append(feats)
			count = count+1
		return overall_feats

featgen = FeatureGen()
#featgen.prepareTrainTestTuples()

overall_feats = featgen.getSIFTTrainFeatures()
feature_file = open("features_12500","wb")
for feats in overall_feats:
	val = ""
	for vals in feats:
		val += (str(vals)+",")
	val = val[0:len(val)-1]
	feature_file.write(val+"\n")


overall_feats = featgen.getSIFTTestFeatures()
feature_file = open("test_features_1000","wb")
for feats in overall_feats:
	val = ""
	for vals in feats:
		val += (str(vals)+",")
	val = val[0:len(val)-1]
	feature_file.write(val+"\n")
