from sklearn.svm import SVC

feature_file = open("features_1000","r")

features = feature_file.readlines()

target_labels = []
feats = []
for feat in features:
	str = feat.split(",")
	target_labels.append(str[len(str)-1])
print target_labels