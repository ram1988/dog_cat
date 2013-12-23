from sklearn.svm import SVC

feature_file = open("features_1000","r")
features = feature_file.readlines()

target_labels = []
feats = []
for feat in features:
	vals = []
	toks = feat[0:len(feat)-3]
	toks = toks.split(",")
	for tok in toks:
		vals.append(int(tok))
	feats.append(vals)
	target_labels.append(feat[len(feat)-2:len(feat)-1])

clf = SVC(kernel="linear", C=0.025)
clf.fit(feats, target_labels)

feature_file = open("test_features_1000","r")
features = feature_file.readlines()

write_res = open("results","wb")
count=1
write_res.write("id,label\n")
for feat in features:
	feat_vals = feat.split(",")
	predict_vals = []
	for val in feat_vals:
		predict_vals.append(int(val))
	write_res.write(str(count)+","+clf.predict(predict_vals)[0]+"\n")
	count=count+1
