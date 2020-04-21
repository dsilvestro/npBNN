import numpy as np

np.random.seed(1234)
classes = 20
features = 10
instances = 110

data = np.array([])
list_prm = []

for k in range(classes):
	shape_prm = np.random.uniform(0.2,5,(features,2))
	x = np.zeros((instances,features+1))
	list_prm.append(shape_prm)
	x[:,0] = k
	for i in range(features):
		x	
		x[:,i+1] = np.random.beta(shape_prm[i,0], shape_prm[i,1], instances)
	if k==0:
		data=x
	else:
		data = np.concatenate((data, x), axis=0)

	
print(list_prm)

feat = data[:,1:]
labels = data[:,0].astype(int)

np.save("betas_training_features.npy", feat)
np.save("betas_training_labels.npy", labels)

# additional test set

instances = 100

for k in range(classes):
	shape_prm = list_prm[k]
	x = np.zeros((instances,features+1))
	#print(shape_prm)
	list_prm.append(shape_prm)
	x[:,0] = k
	for i in range(features):
		x	
		x[:,i+1] = np.random.beta(shape_prm[i,0], shape_prm[i,1], instances)
	if k==0:
		data=x
	else:
		data = np.concatenate((data, x), axis=0)

feat = data[:,1:]
labels = data[:,0].astype(int)

np.save("betas_test_features.npy", feat)
np.save("betas_test_labels.npy", labels)
