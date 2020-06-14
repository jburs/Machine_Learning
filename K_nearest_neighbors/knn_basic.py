# K neareast neighbors
# using euclidian distance sqrt(x1^2+x2^2)
#
#
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}  #2 classes and their features
new_features = [3,7]                                           #which class does it belong to?


def k_nearest_neighbors(data, predict, k=3):
	if len(data) >= k:
		warnings.warn('K is set to a value less than total voting features!') #need more votes than number of features

	distances =[]
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict)) #does pythagoreom theorem, but fast
			distances.append([euclidean_distance, group])

	votes = [i[1] for i in sorted(distances)[:k]]
	print(votes)
	print(Counter(votes).most_common(1))
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1] / k

	return vote_result, confidence

result, confidence = k_nearest_neighbors(dataset, new_features, k=3)
print(result, confidence)


#for i in dataset:
#	for ii in dataset[i]:
#		[plt.scatter(ii[0],ii[1], s=1--, color=i)]
[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=100, color=result)
plt.show()