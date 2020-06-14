# K neareast neighbors
# using euclidian distance sqrt(x1^2+x2^2)
# reference: sentdex: https://pythonprogramming.net/
#
import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random


def k_nearest_neighbors(data, predict, k=3):
	if len(data) >= k:
		warnings.warn('K is set to a value less than total voting features!') #need more votes than number of features

	distances =[]
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict)) #does pythagoreom theorem, but fast
			distances.append([euclidean_distance, group])

	votes = [i[1] for i in sorted(distances)[:k]]
	#print(votes)
	#print(Counter(votes).most_common(1))
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1]/k

	return vote_result, confidence


df = pd.read_csv('data/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True) #df.dropna(inplace=True)  alternitave, but removes entire now, including useful data
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()  #data housekeeping
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]  #goes from 0 to 80% data
test_data = full_data[-int(test_size*len(full_data)):]  #goes from 80% data to 100% data

for i in train_data:
	train_set[i[-1]].append(i[:-1])    # takes final value 2/4, and appends the rest of the list elements up to final value
for i in test_data:
	test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
	for data in test_set[group]:
		vote, confidence = k_nearest_neighbors(train_set, data, k=5)
		if group == vote:
			correct += 1
		else:
			print('incorrect: ', vote, confidence)
		total += 1

print('\n')
print('accuracy:', correct/total)
print('confidence: ', confidence)





