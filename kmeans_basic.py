#
# K means unsupervised clustering 
#
#

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import random 

#X =  np.array([[1,2],[1.5,1.8],[0.1,0.8],[8,8],[9,7],[1,11],[2,10],[0.4,9],[5,5], [2,1], [6.3, 8.9], [4.5, 10.3]])


X = []

for i in range(16):
	a = random.randint(0,10)
	b = random.randint(0,10)
	X.append([a,b])
X=np.array(X)
print(X)

plt.scatter(X[:,0], X[:,1], s=150)
plt.show()

colors = 10*["r","c","b","k"]


class k_means:
	def __init__(self, k=3, tol=5.0, max_iter=300):      #k= centroid num, tol = accuracy required, max_iter = max iterations
		self.k = k
		self.tol = tol
		self.max_iter = max_iter

	def fit(self, data):
		#select k random starting centroids, screen for duplicate centroids
		#need algorithm to prevent duplicates 
		self.centroids = []
		for i in range(self.k):
			rand = random.randrange(len(data))
			cent_init = data[rand]
			self.centroids.append(cent_init.tolist())
			print(i)

		print("Initializing")
		print("0:", self.centroids[0], "1:", self.centroids[1], "2:", self.centroids[2], "\n")


		#while loop for avg_perc_change > tol:
		avg_perc_change = 100
		iterations = 0
		while avg_perc_change > self.tol:
			iterations += 1

			#dictionary for classification, keys =(0, 1 ... k-1)
			cent_dict = {k: [] for k in range(self.k)}

			
			#optimize centroids, iterate through each point and characterize by distance to centroid
			#distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
			for point in X:
				distances = []
				cent_counter = 0
				for centroid in self.centroids:
					distance = np.linalg.norm(point-centroid)
					distances.append([distance, cent_counter])
					cent_counter += 1

				distances = sorted(distances)
				#print(distances)
				cent_dict[distances[0][1]].append(point.tolist())
				
			#print (cent_dict)

			new_centroids = []
			for count in range(self.k):
				new_centroid = (np.mean(cent_dict[count], axis=0))
				new_centroids .append(new_centroid.tolist())

			print("centroids:  ", self.centroids)
			print("new_centroids:  ", new_centroids)

			# percent change of each centroid, averaged
			difference = np.subtract(new_centroids, self.centroids)
			perc_change = np.divide(difference, self.centroids)*100
			avg_perc_change = np.divide(np.sum(abs(perc_change)), (len(perc_change)*len(perc_change[0])))

			print("avg_perc_change: ", avg_perc_change, "iterations: ", iterations, "\n")


			#update centroids
			self.centroids = new_centroids



		#plot with centroids
		cent_plot = np.array(self.centroids)
		plt.scatter(cent_plot[:,0], cent_plot[:,1], marker="x", color='g', s=150)
		plt.scatter(X[:,0], X[:,1], color='r', s=150)
		plt.show()

		print(cent_dict)


	def predict(self, unknown):
		point = np.array(unknown)
		distances = []
		cent_counter = 0
		for centroid in self.centroids:
			distance = np.linalg.norm(point-centroid)
			distances.append([distance, cent_counter])
			cent_counter += 1

		distances = sorted(distances)

		print("this point belongs to centroid: ", distances[0][1], "with a value of: ", self.centroids[distances[0][1]])




clf = k_means()
clf.fit(X)
clf.predict([1,1])