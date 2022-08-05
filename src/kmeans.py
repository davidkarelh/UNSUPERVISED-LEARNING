import math
import random
import pandas as pd
import numpy as np


class KMeans:
    maxIteration = 100

    def __init__(self, dataset : str, k : int) -> None:
        data = pd.read_csv(dataset)
        self.k = k

        self.__features = data.iloc[:, [-2, -1]].values

        self.__centroidsIdxs = random.sample(range(0, len(self.__features)), self.k)
        self.__centroids = [self.__features[element] for element in self.__centroidsIdxs]

        self.__clusters = [[self.__features[element]] for element in self.__centroidsIdxs]
        self.__clustersIdxs = [[element] for element in self.__centroidsIdxs]

        self.__kmeans()
    
    def __kmeans(self):
        for i in range(self.maxIteration):
            old_centroids = self.__centroids

            for idx in range(len(self.__features)):
                if (self.__centroidsIdxs[0] == -1 or idx not in self.__centroidsIdxs):
                    self.__clusterPoint(idx)
            
            new_centroids = self.__calculateNewCentroids()

            self.__centroids = new_centroids            
            allSame = False

            for idx in range(len(old_centroids)):
                allSame = old_centroids[idx][0] == self.__centroids[idx][0] and old_centroids[idx][1] == self.__centroids[idx][1]
                if not allSame:
                    break

            if (allSame):
                break

            if i != self.maxIteration - 1:
                self.__clusters = [[] for j in range(self.k)]
                self.__clustersIdxs = [[] for j in range(self.k)]

        for i in range(len(self.__clusters)):
            print(f"Indeks dan data dari anggota cluster {i + 1} ({len(self.__clusters[i])} anggota):")
            j = 0
            for element in self.__clustersIdxs[i]:
                print(f"\tindeks = {element}, data = {self.__features[element]}")
            print()
        
        print(f"Sum of Squared Errors = {self.__calculateSSE()}")
    
    def __clusterPoint(self, idx : int):
        point = self.__features[idx]
        minDistance = math.inf
        clusterIdx = 0

        for i in range(self.k):
            temp = self.__calculateDistance(point, self.__centroids[i])

            if (temp < minDistance):
                minDistance = temp
                clusterIdx = i

        self.__clusters[clusterIdx].append(point)
        self.__clustersIdxs[clusterIdx].append(idx)

    def __calculateDistance(self, point1, point2):
        return (point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (point1[1] - point2[1])

    def __calculateNewCentroids(self):
        ret = []

        for i in range(self.k):
            x_means = 0
            y_means = 0

            for j in range(len(self.__clusters[i])):
                x_means += self.__clusters[i][j][0]
                y_means += self.__clusters[i][j][1]

            x_means = x_means / float(len(self.__clusters[i]))
            y_means = y_means / float(len(self.__clusters[i]))

            ret.append([x_means, y_means])
        
        self.__centroidsIdxs = [-1 for i in range(self.k)]
        return ret

    def __calculateSSE(self):
        ret = 0

        for i in range(self.k):
            sseCluster = 0
            for element in self.__clusters[i]:
                temp = self.__calculateDistance(self.__centroids[i] , element)
                sseCluster += (temp * temp)
        
            ret += sseCluster
        
        return ret