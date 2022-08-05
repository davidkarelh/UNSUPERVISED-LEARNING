import pandas as pd
from numpy import ndarray


class DBSCAN:
    def __init__(self, dataset : str, epsilon : float, minimalPoints : int) -> None:
        data = pd.read_csv(dataset)
        self.epsilon = epsilon
        self.minimalPoints = minimalPoints

        self.__features = data.iloc[:, [-2, -1]].values
        self.__featuresIndex = [i for i in range(len(self.__features) - 1, -1, -1)]
        self.__clusterIndexes = []

        self.__dbscan()

        clusterCount = 0
        outlierCount = 0
        for element in self.__clusterIndexes:
            if (len(element) >= self.minimalPoints):
                clusterCount += 1
            else:
                outlierCount +=len(element)
        
        
        print(f"\nJumlah cluster: {clusterCount}")
        print(f"Jumlah outlier: {outlierCount}")
        j = 0
        for i in range(len(self.__clusterIndexes)):
            if (len(self.__clusterIndexes[i]) >= self.minimalPoints):
                print(f"Indeks dan data dari anggota cluster {j + 1} ({len(self.__clusterIndexes[i])} anggota):")
                for element in self.__clusterIndexes[i]:
                    print(f"\tindeks = {element}, data = {self.__features[element]}")
                print()
                j += 1
    
    def __dbscan(self):
        j = 0
        for i in range(len(self.__features)):
            if (i in self.__featuresIndex):
                self.__clusterIndexes.append([i])
                self.__featuresIndex.remove(i)
                self.__dbscanRecursive(j, self.__features[i])
                j += 1

            if (len(self.__featuresIndex) == 0):
                break
        
    def __dbscanRecursive(self, idxCluster : int, feature : ndarray):
        for i in range(len(self.__features)):
            if (i in self.__featuresIndex):
                if (self.__countDistance(self.__features[i], feature) <= self.epsilon):
                    self.__clusterIndexes[idxCluster].append(i)
                    self.__featuresIndex.remove(i)
                    self.__dbscanRecursive(idxCluster, self.__features[i])

            if (len(self.__featuresIndex) == 0):
                break

    def __countDistance(self, point1 : ndarray, point2 : ndarray):
        return (point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (point1[1] - point2[1])
    
