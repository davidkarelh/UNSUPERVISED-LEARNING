import os, sys


sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from kmeans import KMeans
from dbscan import DBSCAN


def main():
    mainLoop = True
    while (mainLoop):
        # try:
            print(
                """Daftar Algoritma:
                1. kmeans
                2. kmedoids
                3. dbscan""")
            
            selectedAlgo = int(input("Masukkan algoritma yang ingin digunakan: ")) 

            if (selectedAlgo < 1 or selectedAlgo > 3):
                raise Exception("Invalid input. Enter in the range of 1 to 3.")

            mainLoop = False

            if (selectedAlgo == 1):
                dataset = str(input("Masukkan dataset yang ingin digunakan: "))
                k = int(input("Masukkan nilai k: "))
                clusterer = KMeans(dataset, k)

            elif (selectedAlgo == 2):
                pass
            elif (selectedAlgo == 3):
                dataset = str(input("Masukkan dataset yang ingin digunakan: "))
                epsilon = float(input("Masukkan nilai epsilon: "))
                minimalPoints = int(input("Masukkan nilai minimal points: "))

                clusterer = DBSCAN(dataset, epsilon, minimalPoints)
        
        # except Exception as exc:
        #     print(exc)
        #     print()
    
    pass

if (__name__ == "__main__"):
    main()