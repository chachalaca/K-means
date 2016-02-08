__author__ = 'chachalaca'

from KMeans import KMeans
import numpy as np

import pandas as pd

def main():

    km = KMeans(3)

    iris = pd.read_csv("iris.csv")
    data = np.array(
        iris[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]].values.tolist()
    )

    km.fit(data)

    print("cluster centers: %s" % km.cluster_centers)

    for d in iris.values:
        prediction = km.predict([[
            d[2],
            d[2],
            d[3],
            d[4]
        ]])
        print(d[5]+" - "+str(prediction[0]))



if __name__ == "__main__":
    main()

