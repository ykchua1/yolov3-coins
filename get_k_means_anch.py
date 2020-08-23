import os
import numpy as np
from sklearn.cluster import KMeans

wh_tuple_list = []

for file in os.listdir("./data/scattered_coins/train"): 
      
    if file[-4:] != ".txt" or file[:3] != "IMG": 
        continue 
    else: 
        abspath = os.path.abspath("./data/scattered_coins/train") 
        abspath = os.path.join(abspath, file) 
         
        with open(abspath) as f: 
            coordlist = f.readlines() 
     
        for line in coordlist: 
            wh_tuple_list.append((line.split()[3], line.split()[4]))
            
X = np.array(wh_tuple_list)
kmeans = KMeans(n_clusters=9, random_state=0).fit(X)

# sort by area
clust_centers = kmeans.cluster_centers_
areas = [(i, x[0] * x[1]) for i, x in enumerate(clust_centers)]
areas.sort(key=lambda x: x[1])

area_ranked_ind = [x[0] for x in areas]
clust_centers = clust_centers[area_ranked_ind]

clust_centers *= 416
clust_centers = np.rint(clust_centers)

print(clust_centers)