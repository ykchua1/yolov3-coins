import os
import numpy as np
from sklearn.cluster import KMeans

wh_tuple_list = []

for file in os.listdir("./data/scattered_coins/"): 
      
    if file[-4:] != ".txt" or file[:3] != "IMG": 
        continue 
    else: 
        abspath = os.path.abspath("./data/scattered_coins/") 
        abspath = os.path.join(abspath, file) 
         
        with open(abspath) as f: 
            coordlist = f.readlines() 
     
        for line in coordlist: 
            wh_tuple_list.append((line.split()[3], line.split()[4]))
            
X = np.array(wh_tuple_list)