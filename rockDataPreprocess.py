# (c) 2024 Patrick Nieman and Varun Sahay
# Extracts region boundaries from geospatial dataset for rock classes

import json
import numpy as np
import os

#Load ocean and land datasets
sourceFolder="Sources"
outputFolder="Rock Query"
data=[]
with open(os.path.join("World2.geojsonl.json","r")) as f:
    l=f.readline()
    while l:
        data.append(json.loads(l))
        l=f.readline()

with open(os.path.join("Ocean.geojsonl.json","r")) as f:
    l=f.readline()
    while l:
        data.append(json.loads(l))
        l=f.readline()

#For each shape in each dataset
rocks=[]
rockTypes=[]
polys=[]
holeReferences=[]
with open(os.path.join(outputFolder,"polys.csv","w")) as f:
    for shape in data:
        #Generate text name for reference
        name="%s - %s - %s"%(shape["properties"]["LITHO_EN"],shape["properties"]["STRATI_EN"],shape["properties"]["DESCR_EN"])
        if name not in rocks:
            rocks.append(name)
            rockTypes.append(len(rocks)-1)
        else:
            rockTypes.append(rocks.index(name))
        list=[]

        #Read coordinate points of each polygon, considering holes inside polygons
        try:
            list=np.array(shape["geometry"]["coordinates"]).flatten()
            holeReferences.append(-1)
            f.write(",".join("%s"%a for a in list))
            f.write("\n")
        except: #If the shape contains a hole
            k=0
            backReference=len(rockTypes)-1
            for loop in shape["geometry"]["coordinates"][0]:
                f.write(",".join("%s"%a for a in np.array(loop).flatten()))
                f.write("\n")
                if k>0:
                    rockTypes.append(rocks.index(name))
                    holeReferences.append(backReference)
                else:
                    holeReferences.append(-1)
                k+=1

#Save polygons, rock type of each polygon, and rock type labels
np.savetxt(os.path.join(outputFolder,"rockTypes.csv"),np.array(rockTypes),fmt="%s",delimiter=",")
np.savetxt(os.path.join(outputFolder,"rocks.csv"),np.array(rocks),fmt="%s",delimiter=",")
np.savetxt(os.path.join(outputFolder,"holeReferences.csv"),np.array(holeReferences),fmt="%s",delimiter=",")