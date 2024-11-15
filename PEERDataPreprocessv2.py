# (c) 2024 Patrick Nieman and Varun Sahay
# Preprocesses ground motion records in the PEER NGA-West2 format


import os
import json

folderpath="/Applications/CS230 Data/PEER 1-50"
recordPath="/Applications/CS230 Data/Records"
tsMetadataPath="/Applications/CS230 Data/timeSeriesMetadata.json"
metadataPath="/Applications/CS230 Data/metadataExpanded.csv"

#Load record file names for each record
metadata={}
with open(metadataPath,"r") as f:
    l=f.readline()
    while l:
        l=l.split("||")
        metadata[str(int(round(float(l[0]))))]=l[112:117]
        l=f.readline()

#Utility
def pad(a,n):
    out=""
    for i in range(n-len(a)):
        out=out+"0"
    return out+a

#For each record, collect desired record metadata, including dt and orientation
tsDatabase={}
mink=1e11
count=0
for _, __, files in os.walk(folderpath, topdown=False):
    for file in files:
        if "_SearchResults" not in file and file.endswith(".AT2"):
            rsni=file.split("_")[0].replace("RSN","")
            rsn=pad(rsni,5)
            if metadata[rsni][0].split("\\")[-1] in file:
                direction=int(round(float(metadata[rsni][3])))
            elif metadata[rsni][1].split("\\")[-1] in file:
                direction=int(round(float(metadata[rsni][4])))
            else: #Shown to be lossless
                continue
            f=open(os.path.join(folderpath,file))
            with open(os.path.join(recordPath,"%s-%s.csv"%(rsn,direction)),"w") as g:
                for i in range(4):
                    l=f.readline()
                try:
                    dt=float(l.lower().split("dt=")[1].split("sec")[0].strip())
                except:
                    print(file)
                    f.close()
                    continue
                l=f.readline()
                k=0
                while l:
                    l=l.strip().replace("\n","").replace("   ","  ").split("  ")
                    for a in l:
                        k+=1
                        g.write("%s\n"%float(a))
                    l=f.readline()
                mink=min(k,mink)
                try:
                    tsDatabase[rsn]["d"].append(int(direction))
                    tsDatabase[rsn]["c"].append(k)
                    tsDatabase[rsn]["dt"].append(dt)
                except:
                    try:
                        tsDatabase[rsn]={"d":[int(direction)],"c":[k],"dt":[dt]}
                    except:
                        print(file)
                count+=1
                if count%100==0:
                    print(count)
            f.close()

#Write metadata as json
with open(tsMetadataPath,"w") as f:
    f.write(json.dumps(tsDatabase))

#Note the minimum number of datapoints in any one record
print(len(tsDatabase)," entries")
print("At least ",mink," time steps")