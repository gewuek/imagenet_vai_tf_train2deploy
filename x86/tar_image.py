#! /usr/bin/python3
# coding=utf-8
#####################

import os
import subprocess

tarFileList = []

for fileName in os.listdir("./"):
    if fileName.endswith(".tar"):
        tarFileList.append(fileName)
print(len(tarFileList))
#print(tarFileList[1][:-4])
#subprocess.run(["mkdir", tarFileList[1][:-4]])
#subprocess.run(["tar", "-xvf", tarFileList[1], "-C", tarFileList[1][:-4]])
#subprocess.run(["tar", "-zxvf", tarFileList[0], ])
#subprocess.run(["ls", "-all"])
#subprocess.run(["tar", "-zxvf", tarFileList[0]])
for tarFileName in tarFileList:
    print(tarFileName)
    subprocess.run(["mkdir", tarFileName[:-4]])
    subprocess.run(["tar", "-xf", tarFileName, "-C", tarFileName[:-4]])

