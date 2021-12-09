"""
A series of Utils for dabbot
"""

from os import listdir
from os.path import isfile, join
import random
import simplejson

"""
Get random Dab from the selection of dabs. 
Resized : bool
    If True uses the resized images.
"""


def getDab(Resized=False):
    path = "data/Raw/Dabs/"
    if Resized:
        path = "data/ResizedDabs/Train/Dabs/"
    return path + random.choice([f for f in listdir(path) if isfile(join(path, f))])


"""
Get random non-dab from the selection.
Resized : bool
    If True uses the resized images.
"""


def getNotDab(Resized=False):
    path = "data/Raw/NotDabs/"
    if Resized:
        path = "data/ResizedDabs/Train/NotDabs/"
    return path + random.choice([f for f in listdir(path) if isfile(join(path, f))])


"""
Outputs Data to file.
data : string
    The data to write to the file.
fileName : string
    The Name of the file to save to. 
"""


def outputToFile(data, fileName="Dab_Output.txt"):
    file = open(fileName, "w")
    if type(data) is list:
        simplejson.dump(data, file)
        file.close()
        return
    file.write(data)
    file.close()


"""
Returns the Current Pose Landmarks I want to use. 

Landmarks:
0 - Nose
11 - Left Shoulder
12 - Right Shoulder
13 - Left Elbow
14 - Right Elbow
"""


def dabLandmarks():
    return [0, 11, 12, 13, 14]


"""
Create and populate Folders for Training.
"""


def generateTrainingData():
    # Make Training Directory.
    os.mkdir("/dab_images_in")
    # Make Seperate Directory for both Dab and Non-Dabs.
    os.mkdir("/dab_images_in/dabs")
    os.mkdir("/dab_images_in/not_dabs")
