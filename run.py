import csv
import os
import shutil
from os import listdir
from os.path import isfile, join
import subprocess
from collections import defaultdict


homeDir = "./"
for f in listdir(homeDir):
  if isfile(join(homeDir, f)) or f == "output" or "." in f:
    continue
  print "Processing directory: " + f
  tracks = set()
  for wavFile in listdir(join(homeDir,f)):
    if ".wav" in wavFile:
      tracks.add(wavFile[:-4])
  sortedTracks = sorted(tracks)
  for track in sortedTracks:
    print "Processing track: " + track
    subprocess.call(['/Applications/Praat.app/Contents/MacOS/Praat', '--run', 'merge_and_slice.praat', f, track])
  wordFile = join(homeDir, f, 'word.csv')
  print wordFile
  naming = defaultdict(dict)
  with open(wordFile) as wordCsv:
    cf = csv.reader(wordCsv)
    next(cf)
    for row in cf:
      naming[row[2]][row[0]] = (row[3], row[4])
  print naming
  for track in sortedTracks:
    trackPath = join(homeDir, f, track)
    for trackName in sorted(listdir(trackPath)):
      print trackName
      slicedNum, extension = os.path.splitext(trackName)
      finalName = naming[track][slicedNum][1] 
      finalDir = join(homeDir, "output", f, naming[track][slicedNum][0])
      dstPath = join(finalDir, finalName + extension)
      srcPath = join(trackPath, trackName)
      if not os.path.exists(finalDir):
        os.makedirs(finalDir)
      print srcPath + " ==> " + dstPath
      shutil.copy2(srcPath, dstPath)
