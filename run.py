import csv
import os
import shutil
from os import listdir
from os.path import isfile, join
import subprocess
from collections import defaultdict

badTracks = set()

homeDir = "./"
for f in sorted(listdir(homeDir)):
  if isfile(join(homeDir, f)) or f in ("output", 'analysis', 'hold') or "." in f:
    continue
  print("Processing directory: " + f)
  tracks = set()
  for wavFile in listdir(join(homeDir,f, f+'_01')):
    if ".wav" in wavFile:
      tracks.add(wavFile[:-4])
      print(wavFile)
  for wavFile in listdir(join(homeDir,f, f+'_02')):
    if ".wav" in wavFile:
      tracks.add(wavFile[:-4])
      print(wavFile)
  sortedTracks = sorted(tracks)
  for track in sortedTracks:
    print("Processing track: " + f + " @ " + track)
    praatDir = join(f, track)
    praatOutputDir = join(f, track, track)
    # print(praatOutputDir)
    shutil.rmtree(praatOutputDir, ignore_errors=True)
    subprocess.call(['/Applications/Praat.app/Contents/MacOS/Praat', '--run', 'merge_and_slice.praat', praatDir, track])
  wordFile = join(homeDir, f, 'word.csv')
  print(wordFile)
  naming = defaultdict(dict)
  with open(wordFile) as wordCsv:
    cf = csv.reader(wordCsv)
    next(cf)
    for row in cf:
      naming[row[1]][row[0]] = (row[3], row[4])
  # print(naming)
  for track in sortedTracks:
    trackPath = join(homeDir, f, track, track)
    print(trackPath)
    for trackName in sorted(listdir(trackPath)):
      # print(trackName)
      slicedNum, extension = os.path.splitext(trackName)
      if slicedNum not in naming[track]:
        badTracks.add(track + ' => ' + slicedNum)
        print(badTracks)
        continue
      finalName = naming[track][slicedNum][1] 
      finalDir = join(homeDir, "output", f, naming[track][slicedNum][0])
      dstPath = join(finalDir, finalName + extension)
      srcPath = join(trackPath, trackName)
      if not os.path.exists(finalDir):
        os.makedirs(finalDir)
      # print(srcPath + " ==> " + dstPath)
      shutil.copy2(srcPath, dstPath)

print(sorted(badTracks))
with open('bad_tracks.txt', 'w') as of:
  for row in sorted(badTracks):
    of.write(row+'\n')