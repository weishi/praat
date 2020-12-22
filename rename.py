import csv
import os
import shutil
from os import listdir
from os.path import isfile, join
import subprocess
from collections import defaultdict


with open("word.csv") as wordCsv:
  cf = csv.reader(wordCsv)
  dir = '04'
  next(cf)
  for row in cf:
    src_dir = row[0]
    filename = row[1]
    dest_dir0 = row[5]
    dest_dir1 = row[4]
    dest_dir2 = row[3]
    dest_dir3 = row[2]
    if dest_dir1 == '':
      continue

    source_dir = os.path.join(dir, src_dir)
    dest_dir = os.path.join('output', dest_dir0, dest_dir1, dest_dir2, dest_dir3)
    source_path = os.path.join(source_dir, filename)
    dest_path = os.path.join(dest_dir, filename)
    if not os.path.exists(dest_dir):
      os.makedirs(dest_dir)
    print source_path + "==>" + dest_path
    # shutil.copy(source_path+'.txt', dest_path+'.txt')
    shutil.copy(source_path+'.wav', dest_path+'.wav')
    # shutil.copy(source_path+'-merge.TextGrid', dest_path+'.TextGrid')
    shutil.copy(source_path+'.TextGrid', dest_path+'.TextGrid')

