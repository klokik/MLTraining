#!/usr/bin/python3

import os
import subprocess as subp


def preprocess(file_base_name, tag):
  ifile = open("{0}.list".format(file_base_name), "r")
  ofile = open("{0}.data".format("data/training_data"), "a")
  for line in ifile:
    words = line.split()
    if len(words) == 16:
      words.append(str(tag))
      ofile.write(" ".join(words[1:])+"\n")

  ifile.close()
  ofile.close()

def main():
  subp.check_call("unrar x -o+ data/DIAG.rar data\\", shell=True)
  os.system("cat data/ALLD2/*.POK > data/malignant.list")
  os.system("cat data/ALLD3/*.POK > data/benign.list")
  if os.path.exists("data/training_data.data"):
    os.remove("data/training_data.data")

  preprocess("data/malignant", 1)
  preprocess("data/benign", 0)

if __name__ == "__main__":
  main()

