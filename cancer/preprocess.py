#!/usr/bin/python3

import os
import subprocess as subp


def preprocess(file_base_name, tag):
  ifile = open("{0}.list".format(file_base_name), "r")
  ofile = open("{0}.data".format("training_data"), "a")
  for line in ifile:
    words = line.split()
    if len(words) == 16:
      words.append(str(tag))
      ofile.write(" ".join(words[1:])+"\n")

  ifile.close()
  ofile.close()

def main():
  subp.check_call("unrar x -f DIAG.rar", shell=True)
  os.system("cat ALLD2/*.POK > malignant.list")
  os.system("cat ALLD3/*.POK > benign.list")
  if os.path.exists("training_data.data"):
    os.remove("training_data.data")

  preprocess("malignant", 1)
  preprocess("benign", 0)

if __name__ == "__main__":
  main()

