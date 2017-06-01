#!/usr/bin/python3

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def readData():
  data = []

  #bfile = open("benign.data")
  #for line in bfile:
  #  vals = list(map(float, line.split()))
  #  data.append((vals[1:], 0))
  #bfile.close()

  #mfile = open("malignant.data")
  #for line in mfile:
  #  vals = list(map(float, line.split()))
  #  data.append((vals[1:], 1))
  #mfile.close()

  tfile = open("data/training_data.data")
  for line in tfile:
    vals = list(map(float, line.split()))
    data.append((vals[:-1], vals[-1]))
  tfile.close()

  print("Successfuly read {0} data samples".format(len(data)))

  return data

def levelTopBottom(data, rate):
  n = len(data)
  level = (1 - rate)/2

  xss = sorted(data)

  low = xss[int(n*level)]
  high = xss[int(n*(1-level))]
  
  return (low, high)

def filterList(data, mask):
  return [x for i, x in enumerate(data) if mask[i]]

def plotPair(data, feature_a, feature_b):

  ax_a_ben = []
  ax_b_ben = []
  ax_a_mal = []
  ax_b_mal = []

  for x, tag in data:
    if tag == 0:
      ax_a_ben.append(x[feature_a])
      ax_b_ben.append(x[feature_b])
    else:
      ax_a_mal.append(x[feature_a])
      ax_b_mal.append(x[feature_b])

  # drop top/bottom parts of data
  mask = [True] * len(data)

  for xs in [ax_a_ben, ax_b_ben, ax_a_mal, ax_b_mal]:
    lo, hi = levelTopBottom(xs, 0.9)
    for i, x in enumerate(xs):
      mask[i] &= (x >= lo and x <= hi)

  ax_a_ben = filterList(ax_a_ben, mask)
  ax_b_ben = filterList(ax_b_ben, mask)
  ax_a_mal = filterList(ax_a_mal, mask)
  ax_b_mal = filterList(ax_b_mal, mask)

  plt.plot(ax_a_ben, ax_b_ben, "gx")
  plt.plot(ax_a_mal, ax_b_mal, "r+")

  plt.show()

def plotTriple(data, feature_a, feature_b, feature_c):
  ax_a_ben = []
  ax_b_ben = []
  ax_c_ben = []
  ax_a_mal = []
  ax_b_mal = []
  ax_c_mal = []

  for x, tag in data:
    if tag == 0:
      ax_a_ben.append(x[feature_a])
      ax_b_ben.append(x[feature_b])
      ax_c_ben.append(x[feature_c])
    else:
      ax_a_mal.append(x[feature_a])
      ax_b_mal.append(x[feature_b])
      ax_c_mal.append(x[feature_c])

  # drop top/bottom parts of data
  mask = [True] * len(data)

  for xs in [ax_a_ben, ax_b_ben, ax_c_ben, ax_a_mal, ax_b_mal, ax_c_mal]:
    lo, hi = levelTopBottom(xs, 0.9)
    for i, x in enumerate(xs):
      mask[i] &= (x >= lo and x <= hi)

  ax_a_ben = filterList(ax_a_ben, mask)
  ax_b_ben = filterList(ax_b_ben, mask)
  ax_c_ben = filterList(ax_c_ben, mask)
  ax_a_mal = filterList(ax_a_mal, mask)
  ax_b_mal = filterList(ax_b_mal, mask)
  ax_c_mal = filterList(ax_c_mal, mask)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  ax.scatter(ax_a_ben, ax_b_ben, ax_c_ben, c="g", marker="x")
  ax.scatter(ax_a_mal, ax_b_mal, ax_c_mal, c="r", marker="+")

  plt.show()

def main():
  data = readData()

  for i in range(0, 13):
    for j in range(i+1, 14):
      for k in range(j+1, 15):
        print((i, j, k))
        #print((i, j))
        plt.close()
        plotTriple(data, i, j, k)
        #plotPair(data, i, j)

  pass

if __name__ == "__main__":
  main()

# vim: tabstop=2 shiftwidth=2 expandtab
