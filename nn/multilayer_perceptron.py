import numpy as np
import matplotlib.pyplot as plt
import time

traindata1 = [ ([2,1], 0),
              ([1,1], 0),
              ([0,0], 0),
              ([0.8,1.4], 0),
              ([2.5,2], 0),
              ([0,1], 1),
              ([1,0], 1),
              ([1.5,2], 1),
              ([1.6, 1.6], 1),
              ([0.6,2.2], 1),
              ([-1,-1], 1) ]

traindata21 = [([np.random.rand()*5-1.5,np.random.rand()*5-1.5], 0) for i in range(10)]
traindata22 = [([np.random.rand()*5-1.5,np.random.rand()*5-1.5], 1) for i in range(10)]
traindata2 = traindata21 + traindata22

traindata = traindata2

ls = [2, 8, 8, 8, 2]

plt.ion()
fig = plt.figure()
graph1 = fig.add_subplot(111)

def newWeights():
  return [np.random.rand(ls[i], ls[i-1]+1)*2-1 for i in range(1, len(ls))]

def ReLU(x):
  return max(x, 0)

def dReLU(x):
  if x < 0:
    return 0
  else:
    return 1

def sigmoid(x):
  return 1/(1+np.exp(-x))

def dSigmoid(x):
  return (1-sigmoid(x))*sigmoid(x)

def forward(x, W):
  out = [np.zeros(ls[i]) for i in range(0, len(ls))]
  out[0] = x
  for i in range(1, len(out)):
    inp = np.concatenate(([1],out[i-1]))
    out[i] = np.matmul(W[i-1], inp)
    for j,v in enumerate(out[i]):
      out[i][j] = ReLU(out[i][j])
  return out

def Err(x, y):
  return np.sum(np.power(np.subtract(x, y), 2))/2

def dErr(x, y, n):
  return x[n] - y[n]

def backprop(nn_out, cl_expected, W, rate):
  gr_out = [np.zeros(ls[i]+1) for i in range(0, len(ls))]

  expected = np.zeros(ls[-1])
  expected[cl_expected] = 1

  for i in range(0, ls[-1]):
    gr_out[-1][i+1] = dErr(nn_out[-1], expected, i)
    #gr_out[-1][i+1] = dLoss_i(nn_out[-1], i, cl_expected)

  for i in range(len(ls)-1, 0, -1):
    x = np.concatenate(([1],nn_out[i-1]))
    for j in range(0, ls[i]):
      dL_x_ds = gr_out[i][j+1]*dReLU(np.dot(x, W[i-1][j]))

      gr_out[i-1] = np.add(gr_out[i-1], dL_x_ds*W[i-1][j])
      W[i-1][j] = np.subtract(W[i-1][j], np.multiply(x, rate*dL_x_ds))
      # for k in range(0, ls[i-1]+1):
      #   dxk = dL_x_ds*W[i-1][j][k]
      #   gr_out[i-1][k] += dxk
      #   dwjk = dL_x_ds*x[k]
      #   W[i-1][j][k] -= dwjk*rate

def loss_i(y, n):
  # softmax
  bias = np.max(y)
  return -np.log(np.exp(y[n]-bias)/np.sum(np.exp(y-bias)))

def loss(W):
  acc = 0
  for item in traindata:
    y = forward(item[0], W)[-1]
    acc += loss_i(y, item[1])
  # TODO: regularization
  return acc #/len(traindata)

def lossE(W):
  acc = 0
  for item in traindata:
    y = forward(item[0], W)[-1]
    expected = np.zeros(ls[-1])
    expected[item[1]] = 1
    acc += Err(y, expected)
  return acc

def dLoss_i(y, k, n):
  h = 0.001
  l_l = loss_i(y, n)
  y[k] += h
  l_r = loss_i(y, n)
  y[k] -= h

  return (l_r-l_l)/h
  #bias = np.max(y)
  #S = np.sum(np.exp(y-bias))

  #if k != n:
  #  return -np.exp(y[k])/S
  #else:
  #  return 1-np.exp(y[k])/S

def getGradAtPoint(f, x):
  grad = [np.zeros(li.shape) for li in x]

  h = 0.0001

  for i,xi in enumerate(x):
    for j,xij in enumerate(xi):
      for k,xijk in enumerate(xij):
        x[i][j][k] = xijk - h
        fx_l = f(x)

        x[i][j][k] = xijk + h
        fx_r = f(x)
        # return back value
        x[i][j][k] = xijk

        grad[i][j][k] = (fx_r-fx_l)/(2*h)

  return grad

def learn_random(n):
  bestW = np.multiply(newWeights(), 0.1)
  bestLoss = 100000
  for k in range(n):
    newW = newWeights()
    c_loss = loss(newW)
    if k % 100 == 0:
      print("sample {0}, loss: {1}".format(k, c_loss))
    if c_loss < bestLoss:
      bestLoss = c_loss
      bestW = newW

  print("Best loss {0}".format(bestLoss))
  return bestW

def learn_random_gradient(n):
  W = learn_random(int(n/10))
  bestLoss = 100000
  for k in range(n):
    newW = np.add(W, np.multiply(newWeights(), 0.01))
    c_loss = loss(newW)
    if k % 100 == 0:
      print("sample {0}, loss: {1}".format(k, c_loss))
    if k % 1000 == 0:
      plot(W, 32, False)
    if c_loss < bestLoss:
      bestLoss = c_loss
      W = newW

  print("Best loss {0}".format(bestLoss))
  return W

def learn_num_gradient(n):
  W = learn_random(int(n/3))
  bestLoss = 100000
  learn_rate = 0.005
  for k in range(n):
    grad = getGradAtPoint(loss, W);
    W = np.subtract(W, np.multiply(grad, learn_rate))
    c_loss = loss(W)
    if k % 100 == 0:
       print("sample {0}, loss: {1}".format(k, c_loss))
    bestLoss = c_loss

  print("Best loss {0}".format(bestLoss))
  return W

def learn_backprop(n):
  W = learn_random(int(n/3))
  for k in range(n):
    for item in traindata:
      nn_out = forward(item[0], W)
      backprop(nn_out=nn_out, cl_expected=item[1], W=W, rate=0.0001)

    c_loss = loss(W)
    if k % 10 == 0:
      print("sample {0}, loss: {1}".format(k, c_loss))
      bestLoss = c_loss
    if k % 100 == 0:
      plot(W, 32, False)

  print("Best loss {0}".format(bestLoss))
  return W

def plot(W, n, block):
  xs = np.linspace(-2, 4, num=n)
  ys = np.linspace(-2, 4, num=n)
  xxs, yys = np.meshgrid(xs, ys)
  zs = np.zeros((xs.size, ys.size))

  for i,x in enumerate(xs):
    for j,y in enumerate(ys):
      z = forward([x,y], W)[-1]
      if z[0] > z[1]:
        zs[j, i] = 0
      else:
        zs[j, i] = 1

  #plt.gcf().clear();
  graph1.clear()
  graph1.pcolor(xxs, yys, zs, cmap="winter")
  #graph1.colorbar()
  for item in traindata:
    ptx = [item[0][0]]
    pty = [item[0][1]]
    if item[1] == 0:
      graph1.plot(ptx, pty, "ro")
    else:
      graph1.plot(ptx, pty, "gs")
  fig.canvas.draw()
  #plt.show(block=True)

def main():
  #W = learn_random_gradient(10000)
  W = learn_backprop(5000)

  plot(W, 128, True)
  time.sleep(10)

if __name__ == '__main__':
  main()

