import numpy as np
import matplotlib.pyplot as plt

traindata = [ ([2,1], 0),
              ([1,1], 0),
              #([0,0], 0),
              ([0.8,1.4], 0),
              ([2.5,2], 0),
              ([0,1], 1),
              ([1,0], 1),
              ([1.5,2], 1),
              ([1.6, 1.6], 1),
              ([0.6,2.2], 1),
              ([-1,-1], 1) ];

ls = [2, 2]

out = [np.zeros(ls[i]) for i in range(0, len(ls))]
gr_out = [np.zeros(ls[i]+1) for i in range(0, len(ls))]

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
  out[0] = x
  for i in range(1, len(out)):
    inp = np.concatenate(([1],out[i-1]))
    out[i] = np.matmul(W[i-1], inp)
    for j,v in enumerate(out[i]):
      out[i][j] = ReLU(out[i][j])
  return out[-1]

def backprop(nn_out, cl_expected, W, rate):
  gr_out[-1][1:] = np.ones(ls[-1])
  for i in range(len(ls)-1, 0, -1):
    for j in range(0, ls[i]):
      for k in range(0, ls[i-1]+1):
        print("{0}, {1}, {2}".format(i, j, k))
        x = np.concatenate(([1],out[i-1]))
        dxk = gr_out[i][j]*dReLU(np.dot(x, W[i-1][j]))*W[i-1][j][k]
        gr_out[i-1][j] = dxk
        dwjk = gr_out[i][j]*dReLU(np.dot(x, W[i-1][j]))*x[k]
        W[i-1][j][k] -= dwjk*rate
  print(gr_out)
  exit(0)

def loss_i(y, n):
  # softmax
  bias = np.max(y)
  return -np.log(np.exp(y[n]-bias)/np.sum(np.exp(y-bias)))

def loss(W):
  acc = 0
  for item in traindata:
    y = forward(item[0], W)
    acc += loss_i(y, item[1])
  # TODO: regularization
  return acc

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
  bestW = np.multiply(newWeights(), 0.01)
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
  bestW = learn_random(n/10)
  bestLoss = 100000
  for k in range(n):
    newW = np.add(bestW, np.multiply(newWeights(), 0.01))
    c_loss = loss(newW)
    if k % 100 == 0:
      print("sample {0}, loss: {1}".format(k, c_loss))
    if c_loss < bestLoss:
      bestLoss = c_loss
      bestW = newW

  print("Best loss {0}".format(bestLoss))
  return bestW

def learn_num_gradient(n):
  W = learn_random(n/3)
  bestLoss = 100000
  learn_rate = 0.01
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
  W = learn_random(n/10)
  for k in range(n):
    for item in traindata:
      y = forward(item[0], W)
      backprop(nn_out=y, cl_expected=item[1], W=W, rate=0.01)

    c_loss = loss(W)
    if k % 1 == 0:
      print("sample {0}, loss: {1}".format(k, c_loss))
      bestLoss = c_loss

  print("Best loss {0}".format(bestLoss))
  return W

def main():
  # W = learn_random_gradient(50000)
  W = learn_backprop(10)

  xs = np.linspace(-3, 3, num=100)
  ys = np.linspace(-3, 3, num=100)
  xxs, yys = np.meshgrid(xs, ys)
  zs = np.zeros((xs.size, ys.size))

  for i,x in enumerate(xs):
    for j,y in enumerate(ys):
      z = forward([x,y], W)
      if z[0] > z[1]:
        zs[j, i] = 0
      else:
        zs[j, i] = 1

  plt.pcolor(xxs, yys, zs, cmap="winter")
  plt.colorbar()
  for item in traindata:
    ptx = [item[0][0]]
    pty = [item[0][1]]
    if item[1] == 0:
      plt.plot(ptx, pty, "bo")
    else:
      plt.plot(ptx, pty, "gs")
  #plt.show(block=False)

  plt.show()

if __name__ == '__main__':
  main()

