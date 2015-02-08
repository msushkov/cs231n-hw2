import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  
  # D = product of all di's
  D = np.prod(x.shape[1:])
  X = x.reshape((x.shape[0], D))

  out = np.dot(X, w) + b

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  
  N = x.shape[0]

  # D = product of all di's
  D = np.prod(x.shape[1:])
  X = x.reshape((N, D))

  dx = np.dot(dout, w.T).reshape(x.shape)

  db = np.dot(np.ones((1, N)), dout).reshape(b.shape)

  dw = np.dot(X.T, dout)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  
  # ReLU(x) is max(0, x)
  out = np.maximum(x, 0)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  
  # want dReLU/dx * d(next layer)/dReLU (this is dout)
  # dReLU(x)/dx is I(x >= 0)
  dReLU = x.copy()
  dReLU[dReLU >= 0] = 1
  dReLU[dReLU < 0] = 0
  dx = np.multiply(dout, dReLU)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width WW.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  pad = conv_param['pad']
  stride = conv_param['stride']

  Hprime = 1 + (H + 2 * pad - HH) / stride
  Wprime = 1 + (W + 2 * pad - WW) / stride

  assert Hprime == int(Hprime)
  assert Wprime == int(Wprime)

  out = np.empty((N, F, Hprime, Wprime))

  # for each image
  for i in xrange(N):
    # get the current image and pad it with 0's
    X = np.pad(x[i], ((0, 0), (pad, pad), (pad, pad)), 'constant')

    # X is of shape (C, H + 2pad, W + 2pad)

    # for each filter
    for f in xrange(F):
      W = w[f]
      b_curr = b[f]

      # W is of shape (C, HH, WW)
      # b_curr is a constant

      result = np.empty((Hprime, Wprime))

      for r in xrange(Hprime):
        for c in xrange(Wprime):
          r_start = r * stride
          r_end = r_start + HH

          c_start = c * stride
          c_end = c_start + WW

          x_curr = X[:, r_start:r_end, c_start:c_end]

          # x_curr is of shape (C, HH, WW)

          result[r, c] = np.sum(np.multiply(x_curr, W)) + b_curr

      # result is of shape (Hprime, Wprime)

      out[i, f] = result

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  
  (x, w, b, conv_param) = cache

  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  pad = conv_param['pad']
  stride = conv_param['stride']

  Hprime = 1 + (H + 2 * pad - HH) / stride
  Wprime = 1 + (W + 2 * pad - WW) / stride

  # dout shape is (N, F, Hprime, Wprime)

  assert dout.shape == (N, F, Hprime, Wprime)

  # dx shape is (N, C, H, W)
  # dw shape is (F, C, HH, WW)
  # db shape is (F,)

  dw = np.zeros(w.shape)
  db = np.zeros(b.shape)

  # pad x
  x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

  dx_padded = np.zeros(x_padded.shape)

  for i in xrange(N):
    for c in xrange(C):
      # X is shape (H, W)
      X = x_padded[i, c]
      
      for f in xrange(F):
        # W is shape (HH, WW)
        W = w[f, c]

        # indices in the output volume
        for r1 in xrange(Hprime):
          # indices in the input X
          r1_start = r1 * stride
          r1_end = r1_start + HH

          for r2 in xrange(Wprime):
            # indices in the input X
            r2_start = r2 * stride
            r2_end = r2_start + WW

            x_curr = X[r1_start:r1_end, r2_start:r2_end].copy()

            dx_padded[i, c, r1_start:r1_end, r2_start:r2_end] += W * dout[i, f, r1, r2]

            dw[f, c, 0:HH, 0:WW] += x_curr * dout[i, f, r1, r2]

            db[f] += dout[i, f, r1, r2] # doing this C too many times

  dx = dx_padded[:, :, pad:-pad, pad:-pad]

  db /= C

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  
  N, C, H, W = x.shape

  ht = pool_param['pool_height']
  wd = pool_param['pool_width']
  stride = pool_param['stride']

  Hprime = 1 + (H - ht) / stride
  Wprime = 1 + (W - wd) / stride

  out = np.empty((N, C, Hprime, Wprime))

  for i in xrange(N):
    for c in xrange(C):
      # get the current image
      X = x[i, c]

      # X is of shape (H, W)

      for r1 in xrange(Hprime):
        for r2 in xrange(Wprime):
          r1_start = r1 * stride
          r1_end = r1_start + ht

          r2_start = r2 * stride
          r2_end = r2_start + wd

          x_curr = X[r1_start:r1_end, r2_start:r2_end]

          out[i, c, r1, r2] = np.max(x_curr)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  
  (x, pool_param) = cache

  N, C, H, W = x.shape

  ht = pool_param['pool_height']
  wd = pool_param['pool_width']
  stride = pool_param['stride']

  Hprime = 1 + (H - ht) / stride
  Wprime = 1 + (W - wd) / stride

  # dx has shape (N, C, H, W)
  # dout has shape (N, C, Hprime, Wprime)

  dx = np.zeros(x.shape)

  for i in xrange(N):
    for c in xrange(C):
      # get the current image
      X = x[i, c]

      # X is of shape (H, W)

      global_offset_r1 = 0
      global_offset_r2 = 0

      for r1 in xrange(Hprime):
        for r2 in xrange(Wprime):
          r1_start = r1 * stride
          r1_end = r1_start + ht

          r2_start = r2 * stride
          r2_end = r2_start + wd

          x_curr = X[r1_start:r1_end, r2_start:r2_end].copy()
          curr_max = np.max(x_curr)

          # indicator on x_curr
          x_curr[x_curr != curr_max] = 0
          x_curr[x_curr == curr_max] = 1

          dx[i, c, r1_start:r1_end, r2_start:r2_end] += dout[i, c, r1, r2] * x_curr

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

