import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
gaussian_df = pd.read_csv( '/content/drive/My Drive/gaussian_data.csv' )
test_final_df =  gaussian_df.sample(frac=0.2, random_state=42)
gaussian_df = gaussian_df.drop( test_final_df.index )

X = gaussian_df.drop( "class", axis = 1 )
X_test_final = test_final_df.drop( "class", axis = 1 )

y = gaussian_df["class"]
y = pd.get_dummies(y)

y_test_final = test_final_df["class"]
y_test_final = pd.get_dummies(y_test_final)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

from google.colab import drive
drive.mount('/content/drive')

 def tanh(Z):
   """
   Z : non activated outputs
   Returns (A : 2d ndarray of activated outputs, df: derivative component wise)
   """
   A = np.empty(Z.shape)
   A = 2.0/(1 + np.exp(-2.0*Z)) - 1 # A = np.tanh(Z)
   df = 1-A**2
   return A,df

def softmax(z):
    shiftz = z - np.max(z)
    exps = np.exp(shiftz)
    return exps / np.sum(exps)

def softmax_gradient(z):
    """Computes the gradient of the softmax function."""
    Sz = softmax(z)
    D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
    return D

def cross_entropy_loss(p, y):
    """Cross-entropy loss between predicted and expected probabilities."""
    assert(p.shape == y.shape)
    return -np.sum(y * np.log(p))

def tr(x):
  x = np.array(  [[i] for i in x]  )
  return x
def max_row(row):
  """row: 1D arrayReturns: index of the maximum value in row"""
  return np.where(row == np.amax(row))[0][0]

def mse(a, b):
	error = 0.0
	for i in range(len(a)):
		error += (b[i] - a[i]) ** 2
	return error / len(a)
 
def convert(y_hat):
  y_max = np.max(y_hat)
  y = []
  for i in range(len(y_hat)):
    if y_hat[i] == y_max:
      y.append(1)
    else:
      y.append(0)
  return y

def plt_confusion_matrix( confusion_mtx = [], class_names = ['class-0', 'class-1', 'class-2'] ):
  plt.figure(figsize = (8,8))
  sns.set(font_scale=2) # label size
  ax = sns.heatmap(
      confusion_mtx, annot=True, annot_kws={"size": 30}, # font size
      cbar=False, cmap='Blues', fmt='d', 
      xticklabels=class_names, yticklabels=class_names)
  ax.set(title='', xlabel='Actual', ylabel='Predicted')
  plt.show()

class NeuralNet:
  """
    — un nombre de couches cachées et leur dimensions
    — une liste de matrices de poids
    — une liste de matrices de biais
    — une liste de matrices d’entrées pondérées
    — une liste de matrices d’activations
    — un taux d’apprentissage (η)
    — une fonction d’activation pour les unités des couches cachées
    — un nombre d’epoch pendant lequel entrainer le modéle

  """
  bias = []
  weights = []
  X_train, X_test = [], []
  y_train, y_test = [], []
  hidden_layer_sizes = []
  layer_sizes = []
  A, df = [],[]
  activation = ''
  learning_rate = 0.01
  epoch = 200

  def __init__(self, X_train = None, y_train = None, X_test = None, y_test = None, \
                hidden_layer_sizes = [4,3,2] , activation='identity', learning_rate=0.01, epoch=200):
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test
    self.hidden_layer_sizes = hidden_layer_sizes
    self.activation = activation
    self.learning_rate = learning_rate
    self.epoch = epoch
    self.layer_sizes =  [len ( X_train.columns )] + hidden_layer_sizes + [len( y_train.columns )] 
    self._weights_initialization()

  def _weights_initialization(self):
    self.A = [None] * ( len(self.hidden_layer_sizes) + 1 )
    self.df = [None] * ( len(self.hidden_layer_sizes) + 1 )
    for i in range (1,len(self.layer_sizes) ):
      self.weights.append( np.array( self.layer_sizes[i]*[ self.layer_sizes[i-1]*[None] ] )  )
      # self.weights[i-1] =  np.random.uniform(-.1,.1,[self.layer_sizes[i],self.layer_sizes[i-1]])
      self.weights[i-1] = np.random.normal(loc=0.0,scale=1.0 / np.sqrt(self.layer_sizes[i]),
                                            size=(self.layer_sizes[i], self.layer_sizes[i-1]))
      self.bias.append( np.array( self.layer_sizes[i]*[None] ) )
      self.bias[i-1] = np.random.normal(loc=0.0,scale=1.0, size= self.layer_sizes[i] )
      # self.bias[i-1] = np.random.uniform(-.1,.1, [ self.layer_sizes[i] ] )

  def feed_forward(self, X ,y):
    '''
    Implementation of the Feedforward
    '''
    g = lambda x: tanh(x)
    Z = [None] * len(self.layer_sizes)
    input_layer = X
    for i in range(len(self.hidden_layer_sizes) + 1):
      # Multiplying input_layer by weights for this layer
      Z[i + 1] = np.dot(self.weights[i],input_layer) + self.bias[i]
      # Activation Function
      if( i == len(self.hidden_layer_sizes) ):
        #For calculating the loss
        self.A[i] = softmax(Z[i + 1])
        self.df[i] = softmax_gradient( Z[i + 1] )
      else:
        self.A[i],self.df[i] = g(Z[i + 1])
      # Current output_layer will be next input_layer
      input_layer = self.A[i]
    error = cross_entropy_loss(self.A[-1],y)
    return error , self.A[-1]
  
  def back_propagation(self, X, y):
    # Initialization
    delta = [None] * (len(self.hidden_layer_sizes) + 1)
    dW = [None] * (len(self.hidden_layer_sizes) + 1)
    db = [None] * (len(self.hidden_layer_sizes) + 1)
    # Calculation for last(output) layer
    delta[-1] = np.dot((self.A[-1] - y),self.df[-1])
    dW[-1] = np.transpose(delta[-1] * tr(self.A[-2]))
    db[-1] = delta[-1]
    # Calculation for the rest
    for l in range(len(self.hidden_layer_sizes) -1 , -1, -1):
      delta[l] = np.multiply(np.dot(self.weights[l + 1].T, delta[l + 1]), self.df[l])
      if( l == 0 ):
        dW[l] = np.transpose(delta[l] * tr(X))
      else:
        dW[l] = np.transpose(delta[l] * tr(self.A[l-1]))
      db[l] = delta[l]
    # Updating the Weights and Biases for next epoch
    for l in range(len(self.hidden_layer_sizes) + 1):
      self.weights[l] = self.weights[l] - self.learning_rate*dW[l]
      self.bias[l] = self.bias[l] - self.learning_rate*db[l]
        
  def predict(self, X , y):
    accuracies = []
    prediction = []
    for _X, _y in zip(X.values, y.values):
      err , A = self.feed_forward(_X,_y)
      prediction.append(A)
      accuracies.append( self.accuracy( A, _y ) )
    print("Accuracy: "+ str( np.mean(accuracies) ))
    return accuracies,prediction

  def train (self, X,y):
    mean_error = []
    x = 0
    for i in range(200):
      error = []
      for _X, _y in zip(X.values, y.values):
        x = x + 1
        err, A = self.feed_forward(_X, _y)
        error.append(err)
        self.back_propagation(_X,_y)
      mean_error.append( np.mean( error ) )
      X, y = shuffle(X, y)
    return mean_error

  def accuracy( self, y_pred,y_real ):
    pred = convert(y_pred)
    return (pred == y_real).mean()

  def confusion_matrix(self, y_pred, y_real):
    for i in range(len(y_pred)):
      y_pred[i] = convert(y_pred[i])

    n = y_real.shape[1]
    m = np.array( n*[ n*[0] ] )
    for i in range(len(y_real)):
      if( y_real.values[i][0] == 1 ):
        m[0] += y_pred[i]
      elif(y_real.values[i][1] == 1 ):
        m[1] += y_pred[i]
      else:
        m[2] += y_pred[i]
    plt_confusion_matrix(m)
    return m

nn = NeuralNet(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test)

gr_train = nn.train(nn.X_train , nn.y_train)
gr_test = nn.train(nn.X_test , nn.y_test)

acc,prr = nn.predict(X_test_final,y_test_final)

conf_mx = nn.confusion_matrix(prr,y_test_final)
print()
plt.plot(gr_train, label='train')  
plt.plot(gr_test, label='test')
plt.legend()
plt.show()
print("Last error: "+ str(gr_train[-1]))

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
