import theano
import theano.tensor as T
import numpy as np
from sklearn import linear_model, datasets
from matplotlib import pyplot as plt
# Generate a dataset and plot it
np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)

plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

# Train the logistic rgeression classifier
clf = linear_model.LogisticRegressionCV()
clf.fit(X, y)
 
# Plot the decision boundary
plot_decision_boundary(lambda x: clf.predict(x))
plt.title("Logistic Regression")
plt.show()

# nn_input_dim = 100
# nn_hdim = 50
# nn_output_dim = 2


# # Our data vectors
# X = T.matrix('X') # matrix of doubles
# y = T.lvector('y') # vector of int64

# W1 = theano.shared(np.random.randn(nn_input_dim, nn_hdim), name='W1')
# b1 = theano.shared(np.zeros(nn_hdim), name='b1')
# W2 = theano.shared(np.random.randn(nn_hdim, nn_output_dim), name='W2')
# b2 = theano.shared(np.zeros(nn_output_dim), name='b2')


# print (y * 2).eval({y : [1,1] })