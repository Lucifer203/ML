## Implementing a multilayer perceptron

import numpy as np 

def sigmoid(z):
    return 1./(1+np.exp(-z))

def int_to_onehot(y,num_labels):
    ary = np.zeros((y.shape[0],num_labels))
    for i,val in enumerate(y):
        ary[i,val] = 1
    
    return ary 

# y = np.array([1,2,2,3,1,1,3,3,2,1])
# label = 3

# print(int_to_onehot(y,label))

class NeuralNetMLP:

    def __init__(self,num_features,num_hidden,
                 num_classes,random_seed=123):
        super().__init__()

        self.num_classes = num_classes

        #hidden
        rng = np.random.RandomState(random_seed)

        self.weight_h = rng.normal(loc=0.0,scale=0.1,size=(num_hidden,num_features))
        self.bias_h = np.zeros(num_hidden)

        ##output
        self.weight_out = rng.normal(loc=0.0,scale=0.1,size=(num_classes,num_hidden))
        self.bias_out = np.zeros(num_classes)

    
    def forward(self,X):
        """
        Hidden Layer
        input_dim : [n_hidden,n_features]
                dot [n_features,n_hidden].T
        output dim: [n_examples,n_hidden]
        """
        z_h = np.dot(X,self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        #for output layer
        z_out = np.dot(a_h,self.weight_out.T)+self.bias_out
        a_out = sigmoid(z_out)
        return a_h,a_out
    
    def backward(self,x,a_h,a_out,y):

        #################
        ## Output layer weights
        ##################

        #one -hot encoding

        y_onehot = int_to_onehot(y,self.num_classes)

        ## dLoss/dOutWeights

        d_loss__d_a_out = 2.*(a_out-y_onehot)/y.shape[0]

##sigmoid derivative
        d_a_out__d_z_out = a_out * (1. - a_out)

        delta_out = d_loss__d_a_out * d_a_out__d_z_out

        d_z_out__dw_out = a_h

        d_loss__dw_out = np.dot(delta_out.T,d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out,axis=0)

        d_z_out__a_h = self.weight_out

        d_loss__a_h = np.dot(delta_out,d_z_out__a_h)

        d_a_h__d_z_h = a_h * (1. - a_h)

        d_z_h__d_w_h = x

        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T,
                               d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h*d_a_h__d_z_h),axis=0)

        return (d_loss__dw_out,d_loss__db_out,
                d_loss__d_w_h,d_loss__d_b_h)

model = NeuralNetMLP(num_features=28*28,num_hidden=50,num_classes=10)

