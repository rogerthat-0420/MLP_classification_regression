# MLP class using numpy and pandas 

import numpy as np

class MLP_Classifier:

    def __init__(self, lr=0.01, act_fun='relu', optimizer='sgd', num_hidden=2, num_neurons=[3,3],batch_size=32, num_epochs=100,seed=42,
                 type_class='multi'):
        self.lr=lr
        self.optimizer=optimizer
        self.num_hidden=num_hidden
        self.num_neurons=num_neurons
        self.batch_size=batch_size
        self.num_epochs=num_epochs
        self.seed=seed
        self.type_class=type_class
        self.act_fun=act_fun

        if act_fun=='relu':
            self.activation_func=self.relu
            self.activation_derivative=self.relu_derivative
        elif act_fun=='sigmoid':
            self.activation_func=self.sigmoid
            self.activation_derivative=self.sigmoid_derivative
        elif act_fun=='tanh':
            self.activation_func=self.tanh
            self.activation_derivative=self.tanh_derivative
        elif act_fun=='linear':
            self.activation_func=self.linear
            self.activation_derivative=self.linear

        self.X=None    
        self.Y=None
        self.hidden=None
        self.layer_act=None
        self.layer_z=None
        self.weights=None
        self.biases=None
        self.layer_sizes =None
    
    def set_lr(self, lr):
        self.lr=lr 
    
    def set_activation_func(self, act_fun):
        self.activation_func=act_fun
    
    def set_optimizer(self, optimizer):
        self.optimizer=optimizer
    
    def set_num_hidden(self, num_hidden):
        self.num_hidden=num_hidden
    
    def set_num_neurons(self, num_neurons):
        self.num_neurons=num_neurons
    
    def set_batch_size(self, batch_size):
        self.batch_size=batch_size
    
    def set_num_epochs(self, num_epochs):
        self.num_epochs=num_epochs
    
    def model_parameters(self):
        return {
            'learning_rate': self.lr,
            'activation_function': self.act_fun,
            'optimizer': self.optimizer,
            'num_hidden_layers': self.num_hidden,
            'neurons_per_layer': self.num_neurons,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs
        }
    
    def relu(self, X):
        return np.maximum(0,X)
    
    def relu_derivative(self, X):
        return np.where(X>0,1,0)

    def sigmoid(self, X):
        return 1./(1.+np.exp(-X))
    
    def sigmoid_derivative(self, X):
        return self.sigmoid(X)*(1-self.sigmoid(X))
    
    def tanh(self, X):
        return np.tanh(X)
    
    def tanh_derivative(self, X):
        return 1-self.tanh(X)**2
    
    def softmax(self, X):
        max_values=np.max(X,axis=1)
        max_values=max_values[:,np.newaxis]
        exponent=np.exp(X-max_values)
        exp_sum=np.sum(exponent,axis=1)
        exp_sum=exp_sum[:,np.newaxis]
        return exponent/exp_sum
    
    def linear(self, X):
        return X
    
    def init_layers(self,batch_size):
        self.layer_z=[np.empty((batch_size,layer)) for layer in self.layer_sizes]
        self.layer_act=[np.empty((batch_size,layer)) for layer in self.layer_sizes]
    
    def initialize_params(self):
        np.random.seed(self.seed)
        weights = []
        biases = []
        for i in range(self.layer_sizes.shape[0] - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i+1]
            lim=np.sqrt(2/input_size)
            weight_matrix = np.random.randn(input_size, output_size)*lim
            weights.append(weight_matrix)
            bias_vector = np.random.randn(1, output_size)
            biases.append(bias_vector)
        
        self.weights=weights
        self.biases=biases

    # def to_categorical(self, X):
    #     categorical=np.zeros((X.shape[0],self.Y.shape[1]))
    #     categorical[np.arange(X.shape[0]),X.argmax(axis=1)]=1
    #     return categorical
    
    def loss(self,y_pred,y):
        # Computing the loss along the rows, averaging along the number of samples
        loss=-np.sum(y*np.log(y_pred))/y.shape[0]
        return loss
    
    def binary_loss(self,y_pred,y):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss
    
    def forward_pass(self, X):
        # here X is a batch of samples - (N_samples * d_features)
        
        self.layer_act[0]=X
        self.layer_z[0]=X
        for i in range(len(self.weights)-1):

            X=np.dot(X,self.weights[i])+self.biases[i]
            self.layer_z[i+1]=X
            X=self.activation_func(X)
            self.layer_act[i+1]=X
        
        X=np.dot(X,self.weights[-1])+self.biases[-1]
        self.layer_z[-1]=X

        if(self.type_class=='binary'):
            X=self.sigmoid(X)
        else:
            X=self.softmax(X)

        self.layer_act[-1]=X
        return self.layer_act[-1]
    
    def backward_pass(self,Y,batch_size):

        grad_w=[np.zeros_like(weight) for weight in self.weights]
        grad_b=[np.zeros_like(bias) for bias in self.biases]
        # here Y is the corresponding true labels of the batch of X samples provided in the forward pass
        if self.type_class=='binary':
            error=(self.layer_act[-1]-Y)*self.layer_z[-1]*(1-self.layer_z[-1]) # derivaive of loss function with respect to the output layer using chain rule.
        else:
            error=(self.layer_act[-1]-Y) * 1
        
        grad_w[-1]=np.dot(self.layer_act[-2].T,error)/batch_size
        grad_b[-1]=np.sum(error,axis=0,keepdims=True)/batch_size

        for i in range(len(self.weights)-2,0,-1):
            error=np.dot(error,self.weights[i+1].T) * self.activation_derivative(self.layer_z[i+1])
            grad_w[i]=np.dot(self.layer_act[i].T,error)/batch_size
            grad_b[i]=np.sum(error,axis=0,keepdims=True)/batch_size
        
        error=np.dot(error,self.weights[1].T)*self.activation_derivative(self.layer_z[1])
        grad_w[0]=np.dot(self.layer_act[0].T,error)/batch_size
        grad_b[0]=np.sum(error,axis=0,keepdims=True)/batch_size
        return grad_w,grad_b

    def update_params(self,grad_w,grad_b):

        for i in range(len(self.weights)):
            self.weights[i]-=self.lr*grad_w[i]
            self.biases[i]-=self.lr*grad_b[i]
    
    def fit(self,X_train,Y_train,X_val,Y_val,patience=5):
        self.X=X_train
        num_classes=len(np.unique(Y_train))
        self.Y=np.squeeze(np.eye(num_classes)[Y_train.astype(int).reshape(-1)])
        input_layer_size=X_train.shape[1]
        output_layer_size=self.Y.shape[1]
        self.layer_sizes=np.array([input_layer_size]+self.num_neurons+[output_layer_size]) 
        self.initialize_params()

    def fit_train(self,X_train,Y_train,X_val,Y_val,patience=5):
        self.X=X_train
        num_classes=len(np.unique(Y_train))
        self.Y=np.squeeze(np.eye(num_classes)[Y_train.astype(int).reshape(-1)])
        input_layer_size=X_train.shape[1]
        output_layer_size=self.Y.shape[1]
        self.layer_sizes=np.array([input_layer_size]+self.num_neurons+[output_layer_size]) 
        self.initialize_params()
       
        n_samples=self.X.shape[0]
        best_loss=float('inf')
        patience_counter=0
        best_weights=None
        Y_val=np.squeeze(np.eye(num_classes)[Y_val.astype(int).reshape(-1)])

        if self.optimizer=='gd':
            batch_size=n_samples
        elif self.optimizer=='sgd':
            batch_size=1
        elif self.optimizer=='mini-batch':
            batch_size=self.batch_size

        val_loss_list=[]
        for epoch in range(self.num_epochs):
            indices = np.random.permutation(n_samples)
            X_train_shuffled = self.X[indices]
            Y_train_shuffled = self.Y[indices]

            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                X_batch = X_train_shuffled[i:batch_end]
                Y_batch = Y_train_shuffled[i:batch_end]

                self.init_layers(X_batch.shape[0])
                y_pred = self.forward_pass(X_batch)
                grad_w,grad_b=self.backward_pass(Y_batch,batch_size)
                self.update_params(grad_w,grad_b)

            # self.check_gradients(X_train_shuffled,Y_train_shuffled)

            
            # if self.type_class=='binary':
            #     current_loss = self.binary_loss(y_pred, Y_batch)
            # else:
            #     current_loss = self.loss(y_pred, Y_batch)

            # print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {current_loss}')
            
            self.init_layers(X_val.shape[0])
            y_val_pred = self.forward_pass(X_val)
            
            if self.type_class=='binary':
                val_loss = self.binary_loss(y_val_pred, Y_val)
            else:
                val_loss = self.loss(y_val_pred, Y_val)
            
            val_loss_list.append(val_loss)

            # print(f'Epoch {epoch+1}/{self.num_epochs}, Validation Loss: {val_loss}')

            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = (self.weights.copy(), self.biases.copy())
                patience_counter = 0
            
            else:
                patience_counter+=1
                # print(f'Patience counter: {patience_counter}/{patience}')
            
            if patience_counter >= patience:
                if best_weights is not None:
                    self.weights, self.biases = best_weights
                    print(f"Restoring best weights from epoch {epoch+1-patience}")
                print(f"Early stopping at epoch {epoch+1}")
                break
        return val_loss_list
            
    def predict(self,X_test):
        self.init_layers(X_test.shape[0])
        y_test_pred=self.forward_pass(X_test)
        predictions=np.argmax(y_test_pred,axis=1)
        return predictions
    
    def numerical_gradient(self,X,Y,e=1e-6):
        num_grad_w=[]
        num_grad_b=[]

        for i in range(len(self.weights)):
            grad_w=np.zeros_like(self.weights[i])
            grad_b=np.zeros_like(self.biases[i])

            for j in range(self.weights[i].shape[0]):
                for k in range(self.weights[i].shape[1]):

                    self.init_layers(X.shape[0])
                    self.weights[i][j,k]+=e
                    y_pred=self.forward_pass(X)
                    loss1=self.loss(y_pred,Y)

                    self.init_layers(X.shape[0])
                    self.weights[i][j,k]-=2*e
                    y_pred=self.forward_pass(X)
                    loss2=self.loss(y_pred,Y)

                    self.weights[i][j,k]+=e

                    grad_w[j,k]=(loss1-loss2)/(2*e)
            
            for j in range(self.biases[i].shape[1]):

                self.biases[i][0,j]+=e
                self.init_layers(X.shape[0])
                y_pred=self.forward_pass(X)
                loss1=self.loss(y_pred,Y)

                self.biases[i][0,j]-=2*e
                self.init_layers(X.shape[0])
                y_pred=self.forward_pass(X)
                loss2=self.loss(y_pred,Y)

                self.biases[i][0,j]+=e

                grad_b[0,j]=(loss1-loss2)/(2*e)
            
            num_grad_w.append(grad_w)
            num_grad_b.append(grad_b)
        
        return num_grad_w,num_grad_b
    
    def check_gradients(self,X,Y,threshold=1e-5):
        self.init_layers(X.shape[0])
        y_pred=self.forward_pass(X)
        grad_w,grad_b=self.backward_pass(Y,X.shape[0])

        num_grad_w,num_grad_b=self.numerical_gradient(X,Y)

        for i in range(len(grad_w)):
            difference_w = np.linalg.norm(grad_w[i] - num_grad_w[i])
            difference_b = np.linalg.norm(grad_b[i] - num_grad_b[i])
            
            print(f'Layer {i + 1}:')
            print(f'  Weight Gradient Difference: {difference_w}')
            print(f'  Bias Gradient Difference: {difference_b}')
            
            if difference_w > threshold or difference_b > threshold:
                print('Gradient check failed!')
            else:
                print('Gradient check passed!')
                    







