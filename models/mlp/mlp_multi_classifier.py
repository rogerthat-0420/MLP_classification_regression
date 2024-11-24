# Create a separate MLP class specifically designed for handling multi-label classification tasks. 

import numpy as np

class MLP_Multi_Classifier:

    def __init__(self, lr=0.01, act_fun="relu", optimizer="sgd", num_hidden=2, num_neurons=[8,16],batch_size=32, num_epochs=100, seed=42):
        self.lr = lr
        self.optimizer = optimizer
        self.num_hidden = num_hidden
        self.num_neurons = num_neurons
        self.batch_size=batch_size
        self.num_epochs=num_epochs
        self.seed=seed
        self.act_fun = act_fun

        if act_fun == "relu":
            self.activation_func = self.relu
            self.activation_derivative=self.relu_derivative
        elif act_fun == "sigmoid":
            self.activation_func = self.sigmoid
            self.activation_derivative=self.sigmoid_derivative
        elif act_fun == "tanh":
            self.activation_func = self.tanh
            self.activation_derivative=self.tanh_derivative
        elif act_fun == "linear":
            self.activation_func = self.linear
            self.activation_derivative=self.linear
        
        self.X=None
        self.Y=None
        self.layer_act=None
        self.layer_z=None
        self.weights=None
        self.biases=None
        self.layer_sizes=None
    
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
        X=self.sigmoid(X)
        self.layer_act[-1]=X

        return self.layer_act[-1]
    
    def backward_pass(self,Y,batch_size):

        grad_w=[np.zeros_like(weight) for weight in self.weights]
        grad_b=[np.zeros_like(bias) for bias in self.biases]
        # here Y is the corresponding true labels of the batch of X samples provided in the forward pass

        error=(self.layer_act[-1]-Y)*self.sigmoid_derivative(self.layer_z[-1])
        # derivaive of loss function with respect to the output layer using chain rule.
        
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
    
    def multi_label_loss(self,Y_true,Y_pred):
        num_samples=Y_true.shape[0]
        loss_individual=np.zeros(num_samples)

        for i in range(num_samples):
            loss_individual[i]=-np.sum(Y_true[i]*np.log(Y_pred[i])+(1-Y_true[i])*np.log(1-Y_pred[i]))
        
        return np.mean(loss_individual)
    
    def hamming_accuracy(self,Y_true,Y_pred):
        num_samples=Y_true.shape[0]
        accuracy_individual=np.zeros(num_samples)

        for i in range(num_samples):
           binary_pred=(Y_pred[i]>0.5).astype(int)
           accuracy_individual[i]=np.sum(binary_pred==Y_true[i])
        
        hamming_accuracy=np.sum(accuracy_individual)/(num_samples*Y_true.shape[1])
        return hamming_accuracy
    
    def fit(self, X_train, Y_train, X_val, Y_val):
        self.X=X_train
        self.Y=Y_train
        self.layer_sizes=np.array([X_train.shape[1]]+self.num_neurons+[Y_train.shape[1]])
        self.initialize_params()
    
    def fit_train(self, X_train, Y_train, X_val, Y_val):
        self.X=X_train
        self.Y=Y_train
        self.layer_sizes=np.array([X_train.shape[1]]+self.num_neurons+[Y_train.shape[1]])
        self.initialize_params()

        n_samples=self.X.shape[0]
        patience_counter=0
        best_loss=float('inf')
        best_weights=None

        if self.optimizer =='gd':
            batch_size=n_samples
        elif self.optimizer=='sgd':
            batch_size=1
        elif self.optimizer=='mini-batch':
            batch_size=self.batch_size
        
        for epoch in range(self.num_epochs):
            indices=np.random.permutation(n_samples)
            X_train_shuffled=self.X[indices]
            Y_train_shuffled=self.Y[indices]

            for i in range(0,n_samples,batch_size):
                batch_end=min(i+batch_size,n_samples)
                X_batch=X_train_shuffled[i:batch_end]
                Y_batch=Y_train_shuffled[i:batch_end]

                self.init_layers(X_batch.shape[0])
                Y_pred=self.forward_pass(X_batch)
                grad_w,grad_b=self.backward_pass(Y_batch,batch_size)
                self.update_params(grad_w,grad_b)
            
            self.init_layers(X_val.shape[0])
            y_val_pred=self.forward_pass(X_val)
            val_loss=self.multi_label_loss(Y_val,y_val_pred)
            hamming_val=self.hamming_accuracy(Y_val,y_val_pred)
            # print(f"Epoch {epoch+1}/{self.num_epochs} - Val Loss: {val_loss:.4f} - Hamming Accuracy: {hamming_val:.4f}")

            if val_loss<best_loss:
                best_loss=val_loss
                best_weights=(self.weights.copy(),self.biases.copy())
                patience_counter=0
            
            else:
                patience_counter+=1
            
            if patience_counter>=5:
                if best_weights is not None:
                    self.weights,self.biases=best_weights
                #     print(f"Restoring best weights from epoch {epoch+1-patience_counter}")
                # print(f"Early stopping at epoch {epoch+1}")
                break
    
    def predict(self,X_test):
        self.init_layers(X_test.shape[0])
        y_test_pred = self.forward_pass(X_test)
        binary_pred = (y_test_pred > 0.5).astype(int)
        return binary_pred
