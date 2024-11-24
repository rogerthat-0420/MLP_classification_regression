# In this part, we will make an autoencoder class. Make sure the dimension that your dataset is reduced to using the encoder is the same as the optimal dimensions you reduced it to using PCA in assignment 2. Make an AutoEncoder class which calls the MLP regression class to make neural network model which takes in an input vector, reduces to n-dimensions and reconstructs it back to the original vector. The class must have the following 3 methods:
# • The initialisation method which initialises the model. (5)
# • The fit method which trains the model. (5)
# • The get latent method which will return the reduced dataset.
import os
import sys
import numpy as np

class AutoEncoder:

    def __init__(self, latent_dim,batch_size=256,num_hidden=5,num_neurons=[16,8,8,16],lr=0.01, act_fun='tanh', optimizer='gd',num_epochs=100):

        module_path_mlp_regression = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/mlp'))
        sys.path.append(module_path_mlp_regression)
        from mlp_regression import MLP_Regression

        
        if num_hidden%2==0:
            self.index=(num_hidden+2)//2
            cut=len(num_neurons)//2+1
            encoder_neurons=num_neurons[:cut]
            decoder_neurons = num_neurons[cut:]
        else:
            self.index=(num_hidden+2)//2
            cut=len(num_neurons)//2
            encoder_neurons=num_neurons[:cut]
            decoder_neurons = num_neurons[cut:]
        
        hidden_neurons = encoder_neurons + [latent_dim] + decoder_neurons
        # input -> encoder -> latent -> decoder -> output
        self.autoencoder=MLP_Regression(lr=lr, batch_size=batch_size,act_fun=act_fun, optimizer=optimizer,num_hidden=num_hidden, num_neurons=hidden_neurons, num_epochs=num_epochs)

    def fit(self, X_train,X_val):
        Y_train = X_train
        Y_val = X_val
        self.autoencoder.fit(X_train,Y_train,X_val,Y_val)

        if self.autoencoder.optimizer=='gd':
            batch_size=X_train.shape[0]
        elif self.autoencoder.optimizer=='mini-batch':
            batch_size=self.autoencoder.batch_size
        else:
            batch_size=1
        
        n_samples = X_train.shape[0]
        best_loss=float('inf')
        patience_counter=0
        best_weights=None
        
        for epoch in range(self.autoencoder.num_epochs):
            indices = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[indices]
            Y_train_shuffled = Y_train[indices]
            
            for i in range(0, n_samples, batch_size):
                batch_end=min(i+batch_size,n_samples)
                X_batch = X_train_shuffled[i:batch_end]
                Y_batch = Y_train_shuffled[i:batch_end]

                self.autoencoder.init_layers(X_batch.shape[0])
                y_pred=self.autoencoder.forward_pass(X_batch)
                grad_w,grad_b = self.autoencoder.backward_pass(Y_batch,batch_size)
                self.autoencoder.update_params(grad_w,grad_b)
                val_loss = self.autoencoder.mse_loss(y_pred,Y_batch)
                # print("Epoch: ",epoch," Batch: ",i," Loss: ",val_loss)
            
            self.autoencoder.init_layers(X_val.shape[0])
            y_val_pred=self.autoencoder.forward_pass(X_val)
            val_loss = self.autoencoder.mse_loss(y_val_pred,Y_val)

            self.autoencoder.init_layers(X_train.shape[0])
            y_train_pred=self.autoencoder.forward_pass(X_train)
            train_loss = self.autoencoder.mse_loss(y_train_pred,Y_train)

            print("Epoch: ",epoch," Train Loss: ",train_loss," Val Loss: ",val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_weights=self.autoencoder.weights, self.autoencoder.biases
                patience_counter=0
            else:
                patience_counter+=1
            
            if patience_counter>=5:
                if best_weights is not None:
                    self.autoencoder.weights,self.autoencoder.biases=best_weights
                    print("Early Stopping")
                break

    def get_latent(self, X):
        self.autoencoder.init_layers(X.shape[0])
        reconsructed = self.autoencoder.forward_pass(X)
        return self.autoencoder.layer_act[self.index]
