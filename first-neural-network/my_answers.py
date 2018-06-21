import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        #self.learning_rate = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        #self.activation_function = lambda x : 0  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))  # Replace 0 with your sigmoid calculation here

        self.activation_function = sigmoid


    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        #for i in range(iterations):
        for X, y in zip(features, targets):

            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # Hidden layer
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # Output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer --- AE: the activation function in the output node is f(x) = x, so the input is our output
        
        #print("FPT:")
        #print("X = ",X)
        #print("weights_input_to_hidden = ",self.weights_input_to_hidden)
        #print("weights_hidden_to_output = ",self.weights_hidden_to_output)
        #print("hidden_inputs = ",hidden_inputs)
        #print("hidden_outputs = ",hidden_outputs)
        #print("final_inputs = ",final_inputs)
        #print("final_outputs = ",final_outputs)
        #print("!FPT")
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # Output error
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        # Hidden layer's contribution to the error
        hidden_error = (error * self.weights_hidden_to_output).T
        
        # Backpropagated error terms
        output_error_term = error * 1 ### AE: the output layer activation function is f(x) = x and derivative of that is 1, so that is what we multiply it with

        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        
        #print("error = ",error)
        #print("weights_hidden_to_output = ",self.weights_hidden_to_output)
        #print("hidden_error = ",hidden_error)
        #print("hidden_outputs = ",hidden_outputs)
        #print("delta_weights_i_h = ",delta_weights_i_h)
        #print("hidden_error_term = ",hidden_error_term)
        #print("X = ",X)
        #print("output_error_term = ",output_error_term)
        #print("delta_weights_h_o = ",delta_weights_h_o)
        
        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:, None]
        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]

        #print("BP:")
        #print("error = ",error)
        #print("hidden_error = ",hidden_error)
        #print("output_error_term = ",output_error_term)
        #print("hidden_error_term = ",hidden_error_term)
        #print("delta_weights_i_h = ",delta_weights_i_h)
        #print("delta_weights_h_o = ",delta_weights_h_o)
        #print("!BP")
        
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        #print("weights_hidden_to_output = ",self.weights_hidden_to_output)
        #print("weights_input_to_hidden = ",self.weights_input_to_hidden)
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step
        #print("UW:")
        #print("weights_hidden_to_output = ",self.weights_hidden_to_output)
        #print("learning_rate = ",self.lr)
        #print("delta_weights_h_o = ",delta_weights_h_o)
        #print("n_records = ",n_records)
        #print("weights_input_to_hidden = ",self.weights_input_to_hidden)
        #print("delta_weights_i_h = ",delta_weights_i_h)
        #print("!UW")

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Forward pass here ####
        # Hidden layer
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # Output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################

### Need to choose a number here a level that stops shortly after the validation loss is no longer decreasing
iterations = 7500

### Each batch will contain 128 data points according to Stochastics Gradient Descent script that is used
### According to course notes, the learning rate could be chosen as a divison of 1 / (size of training set)
learning_rate = 0.35#0.0078125

### According to course notes, the best number to choose here is between the number of input and output nodes.
### Hence the number of input nodes is 56, let's try something in between. Too many nodes will overfit the model.
hidden_nodes = 28
output_nodes = 1

### Sets that work well:
### Trains well, and quite confident on the predictions
#iterations = 300
#learning_rate = 0.2
#hidden_nodes = 28

### Trains well and uses few iterations, but not confident on predictions
#iterations = 30
#learning_rate = 0.2
#hidden_nodes = 48

### Trains too well-- both the training and validation losses have long since platoed, but very confident
#iterations = 300
#learning_rate = 0.2
#hidden_nodes = 48

### Noisy train and validation losses, but quite confident:
#iterations = 300
#learning_rate = 0.2
#hidden_nodes = 38

### Very good confidence and seemingly platoed validation loss. 
#iterations = 500
#learning_rate = 0.2
#hidden_nodes = 48

### Excellent confidence, but noisy validation loss, but that's probably ok.
#iterations = 500
#learning_rate = 0.2
#hidden_nodes = 38

### Even more confidence, but noisy validation loss, but that's probably ok.
#iterations = 1000
#learning_rate = 0.2
#hidden_nodes = 38

### Even more confidence, but less noisy validation loss
#iterations = 1000
#learning_rate = 0.2
#hidden_nodes = 28

### Even more confidence and even less noisy validation loss
#iterations = 1000
#learning_rate = 0.1
#hidden_nodes = 28

### Looks like validation loss is still decreasing just a little bit, lets increase iterations
#iterations = 1500
#learning_rate = 0.1
#hidden_nodes = 28

### Validation loss seems to platoueu to 0.45 - 0.46 at around 1500 iterations and certainly doesn't change much anymore after 2000.
#iterations = 2500
#learning_rate = 0.1
#hidden_nodes = 28

### Looks like we can decrease the iterations a little further and fewer hidden nodes also didn't hurt much
#iterations = 1700
#learning_rate = 0.1
#hidden_nodes = 18

### Fewer iterations, fewer hidden nodes, about the same performance.
#iterations = 1500
#learning_rate = 0.1
#hidden_nodes = 13

### Still good, let's cut more hidden nodes out
#iterations = 1500
#learning_rate = 0.1
#hidden_nodes = 10

### Even better result with only 5 hidden nodes. Validation loss got as low as 0.43.
#iterations = 1500
#learning_rate = 0.1
#hidden_nodes = 5

### Slightly higher validation loss
#iterations = 1500
#learning_rate = 0.1
#hidden_nodes = 4

### About the same
#iterations = 1500
#learning_rate = 0.1
#hidden_nodes = 3

### Validation loss going back up
#iterations = 1500
#learning_rate = 0.1
#hidden_nodes = 2

### With 7 hidden nodes validation loss is also back up at 0.46
#iterations = 1500
#learning_rate = 0.1
#hidden_nodes = 7

### With 6 hidden nodes validation loss is 0.455, let's stick with 5 hidden nodes and 1500 iterations. Now let's play a little with learn rate.
#iterations = 1500
#learning_rate = 0.1
#hidden_nodes = 6

### Validation loss down to 0.425
#iterations = 1500
#learning_rate = 0.15
#hidden_nodes = 5

### Validation loss back up to 0.439
#iterations = 1500
#learning_rate = 0.2
#hidden_nodes = 5

### Validation loss down to 0.419
#iterations = 1500
#learning_rate = 0.17
#hidden_nodes = 5

### Validation loss: 0.431
#iterations = 1500
#learning_rate = 0.16
#hidden_nodes = 5

### Validation loss: 0.425
#iterations = 1500
#learning_rate = 0.18
#hidden_nodes = 5

### Validation loss: 0.467
#iterations = 1500
#learning_rate = 0.19
#hidden_nodes = 5

### OK, and the WINNER IS:
#iterations = 1500
#learning_rate = 0.17
#hidden_nodes = 5
### Hopefully it's not a local minimum, but looking at the prediction result, it seems to predict really well. I wonder if another perceptron could be created that tries to optimize these parameters (iterations, learning rate and hidden node count) by running this neural network. Yeah, I know, the complexity would be at least square if not exponential, but still...

### It seems that all these models struggle to predict the reduced demand around 22nd - 26th of December and then 28th - 30th of December. Well, obviously those are days that people often may take off work although they are working days. Perhaps we need an extra input feature to reflect that, but that would require synthesizing it and since it wasn't required in the project, I will probably not have time for that.

### Ok, the above values were a "local minimum" configuration. Better ones can be found. Validation loss is now at 0.202, but it takes a long time to train.
#iterations = 7500
#learning_rate = 0.35#0.0078125
#hidden_nodes = 15#28
#output_nodes = 1

### And and even better result (good enought to submit now) can be achieved with the recommended number of hidden nodes (half of input features, so 28) and with huge enough number of iterations. Training loss: 0.057 ... Validation loss: 0.134
#iterations = 7500
#learning_rate = 0.35
#hidden_nodes = 28
#output_nodes = 1