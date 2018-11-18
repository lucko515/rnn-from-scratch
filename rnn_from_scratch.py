import numpy as np
from tqdm import tqdm

class SimpleRNN(object):

	def __init__(self, vocab_to_idx, idx_to_vocab, vocab_size, learning_rate=0.001, seq_length=30, 
				hidden_layer_size=128, 
				epochs=100, verbose = True, sample_step=500, clip_rate=5):

		'''
		SimpleRNN

		Inputs: vocab_to_idx - dictionery where keys are unique characters (set(text)) from the training text
				idx_to_vocab - dictionery where keys are positions of the unique characters from the training text
				vocab_size - how many unique characters are there in the training text
				learning_rate - number used at the update time, how much we are going to move towards the minima
				seq_length - how many characters we feed at the time (this number is also how many time steps we have at the unrolled network)
				hidden_layer_size -  how many units we have at the hidden layer
				epochs - how many times we are going to train the network
				verbose - if True every 1000 steps you will see the current loss
				smple_step - at how many sequenceses network will sample some characters
				clip_rate - this number will clip gradients below -clip_rate and above clip_rate. This param helps at overcoming Exploding Gradient problem.
		'''

		self.learning_rate = learning_rate
		self.seq_len = seq_length
		self.h_size = hidden_layer_size
		self.epochs = epochs
		self.vocab_size = vocab_size
		self.verbose = verbose
		self.sample_step = sample_step
		self.smooth_loss = -np.log(1.0/self.vocab_size)*self.seq_len
		self.clip_rate = clip_rate
		self.vocab_to_idx = vocab_to_idx
		self.idx_to_vocab = idx_to_vocab

		# Setting up weight and biases for the network
		self.w_ih = np.random.randn(self.vocab_size, self.h_size)*0.01

		self.w_hh = np.random.randn(self.h_size, self.h_size)*0.01
		self.b_hh = np.zeros((1, self.h_size))

		self.w_ho = np.random.randn(self.h_size, self.vocab_size)*0.01
		self.b_ho = np.zeros((1, self.vocab_size))
		#This state will be updated over time
		self.state = np.zeros((1, self.h_size))

		#Memory params for Adagrad
		self.m_w_ih = np.zeros_like(self.w_ih)
		self.m_w_hh = np.zeros_like(self.w_hh)
		self.m_w_ho = np.zeros_like(self.w_ho)
		self.m_b_ho = np.zeros_like(self.b_ho)
		self.m_b_hh = np.zeros_like(self.b_hh)



	def batch_opt(self, X, index):
		'''
		This function is used to created batches for feeding data into the RNN

		Inputs: X - encoded input data (output from encoding_dataset function)
				index - is index of training loop, this index is used to determine at which place should we start our batch

		Outputs:  X_batch_new - one_hot encoded tensor. size: [self.seq_len, 1, self.vocab_size]
				  y_batch_new - is the similar one_hot tensor, but every char is shifted one time step to the right. size:[self.seq_len, 1, self.vocab_size]
		
		'''
		X_batch = X[index:index+self.seq_len]
		y_batch = X[index+1:index+self.seq_len+1]
		X_batch_new = []
		y_batch_new = []
		for i in X_batch:
			one_hot_char = np.zeros((1, self.vocab_size))
			one_hot_char[0][i] = 1
			X_batch_new.append(one_hot_char)

		for i in y_batch:
			one_hot_char = np.zeros((1, self.vocab_size))
			one_hot_char[0][i] = 1
			y_batch_new.append(one_hot_char)

		return X_batch_new, y_batch_new

	def encoding_dataset(self, X):
		'''
		This function is used to encode text to the integers
		Inputs: X - text to be processed in the string format

		Outputs: encoded_data - same sized 'text' as input X but every character is encoded to integer,
								based at integer index in vocab_to_idx
		'''
		enoded_data = []

		for char in X:
			enoded_data.append(self.vocab_to_idx[char])

		return enoded_data

	def fit(self, X):
		
		number_of_full_sequences = len(X) // self.seq_len
		#chopping end of the input text so we have full number of sequences
		cut_X = X[:number_of_full_sequences*self.seq_len]

		#Encoding text
		encoded_cut_X = self.encoding_dataset(cut_X)

		for i in range(self.epochs):

			#Delete tqdm keyword if you don't like the loading bar while training
			for ii in tqdm(range(0, len(encoded_cut_X)-self.seq_len, self.seq_len)): #Batch loop
				X_batch, y_batch = self.batch_opt(encoded_cut_X, ii)

				outputs, probs, states = self.forward(X_batch)

				loss = 0
				for ts  in range(self.seq_len):
					loss += -np.log(probs[ts][0, np.argmax(y_batch[ts])])
					
				self.smooth_loss = self.smooth_loss * 0.999 + loss *0.001
				if self.verbose:
					if ii % 1000 == 0:
						print('Current loss is: {}'.format(self.smooth_loss)) 

				#Here will be a part for Backpropagation
				dW_ih, dW_hh, dW_ho, db_hh, db_ho = self.bacprop(X_batch, y_batch, probs, states)

				#Simple Adagrad update of params (based on SGD)
				for params, derivative_params, memory in zip([self.w_ih, self.w_hh, self.w_ho, self.b_hh, self.b_ho],
															[dW_ih, dW_hh, dW_ho, db_hh, db_ho],
															[self.m_w_ih, self.m_w_hh, self.m_w_ho, self.m_b_hh, self.m_b_ho]):
					memory += derivative_params * derivative_params
					params += -self.learning_rate * derivative_params / np.sqrt(memory + 1e-8)
			

				if ii % self.sample_step == 0:
					#SAMPLE time ^_^
					sampled_string = self.sample(200, 20)
					print(sampled_string)

			print("EPOCH: {}/{}".format(i, self.epochs))

	def forward(self, X):
		'''
		Inputs: X - characters for the network input

		OutputS: outputs - logits from the output layer
				 output_probs - softmax probabilities from the output layer
				 hidden_states - states of the RNN layer in the network
		'''
		current_state = self.state
		outputs = {}
		output_probs = {}
		hidden_states = {}
		hidden_states[-1] = current_state
		#forward prop loop
		for ts in range(self.seq_len):
			current_state = np.tanh(np.dot(X[ts], self.w_ih) + np.dot(current_state, self.w_hh) + self.b_hh)
			hidden_states[ts] = current_state
			outputs[ts] = np.dot(current_state, self.w_ho) + self.b_ho
			output_probs[ts] = np.exp(outputs[ts]) / np.sum(np.exp(outputs[ts])) #softmax

		self.state = current_state
		return outputs, output_probs, hidden_states

	def bacprop(self, X, y, probs, states):
		'''
		Inputs: X - input to the forward step of the  netwrok
				y - targets of the currrent batch
				probs - probability (softmax) outputs from the forward function
				states - states for each time step computed in the forward step

		Outputs: derivatives of every learnable param in the network
		'''
		dW_ih = np.zeros_like(self.w_ih)
		dW_hh = np.zeros_like(self.w_hh)
		dW_ho = np.zeros_like(self.w_ho)

		db_hh = np.zeros_like(self.b_hh)
		db_ho = np.zeros_like(self.b_ho)

		dh_s_next = np.zeros_like(states[0])

		#Backprop through time - loop
		for ts in reversed(range(self.seq_len)):
			
			dy = np.copy(probs[ts])
			dy[0][np.argmax(y[ts])] -= 1
			
			dW_ho += np.dot(states[ts].T, dy)
			db_ho += dy
			dhidden = (1 - states[ts]**2) * (np.dot(dy, self.w_ho.T) + dh_s_next)
			dh_s_next = np.dot(dhidden, self.w_hh.T)
			dW_hh += np.dot(states[ts-1].T, dhidden)
			dW_ih += np.dot(X[ts].T, dhidden)
			db_hh += dhidden

		#Clipping gradients
		for params in [dW_ih, dW_hh, dW_ho, db_hh, db_ho]:
			np.clip(params, -self.clip_rate, self.clip_rate, out=params)
		return dW_ih, dW_hh, dW_ho, db_hh, db_ho

	def sample(self, number_to_sampled, starting_inx):
		'''
		Inputs: number_to_sampled - how many chars do we want to sample
				starting_idx - from which character we want to start sampling

		Outputs: sampled_string - string made of sampled characters with len  == number_to_sampled
		'''

		#We always start sampling with the newest network state
		sampling_state = self.state

		#Setting up starting params for the sampling process
		sampled_string = ""
		x = np.zeros((1, self.vocab_size))
		x[0][starting_inx] = 1

		#Sampling loop
		for i in range(number_to_sampled):
			#Forwad step of the network
			hidden_sample = np.tanh(np.dot(x, self.w_ih) + np.dot(sampling_state, self.w_hh) + self.b_hh)
			output = np.dot(hidden_sample, self.w_ho) + self.b_ho
			probs = np.exp(output)/np.sum(np.exp(output))

			#We find the index with the highest prob
			index = np.random.choice(range(self.vocab_size), p=probs.ravel())
			#setting x-one_hot_vector for the next character
			x = np.zeros((1, self.vocab_size))
			x[0][index] = 1
			#Find the char with the sampled index and concat to the output string
			char = self.idx_to_vocab[index]
			sampled_string += char

		return sampled_string







###############################################

text = open('zero_to_one.txt').read()

vocab = set(text)
vocab_size = len(vocab)

#Creating vocabs for mapping chars and ints in both directions
vocab_to_int = {char:i for i, char in enumerate(vocab)}
int_to_vocab = {i:char for i, char in enumerate(vocab)}


#Setting up hyperparameters for RNN
hidden_layer_size = 128
seq_length = 20
learning_rate = 0.01

model = SimpleRNN(vocab_to_idx=vocab_to_int, 
				idx_to_vocab=int_to_vocab,
				vocab_size=len(vocab), 
				learning_rate=learning_rate, 
				seq_length=25, 
				hidden_layer_size=128, 
				epochs=100, 
				verbose = True, 
				sample_step=500, 
				clip_rate=5)

model.fit(text)



