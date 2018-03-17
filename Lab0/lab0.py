import numpy as np

class Neural:
	def __init__(self):

		self.ni=3	# Input layer node (2 input + 1 bias)
		self.nh=3	# Hidden layer node
		self.no=1	# Output layer node
		self.hiddenlayerN = 2	# Amount of Hidden layer
		self.epoch = 10000

		self.weightToHiddenLayers = np.random.randn(self.hiddenlayerN, self.nh, self.nh)	# 2 * 3 * 3
		self.weightToOutputLayer = np.random.randn(self.nh, self.no)	# 1 * 3

		self.activatedInput = np.ones((self.ni))	# = input
		self.activatedHidden = np.ones((self.hiddenlayerN, self.nh))	# value after activated from hidden layer
		self.activatedOutput = np.ones((self.no))	# = output

		# To incorporate momentum factor, introduce another array for the 'previous change'.
		self.cToHiddenLayer = np.zeros((self.hiddenlayerN, self.nh, self.nh))
		self.cToOutputLayer = np.zeros((self.nh, self.no))
	
	# M : momentum parm, N : learning rate
	def backpropagate(self, inputs, expected, output, N=0.5, M=0.1):

		# Reference : Wiki - backpropagation loss function definition		
		error = expected - output
		output_deltas = error * dsigmoid(self.activatedOutput)
		
		weightToOutputLayers_t = np.copy(self.weightToOutputLayer)

		delta_weight = self.activatedHidden[self.hiddenlayerN - 1].reshape(self.nh, self.no).dot(output_deltas.reshape(1, self.no))
		
		self.weightToOutputLayer += M * self.cToOutputLayer + N * delta_weight
		self.cToOutputLayer = np.copy(delta_weight)
		
		for i in range(self.hiddenlayerN-1, -1, -1):
			
			if(i == self.hiddenlayerN-1):	# When next layer is hidden to output 
				error = weightToOutputLayers_t.dot(output_deltas.reshape(self.no, 1))		
			else:
				error = weightToHiddenLayers_t.dot(hidden_deltas.reshape(self.nh, 1))
			
			hidden_deltas = error.reshape(self.nh) * dsigmoid(self.activatedHidden[i])
			
			if(i == 0):	# When current layer is input to hidden layer
				delta_weight = self.activatedInput.reshape(self.ni, 1).dot(hidden_deltas.reshape(1, self.nh))
			else:
				delta_weight = self.activatedHidden[i - 1].reshape(self.nh, 1).dot(hidden_deltas.reshape(1, self.nh))
			weightToHiddenLayers_t = np.copy(self.weightToHiddenLayers[i])

			self.weightToHiddenLayers[i] += M * self.cToHiddenLayer[i] + N * delta_weight
			self.cToHiddenLayer[i] = np.copy(delta_weight)
						
	def test(self, testX, testY):
		for j in range(testY.size):
			print('For input:', testX[j], ' Output -->', self.runNetwork(testX[j]), '\tTarget: ', testY[j])
	
	def runNetwork(self, feed):
		if(feed.size != self.ni-1):
			print('Error in number of input values.')
		
		np.copyto(self.activatedInput[:feed.size], feed)

		for i in range(self.hiddenlayerN):						
			if(i == 0): # When input Layer to Hidden Layer
				temp = self.activatedInput.dot(self.weightToHiddenLayers[i])
				self.activatedHidden[0] = sigmoid(temp)
			else:
				temp = self.activatedHidden[i-1].dot(self.weightToHiddenLayers[i])
				self.activatedHidden[i] = sigmoid(temp)
		
		temp = self.activatedHidden[self.hiddenlayerN - 1].dot(self.weightToOutputLayer)
		self.activatedOutput = sigmoid(temp)

		return self.activatedOutput


	def trainNetwork(self, trainX, trainY):
		for i in range(self.epoch):
			for j in range(trainY.size):
				inputs = trainX[j]
				out = self.runNetwork(inputs)
				expected = trainY[j]
				self.backpropagate(inputs,expected,out)
		self.test(trainX, trainY)

# End of class.

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def dsigmoid(y):
	return y * (1 - y)

def main():
	X = np.array([
		[0.0,0.0],
		[0.0,1.0],
		[1.0,0.0],
		[1.0,1.0]
	])
	Y = np.array([0.0,1.0,1.0,0.0])

	newNeural = Neural()
	newNeural.trainNetwork(X, Y)

if __name__ == "__main__":
	main()