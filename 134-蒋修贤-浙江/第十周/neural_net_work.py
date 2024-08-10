#!/usr/bin/env python3
import scipy.special
import numpy as np

class NeuralNetWork:
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		self.lr = learningrate
		self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
		self.who = np.random.rand(self.onodes, self.hnodes) - 0.5
		
		self.activation_function = lambda x: scipy.special.expit(x)
		pass
		
	def train(self, inputs_list, targets_list):
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(targets_list, ndmin=2).T
		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)
		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)
		
		output_errors = targets - final_outputs
		hidden_errors = np.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
		self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
										np.transpose(hidden_outputs))
		self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
										np.transpose(inputs))
		pass
		
	def query(self,inputs):
		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)
		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)
		return final_outputs
	
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("./dataset/mnist_train.csv",'r')
training_data_list= training_data_file.readlines()
training_data_file.close()

epochs = 5
for e in range(epochs):
	for record in training_data_list:
		all_values =record.split(',')
		inputs = (np.asfarray(all_values[1:]))/255
		targets = np.zeros(output_nodes)
		targets[int(all_values[0])] = 1
		n.train(inputs, targets)
		
test_data_file = open('./dataset/mnist_test.csv')
test_data_list = test_data_file.readlines()
test_data_file.close()

scores =[]
for record in test_data_list:
	all_values = record.split(',')
	correct_number = int(all_values[0])
	print("true ",correct_number)
	inputs = (np.asfarray(all_values[1:])) / 255
	outputs = n.query(inputs)
	label =np.argmax(outputs)
	print("argmax ", label)
	if label == correct_number:
		scores.append(1)
	else:
		scores.append(0)
print(scores)

# 计算成功率
scores_array = np.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)