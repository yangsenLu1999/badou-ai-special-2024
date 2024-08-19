import numpy as np
import scipy.special
# -*- coding: utf-8 -*-

class NaturalNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        self.lr=learningrate

        self.wih=np.random.rand(self.hnodes,self.inodes)-0.5
        self.who=np.random.rand(self.onodes,self.hnodes)-0.5

        self.activaityfunc=lambda x:scipy.special.expit(x)
        pass

    def train(self,inputlist,targetlist):
        input=np.array(inputlist, ndmin=2).T
        target = np.array(targetlist, ndmin=2).T

        hidden_input=np.dot(self.wih,input)
        hidden_output=self.activaityfunc(hidden_input)
        outcome_input=np.dot(self.who,hidden_output)
        outcome_output = self.activaityfunc(outcome_input)

        outcome_error=target-outcome_output
        hidden_error=np.dot(self.who.T,outcome_error*outcome_output*(1-outcome_output))

        self.who += self.lr*np.dot(outcome_error*outcome_output*(1-outcome_output),hidden_output.T)
        self.wih += self.lr*np.dot(hidden_error*hidden_output*(1-hidden_output),input.T)
        pass

    def query(self,inputlist):
        input = np.array(inputlist, ndmin=2).T
        hidden_input = np.dot(self.wih, input)
        hidden_output = self.activaityfunc(hidden_input)
        outcome_input = np.dot(self.who, hidden_output)
        outcome_output = self.activaityfunc(outcome_input)

        return outcome_output

if __name__=="__main__":

    in1=784
    hi1=200
    out1=10
    lrate=0.1

    train_date=open("dataset/mnist_train.csv","r")
    train_date_list=train_date.readlines()
    train_date.close()
    n=NaturalNetwork(in1,hi1,out1,lrate)

    epoch=5
    for e in range(epoch):
        for i in train_date_list:
            data_all=i.split(",")
            inputs=np.asfarray(data_all[1:])/255*0.99+0.01
            target=np.zeros(out1)+0.01
            target[int(data_all[0])]=0.99
            n.train(inputs,target)

    test_date = open("dataset/mnist_test.csv", "r")
    test_date_list = test_date.readlines()
    test_date.close()
    scores = []
    for i in train_date_list:
        data_all=i.split(",")
        inputs=np.asfarray(data_all[1:])/255*0.99+0.01
        outcome=n.query(inputs)

        correct_number=int(data_all[0])
        print("correct_number is:", correct_number)
        label=np.argmax(outcome)
        print("AI think correct number is:", label)

        if label==correct_number:
            scores.append(1)
        else:
            scores.append(0)

    print("AI scores is:",scores)

    s=sum(scores)/len(scores)
    print("AI accuracy is:",s)






