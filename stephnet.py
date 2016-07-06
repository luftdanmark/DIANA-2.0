# Stephanie Taylor
# Spring 2014
# neural_networks.py

import random
import numpy as np
from data import Data

def computeConfusionMatrix(actual,predictions):
 """creates the confusion matrix for the class data"""
 ii = { }
 if 51 - 51: IiI1i11I
 Iii1I1 = len ( actual )
 if 73 - 73: II1I1iiiiii * ooo0OO / o0OO00 / oo
 i1iII1IiiIiI1 = int(np.max(actual)) + 1
 o0OoOoOO00 = np . matrix ( np . zeros ( ( i1iII1IiiIiI1 , i1iII1IiiIiI1 ) ) )
 for I11i in range ( Iii1I1 ) :
  o0OoOoOO00 [ int ( actual [ I11i ] ) , int ( predictions [ I11i ] ) ] = o0OoOoOO00 [ int ( actual [ I11i ] ) , int ( predictions [ I11i ] ) ] + 1
 return o0OoOoOO00

#-------------------------------------------------------------------------------------
#  Routines and Classes for Neural Networks
#-------------------------------------------------------------------------------------

# base class for neurons.
class BaseNeuron:
    def __init__( self, id ):
        self.id = id
        self.output_nodes = []

    # override this!
    def getOutput( self ):
        return None

    def addOutputNode( self, node ):
        self.output_nodes.append( node )

# constant has no input or id, but can send signals.
# this is for bias/threshold neurons and data features.
class ConstantNeuron(BaseNeuron):
    def __init__( self, id, value ):
        BaseNeuron.__init__(self, id )
        self.value = value
        self.output_nodes = []

    def setValue( self, value ):
        self.value = value

    def getOutput( self ):
        return self.value

    def computeOutputFast(self):
        return self.value

    def dump( self):
        print "Constant ", self.value

# specific class for bias/threshold neuron
class ThresholdNeuron(ConstantNeuron):
    def __init__( self ):
        ConstantNeuron.__init__( self, -1, 1.0 )

# class for neuron at any layer but the first
class Neuron(BaseNeuron):
    def __init__( self, id ):
        BaseNeuron.__init__(self,id)
        # the first input will be the fixed threshold thing
        # input is the list of nodes sending info to this node
        # weights is the list of weights associated with that
        # info
        self.input_nodes = [ThresholdNeuron()]
        self.weights = [random.random()]

    def addInputNode( self, node ):
        self.input_nodes.append( node )
        self.weights.append( random.random() )

    # Return the weight of the connection from input_node
    # to this node.
    def getWeightFromInputNode( self, input_node ):
        for i in range(len(self.input_nodes)):
            if self.input_nodes[i] == input_node:
                return self.weights[i]

    # compute a column vector of weight changes
    # the output should be a num_inputs x 1 matrix
    # this doesn't update any fields
    # this is unscaled in that it doesn't take eta (the learning
    # rate into account)
    # this assumes the value field of the input nodes is up to date
    # (which it will be if forwardPropagate has been run)
    # It also assumes that self.delta is up to date (i.e. computeDelta
    # has been run)
    def computeUnscaledWeightChanges( self ):
        M = len(self.input_nodes)
        wd = np.zeros( (M,1) )
        for i in range(M):
            wd[i,0] = self.delta * self.input_nodes[i].value
        return wd

    # get the output by recursing backwards through the
    # network.
    def getOutput( self ):
        # integrate the inputs (weighted sums)
        weights = np.array( self.weights )
        inputs = np.zeros( weights.shape )
        for i in range( inputs.size ):
            inputs[i] = self.input_nodes[i].getOutput()
        integrated_input = np.dot( weights, inputs )
        #print "N",self.id,"w, i, w dot i", weights, inputs, integrated_input
        ret = 1.0 / (1 + np.exp( -integrated_input ) )
        self.value = ret
        return ret

    # get compute output, assuming we are in the midst of
    # forward propagation and the inputs have had their
    # outputs computed already.
    def computeOutputFast(self, verbose=False):
        # integrate the inputs (weighted sums)
        weights = np.array( self.weights )
        inputs = np.zeros( weights.shape )
        for i in range( inputs.size ):
            inputs[i] = self.input_nodes[i].value
        integrated_input = np.dot( weights, inputs )
        #print "N",self.id,"w, i, w dot i", weights, inputs, integrated_input
        ret = 1.0 / (1 + np.exp( -integrated_input ) )
        if verbose:
            print "Node %d, input= %f, output=%f" % (self.id,integrated_input,ret)
        self.value = ret
        return ret

    def dump( self ):
        print "Node ",self.id
        print "Inputs\tWeights"
        for i in range(len(self.input_nodes)):
            print self.input_nodes[i].id,"\t",self.weights[i]

class OutputNeuron(Neuron):
    # compute delta. Store the result in a delta field and return it.
    def computeDelta( self, true_value ):
        # since the output is a sigmoid, the deriv of the output
        # is output * (1-output)
        output_deriv = self.value * (1.0-self.value)
        self.delta = (true_value-self.value) * output_deriv
        return self.delta

class HiddenNeuron(Neuron):
    # compute delta. Store the result in a delta field and return it.
    def computeDelta( self ):
        # since the output is a sigmoid, the deriv of the output
        # is output * (1-output)
        output_deriv = self.value * (1.0-self.value)
        output_weights = 0.0
        for output_node in self.output_nodes:
            output_weights += output_node.delta * output_node.getWeightFromInputNode( self )
        self.delta = output_weights * output_deriv
        return self.delta

class Perceptron():
    def __init__(self, Ninput, Noutput):
        self.input_layer = []
        self.output_layer = []
        for i in range(Noutput):
            self.output_layer.append( OutputNeuron( id = i ) )
        for i in range(Ninput):
            self.input_layer.append( ConstantNeuron( id = Noutput+i, value = 0.0 ) )
        # wire up the nodes. Each input layer neuron sends its output
        # to each output layer node
        for output_node in self.output_layer:
            for input_node in self.input_layer:
                output_node.addInputNode( input_node )
                input_node.addOutputNode( output_node )

    # Return the outputs from forward propagation using
    # the given (1xF) data point
    def forwardPropagate( self, data_pt, verbose=False ):
        for i in range(data_pt.shape[1]):
            self.input_layer[i].setValue( data_pt[0,i] )
            #print input_layer[i].id, input_layer[i].value
        outputs = []
        for output_node in self.output_layer:
            outputs.append( output_node.computeOutputFast(verbose) )
        return outputs

    # a backward propagation begins with a forward propagation.
    # input:
    #     data_pt: 1 x F matrix
    #     expected_output: 1 x C matrix
    # Return error (a scalar)
    # Return the proposed changes to the weights
    def backwardPropagate( self, data_pt, expected_output ):
        actual_output = np.matrix( [self.forwardPropagate( data_pt )] )
        C = expected_output.shape[1]
        F = data_pt.shape[1]
        mse = np.sum(np.power( actual_output - expected_output, 2.0 )) / float(C)
        # compute output deltas first (this is important, because the hidden
        # node deltas depend on it)
        output_deltas = np.zeros( (C,1) )
        for i in range(C):
            output_deltas[i,0] = self.output_layer[i].computeDelta( expected_output[0,i] )
        # compute weight changes for links going into the output nodes
        # the inputs are the input nodes + a threshold node,
        # We need one weight per input, per output node
        # rows: which input node (0 = bias, 1=self.input_layer[0], etc.)
        # columns: which output node
        unscaled_wd_output = np.zeros( (F+1,C) )
        for j in range(C):
            column = self.output_layer[j].computeUnscaledWeightChanges()
            for i in range(F+1):
                unscaled_wd_output[i,j] = column[i,0]

        return mse, unscaled_wd_output

    # perform backpropagation for each data pt, updating the weights
    # after each point.
    # input:
    #     all_data: N x F matrix
    #     all_expected_output: N x C matrix
    def onlineEpoch(self, all_data, all_expected_output, learning_rate, verbose=False):
        N = all_data.shape[0]
        for i in range(N):
            (mse, unscaled_wd_output) = self.backwardPropagate( all_data[i,:], all_expected_output[i,:] )
            for to_idx in range(len(self.output_layer)):
                output_node = self.output_layer[to_idx]
                for from_idx in range(len(output_node.weights)):
                    output_node.weights[from_idx] += learning_rate * unscaled_wd_output[from_idx,to_idx]
            if verbose:
                print "\n-------------Weights after training data point",i,"----------------"
                self.dump()

    # print weights and other info to the terminal
    def dump( self ):
        for n in self.input_layer:
            n.dump()
        for n in self.output_layer:
            n.dump()

    # Classify the given data and print the confusion matrix
    # input:
    #     all_data: N x F matrix
    #     class_vals: N x 1 matrix
    def computeMeanError( self, all_data, all_expected_output ):
        N = all_data.shape[0]
        C = all_expected_output.shape[1]
        F = all_data.shape[1]
        all_mse = np.zeros( (N,1) )
        for i in range(N):
            actual_output = np.matrix( [self.forwardPropagate( all_data[i,:], )] )
            all_mse[i] = np.sum(np.power( actual_output - all_expected_output[i,:], 2.0 )) / float(C)
        return np.mean(all_mse)


    def printConfusionMatrix( self, all_data, class_vals ):
        N = all_data.shape[0]
        pred_classes = np.zeros( (N,1) )
        for i in range(N):
            output = self.forwardPropagate( all_data[i,:] )
            pred_classes[i,0] = np.argmax( output )

        cm = computeConfusionMatrix( class_vals, pred_classes )
        print "Confusion Matrix"
        print cm

class PerceptronForANDData(Perceptron):

    def __init__(self):
        # load in data
        self.ddata = np.matrix( [ [0.1,0.1], [0.1,0.9], [0.9,0.1], [0.9,0.9] ] )
        N = self.ddata.shape[0]
        F = self.ddata.shape[1]
        self.classes = [['False'],['False'],['False'],['True']]
        self.class_names = np.unique( np.array(self.classes) )
        self.class_vals = np.zeros( (N,1) )
        for cidx in range(len(self.classes)):
            for i in range(self.class_names.size):
                if self.classes[cidx][0] == self.class_names[i]:
                    self.class_vals[cidx,0] = i

        # construct FF network
        Perceptron.__init__( self, F, self.class_names.size )

    # instead of training the network, use Weka's weights
    def useStephWeights(self):
        # for non-normalized data, using Weka's weights
        # Node 0
        self.output_layer[0].weights = [7.0, -5.0, -5.0]
        # Node 1
        self.output_layer[1].weights = [-7.0, 5.2, 5.2]

    def printOutputFromTrialData( self ):
        self.printConfusionMatrix( self.ddata, self.class_vals )

        print "----- following one forward propagation -------"
        print "input", self.ddata[0,:]
        output = self.forwardPropagate( self.ddata[0,:], verbose = True )
        print "input", self.ddata[1,:]
        output = self.forwardPropagate( self.ddata[1,:], verbose = True )
        print "input", self.ddata[2,:]
        output = self.forwardPropagate( self.ddata[2,:], verbose = True )
        print "input", self.ddata[3,:]
        output = self.forwardPropagate( self.ddata[3,:], verbose = True )

class MultilayerPerceptron():
    def __init__(self, Ninput, Noutput):
        self.input_layer = []
        self.hidden_layer = []
        self.output_layer = []
        for i in range(Noutput):
            self.output_layer.append( OutputNeuron( id = i ) )
        for i in range(Noutput):
            self.hidden_layer.append( HiddenNeuron( id = i+Noutput ) )
        for i in range(Ninput):
            self.input_layer.append( ConstantNeuron( id = Noutput*2+i, value = 0.0 ) )
        # wire up the nodes. Each input layer neuron sends its output
        # to each hidden layer node. Each hidden layer node sends its
        # output to each output layer node
        for output_node in self.output_layer:
            for hidden_node in self.hidden_layer:
                output_node.addInputNode( hidden_node )
                hidden_node.addOutputNode( output_node )
        for hidden_node in self.hidden_layer:
            for input_node in self.input_layer:
                hidden_node.addInputNode( input_node )
                input_node.addOutputNode( hidden_node )

    # Return the outputs from forward propagation using
    # the given (1xF) data point
    def forwardPropagate( self, data_pt, verbose=False ):
        for i in range(data_pt.shape[1]):
            self.input_layer[i].setValue( data_pt[0,i] )
            #print input_layer[i].id, input_layer[i].value
        for node in self.hidden_layer:
            node.computeOutputFast(verbose)
        outputs = []
        for output_node in self.output_layer:
            outputs.append( output_node.computeOutputFast(verbose) )
        return outputs

    # a backward propagation begins with a forward propagation.
    # input:
    #     data_pt: 1 x F matrix
    #     expected_output: 1 x C matrix
    # Return error (a scalar)
    # Return the proposed changes to the weights
    def backwardPropagate( self, data_pt, expected_output ):
        actual_output = np.matrix( [self.forwardPropagate( data_pt )] )
        C = expected_output.shape[1]
        F = data_pt.shape[1]
        print C, F
        mse = np.sum(np.power( actual_output - expected_output, 2.0 )) / float(C)
        # compute output deltas first (this is important, because the hidden
        # node deltas depend on it)
        output_deltas = np.zeros( (C,1) )
        for i in range(C):
            output_deltas[i,0] = self.output_layer[i].computeDelta( expected_output[0,i] )
        hidden_deltas = np.zeros( (C,1) )
        for i in range(C):
            hidden_deltas[i,0] = self.hidden_layer[i].computeDelta()
        # compute weight changes for links going into the output nodes
        # the inputs are the hidden nodes + a threshold node,
        # We need one weight per input, per output node
        # rows: which input node (0 = threshold, 1=self.hidden_layer[0], etc.)
        # columns: which output node
        unscaled_wd_output = np.zeros( (C+1,C) )
        for j in range(C):
            column = self.output_layer[j].computeUnscaledWeightChanges()
            for i in range(C+1):
                unscaled_wd_output[i,j] = column[i,0]
        # compute weight changes for links going into the hidden nodes
        # the inputs are the input nodes + a threshold node,
        # We need one weight per input, per output node
        # rows: which input node (0 = threshold, 1=self.hidden_layer[0], etc.)
        # columns: which output node
        unscaled_wd_hidden = np.zeros( (F+1,C) )
        for j in range(C):
            column = self.hidden_layer[j].computeUnscaledWeightChanges()
            for i in range(F+1):
                unscaled_wd_hidden[i,j] = column[i,0]

        return mse, unscaled_wd_hidden, unscaled_wd_output
#         print 'output deltas ', output_deltas
#         print 'hidden deltas', hidden_deltas
#         print 'unscale wd (from_hidden_node_i-1,to_output_node_j)'
#         print unscaled_wd_output
#         print 'unscale wd (from_input_node_i-1,to_hidden_node_j)'
#         print unscaled_wd_hidden
#         print 'mean squared error', mse

    # perform backpropagation for each data pt, updating the weights
    # after each point.
    # input:
    #     all_data: N x F matrix
    #     all_expected_output: N x C matrix
    def onlineEpoch(self, all_data, all_expected_output, learning_rate, verbose=False):
        N = all_data.shape[0]
        for i in range(N):
            (mse, unscaled_wd_hidden, unscaled_wd_output) = self.backwardPropagate( all_data[i,:], all_expected_output[i,:] )
            for to_idx in range(len(self.output_layer)):
                output_node = self.output_layer[to_idx]
                for from_idx in range(len(output_node.weights)):
                    output_node.weights[from_idx] += learning_rate * unscaled_wd_output[from_idx,to_idx]
            for to_idx in range(len(self.hidden_layer)):
                hidden_node = self.hidden_layer[to_idx]
                for from_idx in range(len(hidden_node.weights)):
                    hidden_node.weights[from_idx] += learning_rate * unscaled_wd_hidden[from_idx,to_idx]
            if verbose:
                print "\n-------------Weights after training data point",i,"----------------"
                self.dump()

    # perform backpropagation for each data pt, updating the weights
    # after each point. But do the points in a random order.
    # input:
    #     all_data: N x F matrix
    #     all_expected_output: N x C matrix
    def stochasticEpoch(self, all_data, all_expected_output, learning_rate):
        N = all_data.shape[0]
        for i in np.random.permutation(N):
            (mse, unscaled_wd_hidden, unscaled_wd_output) = self.backwardPropagate( all_data[i,:], all_expected_output[i,:] )
            for to_idx in range(len(self.output_layer)):
                output_node = self.output_layer[to_idx]
                for from_idx in range(len(output_node.weights)):
                    output_node.weights[from_idx] += learning_rate * unscaled_wd_output[from_idx,to_idx]
            for to_idx in range(len(self.hidden_layer)):
                hidden_node = self.hidden_layer[to_idx]
                for from_idx in range(len(hidden_node.weights)):
                    hidden_node.weights[from_idx] += learning_rate * unscaled_wd_hidden[from_idx,to_idx]

    # perform backpropagation for each data pt, summing the weight-changes
    # Update the weights at the end of the epoch.
    # input:
    #     all_data: N x F matrix
    #     all_expected_output: N x C matrix
    def batchEpoch(self, all_data, all_expected_output, learning_rate):
        N = all_data.shape[0]
        F = all_data.shape[1]
        C = all_expected_output.shape[1]
        sum_unscaled_wd_output = np.zeros( (C+1,C) )
        sum_unscaled_wd_hidden = np.zeros( (F+1,C) )
        for i in range(N):
            (mse, unscaled_wd_hidden, unscaled_wd_output) = self.backwardPropagate( all_data[i,:], all_expected_output[i,:] )
            sum_unscaled_wd_output += unscaled_wd_output
            sum_unscaled_wd_hidden += unscaled_wd_hidden
        for to_idx in range(len(self.output_layer)):
            output_node = self.output_layer[to_idx]
            for from_idx in range(len(output_node.weights)):
                output_node.weights[from_idx] += learning_rate * sum_unscaled_wd_output[from_idx,to_idx]
        for to_idx in range(len(self.hidden_layer)):
            hidden_node = self.hidden_layer[to_idx]
            for from_idx in range(len(hidden_node.weights)):
                hidden_node.weights[from_idx] += learning_rate * sum_unscaled_wd_hidden[from_idx,to_idx]

    # Return the mean error of the output for all inputs.
    # The error is the mean squared error.
    def computeMeanError( self, all_data, all_expected_output ):
        N = all_data.shape[0]
        C = all_expected_output.shape[1]
        F = all_data.shape[1]
        all_mse = np.zeros( (N,1) )
        for i in range(N):
            actual_output = np.matrix( [self.forwardPropagate( all_data[i,:], )] )
            all_mse[i] = np.sum(np.power( actual_output - all_expected_output[i,:], 2.0 )) / float(C)
        return np.mean(all_mse)

    # Classify the given data and print the confusion matrix
    # input:
    #     all_data: N x F matrix
    #     class_vals: N x 1 matrix
    def printConfusionMatrix( self, all_data, class_vals ):
        N = all_data.shape[0]
        pred_classes = np.zeros( (N,1) )
        for i in range(N):
            output = self.forwardPropagate( all_data[i,:] )
            pred_classes[i,0] = np.argmax( output )

        cm = computeConfusionMatrix( class_vals, pred_classes )
        print "Confusion Matrix"
        print cm


    # print weights and other info to the terminal
    def dump( self ):
        for n in self.input_layer:
            n.dump()
        for n in self.hidden_layer:
            n.dump()
        for n in self.output_layer:
            n.dump()


class MultilayerPerceptronForXorData(MultilayerPerceptron):
# class for XOR. You can use the preset weights or train it

    def __init__(self):
        # load in data
        self.ddata = np.matrix( [ [0.1,0.1], [0.1,0.9], [0.9,0.1], [0.9,0.9] ] )
        N = self.ddata.shape[0]
        F = self.ddata.shape[1]
        self.classes = [['False'],['True'],['True'],['False']]
        self.class_names = np.unique( np.array(self.classes) )
        self.class_vals = np.zeros( (N,1) )
        for cidx in range(len(self.classes)):
            for i in range(self.class_names.size):
                if self.classes[cidx][0] == self.class_names[i]:
                    self.class_vals[cidx,0] = i

        # construct FF network
        MultilayerPerceptron.__init__( self, F, self.class_names.size )

    # instead of training the network, use Weka's weights
    def useWekaWeights(self):
        # for non-normalized data, using Weka's weights
        # Node 0
        self.output_layer[0].weights = [5.06, -10.45, 10.35]
        # Node 1
        self.output_layer[1].weights = [-5.06, 10.45, -10.35]
        # Node 2
        self.hidden_layer[0].weights = [9.5,-6.8,-6.8]
        # Node 3
        self.hidden_layer[1].weights = [5.9,-11.03,-11.05]

    def printOutputFromTrialData( self ):
        self.printConfusionMatrix( self.ddata, self.class_vals )

        print "----- following one forward propagation -------"
        print "input", self.ddata[0,:]
        output = self.forwardPropagate( self.ddata[0,:], verbose = True )
        print "input", self.ddata[1,:]
        output = self.forwardPropagate( self.ddata[1,:], verbose = True )
        print "input", self.ddata[2,:]
        output = self.forwardPropagate( self.ddata[2,:], verbose = True )
        print "input", self.ddata[3,:]
        output = self.forwardPropagate( self.ddata[3,:], verbose = True )

class MultilayerPerceptronForMysteryData(MultilayerPerceptron):
# class for XOR. You can use the preset weights or train it

    def __init__(self, add_noise = False):
        # load in data
        N = 100;
        F = 2;
        self.ddata = np.matrix( np.zeros( (N, F) ) );
        for i in range(N):
            for j in range(F):
                self.ddata[i,j] = random.random()
        self.classes = np.matrix( np.zeros((N,1)) );
        for i in range(N):
            if self.ddata[i,0] < self.ddata[i,1]:
                self.classes[i,0] = 1
            if add_noise and random.random() < 0.1:
                self.classes[i,0] = abs(self.classes[i,0]-1)
        self.class_vals = self.classes.copy()
        self.class_names = ['0','1']

        # construct FF network
        MultilayerPerceptron.__init__( self, F, len(self.class_names) )

    def printConfusionMatrixFromTrialData( self ):
        self.printConfusionMatrix( self.ddata, self.class_vals )

def test_first_NN():
    n0 = Neuron( 0 )
    in1 = ConstantNeuron( 1, 0.5 )
    in2 = ConstantNeuron( 2, 0.5 )
    n0.addInputNode( in1 )
    n0.addInputNode( in2 )
    print n0.getOutput()

def test_AND_NN():
    network = PerceptronForANDData( )
    network.useStephWeights()
    print "\n-------------------network ----------------------\n"
    network.dump()
    network.printOutputFromTrialData()

def train_AND_NN(learning_rate=0.3):
    network = PerceptronForANDData( )
    print "\n-------------------network at beginning ----------------------\n"
    network.dump()
    network.printOutputFromTrialData()
    # train the network
    # the expected output is that there is a one in the entry for the
    # given class value.
    # e.g. if there are 3 possible classes and it it class 1 (out of 0, 1, 2)
    # the the expected output for that data pt is (0,1,0)
    all_expected_output = np.matrix(np.zeros( (network.ddata.shape[0], network.class_names.size) )) + 0.1
    for i in range(all_expected_output.shape[0]):
        all_expected_output[i,network.class_vals[i,0]] = 0.9
    NUM_ITERS = 5000
    errs = np.zeros( (NUM_ITERS, 1) )
    for i in range(NUM_ITERS):
        network.onlineEpoch(network.ddata, all_expected_output, learning_rate)
        errs[i,0] = network.computeMeanError( network.ddata, all_expected_output )
        print "error after epoch", i,"=",errs[i,0]
    # what are the final errors?
    print "\n------------------- trained network ----------------------\n"
    network.dump()
    network.printOutputFromTrialData()

    # plot the error changing over time
    import matplotlib.pyplot as plt
    plt.plot( np.matrix( [range(NUM_ITERS)] ).T, errs )
    plt.xlabel( "Epoch" )
    plt.ylabel( "Error" )
    plt.show()

def test_xor_NN(learning_rate=0.3):
    network = MultilayerPerceptronForXorData( )
    print "\n-------------------network at beginning ----------------------\n"
    network.dump()
    network.printOutputFromTrialData()
    # train the network
    # the expected output is that there is a one in the entry for the
    # given class value.
    # e.g. if there are 3 possible classes and it it class 1 (out of 0, 1, 2)
    # the the expected output for that data pt is (0,1,0)
    all_expected_output = np.matrix(np.zeros( (network.ddata.shape[0], network.class_names.size) )) + 0.1
    for i in range(all_expected_output.shape[0]):
        all_expected_output[i,network.class_vals[i,0]] = 0.9
    NUM_ITERS = 5000
    errs = np.zeros( (NUM_ITERS, 1) )
    for i in range(NUM_ITERS):
        network.onlineEpoch(network.ddata, all_expected_output, learning_rate)
#         network.stochasticEpoch(network.ddata, all_expected_output, learning_rate)
#         network.batchEpoch(network.ddata, all_expected_output, learning_rate)
        errs[i,0] = network.computeMeanError( network.ddata, all_expected_output )
        print "error after epoch", i,"=",errs[i,0]
    # what are the final errors?
    print "\n------------------- trained network ----------------------\n"
    network.dump()
    network.printOutputFromTrialData()

    # plot the error changing over time
    import matplotlib.pyplot as plt
    plt.plot( np.matrix( [range(NUM_ITERS)] ).T, errs )
    plt.xlabel( "Epoch" )
    plt.ylabel( "Error" )
    plt.show()

def test_trained_xor_NN():
    network = MultilayerPerceptronForXorData( )
    # what are the original errors?
    network.printOutputFromTrialData()
    # "train" it by just putting in Weka's results
    network.useWekaWeights()
    # what are the final errors?
    network.printOutputFromTrialData()
    return

def train_mystery(learning_rate, add_noise):
    random.seed(0) # make it so every time this runs, it will use the same
       # set of "random" numbers
    network = MultilayerPerceptronForMysteryData(add_noise)
    print "\n------------------- untrained network ----------------------\n"
    network.dump()
    network.printConfusionMatrixFromTrialData()
    all_expected_output = np.matrix(np.zeros( (network.ddata.shape[0], len(network.class_names)) )) + 0.1
    for i in range(all_expected_output.shape[0]):
        all_expected_output[i,network.class_vals[i,0]] = 0.9
    NUM_ITERS = 500
    errs = np.zeros( (NUM_ITERS, 1) )
    for i in range(NUM_ITERS):
        network.onlineEpoch(network.ddata, all_expected_output, learning_rate)
#         network.stochasticEpoch(network.ddata, all_expected_output, learning_rate)
#         network.batchEpoch(network.ddata, all_expected_output, learning_rate)
        errs[i,0] = network.computeMeanError( network.ddata, all_expected_output )
        print "error after epoch", i,"=",errs[i,0]
    # what are the final errors?
    print "\n------------------- trained network ----------------------\n"
    network.printConfusionMatrixFromTrialData()
    return network

    # plot the error changing over time
#     import matplotlib.pyplot as plt
#     plt.plot( np.matrix( [range(NUM_ITERS)] ).T, errs )
#     plt.xlabel( "Epoch" )
#     plt.ylabel( "Error" )
#     plt.show()

def test_mystery( net ):
    # create 100 test points
    N = 100;
    F = 2;
    random.seed(1)
    ddata = np.matrix( np.zeros( (N, F) ) );
    for i in range(N):
        for j in range(F):
            ddata[i,j] = random.random()
    class_vals = np.matrix( np.zeros((N,1)) );
    for i in range(N):
        if ddata[i,0] < ddata[i,1]:
            class_vals[i,0] = 1
    print "\n------------------- test data ----------------------\n"
    net.printConfusionMatrix( ddata, class_vals )

def test_multilayer():
    ninput = 4
    noutput = 3
    data = Data("iris_proj8_all.csv")
    X = data.getData(data.getHeaders())[:,:4]
    y = data.getData(data.getHeaders())[:,4]

    net = MultilayerPerceptron(ninput, noutput)
    net.onlineEpoch(X, y, 0.3, verbose=True)


if __name__ == '__main__':
    #test_first_NN()
    #test_trained_xor_NN(weka_weights=True)
    #test_xor_NN(learning_rate = 0.3)
    #net = train_mystery(learning_rate=1, add_noise=True)
    #test_mystery( net )
    #test_AND_NN()
    test_multilayer()
