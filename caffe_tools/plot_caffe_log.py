import os
import sys
import numpy as np

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import math
import pylab
import sys
import argparse
import re
from pylab import figure, show, legend, ylabel

from mpl_toolkits.axes_grid1 import host_subplot


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Makes a plot from Caffe output')
  parser.add_argument('training_log', help='file of captured stdout and stderr')
  parser.add_argument('output_path', help='folder where the generated plots will be stored')
  parser.add_argument('output_files_prefix', help='prefix for the output files')
  args = parser.parse_args()

  output_path = args.output_path
  if output_path[-1] != '/':
    output_path = output_path + '/'
  
  f = open(args.training_log, 'r')

  training_iterations = []
  training_loss = []
  training_accuracy = []

  test_iterations = []
  test_accuracy = []
  test_loss = []

  for line in f:

    if '] Iteration ' in line and 'loss = ' in line:
      arr = re.findall(r'ion \b\d+\b,', line)
      training_iterations.append(int(arr[0].strip(',')[4:]))
      #training_loss.append(float(line.strip().split(' = ')[-1]))
      #check_train = True

    if '] Iteration ' in line and 'Testing net' in line:
      arr = re.findall(r'ion \b\d+\b,', line)
      test_iterations.append(int(arr[0].strip(',')[4:]))
      #check_test = True

    if 'Test net output' in line and 'loss' in line:
      test_loss.append(float(line.strip().split(' ')[-2]))

    if 'Test net output' in line and 'accuracy' in line:
      test_accuracy.append(float(line.strip().split(' ')[-1]))

    if 'Train net output' in line and 'loss' in line:
      training_loss.append(float(line.strip().split(' ')[-2]))

    if 'Train net output' in line and 'accuracy' in line:
      training_accuracy.append(float(line.strip().split(' ')[-1]))

  print 'train iterations len: ', len(training_iterations)
  print 'train loss len: ', len(training_loss)
  print 'train accuracy len: ', len(training_accuracy)
  print 'test loss len: ', len(test_loss)
  print 'test iterations len: ', len(test_iterations)
  print 'test accuracy len: ', len(test_accuracy)

  if len(training_iterations) != len(training_accuracy): #awaiting test...
    print 'mis-match (training)'
    print len(training_iterations[0:-1])
    training_iterations = training_iterations[0:-1]
    test_iterations = test_iterations[0:-1]
    test_accuracy = test_accuracy[0:-1]
    test_loss = test_loss[0:-1]

  '''if len(test_iterations) != len(test_accuracy): #awaiting test...
    print 'mis-match (test)'
    print len(test_iterations[0:-1])
    test_iterations = test_iterations[0:-1]'''

  f.close()
  

  # Plot loss
  host = host_subplot(111)#, axes_class=AA.Axes)
  plt.subplots_adjust(right=0.75)

  #par1 = host.twinx()

  host.set_xlabel("Iterations")
  host.set_ylabel("Loss")
  #par1.set_ylabel("validation accuracy")
 
  if len(training_loss) > 0:
    p1, = host.plot(training_iterations, training_loss, label="Train")
  if len(test_loss) > 0:
    p3, = host.plot(test_iterations, test_loss, label="Validation")
  #p2, = par1.plot(test_iterations, test_accuracy, label="validation accuracy")
  #p4, = par1.plot(training_iterations, training_accuracy, label="train accuracy")

  host.legend(loc=1)

  #host.axis["left"].label.set_color(p1.get_color())
  #par1.axis["right"].label.set_color(p2.get_color())

  plt.draw()
  plt.show()

  plt.savefig(output_path + args.output_files_prefix + '_loss.png')

  # Plot accuracy
  plt.clf()
  host = host_subplot(111)#, axes_class=AA.Axes)
  plt.subplots_adjust(right=0.75)

  #par1 = host.twinx()

  host.set_xlabel("Iterations")
  host.set_ylabel("Accuracy")
  #par1.set_ylabel("validation accuracy")
 
  if len(training_accuracy) > 0:
    p1, = host.plot(training_iterations, training_accuracy, label="Train")
  if len(test_accuracy) > 0:
    p3, = host.plot(test_iterations, test_accuracy, label="Validation")
  #p2, = par1.plot(test_iterations, test_accuracy, label="validation accuracy")
  #p4, = par1.plot(training_iterations, training_accuracy, label="train accuracy")

  host.legend(loc=4)

  #host.axis["left"].label.set_color(p1.get_color())
  #par1.axis["right"].label.set_color(p2.get_color())

  plt.draw()
  plt.show()

  plt.savefig(output_path + args.output_files_prefix + '_accuracy.png')


  # Plot accuracy+loss
  plt.clf()  # clear the previous plot
  host = host_subplot(111)#, axes_class=AA.Axes)
  plt.subplots_adjust(right=0.75)

  par1 = host.twinx()

  host.set_xlabel("Iterations")
  host.set_ylabel("Loss")
  par1.set_ylabel("Accuracy")
 

  ncol = 0
  if len(training_loss) > 0:
    p1, = host.plot(training_iterations, training_loss, label="Training loss")
    ncol += 1
  if len(test_loss) > 0:
    p3, = host.plot(test_iterations, test_loss, label="Validation loss")
    ncol += 1
  if len(training_accuracy) > 0:
    p2, = par1.plot(training_iterations, training_accuracy, label="Training accuracy")
    ncol += 1
  if len(test_accuracy) > 0:
    p4, = par1.plot(test_iterations, test_accuracy, label="Validation accuracy")
    ncol += 1

  #host.legend(loc=4)

  #host.axis["left"].label.set_color(p1.get_color())
  #par1.axis["right"].label.set_color(p2.get_color())

  # Shrink current axis's height by 10% on the bottom
  box = host.get_position()
  host.set_position([box.x0, box.y0 + box.height * 0.1,
                   box.width, box.height * 0.9])

  # Put a legend below current axis
  host.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=ncol/2)

  plt.draw()
  plt.show()

  plt.savefig(output_path + args.output_files_prefix + '_both.png')