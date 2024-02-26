#include <time.h>

#define MATRIX_IMPL
#define NEURAL_NET_IMPL
#define COMPUTE_IMPL

/*
  Order of includes matters if one header uses 
  any content of another header. For example,
  if I move compute.h on top of neuralnet.h
  I'll get errors
*/
#include "matrix.h"
#include "neuralnet.h"
#include "compute.h"
#include "samples.h"

int main() {
  srand(100);

  float *td = xor_train_data;

  //Stride is used to divide linear array into
  //multiple sub arrays depicting a 2D array.
  size_t stride = 3;
  size_t n = 4;
  //size_t n = sizeof(xor_train_data)/sizeof(xor_train_data[0])/stride;

  Matrix ti = {
    .rows = n,
    .cols = 2,
    .stride = stride,
    .start = td
  };

  Matrix to = {
    .rows = n,
    .cols = 1,
    .stride = stride,
    .start = td + 2 //td[2]
  };

  //We use epsilon to find the local minimum
  //of our model. This number is a 'magic' number.
  //It means that I just found this number through
  //trial and error. This variable is used for finite
  //difference
  float eps = 1e-1;
  float learnRate = 1;

  //First element is the number of inputs
  //Second to before last element is the number of hidden layers
  //last element is output layer
  size_t nModel[] = {2, 2, 1};
  NeuralNetwork neuralNet = createNetwork(nModel, ARRAY_LENGTH(nModel));
  NeuralNetwork gradient = createNetwork(nModel, ARRAY_LENGTH(nModel));
  randNetwork(neuralNet, 0, 1);

  /** Train **/

  printf("Cost %f\n", computeCost(neuralNet, ti, to));
  for(int i = 0; i < 5000; i++) {
    //computeFiniteDiff(neuralNet, gradient, eps, ti, to);
    backProp(neuralNet, gradient, ti, to);
    trainNetwork(neuralNet, gradient, learnRate);
    //printf("Cost %f\n", computeCost(neuralNet, ti, to));    
  }
  printNetwork(gradient, "gradient");
  printf("Cost %f\n", computeCost(neuralNet, ti, to));

  /** **/

  for(size_t i = 0; i < 2; i++) {
    for(size_t j = 0; j < 2; j++) {
      INPUT_LAYER_NN(neuralNet).
        start[getCell(INPUT_LAYER_NN(neuralNet), 0, 0)] = i;
      INPUT_LAYER_NN(neuralNet).
        start[getCell(INPUT_LAYER_NN(neuralNet), 0, 1)] = j;
      forwardNetwork(neuralNet);

      //%zu is used to display size_t value on the console.
      //%ld can also be used.
      printf(
        "%zu | %zu = %f\n", i, j, 
        OUTPUT_LAYER_NN(neuralNet).
          start[getCell(OUTPUT_LAYER_NN(neuralNet), 0, 0)]
      );
    }
  }
  
}