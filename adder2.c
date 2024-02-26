#define MATRIX_IMPL
#define NEURAL_NET_IMPL
#define COMPUTE_IMPL

#include <string.h>
#include <stdbool.h>

#include "matrix.h"
#include "neuralnet.h"
#include "compute.h"

int main(int argc, char *argv[]) {
  //Number of bits allowed. If sum of bits 
  //of adder is more this bit, it means that
  //the sum overflows. 
  const size_t BITS = 2;

  //left shift to number of bits.
  //This is just like multiplying number by 2
  //except if value is 1. Take a look at these
  //expessions: 
  //1<<1 = 1 0 = 1+1 = 2; 1<<2 = 1 0 0 = (1+1)*2 = 4;
  //1<<3 = 1 0 0 = (1+1)*2*2 = 8 
  size_t n = (1<<BITS);
  size_t rows = n*n;

  size_t inputCols = 2*BITS;
  size_t outputCols = BITS+1;

  //This holds two pairs of bits that will
  //be added
  Matrix ti = matrixAlloc(rows, inputCols);
  
  //This holds the sum of pairs of bits
  //and the last column is for the carry
  Matrix to = matrixAlloc(rows, outputCols);

  //Loop rows
  for(size_t i = 0; i < ti.rows; i++) {
    //if n = 4
    //i = 1-3; x = 0 
    //i = 4-16; x = greater than 0
    //first bit set
    size_t x = i/n;
    //if n = 4
    //i = 1-3; y = 1-3
    //i = 4-16; y = remainder
    //second bit set
    size_t y = i%n;

    //sum of both bit set
    size_t z = x + y;

    //loop columns and fill up the matrix
    for(int j = 0; j < BITS; j++) {
      //Note: use &1 to ignore all bits except
      //for the leftmost bit. For example:
      //1011 & 0001 = 0001
      
      //Note: bit arrangement in each row of
      //'ti' doesn't matter because we're just 
      //doing binary addition. For example, 2 = 1 0
      //but in this loop it will be arranged
      //'0 1' because we extract the first leftmost
      //bit of a number and this loop arrange bits
      //from left to right. Adding '1 0' or '0 1'
      //will have equal answer.

      //Extract leftmost bit
      //Example:
      //x = 1 0 1
      //x>>0 = 1 0 1
      //(x>>0)&1 = 1 0 1 & 0 0 1 = 0 0 1 = [1]
      //x>>1 = 1 0
      //(x>>1)&1 = 1 0 & 0 1 = 0 0 = [0]
      //x>>2 = 1
      //(x>>2)&1 = 1 & 1 = [1]

      ti.start[getCell(ti, i, j)] = (x>>j)&1;
      //Transfer y's leftmost bit to the next sub-array
      ti.start[getCell(ti, i, j + BITS)] = (y>>j)&1;
      
      to.start[getCell(to, i, j)] = (z>>j)&1;
    }

    //Put the carry value in the last column of 'to'.
    //If the sum is greater than the number limit,
    //the sum overflows. Thus, the condition returns
    //0. Else returns 1. For example, if the max bits
    //of adder is 2, value 3(binary = 1 1) is the 
    //max number that we can store. Values greater than
    //3 makes the sum overflows. For example, 
    //4(binary = 1 0 0) has 3 bits. The '1' rightmost bit
    //is the carry.
    to.start[getCell(to, i, BITS)] = z >= n;
  }

  float learnRate = 1;
  size_t nModel[] = {BITS*2, BITS*2+1, BITS+1};
  NeuralNetwork neuralNet = createNetwork(nModel, ARRAY_LENGTH(nModel));
  NeuralNetwork gradient = createNetwork(nModel, ARRAY_LENGTH(nModel));
  randNetwork(neuralNet, 0, 1);

  char gradType = 'b';
  if(argv[1] != NULL) {
    if(strcmp(argv[1], "b") == 0) {
      gradType = 'b';
    }
    else if(strcmp(argv[1], "f") == 0) {
      gradType = 'f';
    }
    else gradType = 'b'; //default
  } else gradType = 'b'; //default


  printf("Cost Before Training: %f\n", computeCost(neuralNet, ti, to));
  for(int i = 0; i < 10*1000; i++) {
    //Try comparing the performance of finite diff and
    //back propagation by using one of them at a time.
    if(argv[1] != NULL) {
      if(strcmp(argv[1], "b") == 0) {
        backProp(neuralNet, gradient, ti, to);
      }
      else if(strcmp(argv[1], "f") == 0) {
        computeFiniteDiff(neuralNet, gradient, 1e-1, ti, to);
      }
      else backProp(neuralNet, gradient, ti, to); //default
    }
    else {
      //default
      backProp(neuralNet, gradient, ti, to);
    }

    trainNetwork(neuralNet, gradient, learnRate);
    printf("%d: Cost(Training): %f\n", i, computeCost(neuralNet, ti, to));
  }
  printf("Cost After Training: %f\n", computeCost(neuralNet, ti, to));

  if(gradType == 'b') {
    printf("\nGradient used: Back Propagation\n\n");
  }
  else if(gradType == 'f') {
    printf("\nGradient used: Finite Difference\n\n");
  }

  size_t fails = 0;
  size_t total = 0;
  for(size_t x = 0; x < n; x++) {
    for(size_t y = 0; y < n; y++) {
      size_t sum = x + y;

      printf("%zu + %zu = ", x, y);

      for(int j = 0; j < BITS; j++) {

        INPUT_LAYER_NN(neuralNet).
          start[getCell(INPUT_LAYER_NN(neuralNet), 0, j)] = (x>>j)&1;
        INPUT_LAYER_NN(neuralNet).
          start[getCell(INPUT_LAYER_NN(neuralNet), 0, j + BITS)] = (y>>j)&1;
        
      }
      forwardNetwork(neuralNet);
      total++;

      if(OUTPUT_LAYER_NN(neuralNet).
          start[getCell(OUTPUT_LAYER_NN(neuralNet), 0, BITS)] > 0.5f) {

        if(sum < n) {
          printf("%zu + %zu = %zu %s", 
            x, y, sum, "| Expected: overflow | Actual: no overflow");
          fails++;
        } else printf("overflow\n");

      } else {
        size_t z = 0;
        for(int j = 0; j < BITS; j++) {
          //Neural network output an approximation between 0 to 1.
          //In this model's equation, if output leans toward 1, 
          //the model predicts that the output is 1. Otherwise, 
          //output is 0. The model's output gets closer to 1 or 0
          //if the model gets more training. Thus, we use 0.5f to decide if
          //a bit should be 1 or 0.
          size_t bit = 
              OUTPUT_LAYER_NN(neuralNet).
                start[getCell(OUTPUT_LAYER_NN(neuralNet), 0, j)] > 0.5f;

          //extract bits to get the sum of the adder.
          //Example:
          //first bit = 1; z = 0
          //bit<<0 = 1 -> z | bit<<0 = 1 | 0 = 1
          //second bit = 0; z = 1
          //bit<<1 = 0 0 -> z | bit<<1 = 0 0 | 1 = 0 1
          //third bit = 1; z = 0 1
          //bit<<2 = 1 0 0 -> z | bit<<2 = 1 0 0 |  0 1 = 1 0 1
          //sum = 5 
          z |= bit<<j;
        }

        if(z != sum) {
          printf("Actual Sum: %zu | Expected Sum: %zu (wrong answer)\n", z, sum);
          fails++;
        }
        else printf("%zu\n", z);
      }   

    }
  }

  printf("\nfails/total = error rate\n");
  printf("%zu / %zu = %.2f%s\n", fails, total, ((float)fails/(float)total)*100, "%");

}