#ifndef NEURAL_NET_H
#define NEURAL_NET_H

typedef struct {
  size_t count;
  //Input layer is the first element
  //of this array. Other elements are hidden layers
  //last element is the output layer. This matrix
  //stores the result of activation function like 
  //sigmoid
  Matrix *layers;
  Matrix *weights;
  Matrix *biases;

} NeuralNetwork;

/*
  Params:
  size_t *nModel = model array = {input-layer, hidden-layer, ..., output-layer}
  size_t modelCount = length of model array
*/
NeuralNetwork createNetwork(size_t *nModel, size_t modelCount);
void printNetwork(NeuralNetwork n, const char *name);
void randNetwork(NeuralNetwork n, float rStart, float rEnd);
void forwardNetwork(NeuralNetwork n);
void resetNetwork(NeuralNetwork n);

#define PRINT_NN(n) printNetwork(n, #n)
#define INPUT_LAYER_NN(n) (n).layers[0]
//No need to to n.count-1 because count is
//already reduced by one when our neural network
//is initialized
#define OUTPUT_LAYER_NN(n) (n).layers[(n).count]

#endif


#ifdef NEURAL_NET_IMPL

void randNetwork(NeuralNetwork n, float rStart, float rEnd) {
  for(size_t i = 0; i < n.count; i++) {
    randMatrix(n.weights[i], rStart, rEnd);
    randMatrix(n.biases[i], rStart, rEnd);
  }
}

void printNetwork(NeuralNetwork n, const char *name) {
  printf("--Neural Network: %s--\n", name);

  char buf[125];
  for(size_t i = 0; i < n.count; i++) {
    snprintf(buf, sizeof(buf), "ws%zu", i);
    printMatrix(*n.weights, buf);
    printf("\n");
    snprintf(buf, sizeof(buf), "bs%zu", i);
    printMatrix(*n.biases, buf);
    printf("\n");
  }

  printf("-- --\n\n");
}

NeuralNetwork createNetwork(size_t *nModel, size_t modelCount) {
  NeuralNetwork nn;

  ASSERT_NN(modelCount > 0);

  nn.count = modelCount - 1;
  //Create an array of weights excluding the input layer.
  //hidden and output layers are required to have weights
  //because they are neurons.
  nn.weights = NN_MALLOC(sizeof(*nn.weights) * nn.count);
  ASSERT_NN(nn.weights != NULL);

  //Create an array of biases excluding the input layer
  //hidden and output layers are required to have biases
  //because they are neurons.
  nn.biases = NN_MALLOC(sizeof(*nn.biases) * nn.count);
  ASSERT_NN(nn.biases != NULL);

  //Create an array of layers of neural network. This
  //includes input, hidden and output layers
  nn.layers = NN_MALLOC(sizeof(*nn.layers) * modelCount);
  ASSERT_NN(nn.layers != NULL);

  //Set input layer and number of inputs
  nn.layers[0] = matrixAlloc(1, nModel[0]);

  for(size_t i = 1; i < modelCount; i++) {
    //weights matrix rows must be equal to their corresponding
    //input layers in order to make dot product work. In order to create
    //a neuron, we need to multiply input layer to weights and then
    //biases are added to each product of the previous operation.
    //Inputs in this context can be the inputs in input layer or
    //the inputs of neurons in hidden layers. Two weights if equal
    //to neuron. For example, 2x2 weight metrix is equal to two neurons
    //with two inputs in input layer. Each neuron has a connection to
    //each input and the connections contain the weights.
    nn.weights[i-1] = matrixAlloc(nn.layers[i-1].cols, nModel[i]);
    //Biases is always a single row. Each neuron has one bias.
    nn.biases[i-1] = matrixAlloc(1, nModel[i]);
    //set hidden layers
    nn.layers[i] = matrixAlloc(1, nModel[i]);
  }

  return nn;
}

void forwardNetwork(NeuralNetwork n) {

  /*
    At 0, we multiply input layer with weights and
    then store the value to the next layer which is
    the hidden layer. Then, add each neuron's bias
    to each product of the previous operation and
    then use activation function to each result in
    order to clamp values from 0 to 1.

    At next iteration, we multiply the previous result
    with the next set of weights and then add biases
    and activation function and repeat.
  */
  for(size_t i = 0; i < n.count; i++) {
    matrixDot(n.layers[i+1], n.layers[i], n.weights[i]);
    matrixSum(n.layers[i+1], n.biases[i]);
    applySigmoid(n.layers[i+1]);
  }
}

void resetNetwork(NeuralNetwork n) {

  for(size_t i = 0; i < n.count; i++) {
    fillMatrix(n.weights[i], 0);
    fillMatrix(n.biases[i], 0);
    fillMatrix(n.layers[i], 0);
  }
  fillMatrix(n.weights[n.count], 0);
}

#endif