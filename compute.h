#ifndef COMPUTE_H
#define COMPUTE_H

void computeLearnRate(Matrix *n, Matrix *g, float rate, size_t index);

void computeGradient(
  NeuralNetwork n,
  Matrix *m, 
  Matrix *mg,
  Matrix ti,
  Matrix to,
  float eps, 
  float costVal,
  size_t index
);

float computeCost(NeuralNetwork n, Matrix tInput, Matrix tOutput);

void computeFiniteDiff(
  NeuralNetwork n, 
  NeuralNetwork gradient, 
  float eps, 
  Matrix ti, 
  Matrix to
);

void trainNetwork(NeuralNetwork n, NeuralNetwork g, float rate);

void backProp(NeuralNetwork n, NeuralNetwork g, Matrix tInput, Matrix tOutput);

#endif

#ifdef COMPUTE_IMPL

void computeLearnRate(Matrix *n, Matrix *g, float rate, size_t index) {
  for(size_t i = 0; i < n[index].rows; i++) {
    for(size_t j = 0; j < n[index].cols; j++) {
      n[index].start[getCell(n[index], i, j)] -= 
        rate*g[index].start[getCell(g[index], i, j)];
    }
  }
}

/*
  Params:
  n = Neural Network
  m = matrix
  mg = matrix(gradient)
  eps = epsilon
  costVal = initial cost

  gradient(finite difference) formula:
  cost(params + eps) - cost(params) / eps

  Take note that the real gradient descent 
  formula uses real derivative. In our case
  we use epsilon instead. The disadvantage of
  finite difference is that we need to guess
  an epsilon that is compatible to our model.
  
  We can test if the epsilon is compatible
  to our model if the result of cost function
  is going down to 0
*/
void computeGradient(
  NeuralNetwork n,
  Matrix *m, 
  Matrix *mg,
  Matrix ti,
  Matrix to,
  float eps, 
  float costVal,
  size_t index
) {
  float saved;

  for(size_t i = 0; i < m[index].rows; i++) {
    for(size_t j = 0; j < m[index].cols; j++) {
      //save the original value first
      saved = m[index].start[getCell(m[index], i, j)];
      //add epsilon
      m[index].start[getCell(m[index], i, j)] += eps;

      //Store the gradient
      mg[index].start[getCell(m[index], i, j)] = 
        (computeCost(n, ti, to) - costVal)/eps;

      //return the original value back
      m[index].start[getCell(m[index], i, j)] = saved;
    }
  }

}

float computeCost(NeuralNetwork n, Matrix tInput, Matrix tOutput) {
  ASSERT_NN(tInput.rows == tOutput.rows);
  ASSERT_NN(tOutput.cols == OUTPUT_LAYER_NN(n).cols);

  size_t r = tInput.rows;
  
  float costVal = 0;
  for(size_t i = 0; i < r; i++) {
    Matrix inputRow = getMatrixRow(tInput, i);
    Matrix outputRow = getMatrixRow(tOutput, i);

    matrixCopy(INPUT_LAYER_NN(n), inputRow);
    forwardNetwork(n);

    size_t c = tOutput.cols;
    for(size_t j = 0; j < c; j++) {
      float diff = 
        OUTPUT_LAYER_NN(n).start[getCell(OUTPUT_LAYER_NN(n), 0, j)] - 
        outputRow.start[getCell(outputRow, 0, j)];
      costVal += diff*diff;
    }
  }

  return costVal/r;
}

/*
  An alternative to back propagation. Easy to implement
  but it's inaccurate when dealing with larger models
*/
void computeFiniteDiff(
  NeuralNetwork n, 
  NeuralNetwork gradient, 
  float eps, 
  Matrix ti, 
  Matrix to
) {
  //Initial cost
  float costVal = computeCost(n, ti, to);

  for(size_t i = 0; i < n.count; i++) { 
    computeGradient(n, n.weights, gradient.weights, ti, to, eps, costVal, i);
    computeGradient(n, n.biases, gradient.biases, ti, to, eps, costVal, i);
  }
}

void trainNetwork(NeuralNetwork n, NeuralNetwork g, float rate) {
  for(size_t i = 0; i < n.count; i++) {
    computeLearnRate(n.weights, g.weights, rate, i);
    computeLearnRate(n.biases, g.biases, rate, i);
  }
}

//Back Propagation
void backProp(NeuralNetwork n, NeuralNetwork g, Matrix tInput, Matrix tOutput) {
  ASSERT_NN(tInput.rows == tOutput.rows);
  ASSERT_NN(OUTPUT_LAYER_NN(n).cols == tOutput.cols);
  size_t r = tInput.rows;
  size_t c = tOutput.cols;

  resetNetwork(g);

  //loop through the row of samples
  for(size_t i = 0; i < r; i++) {
    //Copy a training input row to first layer.
    matrixCopy(INPUT_LAYER_NN(n), getMatrixRow(tInput, i));
    //Activate neural network
    forwardNetwork(n);

    //reset layers value to 0.
    for(int j = 0; j < n.count; j++) {
      fillMatrix(g.layers[j], 0);
    }

    //Loop through training output columns
    for(size_t j = 0; j < c; j++) {
      //Store the difference between actual output and expected output
      //in output layer of gradient neural network. This network is 
      //a copy of original network. In the current state of this
      //library as the time of writing, output layer is linear and
      //only having one row.
      OUTPUT_LAYER_NN(g).start[getCell(OUTPUT_LAYER_NN(g), 0, j)] = 
        OUTPUT_LAYER_NN(n).start[getCell(OUTPUT_LAYER_NN(n), 0, j)] -
        tOutput.start[getCell(tOutput, i, j)];
    }
    
  
    //layers and biases of this neural network structure 
    //only have 1 row.
    //Loop through layers backwards. Thus, starting from the output
    //layer.
    for(size_t l = n.count; l > 0; l--) {
      //Loop through each column of the layer and compute
      //the bias derivative of each neuron in the layer
      for(size_t j = 0; j < n.layers[l].cols; j++) {
        //activation function value in the neuron of the current
        //layer
        float a = n.layers[l].start[getCell(n.layers[l], 0, j)];
        //partial derivative of cost function of next neuron
        //with respect to the current activation: ∂ai^(l)C^(l+1).
        //If loop is in output layer, that value of this variable
        //is the difference that we computed above when we traverse
        //the output layer;
        float da = g.layers[l].start[getCell(g.layers[l], 0, j)];
        //add derivative of the current cost function with respect
        //to current bias: ∂b(l)C^(1) = 2*∂ai^(l)C^(l+1)*(1-a).
        //Put the result in the previous neuron in the previous layer in
        //gradient matrix.
        g.biases[l-1].start[getCell(g.biases[l-1], 0, j)] += 2*da*a*(1-a);

        //loop through the neurons of previous layer of 'n' network
        for(size_t k = 0; k < n.layers[l-1].cols; k++) {
          //previous activation
          float pa = n.layers[l-1].start[getCell(n.layers[l-1], 0, k)];
          //previous weight
          float w = n.weights[l-1].start[getCell(n.weights[l-1], k, j)];
          //add derivative of the current cost function with respect
          //to current weight: ∂wi^(l)C^(l) = 2*∂ai^(l)C^(l+1)*(1-a)*a^(l-1)
          g.weights[l-1].start[getCell(g.weights[l-1], k, j)] += 2*da*a*(1-a)*pa;
          //add derivative of the current cost function with respect to
          //previous activation function ∂ai^(l-1)C^(l) = 
          //2*∂ai^(l)C^(l+1)*(1-a)*a^(l-1)*w^(l)
          g.layers[l-1].start[getCell(g.layers[l-1], 0, k)] += 2*da*a*(1-a)*w;
        }
      }
    }
  }

  //Loop through the model
  for(size_t i = 0; i < g.count; i++) {
    //Loop through all weights
    for(size_t j = 0; j < g.weights[i].rows; j++) {
      for(size_t k = 0; k < g.weights[i].cols; k++) {
        //Divide each weight by the number of input training data
        //in rows. This completes the gradient descent formula.
        g.weights[i].start[getCell(g.weights[i], j, k)] /= r;
      }
    }

    //Loop through all biases
    for(size_t j = 0; j < g.biases[i].rows; j++) {
      for(size_t k = 0; k < g.biases[i].cols; k++) {
        //Divide each biases by the number of input training data
        //in rows. This completes the gradient descent formula.
        g.biases[i].start[getCell(g.biases[i], j, k)] /= r;
      }
    }
  }
}

#endif