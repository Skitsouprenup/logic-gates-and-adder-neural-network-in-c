#include <stdio.h>
#include <math.h>

#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
  size_t rows;
  size_t cols;
  size_t stride;
  //first element pointer
  //of the matrix. This is a
  //linear array that acts
  //as 2D array. stride is used
  //to divide this array into multiple
  //array. Thus, acting as 2D array
  float *start;
} Matrix;

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif

#ifndef ASSERT_NN
#include <assert.h>
#define ASSERT_NN assert
#endif

float randFloat();
float sigmoid(float x);

Matrix matrixAlloc(size_t rows, size_t cols);
void matrixDot(Matrix dst, Matrix a, Matrix b);
void matrixSum(Matrix dst, Matrix a);
size_t getCell(Matrix matrix, size_t row, size_t col);
void printMatrix(Matrix matrix, const char *label);
void randMatrix(Matrix matrix, float rStart, float rEnd);
void fillMatrix(Matrix matrix, float value);
void matrixCopy(Matrix dst, Matrix src);
void applySigmoid(Matrix matrix);
Matrix getMatrixRow(Matrix m, size_t row);

//Add parentheses in-between the variable in order to
//prevent any problems when the content of variable is
//an expression.
#define ARRAY_LENGTH(a) sizeof((a))/sizeof((a)[0])

//#l converts any characters to char literal even
//it's a reserved keyword like *, &, etc.
//It also converts variable into its variable name
#define PRINT_MATRIX(m) printMatrix(m,#m);

#endif

#ifdef MATRIX_IMPL

float sigmoid(float x) {
  return 1.0f/(1.0f + expf(-x));
}

float randFloat() {
  return (float)rand()/(float)RAND_MAX;
}

size_t getCell(Matrix matrix, size_t row, size_t col) {
  //row*matrix.cols is a way to get to the first
  //element of each pseudo-array in linear array
  return row*matrix.stride + col;
}

Matrix matrixAlloc(size_t rows, size_t cols) {
  Matrix matrix;
  matrix.rows = rows;
  matrix.cols = cols;
  matrix.stride = cols;
  matrix.start = NN_MALLOC(sizeof(Matrix) * rows * cols);
  ASSERT_NN(matrix.start != NULL);
  return matrix;
}
void matrixDot(Matrix dst, Matrix a, Matrix b){

  //Reference: https://www.mathsisfun.com/algebra/matrix-multiplying.html
  //To multiply matrix using dot product, we make
  //sure that the columns of the first matrix is equal
  //to the rows of second matrix
  ASSERT_NN(a.cols == b.rows);
  //The final matrix must have rows that is equal to
  //the first matrix rows and columns that is equal to
  //second matrix
  ASSERT_NN(dst.rows == a.rows);
  ASSERT_NN(dst.cols == b.cols);

  size_t refMat = a.cols;
  for(size_t i = 0; i < dst.rows; i++) {
    for(size_t j = 0; j < dst.cols; j++) {

      dst.start[getCell(dst, i, j)] = 0;
      for(size_t k = 0; k < refMat; k++) {
        //Read the explanation above to understand this
        //statement. Traverse 'a' matrix from left to right
        //and traverse 'b' matrix from top to bottom. Multiply
        //each value in 'a' and 'b' during traversal and
        //sum the products of each multiplication.
        dst.start[getCell(dst, i, j)] +=
          a.start[getCell(a, i, k)] * b.start[getCell(b, k, j)];
      }
    }
  }
}
void matrixSum(Matrix dst, Matrix matrix){
  ASSERT_NN(dst.rows == matrix.rows);
  ASSERT_NN(dst.cols == matrix.cols);

  for(size_t i = 0; i < matrix.rows; i++) {
    for(size_t j = 0; j < matrix.cols; j++) {
      dst.start[getCell(dst, i, j)] +=
      matrix.start[getCell(matrix, i, j)];
    }
  }
}
void printMatrix(Matrix matrix, const char *label){
  printf("%s\n", label);
  for(size_t i = 0; i < matrix.rows; i++) {
    for(size_t j = 0; j < matrix.cols; j++) {
      printf("%f ", matrix.start[getCell(matrix, i, j)]);
    }
    printf("\n");
  }
  printf("\n");
}
void randMatrix(Matrix matrix, float rStart, float rEnd) {
  for(int i = 0; i < matrix.rows; i++) {
    for(int j = 0; j < matrix.cols; j++) {
      matrix.start[getCell(matrix, i, j)] = 
        randFloat()*(rEnd - rStart) + rStart;
    }
  }
}
void fillMatrix(Matrix matrix, float value) {
  for(int i = 0; i < matrix.rows; i++) {
    for(int j = 0; j < matrix.cols; j++) {
      matrix.start[getCell(matrix, i, j)] = value;
    }
  }
}
void matrixCopy(Matrix dst, Matrix src) {
  ASSERT_NN(dst.rows == src.rows);
  ASSERT_NN(dst.cols == src.cols);

  for(int i = 0; i < dst.rows; i++) {
    for(int j = 0; j < dst.cols; j++) {
      dst.start[getCell(dst, i, j)] =
        src.start[getCell(src, i, j)];
    }
  }
}
Matrix getMatrixRow(Matrix m, size_t row) {

  return (Matrix){
    .rows = 1,
    .cols = m.cols,
    .stride = m.stride,
    .start = &m.start[getCell(m, row, 0)]
  };

}

void applySigmoid(Matrix matrix) {
  for(int i = 0; i < matrix.rows; i++) {
    for(int j = 0; j < matrix.cols; j++) {
      matrix.start[getCell(matrix, i, j)] =
        sigmoid(matrix.start[getCell(matrix, i, j)]);
    }
  }
}

#endif