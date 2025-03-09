
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeSYMGS_ref.cpp

 HPCG routine
 */

#include "local_int_t.h"
#include <altivec.h>
#include "pveclib/vec_f64_ppc.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/*!
  Computes one step of symmetric Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  x should be initially zero on the first GS sweep, but we do not attempt to exploit this fact.
  - We then perform one back sweep.
  - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSYMGS
*/
int ComputeSYMGS_ref_c( 
    const local_int_t nrow,
    double ** restrict matrixDiagonal,  // An array of pointers to the diagonal entries A.matrixValues
    const double * const restrict rv,
    double * const restrict xv,
    double ** const restrict matrixValues,
    local_int_t ** const restrict mtxIndL,
    char * restrict nonzerosInRow)
{
  
  for (local_int_t i=0; i< nrow; i++) {
    const double * const currentValues = matrixValues[i];
    const local_int_t * const currentColIndices = mtxIndL[i];
    const int currentNumberOfNonzeros = nonzerosInRow[i];
    const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i]; // RHS value


    int curNNZ2 = currentNumberOfNonzeros - (currentNumberOfNonzeros % 2);

    vf64_t sum_v = {0.0, 0.0};

    for (int j=0; j< curNNZ2; j+= 2) {
      const local_int_t * const curCol = &currentColIndices[j];
      //sum -= currentValues[j] * xv[curCol];
      //const long long curCol0 = curCol[0];
      //const long long curCol1 = curCol[1];

      vf64_t xv_v = vec_vglfdso(xv, curCol[0]*sizeof(double), curCol[1]*sizeof(double));
//      int rank;
//      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//      if(rank == 0 && j == 0)
//        printf("%f %f %f %f \n", xv_v[0], xv_v[1], xv[curCol[0]], xv[curCol[1]]);

//      exit(0);
//      sum_v[0] -= currentValues[j] * xv[curCol[0]];
//      sum_v[1] -= currentValues[j+1] * xv[curCol[1]];
      vf64_t * const cv = (vf64_t * const)(&currentValues[j]);

      sum_v -= (*cv) * xv_v;
 
      //sum_v[0] -= currentValues[j] * xv_v[0]; //xv[curCol[0]];
      //sum_v[1] -= currentValues[j+1] * xv_v[1]; //xv[curCol[1]];
    }

    sum += sum_v[0] + sum_v[1];

    for (int j = curNNZ2; j < currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }
 
    sum += xv[i]*currentDiagonal; // Remove diagonal contribution from previous loop

    xv[i] = sum/currentDiagonal;

  }


  // Now the back sweep.

  for (local_int_t i=nrow-1; i>=0; i--) {
    const double * const currentValues = matrixValues[i];
    const local_int_t * const currentColIndices = mtxIndL[i];
    const int currentNumberOfNonzeros = nonzerosInRow[i];
    const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i]; // RHS value

    for (int j = 0; j< currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j]*xv[curCol];
    }
    sum += xv[i]*currentDiagonal; // Remove diagonal contribution from previous loop

    xv[i] = sum/currentDiagonal;
  }

    /*
    for (local_int_t i=0; i< nrow; i++) {
    const double * const currentValues = matrixValues[i];
    const local_int_t * const currentColIndices = mtxIndL[i];
    const int currentNumberOfNonzeros = nonzerosInRow[i];
    const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i]; // RHS value

    for (int j=0; j< currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }

    int const curNNZ2 = (currentNumberOfNonzeros / 2) * 2;
    vf64_t sum_v = {0.0, 0.0};

    for (int j = 0; j< curNNZ2; j += 2) 
    {
      const long long curCol0 = currentColIndices[j];
      const long long curCol1 = currentColIndices[j+1];

      vf64_t xv_v = vec_vglfdso(xv, curCol0, curCol1);
      vf64_t * const cv = (vf64_t * const)(&currentValues[j]);

      sum_v -= (*cv) * xv_v;
    }

    for (int j = curNNZ2; j < currentNumberOfNonzeros; ++j) 
    {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }

    sum += xv[i]*currentDiagonal + sum_v[0] + sum_v[1]; // Remove diagonal contribution from previous loop

    xv[i] = sum/currentDiagonal;
    */

  return 0;
}

