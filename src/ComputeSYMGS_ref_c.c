
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

int min(int a, int b) {
    return (a < b) ? a : b;
}

#define BLOCK 128 

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
  local_int_t k;
  for (k=0; k<(nrow/BLOCK)*BLOCK; k += BLOCK) 
  {
//    for(local_int_t i = 0; i < BLOCK/16; ++i)
//    {
//      __builtin_prefetch(&rv[k + i*16], 0, 1);
//      __builtin_prefetch(&xv[k + i*16], 1, 1);
//      __builtin_prefetch(&nonzerosInRow[k + i*16], 0, 1);
//      __builtin_prefetch(&matrixValues[k + i*16], 0, 1);
//      __builtin_prefetch(&mtxIndL[k + i*16], 0, 1);
//    }

    for (local_int_t i=k; i< k+BLOCK; i++) 
    {
      const double * const currentValues = matrixValues[i];
      const local_int_t * const currentColIndices = mtxIndL[i];
      const int currentNumberOfNonzeros = nonzerosInRow[i];
      const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
      //double sum = rv[i]; // RHS value


      int curNNZ2 = currentNumberOfNonzeros - (currentNumberOfNonzeros % 2);

      //vf64_t sum_v = {0.0, 0.0};
      vf64_t sum_v = {rv[i], xv[i]*currentDiagonal};

      for (int j=0; j< curNNZ2; j+= 2) {
        const local_int_t * const curCol = &currentColIndices[j];

        vf64_t xv_v = vec_vglfdso(xv, curCol[0]*sizeof(double), curCol[1]*sizeof(double));
        vf64_t * const cv = (vf64_t * const)(&currentValues[j]);

        sum_v -= (*cv) * xv_v;
      }

      //sum += sum_v[0] + sum_v[1];

      if (curNNZ2 != currentNumberOfNonzeros)
        sum_v[0] -= currentValues[curNNZ2] * xv[currentColIndices[curNNZ2]];

      //sum += xv[i]*currentDiagonal; // Remove diagonal contribution from previous loop

      xv[i] = (sum_v[0] + sum_v[1])/currentDiagonal;
    }
  }

  for (local_int_t i=k; i<nrow; ++i) 
  {
    const double * const currentValues = matrixValues[i];
    const local_int_t * const currentColIndices = mtxIndL[i];
    const int currentNumberOfNonzeros = nonzerosInRow[i];
    const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i]; // RHS value


    int curNNZ2 = currentNumberOfNonzeros - (currentNumberOfNonzeros % 2);

    vf64_t sum_v = {0.0, 0.0};

    for (int j=0; j< curNNZ2; j+= 2) {
      const local_int_t * const curCol = &currentColIndices[j];

      vf64_t xv_v = vec_vglfdso(xv, curCol[0]*sizeof(double), curCol[1]*sizeof(double));
      vf64_t * const cv = (vf64_t * const)(&currentValues[j]);

      sum_v -= (*cv) * xv_v;
    }

    sum += sum_v[0] + sum_v[1];

    if (curNNZ2 != currentNumberOfNonzeros)
      sum -= currentValues[curNNZ2] * xv[currentColIndices[curNNZ2]];

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

    
  return 0;
}

