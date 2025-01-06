
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
    double ** matrixDiagonal,  // An array of pointers to the diagonal entries A.matrixValues
    const double * const rv,
    double * const xv,
    const double ** const matrixValues,
    const local_int_t ** const mtxIndL,
    const int * nonzerosInRow)
{


  for (local_int_t i=0; i< nrow; i++) {
    const double * const currentValues = matrixValues[i];
    const local_int_t * const currentColIndices = mtxIndL[i];
    const int currentNumberOfNonzeros = nonzerosInRow[i];
    const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i]; // RHS value

    /*
    for (int j=0; j< currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }
  */

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

  }

#if 1

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
#endif

  return 0;
}

