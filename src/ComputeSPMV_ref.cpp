
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
 @file ComputeSPMV_ref.cpp

 HPCG routine
 */

#include "ComputeSPMV_ref.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>

static int constexpr VEC_SIZE=8;

int static ComputeSPMV_ref_opt(double ** __restrict__ AmatrixValues, local_int_t ** __restrict__ AmtxIndL, 
                               char * __restrict__ AnonzerosInRow, double const * __restrict__ xv, double * __restrict__ yv, local_int_t const  nrow)
{
  auto op = [&](local_int_t i)
  {
    double sum[2] = {0.0, 0.0};
    const double * const cur_vals = AmatrixValues[i];
    const local_int_t * const cur_inds = AmtxIndL[i];
    const int cur_nnz = AnonzerosInRow[i];

      /*
    for (int j=0; j< cur_nnz; j++)
      sum += cur_vals[j]*xv[cur_inds[j]];
    */
    for (int j=0; j < (cur_nnz/2)*2; j += 2)
    {
      sum[0] += cur_vals[j]*xv[cur_inds[j]];
      sum[1] += cur_vals[j+1]*xv[cur_inds[j+1]];
    }

    for (int j=(cur_nnz/2)*2; j < cur_nnz; ++j)
       sum[0] += cur_vals[j]*xv[cur_inds[j]];
 
    yv[i] = sum[0] + sum[1];
  };

  int const nrow_v = (nrow / VEC_SIZE) * VEC_SIZE;

  for (local_int_t i_outer =0; i_outer < nrow_v; i_outer += VEC_SIZE)  
  {
    for (int i = i_outer; i < i_outer + VEC_SIZE; ++i)
    {
       op(i);
    }
  }

  for (local_int_t i = nrow_v; i < nrow; ++i)
  {
    op(i);
  }
  
  
#if 0 
  {
    double sum = 0.0;
    const double * const cur_vals = AmatrixValues[i];
    const local_int_t * const cur_inds = AmtxIndL[i];
    const int cur_nnz = AnonzerosInRow[i];

    for (int j=0; j< cur_nnz; j++)
      sum += cur_vals[j]*xv[cur_inds[j]];
    yv[i] = sum;
  }
#endif
 

  return 0;
}

/*!
  Routine to compute matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This is the reference SPMV implementation.  It CANNOT be modified for the
  purposes of this benchmark.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV
*/
int ComputeSPMV_ref( const SparseMatrix & A, Vector & x, Vector & y) {

  assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength>=A.localNumberOfRows);

#ifndef HPCG_NO_MPI
    ExchangeHalo(A,x);
#endif
  const double * const xv = x.values;
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;
#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif

/*
  for (local_int_t i=0; i< nrow; i++)  {
    double sum = 0.0;
    const double * const cur_vals = A.matrixValues[i];
    const local_int_t * const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];

    for (int j=0; j< cur_nnz; j++)
      sum += cur_vals[j]*xv[cur_inds[j]];
    yv[i] = sum;
  }
*/
  ComputeSPMV_ref_opt(&A.matrixValues[0], &A.mtxIndL[0], 
                               &A.nonzerosInRow[0], xv, yv, nrow);

  return 0;
}
