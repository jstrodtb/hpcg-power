#include <vector>
#include <algorithm>
#include <iostream>
#include <queue>


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
 @file OptimizeProblem.cpp

 HPCG routine
 */

#include "OptimizeProblem.hpp"


std::vector<int> reverseCuthillMcKee(int n, const char *nonzerosInRow, local_int_t **mtxIndL ) {
  std::vector<int> degree(n, 0);
  std::vector<std::vector<int>> adj(n);
    
    // Build adjacency list and compute degrees
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < nonzerosInRow[i]; ++j) {
            int neighbor = mtxIndL[i][j];
            adj[i].push_back(neighbor);
            degree[i]++;
        }
    }
    
    std::vector<int> rcm_order;
    std::vector<bool> visited(n, false);
    
    // Find starting node (minimum degree)
    int start = std::min_element(degree.begin(), degree.end()) - degree.begin();
    
    std::queue<int> q;
    q.push(start);
    visited[start] = true;
    
    // Perform BFS
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        rcm_order.push_back(node);
        
        // Sort neighbors by increasing degree before pushing them to the queue
        std::vector<int> neighbors;
        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                neighbors.push_back(neighbor);
            }
        }
        sort(neighbors.begin(), neighbors.end(), [&](int a, int b) { return degree[a] < degree[b]; });
        
        for (int neighbor : neighbors) {
            q.push(neighbor);
            visited[neighbor] = true;
        }
    }
    
    // Reverse the order for Reverse Cuthill-McKee
    reverse(rcm_order.begin(), rcm_order.end());
    return rcm_order;
}


/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with all necessary CG vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed in future CG iteration
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
*/
int OptimizeProblem(SparseMatrix & A, CGData & data, Vector & b, Vector & x, Vector & xexact) {

  auto rcm_order = reverseCuthillMcKee(A.localNumberOfRows, A.nonzerosInRow, A.mtxIndL); 

  // This function can be used to completely transform any part of the data structures.
  // Right now it does nothing, so compiling with a check for unused variables results in complaints

#if defined(HPCG_USE_MULTICOLORING)
  const local_int_t nrow = A.localNumberOfRows;
  std::vector<local_int_t> colors(nrow, nrow); // value `nrow' means `uninitialized'; initialized colors go from 0 to nrow-1
  int totalColors = 1;
  colors[0] = 0; // first point gets color 0

  // Finds colors in a greedy (a likely non-optimal) fashion.

  for (local_int_t i=1; i < nrow; ++i) {
    if (colors[i] == nrow) { // if color not assigned
      std::vector<int> assigned(totalColors, 0);
      int currentlyAssigned = 0;
      const local_int_t * const currentColIndices = A.mtxIndL[i];
      const int currentNumberOfNonzeros = A.nonzerosInRow[i];

      for (int j=0; j< currentNumberOfNonzeros; j++) { // scan neighbors
        local_int_t curCol = currentColIndices[j];
        if (curCol < i) { // if this point has an assigned color (points beyond `i' are unassigned)
          if (assigned[colors[curCol]] == 0)
            currentlyAssigned += 1;
          assigned[colors[curCol]] = 1; // this color has been used before by `curCol' point
        } // else // could take advantage of indices being sorted
      }

      if (currentlyAssigned < totalColors) { // if there is at least one color left to use
        for (int j=0; j < totalColors; ++j)  // try all current colors
          if (assigned[j] == 0) { // if no neighbor with this color
            colors[i] = j;
            break;
          }
      } else {
        if (colors[i] == nrow) {
          colors[i] = totalColors;
          totalColors += 1;
        }
      }
    }
  }

  std::vector<local_int_t> counters(totalColors);
  for (local_int_t i=0; i<nrow; ++i)
    counters[colors[i]]++;

  // form in-place prefix scan
  local_int_t old=counters[0], old0;
  for (local_int_t i=1; i < totalColors; ++i) {
    old0 = counters[i];
    counters[i] = counters[i-1] + old;
    old = old0;
  }
  counters[0] = 0;

  // translate `colors' into a permutation
  for (local_int_t i=0; i<nrow; ++i) // for each color `c'
    colors[i] = counters[colors[i]]++;
#endif

  return 0;
}

// Helper function (see OptimizeProblem.hpp for details)
double OptimizeProblemMemoryUse(const SparseMatrix & A) {

  return 0.0;

}
