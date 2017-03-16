/* -*- compile-command: "R CMD INSTALL .." -*- */

#include "solver.h"
#include "R.h"

extern "C" {

  void compute_optimal_costs_interface
  (int *n_data, 
   double *lower_vec, double *upper_vec,
   double *margin, int *loss,
   int *moves_vec, double *pred_vec, double *cost_vec
   ){
    int status = compute_optimal_costs
      (*n_data, lower_vec, upper_vec,
       *margin, *loss, moves_vec, pred_vec, cost_vec);
    if(status != 0){
      error("non-zero return code from compute_optimal_costs");
    }
  }
  
}
    
