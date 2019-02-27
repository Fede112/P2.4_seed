/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2016 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 */



#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>



#include <cmath>
#include <deal.II/base/function_parser.h>

using namespace dealii;



class Step3
{
public:
  Step3 ();

  void run ();


private:
  void make_grid ();
  void setup_system ();
  void assemble_system ();
  void solve ();
  void output_results () const;
  void compute_error();

  Triangulation<2>     triangulation;
  FE_Q<2>              fe;
  DoFHandler<2>        dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double>       solution;
  Vector<double>       system_rhs;
};


Step3::Step3 ()
  :
  fe (1),
  dof_handler (triangulation)
{}


double func(Point<2> real_q)
{
   return 40.*M_PI*M_PI*sin(2*M_PI*real_q[0])* sin(6*M_PI*real_q[1]);
}


void Step3::make_grid ()
{
  GridGenerator::hyper_cube (triangulation, -1, 1);
  triangulation.refine_global (5);

  std::cout << "Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl;
}




void Step3::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  std::cout << "Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit (sparsity_pattern);

  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
}

void Step3::compute_error()
{
   // Define some constants that will be used by the function parser
  std::map<std::string,double> constants;
  constants["pi"] = numbers::PI;
  // Define the variables that will be used inside the expressions
  std::string variables = "x,y";
  // Define the expressions of the individual components of a
  // vector valued function with two components:
  std::string expression = "sin(2*pi*x)*sin(6*pi*y)";

  // function parser with 3 variables and 2 components
  FunctionParser<2> fp(1);
  // And populate it with the newly created objects.
  fp.initialize(variables, expression, constants);
  
  // interpolation of exact solution
  Vector<double> sol_inter(dof_handler.n_dofs());
  Vector<double> cell_diff(dof_handler.n_dofs());
  double error_quad;

  VectorTools::interpolate(dof_handler, fp, sol_inter);

  // norm in the quadrature points should be >= than calculating them in the vertices
  std::cout << "Error on verteces norm (because the mesh is rectangular of order 1): " << ( sol_inter -= solution ).linfty_norm() << std::endl;

  QGauss<2> q_gauss2(2);


  VectorTools::integrate_difference( dof_handler, solution, fp, cell_diff, q_gauss2,  VectorTools::NormType::Linfty_norm );

  error_quad = VectorTools::compute_global_error(triangulation, cell_diff, VectorTools::NormType::Linfty_norm);

  std::cout << "Error on quadrature points norm: " << error_quad << std::endl;

// void VectorTools::integrate_difference  ( const Mapping< dim, spacedim > &  mapping,
// const DoFHandler< dim, spacedim > &   dof,
// const InVector &  fe_function,
// const Function< spacedim, double > &  exact_solution,
// OutVector &   difference,
// const Quadrature< dim > &   q,
// const NormType &  norm,
// const Function< spacedim, double > *  weight = 0,
// const double  exponent = 2. 
// )

}

void Step3::assemble_system ()
{
  // Point<2> real_q;
  QGauss<2>  quadrature_formula(2);
  FEValues<2> fe_values (fe, quadrature_formula,
                         update_values | update_gradients | update_JxW_values  | update_quadrature_points);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  DoFHandler<2>::active_cell_iterator cell = dof_handler.begin_active();
  DoFHandler<2>::active_cell_iterator endc = dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      // fe_values is constructed once outside the loop
      // then it is reinit for each cell
      fe_values.reinit (cell);

      cell_matrix = 0;
      cell_rhs = 0;

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
          const auto& real_q = fe_values.quadrature_point(q_index);
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) *
                                   fe_values.shape_grad (j, q_index) *
                                   fe_values.JxW (q_index));

          for (unsigned int i=0; i<dofs_per_cell; ++i)

            cell_rhs(i) += (fe_values.shape_value (i, q_index) * func(real_q) *
                            fe_values.JxW (q_index));
        }
      cell->get_dof_indices (local_dof_indices);

      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          system_matrix.add (local_dof_indices[i],
                             local_dof_indices[j],
                             cell_matrix(i,j));

      for (unsigned int i=0; i<dofs_per_cell; ++i)
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }


  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            ZeroFunction<2>(),
                                            boundary_values);
  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);
}



void Step3::solve ()
{
  SolverControl           solver_control (1000, 1e-12);
  SolverCG<>              solver (solver_control);

  solver.solve (system_matrix, solution, system_rhs,
                PreconditionIdentity());
}



void Step3::output_results () const
{
  DataOut<2> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");
  data_out.build_patches ();

  std::ofstream output ("solution.svg");
  data_out.write_svg (output);
}



void Step3::run ()
{
  make_grid ();
  setup_system ();
  assemble_system ();
  solve ();
  output_results ();
  compute_error();
}









int main ()
{

  // Original Main
  deallog.depth_console (2);
  Step3 laplace_problem;
  laplace_problem.run ();


  return 0;
}
