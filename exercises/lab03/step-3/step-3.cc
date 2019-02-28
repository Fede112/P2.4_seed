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

// Thread libraries
#include <deal.II/base/work_stream.h>
#include <deal.II/base/multithread_info.h>


// Output the grid
#include <deal.II/grid/grid_out.h>

// impose constrains to ensure continuity of the solution
// #include <deal.II/lac/affine_constraints.h>
// refine cells which are flag base on an error estimator per cell
#include <deal.II/grid/grid_refinement.h>
// estimate the error per cell
#include <deal.II/numerics/error_estimator.h>


// MPI

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/lac/petsc_vector.h>

#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>

#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_renumbering.h>


using namespace dealii;



struct ScratchData {
  std::vector<double> rhs_values;
  FEValues<2> fe_values;

  ScratchData (const FiniteElement<2> &fe, const Quadrature<2> &quadrature,
    const UpdateFlags update_flags) 
  : rhs_values (quadrature.size()), 
  fe_values (fe, quadrature, update_flags)
  {}
  
  
  ScratchData (const ScratchData &rhs)
  : rhs_values (rhs.rhs_values),
  fe_values (rhs.fe_values.get_fe(),
  rhs.fe_values.get_quadrature(),
  rhs.fe_values.get_update_flags())
  {}
};

struct PerTaskData
{
  FullMatrix<double> cell_matrix;
  Vector<double> cell_rhs;
  std::vector<unsigned int> dof_indices;

  PerTaskData (const FiniteElement<2> &fe)
  : cell_matrix(fe.dofs_per_cell, fe.dofs_per_cell), 
    cell_rhs(fe.dofs_per_cell), dof_indices(fe.dofs_per_cell)
  {}
};


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
  // results () const;
  void compute_error();

  void assemble_on_one_cell (const typename DoFHandler<2>::active_cell_iterator &cell, ScratchData &scratch, PerTaskData &data);
  void copy_local_to_global (const PerTaskData &data);




  // // SparsityPattern      sparsity_pattern;
  // // SparseMatrix<double> system_matrix;

  // // Vector<double>       solution;
  // // Vector<double>       system_rhs;

  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;

  ConditionalOStream pcout;

  // ConstraintMatrix     hanging_node_constraints;
  PETScWrappers::MPI::SparseMatrix system_matrix;
  PETScWrappers::MPI::Vector       solution;
  PETScWrappers::MPI::Vector       system_rhs;

  Triangulation<2>     triangulation;
  DoFHandler<2>        dof_handler;
  FE_Q<2>              fe;


};




// // Constructor of Step3
// Step3::Step3 ()
//   :
//   // decide finite elements degree
//   fe (1),
//    Note that the triangulation isn't set up with a mesh at all at the present time, 
//   but the DoFHandler doesn't care: it only wants to know which triangulation it will be associated with,
//    and it only starts to care about an actual mesh once you try to distribute degree of freedom on the mesh 
//    using the distribute_dofs() function.) All the other member variables of the Step3 class have a default constructor which does all we want. 
//   dof_handler (triangulation)
// {}


Step3::Step3 ()
  :
  // for now I hard code de communicator
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout (std::cout, (this_mpi_process == 0)),
  dof_handler (triangulation),
  // fe (FE_Q<2>(1), 2)
  fe (1)
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

  // define triangulation (all mpi processes own the same triangulation)
  GridTools::partition_triangulation (n_mpi_processes, triangulation);

  // distribute dofs between processes
  dof_handler.distribute_dofs (fe);
  std::cout << "Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

  // For hanging nodes:
  // hanging_node_constraints.clear ();
  // DoFTools::make_hanging_node_constraints (dof_handler,
  //                                          hanging_node_constraints);
  // hanging_node_constraints.close ();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, dsp);
  // sparsity_pattern.copy_from(dsp);

  const std::vector<IndexSet> locally_owned_dofs_per_proc = DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
  const IndexSet locally_owned_dofs = locally_owned_dofs_per_proc[this_mpi_process];
  system_matrix.reinit (locally_owned_dofs,
                        locally_owned_dofs,
                        dsp,
                        mpi_communicator);
  solution.reinit (locally_owned_dofs, mpi_communicator);
  system_rhs.reinit (locally_owned_dofs, mpi_communicator);
}


// void Step3<2>::assemble_system () {
//   int n_virtual_cores = 4;
//   Threads::ThreadGroup<void> threads;
//   std::vector<std::pair<cell_iterator, cell_iterator> >
//   sub_ranges = Threads::split_range (dof_handler.begin_active(), dof_handler.end(), n_virtual_cores);
  
//   for (t=0; t<n_virtual_cores; ++t)
//     threads += Threads::new_thread (&Step3<2>::assemble_on_cell_range, this, sub_ranges[t].first, sub_ranges[t].second);
//   threads.join_all ();

// }



void Step3::assemble_on_one_cell (
const typename DoFHandler<2>::active_cell_iterator &cell,
ScratchData &scratch,
PerTaskData &data)
{
  scratch.fe_values.reinit (cell);

  data.cell_matrix = 0;
  data.cell_rhs = 0;

  for (unsigned int q_index=0; q_index<scratch.fe_values.get_quadrature().size(); ++q_index)
  {
    const auto& real_q = scratch.fe_values.quadrature_point(q_index);
    for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
      for (unsigned int j=0; j<fe.dofs_per_cell; ++j)
      {
        data.cell_matrix(i,j) += (scratch.fe_values.shape_grad (i, q_index) *
                             scratch.fe_values.shape_grad (j, q_index) *
                             scratch.fe_values.JxW (q_index));
      }

    for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
    {
      data.cell_rhs(i) += (scratch.fe_values.shape_value (i, q_index) * func(real_q) *
                      scratch.fe_values.JxW (q_index));
    }
  }

  cell->get_dof_indices (data.dof_indices);
}



void Step3::copy_local_to_global (const PerTaskData &data)
{
  // for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
  for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
    for (unsigned int j=0; j<fe.dofs_per_cell; ++j)
      // data,dof_indeices[i] are already global indeces
      system_matrix.add (data.dof_indices[i], data.dof_indices[j],
      data.cell_matrix(i,j));

  for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
    system_rhs(data.dof_indices[i]) += data.cell_rhs(i);

}

void Step3::assemble_system ()
{


  QGauss<2>  quadrature_formula(2);
  // FEValues<2> fe_values (fe, quadrature_formula,
                         // update_values | update_gradients | update_JxW_values  | update_quadrature_points);

  // Initialize scratch and data
  // He creates de fe_values
  ScratchData scratch( fe, quadrature_formula, update_values | update_gradients | update_JxW_values  | update_quadrature_points );
  PerTaskData data( fe );

  // const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  // const unsigned int   n_q_points    = quadrature_formula.size();

  // FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  // Vector<double>       cell_rhs (dofs_per_cell);

  // from local to global indeces
  // std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  DoFHandler<2>::active_cell_iterator cell = dof_handler.begin_active();
  DoFHandler<2>::active_cell_iterator endc = dof_handler.end();

  for (; cell!=endc; ++cell)
  {
    if (cell->subdomain_id() == this_mpi_process)
    {
    assemble_on_one_cell(cell, scratch, data);
    // copy_local_to_global(data);
    }
  }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  // WorkStream::run ( dof_handler.begin_active(),
  // dof_handler.end(),
  // *this,
  // &Step3::assemble_on_one_cell,
  // &Step3::copy_local_to_global,
  // scratch,
  // data );

  // std::map<types::global_dof_index,double> boundary_values;
  // VectorTools::interpolate_boundary_values (dof_handler,
  //                                           0,
  //                                           ZeroFunction<2>(),
  //                                           boundary_values);
  // MatrixTools::apply_boundary_values (boundary_values,
  //                                     system_matrix,
  //                                     solution,
  //                                     system_rhs);
  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            ZeroFunction<2>(),
                                            boundary_values);
  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs,
                                      false);


}





void Step3::solve ()//{std::cout << "aca!" << std::endl;}
{
  // SolverControl           solver_control (1000, 1e-12);
  // // SolverControl solver_control (solution.size(), 1e-8*system_rhs.l2_norm());
  // // SolverCG<>              solver (solver_control);
  


  // PETScWrappers::SolverCG cg (solver_control, mpi_communicator);
  // PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
  // cg.solve (system_matrix, solution, system_rhs, preconditioner);

  // Vector<double> localized_solution (solution);
  // solution = localized_solution;
  // return solver_control.last_step();
  // // solver.solve (system_matrix, solution, system_rhs,
  //               // PreconditionIdentity());
}



void Step3::output_results () const
{

  if (this_mpi_process == 0)
  {
  
  DataOut<2> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");
  data_out.build_patches ();

  std::ofstream output ("solution.svg");
  data_out.write_svg (output);

  }
}


void Step3::run ()
{
  make_grid ();
  setup_system ();
  assemble_system ();
  solve ();
  output_results ();
  // compute_error();
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
  // std::cout << "Error on verteces norm (because the mesh is rectangular of order 1): " << ( sol_inter -= solution ).linfty_norm() << std::endl;

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






int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  // Original Main
  deallog.depth_console (2);
  Step3 laplace_problem;
  laplace_problem.run ();


  return 0;
}
