
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>



#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

using namespace dealii;

double const FREQUENCY = 3.0 * dealii::numbers::PI;

template <int dim>
class AnalyticalRHS : public dealii::Function<dim>
{
public:
  AnalyticalRHS()
    : dealii::Function<dim>(1, 0.0)
  {}

  double
  value(const dealii::Point<dim> &p, const unsigned int) const final
  {
    double result = FREQUENCY * FREQUENCY * dim;
    for(unsigned int d = 0; d < dim; ++d)
      result *= std::sin(FREQUENCY * p[d]);

    return result;
  }

};


template <int dim>
class AnalyticalSolution : public dealii::Function<dim>
{
public:
  AnalyticalSolution()
    : dealii::Function<dim>(1, 0.0)
  {}

  double
  value(const dealii::Point<dim> &p, const unsigned int) const final
  {
    double result = 1.0;
    for(unsigned int d = 0; d < dim; ++d)
      result *= std::sin(FREQUENCY * p[d]);

    return result;
  }

};

template <int dim, typename Number>
VectorizedArray<Number>
evaluate_function(const Function<dim>                       &function,
                         const Point<dim, VectorizedArray<Number>> &p_vectorized)
{
  AssertDimension(function.n_components, 1);
  VectorizedArray<Number> result;
  for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
    {
      Point<dim> p;
      for (unsigned int d = 0; d < dim; ++d)
        p[d] = p_vectorized[d][v];
      result[v] = function.value(p);
    }
  return result;
}




template <bool transpose_matrix, bool add, typename Number, typename Number2>
void
apply_matrix_vector_product_2(const Number2 *matrix,
                              const Number  *in0,
                              Number        *out0,
                              const int      n_rows,
                              const int      n_columns)
{
  const int mm = transpose_matrix ? n_rows : n_columns,
            nn = transpose_matrix ? n_columns : n_rows;
  Assert(n_rows > 0 && n_columns > 0,
         ExcInternalError("Empty evaluation task!"));
  Assert(n_rows > 0 && n_columns > 0,
         ExcInternalError("The evaluation needs n_rows, n_columns > 0, but " +
                          std::to_string(n_rows) + ", " +
                          std::to_string(n_columns) + " was passed!"));


  const Number *in1  = in0 + mm;
  Number       *out1 = out0 + nn;

  int nn_regular = (nn / 5) * 5;
  for (int col = 0; col < nn_regular; col += 5)
    {
      Number res[10];
      if (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + col;
          const Number   a = in0[0], b = in1[0];
          Number         m = matrix_ptr[0];
          res[0]           = m * a;
          res[5]           = m * b;
          m                = matrix_ptr[1];
          res[1]           = m * a;
          res[6]           = m * b;
          m                = matrix_ptr[2];
          res[2]           = m * a;
          res[7]           = m * b;
          m                = matrix_ptr[3];
          res[3]           = m * a;
          res[8]           = m * b;
          m                = matrix_ptr[4];
          res[4]           = m * a;
          res[9]           = m * b;
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              const Number a = in0[i], b = in1[i];
              m = matrix_ptr[0];
              res[0] += m * a;
              res[5] += m * b;
              m = matrix_ptr[1];
              res[1] += m * a;
              res[6] += m * b;
              m = matrix_ptr[2];
              res[2] += m * a;
              res[7] += m * b;
              m = matrix_ptr[3];
              res[3] += m * a;
              res[8] += m * b;
              m = matrix_ptr[4];
              res[4] += m * a;
              res[9] += m * b;
            }
        }
      else
        {
          const Number2 *matrix_0 = matrix + col * n_columns;
          const Number2 *matrix_1 = matrix + (col + 1) * n_columns;
          const Number2 *matrix_2 = matrix + (col + 2) * n_columns;
          const Number2 *matrix_3 = matrix + (col + 3) * n_columns;
          const Number2 *matrix_4 = matrix + (col + 4) * n_columns;

          const Number a = in0[0], b = in1[0];
          Number       m = matrix_0[0];
          res[0]         = m * a;
          res[5]         = m * b;
          m              = matrix_1[0];
          res[1]         = m * a;
          res[6]         = m * b;
          m              = matrix_2[0];
          res[2]         = m * a;
          res[7]         = m * b;
          m              = matrix_3[0];
          res[3]         = m * a;
          res[8]         = m * b;
          m              = matrix_4[0];
          res[4]         = m * a;
          res[9]         = m * b;
          for (int i = 1; i < mm; ++i)
            {
              const Number a = in0[i], b = in1[i];
              m = matrix_0[i];
              res[0] += m * a;
              res[5] += m * b;
              m = matrix_1[i];
              res[1] += m * a;
              res[6] += m * b;
              m = matrix_2[i];
              res[2] += m * a;
              res[7] += m * b;
              m = matrix_3[i];
              res[3] += m * a;
              res[8] += m * b;
              m = matrix_4[i];
              res[4] += m * a;
              res[9] += m * b;
            }
        }
      if (add)
        {
          out0[0] += res[0];
          out0[1] += res[1];
          out0[2] += res[2];
          out0[3] += res[3];
          out0[4] += res[4];
          out1[0] += res[5];
          out1[1] += res[6];
          out1[2] += res[7];
          out1[3] += res[8];
          out1[4] += res[9];
        }
      else
        {
          out0[0] = res[0];
          out0[1] = res[1];
          out0[2] = res[2];
          out0[3] = res[3];
          out0[4] = res[4];
          out1[0] = res[5];
          out1[1] = res[6];
          out1[2] = res[7];
          out1[3] = res[8];
          out1[4] = res[9];
        }
      out0 += 5;
      out1 += 5;
    }
  if (nn - nn_regular == 4)
    {
      Number res[8];
      if (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + nn_regular;
          const Number   a = in0[0], b = in1[0];
          Number         m = matrix_ptr[0];
          res[0]           = m * a;
          res[4]           = m * b;
          m                = matrix_ptr[1];
          res[1]           = m * a;
          res[5]           = m * b;
          m                = matrix_ptr[2];
          res[2]           = m * a;
          res[6]           = m * b;
          m                = matrix_ptr[3];
          res[3]           = m * a;
          res[7]           = m * b;
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              const Number a = in0[i], b = in1[i];
              m = matrix_ptr[0];
              res[0] += m * a;
              res[4] += m * b;
              m = matrix_ptr[1];
              res[1] += m * a;
              res[5] += m * b;
              m = matrix_ptr[2];
              res[2] += m * a;
              res[6] += m * b;
              m = matrix_ptr[3];
              res[3] += m * a;
              res[7] += m * b;
            }
        }
      else
        {
          const Number2 *matrix_0 = matrix + nn_regular * n_columns;
          const Number2 *matrix_1 = matrix + (nn_regular + 1) * n_columns;
          const Number2 *matrix_2 = matrix + (nn_regular + 2) * n_columns;
          const Number2 *matrix_3 = matrix + (nn_regular + 3) * n_columns;

          const Number a = in0[0], b = in1[0];
          Number       m = matrix_0[0];
          res[0]         = m * a;
          res[4]         = m * b;
          m              = matrix_1[0];
          res[1]         = m * a;
          res[5]         = m * b;
          m              = matrix_2[0];
          res[2]         = m * a;
          res[6]         = m * b;
          m              = matrix_3[0];
          res[3]         = m * a;
          res[7]         = m * b;
          for (int i = 1; i < mm; ++i)
            {
              const Number a = in0[i], b = in1[i];
              m = matrix_0[i];
              res[0] += m * a;
              res[4] += m * b;
              m = matrix_1[i];
              res[1] += m * a;
              res[5] += m * b;
              m = matrix_2[i];
              res[2] += m * a;
              res[6] += m * b;
              m = matrix_3[i];
              res[3] += m * a;
              res[7] += m * b;
            }
        }
      if (add)
        {
          out0[0] += res[0];
          out0[1] += res[1];
          out0[2] += res[2];
          out0[3] += res[3];
          out1[0] += res[4];
          out1[1] += res[5];
          out1[2] += res[6];
          out1[3] += res[7];
        }
      else
        {
          out0[0] = res[0];
          out0[1] = res[1];
          out0[2] = res[2];
          out0[3] = res[3];
          out1[0] = res[4];
          out1[1] = res[5];
          out1[2] = res[6];
          out1[3] = res[7];
        }
    }
  if (nn - nn_regular == 3)
    {
      Number res0, res1, res2, res3, res4, res5;
      if (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + nn_regular;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[1] * in0[0];
          res2                      = matrix_ptr[2] * in0[0];
          res3                      = matrix_ptr[0] * in1[0];
          res4                      = matrix_ptr[1] * in1[0];
          res5                      = matrix_ptr[2] * in1[0];
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              res0 += matrix_ptr[0] * in0[i];
              res1 += matrix_ptr[1] * in0[i];
              res2 += matrix_ptr[2] * in0[i];
              res3 += matrix_ptr[0] * in1[i];
              res4 += matrix_ptr[1] * in1[i];
              res5 += matrix_ptr[2] * in1[i];
            }
        }
      else
        {
          const Number2 *matrix_0 = matrix + nn_regular * n_columns;
          const Number2 *matrix_1 = matrix + (nn_regular + 1) * n_columns;
          const Number2 *matrix_2 = matrix + (nn_regular + 2) * n_columns;

          res0 = matrix_0[0] * in0[0];
          res1 = matrix_1[0] * in0[0];
          res2 = matrix_2[0] * in0[0];
          res3 = matrix_0[0] * in1[0];
          res4 = matrix_1[0] * in1[0];
          res5 = matrix_2[0] * in1[0];
          for (int i = 1; i < mm; ++i)
            {
              res0 += matrix_0[i] * in0[i];
              res1 += matrix_1[i] * in0[i];
              res2 += matrix_2[i] * in0[i];
              res3 += matrix_0[i] * in1[i];
              res4 += matrix_1[i] * in1[i];
              res5 += matrix_2[i] * in1[i];
            }
        }
      if (add)
        {
          out0[0] += res0;
          out0[1] += res1;
          out0[2] += res2;
          out1[0] += res3;
          out1[1] += res4;
          out1[2] += res5;
        }
      else
        {
          out0[0] = res0;
          out0[1] = res1;
          out0[2] = res2;
          out1[0] = res3;
          out1[1] = res4;
          out1[2] = res5;
        }
    }
  else if (nn - nn_regular == 2)
    {
      Number res0, res1, res2, res3;
      if (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + nn_regular;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[1] * in0[0];
          res2                      = matrix_ptr[0] * in1[0];
          res3                      = matrix_ptr[1] * in1[0];
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              res0 += matrix_ptr[0] * in0[i];
              res1 += matrix_ptr[1] * in0[i];
              res2 += matrix_ptr[0] * in1[i];
              res3 += matrix_ptr[1] * in1[i];
            }
        }
      else
        {
          const Number2 *matrix_0 = matrix + nn_regular * n_columns;
          const Number2 *matrix_1 = matrix + (nn_regular + 1) * n_columns;

          res0 = matrix_0[0] * in0[0];
          res1 = matrix_1[0] * in0[0];
          res2 = matrix_0[0] * in1[0];
          res3 = matrix_1[0] * in1[0];
          for (int i = 1; i < mm; ++i)
            {
              res0 += matrix_0[i] * in0[i];
              res1 += matrix_1[i] * in0[i];
              res2 += matrix_0[i] * in1[i];
              res3 += matrix_1[i] * in1[i];
            }
        }
      if (add)
        {
          out0[0] += res0;
          out0[1] += res1;
          out1[0] += res2;
          out1[1] += res3;
        }
      else
        {
          out0[0] = res0;
          out0[1] = res1;
          out1[0] = res2;
          out1[1] = res3;
        }
    }
  else if (nn - nn_regular == 1)
    {
      Number res0, res1;
      if (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + nn_regular;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[0] * in1[0];
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              res0 += matrix_ptr[0] * in0[i];
              res1 += matrix_ptr[0] * in1[i];
            }
        }
      else
        {
          const Number2 *matrix_ptr = matrix + nn_regular * n_columns;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[0] * in1[0];
          for (int i = 1; i < mm; ++i)
            {
              res0 += matrix_ptr[i] * in0[i];
              res1 += matrix_ptr[i] * in1[i];
            }
        }
      if (add)
        {
          out0[0] += res0;
          out1[0] += res1;
        }
      else
        {
          out0[0] = res0;
          out1[0] = res1;
        }
    }
}

template <bool transpose_matrix, bool add, typename Number, typename Number2>
void
apply_matrix_vector_product_3(const Number2 *matrix,
                              const Number  *in0,
                              Number        *out0,
                              const int      n_rows,
                              const int      n_columns)
{
  const int mm = transpose_matrix ? n_rows : n_columns,
            nn = transpose_matrix ? n_columns : n_rows;
  Assert(n_rows > 0 && n_columns > 0,
         ExcInternalError("Empty evaluation task!"));
  Assert(n_rows > 0 && n_columns > 0,
         ExcInternalError("The evaluation needs n_rows, n_columns > 0, but " +
                          std::to_string(n_rows) + ", " +
                          std::to_string(n_columns) + " was passed!"));

  const Number *in1 = in0 + mm, *in2 = in1 + mm;
  Number       *out1 = out0 + nn, *out2 = out1 + nn;


  int nn_regular = (nn / 4) * 4;
  for (int col = 0; col < nn_regular; col += 4)
    {
      Number res[12];
      if (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + col;
          const Number   a = in0[0], b = in1[0], c = in2[0];
          Number         m = matrix_ptr[0];
          res[0]           = m * a;
          res[4]           = m * b;
          res[8]           = m * c;
          m                = matrix_ptr[1];
          res[1]           = m * a;
          res[5]           = m * b;
          res[9]           = m * c;
          m                = matrix_ptr[2];
          res[2]           = m * a;
          res[6]           = m * b;
          res[10]          = m * c;
          m                = matrix_ptr[3];
          res[3]           = m * a;
          res[7]           = m * b;
          res[11]          = m * c;
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              const Number a = in0[i], b = in1[i], c = in2[i];
              m = matrix_ptr[0];
              res[0] += m * a;
              res[4] += m * b;
              res[8] += m * c;
              m = matrix_ptr[1];
              res[1] += m * a;
              res[5] += m * b;
              res[9] += m * c;
              m = matrix_ptr[2];
              res[2] += m * a;
              res[6] += m * b;
              res[10] += m * c;
              m = matrix_ptr[3];
              res[3] += m * a;
              res[7] += m * b;
              res[11] += m * c;
            }
        }
      else
        {
          const Number2 *matrix_0 = matrix + col * n_columns;
          const Number2 *matrix_1 = matrix + (col + 1) * n_columns;
          const Number2 *matrix_2 = matrix + (col + 2) * n_columns;
          const Number2 *matrix_3 = matrix + (col + 3) * n_columns;

          const Number a = in0[0], b = in1[0], c = in2[0];
          Number       m = matrix_0[0];
          res[0]         = m * a;
          res[4]         = m * b;
          res[8]         = m * c;
          m              = matrix_1[0];
          res[1]         = m * a;
          res[5]         = m * b;
          res[9]         = m * c;
          m              = matrix_2[0];
          res[2]         = m * a;
          res[6]         = m * b;
          res[10]        = m * c;
          m              = matrix_3[0];
          res[3]         = m * a;
          res[7]         = m * b;
          res[11]        = m * c;
          for (int i = 1; i < mm; ++i)
            {
              const Number a = in0[i], b = in1[i], c = in2[i];
              m = matrix_0[i];
              res[0] += m * a;
              res[4] += m * b;
              res[8] += m * c;
              m = matrix_1[i];
              res[1] += m * a;
              res[5] += m * b;
              res[9] += m * c;
              m = matrix_2[i];
              res[2] += m * a;
              res[6] += m * b;
              res[10] += m * c;
              m = matrix_3[i];
              res[3] += m * a;
              res[7] += m * b;
              res[11] += m * c;
            }
        }
      if (add)
        {
          out0[0] += res[0];
          out0[1] += res[1];
          out0[2] += res[2];
          out0[3] += res[3];
          out1[0] += res[4];
          out1[1] += res[5];
          out1[2] += res[6];
          out1[3] += res[7];
          out2[0] += res[8];
          out2[1] += res[9];
          out2[2] += res[10];
          out2[3] += res[11];
        }
      else
        {
          out0[0] = res[0];
          out0[1] = res[1];
          out0[2] = res[2];
          out0[3] = res[3];
          out1[0] = res[4];
          out1[1] = res[5];
          out1[2] = res[6];
          out1[3] = res[7];
          out2[0] = res[8];
          out2[1] = res[9];
          out2[2] = res[10];
          out2[3] = res[11];
        }
      out0 += 4;
      out1 += 4;
      out2 += 4;
    }
  if (nn - nn_regular == 3)
    {
      Number res0, res1, res2, res3, res4, res5, res6, res7, res8;
      if (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + nn_regular;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[1] * in0[0];
          res2                      = matrix_ptr[2] * in0[0];
          res3                      = matrix_ptr[0] * in1[0];
          res4                      = matrix_ptr[1] * in1[0];
          res5                      = matrix_ptr[2] * in1[0];
          res6                      = matrix_ptr[0] * in2[0];
          res7                      = matrix_ptr[1] * in2[0];
          res8                      = matrix_ptr[2] * in2[0];
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              res0 += matrix_ptr[0] * in0[i];
              res1 += matrix_ptr[1] * in0[i];
              res2 += matrix_ptr[2] * in0[i];
              res3 += matrix_ptr[0] * in1[i];
              res4 += matrix_ptr[1] * in1[i];
              res5 += matrix_ptr[2] * in1[i];
              res6 += matrix_ptr[0] * in2[i];
              res7 += matrix_ptr[1] * in2[i];
              res8 += matrix_ptr[2] * in2[i];
            }
        }
      else
        {
          const Number2 *matrix_0 = matrix + nn_regular * n_columns;
          const Number2 *matrix_1 = matrix + (nn_regular + 1) * n_columns;
          const Number2 *matrix_2 = matrix + (nn_regular + 2) * n_columns;

          res0 = matrix_0[0] * in0[0];
          res1 = matrix_1[0] * in0[0];
          res2 = matrix_2[0] * in0[0];
          res3 = matrix_0[0] * in1[0];
          res4 = matrix_1[0] * in1[0];
          res5 = matrix_2[0] * in1[0];
          res6 = matrix_0[0] * in2[0];
          res7 = matrix_1[0] * in2[0];
          res8 = matrix_2[0] * in2[0];
          for (int i = 1; i < mm; ++i)
            {
              res0 += matrix_0[i] * in0[i];
              res1 += matrix_1[i] * in0[i];
              res2 += matrix_2[i] * in0[i];
              res3 += matrix_0[i] * in1[i];
              res4 += matrix_1[i] * in1[i];
              res5 += matrix_2[i] * in1[i];
              res6 += matrix_0[i] * in2[i];
              res7 += matrix_1[i] * in2[i];
              res8 += matrix_2[i] * in2[i];
            }
        }
      if (add)
        {
          out0[0] += res0;
          out0[1] += res1;
          out0[2] += res2;
          out1[0] += res3;
          out1[1] += res4;
          out1[2] += res5;
          out2[0] += res6;
          out2[1] += res7;
          out2[2] += res8;
        }
      else
        {
          out0[0] = res0;
          out0[1] = res1;
          out0[2] = res2;
          out1[0] = res3;
          out1[1] = res4;
          out1[2] = res5;
          out2[0] = res6;
          out2[1] = res7;
          out2[2] = res8;
        }
    }
  else if (nn - nn_regular == 2)
    {
      Number res0, res1, res2, res3, res4, res5;
      if (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + nn_regular;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[1] * in0[0];
          res2                      = matrix_ptr[0] * in1[0];
          res3                      = matrix_ptr[1] * in1[0];
          res4                      = matrix_ptr[0] * in2[0];
          res5                      = matrix_ptr[1] * in2[0];
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              res0 += matrix_ptr[0] * in0[i];
              res1 += matrix_ptr[1] * in0[i];
              res2 += matrix_ptr[0] * in1[i];
              res3 += matrix_ptr[1] * in1[i];
              res4 += matrix_ptr[0] * in2[i];
              res5 += matrix_ptr[1] * in2[i];
            }
        }
      else
        {
          const Number2 *matrix_0 = matrix + nn_regular * n_columns;
          const Number2 *matrix_1 = matrix + (nn_regular + 1) * n_columns;

          res0 = matrix_0[0] * in0[0];
          res1 = matrix_1[0] * in0[0];
          res2 = matrix_0[0] * in1[0];
          res3 = matrix_1[0] * in1[0];
          res4 = matrix_0[0] * in2[0];
          res5 = matrix_1[0] * in2[0];
          for (int i = 1; i < mm; ++i)
            {
              res0 += matrix_0[i] * in0[i];
              res1 += matrix_1[i] * in0[i];
              res2 += matrix_0[i] * in1[i];
              res3 += matrix_1[i] * in1[i];
              res4 += matrix_0[i] * in2[i];
              res5 += matrix_1[i] * in2[i];
            }
        }
      if (add)
        {
          out0[0] += res0;
          out0[1] += res1;
          out1[0] += res2;
          out1[1] += res3;
          out2[0] += res4;
          out2[1] += res5;
        }
      else
        {
          out0[0] = res0;
          out0[1] = res1;
          out1[0] = res2;
          out1[1] = res3;
          out2[0] = res4;
          out2[1] = res5;
        }
    }
  else if (nn - nn_regular == 1)
    {
      Number res0, res1, res2;
      if (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + nn_regular;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[0] * in1[0];
          res2                      = matrix_ptr[0] * in2[0];
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              res0 += matrix_ptr[0] * in0[i];
              res1 += matrix_ptr[0] * in1[i];
              res2 += matrix_ptr[0] * in2[i];
            }
        }
      else
        {
          const Number2 *matrix_ptr = matrix + nn_regular * n_columns;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[0] * in1[0];
          res2                      = matrix_ptr[0] * in2[0];
          for (int i = 1; i < mm; ++i)
            {
              res0 += matrix_ptr[i] * in0[i];
              res1 += matrix_ptr[i] * in1[i];
              res2 += matrix_ptr[i] * in2[i];
            }
        }
      if (add)
        {
          out0[0] += res0;
          out1[0] += res1;
          out2[0] += res2;
        }
      else
        {
          out0[0] = res0;
          out1[0] = res1;
          out2[0] = res2;
        }
    }
}

template <bool transpose_matrix, bool add, typename Number, typename Number2>
void
apply_matrix_vector_product_4(const Number2 *matrix,
                              const Number  *in0,
                              Number        *out0,
                              const int      n_rows,
                              const int      n_columns)
{
  const int mm = transpose_matrix ? n_rows : n_columns,
            nn = transpose_matrix ? n_columns : n_rows;
  Assert(n_rows > 0 && n_columns > 0,
         ExcInternalError("Empty evaluation task!"));
  Assert(n_rows > 0 && n_columns > 0,
         ExcInternalError("The evaluation needs n_rows, n_columns > 0, but " +
                          std::to_string(n_rows) + ", " +
                          std::to_string(n_columns) + " was passed!"));

  const Number *in1 = in0 + mm, *in2 = in1 + mm, *in3 = in2 + mm;
  Number       *out1 = out0 + nn, *out2 = out1 + nn, *out3 = out2 + nn;

  int nn_regular = (nn / 4) * 4;
  for (int col = 0; col < nn_regular; col += 4)
    {
      ndarray<Number, 4, 4> res;
      if (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + col;
          const Number   a = in0[0], b = in1[0], c = in2[0], d = in3[0];
          for (unsigned int k = 0; k < 4; ++k)
            {
              const Number m = matrix_ptr[k];
              res[0][k]      = m * a;
              res[1][k]      = m * b;
              res[2][k]      = m * c;
              res[3][k]      = m * d;
            }
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              const Number a = in0[i], b = in1[i], c = in2[i], d = in3[i];
              for (unsigned int k = 0; k < 4; ++k)
                {
                  const Number m = matrix_ptr[k];
                  res[0][k] += m * a;
                  res[1][k] += m * b;
                  res[2][k] += m * c;
                  res[3][k] += m * d;
                }
            }
        }
      else
        {
          const Number2 *matrix_0 = matrix + col * n_columns;
          const Number2 *matrix_1 = matrix + (col + 1) * n_columns;
          const Number2 *matrix_2 = matrix + (col + 2) * n_columns;
          const Number2 *matrix_3 = matrix + (col + 3) * n_columns;

          const Number a = in0[0], b = in1[0], c = in2[0], d = in3[0];
          Number       m = matrix_0[0];
          res[0][0]      = m * a;
          res[1][0]      = m * b;
          res[2][0]      = m * c;
          res[3][0]      = m * d;
          m              = matrix_1[0];
          res[0][1]      = m * a;
          res[1][1]      = m * b;
          res[2][1]      = m * c;
          res[3][1]      = m * d;
          m              = matrix_2[0];
          res[0][2]      = m * a;
          res[1][2]      = m * b;
          res[2][2]      = m * c;
          res[3][2]      = m * d;
          m              = matrix_3[0];
          res[0][3]      = m * a;
          res[1][3]      = m * b;
          res[2][3]      = m * c;
          res[3][3]      = m * d;
          for (int i = 1; i < mm; ++i)
            {
              const Number a = in0[i], b = in1[i], c = in2[i], d = in3[i];
              m = matrix_0[i];
              res[0][0] += m * a;
              res[1][0] += m * b;
              res[2][0] += m * c;
              res[3][0] += m * d;
              m = matrix_1[i];
              res[0][1] += m * a;
              res[1][1] += m * b;
              res[2][1] += m * c;
              res[3][1] += m * d;
              m = matrix_2[i];
              res[0][2] += m * a;
              res[1][2] += m * b;
              res[2][2] += m * c;
              res[3][2] += m * d;
              m = matrix_3[i];
              res[0][3] += m * a;
              res[1][3] += m * b;
              res[2][3] += m * c;
              res[3][3] += m * d;
            }
        }
      for (unsigned int i = 0; i < 4; ++i)
        {
          if (add)
            {
              out0[i] += res[0][i];
              out1[i] += res[1][i];
              out2[i] += res[2][i];
              out3[i] += res[3][i];
            }
          else
            {
              out0[i] = res[0][i];
              out1[i] = res[1][i];
              out2[i] = res[2][i];
              out3[i] = res[3][i];
            }
        }
      out0 += 4;
      out1 += 4;
      out2 += 4;
      out3 += 4;
    }
  if (nn - nn_regular == 3)
    {
      Number res0, res1, res2, res3, res4, res5, res6, res7, res8, res9, res10,
        res11;
      if (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + nn_regular;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[1] * in0[0];
          res2                      = matrix_ptr[2] * in0[0];
          res3                      = matrix_ptr[0] * in1[0];
          res4                      = matrix_ptr[1] * in1[0];
          res5                      = matrix_ptr[2] * in1[0];
          res6                      = matrix_ptr[0] * in2[0];
          res7                      = matrix_ptr[1] * in2[0];
          res8                      = matrix_ptr[2] * in2[0];
          res9                      = matrix_ptr[0] * in3[0];
          res10                     = matrix_ptr[1] * in3[0];
          res11                     = matrix_ptr[2] * in3[0];
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              res0 += matrix_ptr[0] * in0[i];
              res1 += matrix_ptr[1] * in0[i];
              res2 += matrix_ptr[2] * in0[i];
              res3 += matrix_ptr[0] * in1[i];
              res4 += matrix_ptr[1] * in1[i];
              res5 += matrix_ptr[2] * in1[i];
              res6 += matrix_ptr[0] * in2[i];
              res7 += matrix_ptr[1] * in2[i];
              res8 += matrix_ptr[2] * in2[i];
              res9 += matrix_ptr[0] * in3[i];
              res10 += matrix_ptr[1] * in3[i];
              res11 += matrix_ptr[2] * in3[i];
            }
        }
      else
        {
          const Number2 *matrix_0 = matrix + nn_regular * n_columns;
          const Number2 *matrix_1 = matrix + (nn_regular + 1) * n_columns;
          const Number2 *matrix_2 = matrix + (nn_regular + 2) * n_columns;

          res0  = matrix_0[0] * in0[0];
          res1  = matrix_1[0] * in0[0];
          res2  = matrix_2[0] * in0[0];
          res3  = matrix_0[0] * in1[0];
          res4  = matrix_1[0] * in1[0];
          res5  = matrix_2[0] * in1[0];
          res6  = matrix_0[0] * in2[0];
          res7  = matrix_1[0] * in2[0];
          res8  = matrix_2[0] * in2[0];
          res9  = matrix_0[0] * in3[0];
          res10 = matrix_1[0] * in3[0];
          res11 = matrix_2[0] * in3[0];
          for (int i = 1; i < mm; ++i)
            {
              res0 += matrix_0[i] * in0[i];
              res1 += matrix_1[i] * in0[i];
              res2 += matrix_2[i] * in0[i];
              res3 += matrix_0[i] * in1[i];
              res4 += matrix_1[i] * in1[i];
              res5 += matrix_2[i] * in1[i];
              res6 += matrix_0[i] * in2[i];
              res7 += matrix_1[i] * in2[i];
              res8 += matrix_2[i] * in2[i];
              res9 += matrix_0[i] * in3[i];
              res10 += matrix_1[i] * in3[i];
              res11 += matrix_2[i] * in3[i];
            }
        }
      if (add)
        {
          out0[0] += res0;
          out0[1] += res1;
          out0[2] += res2;
          out1[0] += res3;
          out1[1] += res4;
          out1[2] += res5;
          out2[0] += res6;
          out2[1] += res7;
          out2[2] += res8;
          out3[0] += res9;
          out3[1] += res10;
          out3[2] += res11;
        }
      else
        {
          out0[0] = res0;
          out0[1] = res1;
          out0[2] = res2;
          out1[0] = res3;
          out1[1] = res4;
          out1[2] = res5;
          out2[0] = res6;
          out2[1] = res7;
          out2[2] = res8;
          out3[0] = res9;
          out3[1] = res10;
          out3[2] = res11;
        }
    }
  else if (nn - nn_regular == 2)
    {
      Number res0, res1, res2, res3, res4, res5, res6, res7;
      if (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + nn_regular;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[1] * in0[0];
          res2                      = matrix_ptr[0] * in1[0];
          res3                      = matrix_ptr[1] * in1[0];
          res4                      = matrix_ptr[0] * in2[0];
          res5                      = matrix_ptr[1] * in2[0];
          res6                      = matrix_ptr[0] * in3[0];
          res7                      = matrix_ptr[1] * in3[0];
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              res0 += matrix_ptr[0] * in0[i];
              res1 += matrix_ptr[1] * in0[i];
              res2 += matrix_ptr[0] * in1[i];
              res3 += matrix_ptr[1] * in1[i];
              res4 += matrix_ptr[0] * in2[i];
              res5 += matrix_ptr[1] * in2[i];
              res6 += matrix_ptr[0] * in3[i];
              res7 += matrix_ptr[1] * in3[i];
            }
        }
      else
        {
          const Number2 *matrix_0 = matrix + nn_regular * n_columns;
          const Number2 *matrix_1 = matrix + (nn_regular + 1) * n_columns;

          res0 = matrix_0[0] * in0[0];
          res1 = matrix_1[0] * in0[0];
          res2 = matrix_0[0] * in1[0];
          res3 = matrix_1[0] * in1[0];
          res4 = matrix_0[0] * in2[0];
          res5 = matrix_1[0] * in2[0];
          res6 = matrix_0[0] * in3[0];
          res7 = matrix_1[0] * in3[0];
          for (int i = 1; i < mm; ++i)
            {
              res0 += matrix_0[i] * in0[i];
              res1 += matrix_1[i] * in0[i];
              res2 += matrix_0[i] * in1[i];
              res3 += matrix_1[i] * in1[i];
              res4 += matrix_0[i] * in2[i];
              res5 += matrix_1[i] * in2[i];
              res6 += matrix_0[i] * in3[i];
              res7 += matrix_1[i] * in3[i];
            }
        }
      if (add)
        {
          out0[0] += res0;
          out0[1] += res1;
          out1[0] += res2;
          out1[1] += res3;
          out2[0] += res4;
          out2[1] += res5;
          out3[0] += res6;
          out3[1] += res7;
        }
      else
        {
          out0[0] = res0;
          out0[1] = res1;
          out1[0] = res2;
          out1[1] = res3;
          out2[0] = res4;
          out2[1] = res5;
          out3[0] = res6;
          out3[1] = res7;
        }
    }
  else if (nn - nn_regular == 1)
    {
      Number res0, res1, res2, res3;
      if (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + nn_regular;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[0] * in1[0];
          res2                      = matrix_ptr[0] * in2[0];
          res3                      = matrix_ptr[0] * in3[0];
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              res0 += matrix_ptr[0] * in0[i];
              res1 += matrix_ptr[0] * in1[i];
              res2 += matrix_ptr[0] * in2[i];
              res3 += matrix_ptr[0] * in3[i];
            }
        }
      else
        {
          const Number2 *matrix_ptr = matrix + nn_regular * n_columns;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[0] * in1[0];
          res2                      = matrix_ptr[0] * in2[0];
          res3                      = matrix_ptr[0] * in3[0];
          for (int i = 1; i < mm; ++i)
            {
              res0 += matrix_ptr[i] * in0[i];
              res1 += matrix_ptr[i] * in1[i];
              res2 += matrix_ptr[i] * in2[i];
              res3 += matrix_ptr[i] * in3[i];
            }
        }
      if (add)
        {
          out0[0] += res0;
          out1[0] += res1;
          out2[0] += res2;
          out3[0] += res3;
        }
      else
        {
          out0[0] = res0;
          out1[0] = res1;
          out2[0] = res2;
          out3[0] = res3;
        }
    }
}

template <int dim_, typename Number = double>
class Operator : public Subscriptor
{
public:
  using value_type = Number;
  using number     = Number;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  static const int dim = dim_;
  static const int n_components = 1;

  using FECellIntegrator = FEEvaluation<dim, -1, 0, n_components, Number>;
  using FEFaceIntegrator = FEFaceEvaluation<dim, -1, 0, n_components, Number>;

  void
  reinit(const Mapping<dim>              &mapping,
         const DoFHandler<dim>           &dof_handler,
         const Quadrature<dim>           &quad,
         const AffineConstraints<number> &constraints,
         const unsigned int mg_level         = numbers::invalid_unsigned_int,
         const bool         ones_on_diagonal = false)
  {
    this->constraints.copy_from(constraints);

    typename MatrixFree<dim, number>::AdditionalData data;
    data.mapping_update_flags = update_gradients | update_quadrature_points;

    data.mg_level = mg_level;

    matrix_free.reinit(mapping, dof_handler, constraints, quad, data);

    constrained_indices.clear();

    if (ones_on_diagonal)
      for (auto i : this->matrix_free.get_constrained_dofs())
        constrained_indices.push_back(i);


    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::cout << "Sizes shape info: "
                  << matrix_free.get_shape_info()
                       .data[0]
                       .shape_values.memory_consumption()
                  << " "
                  << matrix_free.get_shape_info()
                       .data[0]
                       .shape_gradients.memory_consumption()
                  << " " << dof_handler.get_fe().dofs_per_cell << " "
                  << matrix_free.get_shape_info().n_q_points << " "
                  << matrix_free.get_shape_info().dofs_per_component_on_cell
                  << " " << matrix_free.get_dof_info(0).dof_indices.size()
                  << " " << dof_handler.get_triangulation().n_active_cells()
                  << " "
                  << dof_handler.get_triangulation().n_global_active_cells()
                  << " " << dof_handler.n_dofs() << std::endl;
        std::cout
          << "Mapping info: "
          << matrix_free.get_mapping_info()
               .cell_data[0]
               .data_index_offsets.size()
          << " n_jacobians "
          << matrix_free.get_mapping_info().cell_data[0].jacobians[0].size()
          << std::endl;
      }

    {
      constexpr unsigned int n_lanes = VectorizedArray<number>::size();
      manual_dof_indices.reinit(
        matrix_free.n_cell_batches(),
        matrix_free.get_dof_handler().get_fe().dofs_per_cell * n_lanes,
        true);
      manual_dof_indices.fill(numbers::invalid_unsigned_int);
      std::vector<types::global_dof_index> dof_indices(
        matrix_free.get_dof_handler().get_fe().dofs_per_cell);

      dof_indices_have_constraints.clear();
      dof_indices_have_constraints.resize(matrix_free.n_cell_batches());

      for (unsigned int c = 0; c < matrix_free.n_cell_batches(); ++c)
        {
          bool has_constraints =
            matrix_free.n_active_entries_per_cell_batch(c) < n_lanes;
          for (unsigned int v = 0;
              v < matrix_free.n_active_entries_per_cell_batch(c);
              ++v)
            {
              matrix_free.get_cell_iterator(c, v)->get_dof_indices(dof_indices);
              for (unsigned int i = 0; i < dof_indices.size(); ++i)
                if (!constraints.is_constrained(dof_indices[i]))
                  manual_dof_indices(c, i * n_lanes + v) =
                    matrix_free.get_dof_info()
                      .vector_partitioner->global_to_local(dof_indices[i]);
                else
                  has_constraints = true;
            }
          dof_indices_have_constraints[c] = has_constraints;
        }
      }
  }

  virtual types::global_dof_index
  m() const
  {
    if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
      return this->matrix_free.get_dof_handler().n_dofs(
        this->matrix_free.get_mg_level());
    else
      return this->matrix_free.get_dof_handler().n_dofs();
  }

  Number
  el(unsigned int, unsigned int) const
  {
    DEAL_II_NOT_IMPLEMENTED();
    return 0;
  }

  virtual void
  initialize_dof_vector(VectorType &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }

  virtual void
  vmult_reference(VectorType &dst, const VectorType &src) const
  {
    this->matrix_free.cell_loop(
      &Operator::do_cell_integral_range, this, dst, src, true);

    for (unsigned int i = 0; i < constrained_indices.size(); ++i)
      dst.local_element(constrained_indices[i]) =
        src.local_element(constrained_indices[i]);
  }


  virtual void
  vmult(VectorType &dst, const VectorType &src) const
  {
    this->matrix_free.cell_loop(
      &Operator::do_cell_integral_manual, this, dst, src, true);

    for (unsigned int i = 0; i < constrained_indices.size(); ++i)
      dst.local_element(constrained_indices[i]) =
        src.local_element(constrained_indices[i]);
  }


  virtual void
  rhs(VectorType &dst) const
  {
    const VectorType dummy;
    this->matrix_free.cell_loop(
      &Operator::rhs_range, this, dst, dummy, true);
  }


  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    vmult(dst, src);
  }

  const MatrixFree<dim, number> &
  get_matrix_free() const
  {
    return matrix_free;
  }

  void
  compute_inverse_diagonal(VectorType &diagonal_vector) const
  {
    matrix_free.initialize_dof_vector(diagonal_vector);

    MatrixFreeTools::compute_diagonal<dim, -1, 0, 1, number, VectorizedArray<number>>(
      matrix_free,
      diagonal_vector,
      [&](auto &phi) { do_cell_integral_local(phi); },
      {},
      {},
      0,
      0,
      0);

    for (unsigned int i = 0; i < diagonal_vector.locally_owned_size(); ++i)
      {
        if (std::abs(diagonal_vector.local_element(i)) > 1.0e-10)
          diagonal_vector.local_element(i) = 1.0 / diagonal_vector.local_element(i);
        else
          diagonal_vector.local_element(i) = 1.0;
      }
  }

  void
  get_system_matrix(TrilinosWrappers::SparseMatrix &system_matrix)
  {
    const auto &dof_handler = matrix_free.get_dof_handler();

    TrilinosWrappers::SparsityPattern dsp(
      dof_handler.locally_owned_dofs(),
      dof_handler.locally_owned_dofs(),
      DoFTools::extract_locally_relevant_dofs(dof_handler),
      dof_handler.get_triangulation().get_communicator());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);

    dsp.compress();
    system_matrix.reinit(dsp);
    system_matrix = 0.;

    MatrixFreeTools::compute_matrix<dim, -1, 0, 1, number, VectorizedArray<number>>(
      matrix_free,
      constraints,
      system_matrix,
      [&](auto &phi) { do_cell_integral_local(phi); },
      {},
      {},
      0,
      0,
      0);
  }

private:  
  void
  rhs_range(
    const MatrixFree<dim, number>               &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FECellIntegrator integrator(matrix_free, range);
    AnalyticalRHS<dim> rhs;

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        integrator.reinit(cell);
        

        for (unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          const auto f = evaluate_function(rhs, integrator.quadrature_point(q));
          integrator.submit_value(f, q);
        }

        integrator.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

 
  void
  do_cell_integral_range(
    const MatrixFree<dim, number>               &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FECellIntegrator integrator(matrix_free, range);

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        integrator.reinit(cell);
        integrator.gather_evaluate(src, EvaluationFlags::gradients);

        for (unsigned int q = 0; q < integrator.n_q_points; ++q)
          integrator.submit_gradient(integrator.get_gradient(q), q);

        integrator.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }

  void
  do_cell_integral_local(
    FEEvaluation<dim, -1, 0, n_components, Number> &integrator) const
  { 
        integrator.evaluate(EvaluationFlags::gradients);

        for (unsigned int q = 0; q < integrator.n_q_points; ++q)
          integrator.submit_gradient(integrator.get_gradient(q), q);

        integrator.integrate(EvaluationFlags::gradients);   
  }

 
  void
  do_cell_integral_manual(
    const MatrixFree<dim, number>               &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    AlignedVector<VectorizedArray<number>> *scratch_data =
      matrix_free.acquire_scratch_data();
    const internal::MatrixFreeFunctions::ShapeInfo<number> &shape_info =
      matrix_free.get_shape_info();
    const unsigned int dofs_per_cell  = shape_info.dofs_per_component_on_cell;
    constexpr unsigned int batch_size = 4;
    const unsigned int     n_q_points = shape_info.n_q_points;
    constexpr unsigned int n_lanes    = VectorizedArray<number>::size();

    const auto   &mapping_data = matrix_free.get_mapping_info().cell_data[0];
    const number *quadrature_weights =
      mapping_data.descriptor[0].quadrature_weights.data();

    scratch_data->resize_fast(batch_size * (dim * n_q_points + dofs_per_cell));
    VectorizedArray<number> *values_dofs = scratch_data->begin();
    VectorizedArray<number> *gradients_quad =
      scratch_data->begin() + batch_size * dofs_per_cell;

    for (unsigned int cell = range.first; cell < range.second;
         cell += batch_size)
      {
        // read dof values
        const unsigned int my_batch_size =
          cell + batch_size <= range.second ? batch_size : range.second - cell;
        const unsigned int *dof_indices = &manual_dof_indices(cell, 0);
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            const number *src_ptr = src.begin();
            if (dof_indices_have_constraints[cell + batch])
              {
                for (unsigned int i = 0; i < dofs_per_cell;
                     ++i, dof_indices += n_lanes)
                  {
#if 1
                    values_dofs[batch * dofs_per_cell + i] = {};
                    for (unsigned int v = 0; v < n_lanes; ++v)
                      if (dof_indices[v] != numbers::invalid_unsigned_int)
                        values_dofs[batch * dofs_per_cell + i][v] =
                          src_ptr[dof_indices[v]];
#else
                    values_dofs[batch * dofs_per_cell + i].gather(src_ptr,
                                                                  dof_indices);
#endif
                  }
              }
            else
              for (unsigned int i = 0; i < dofs_per_cell;
                   ++i, dof_indices += n_lanes)
                {
                  values_dofs[batch * dofs_per_cell + i] = {};
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    values_dofs[batch * dofs_per_cell + i][v] =
                      src_ptr[dof_indices[v]];
                }
          }

        // interpolate
        apply_matrix_vector_product_4<true, false>(
          shape_info.data[0].shape_gradients.data(),
          values_dofs,
          gradients_quad,
          dofs_per_cell,
          n_q_points * dim);

        // quadrature point operation
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            const unsigned int offsets =
              mapping_data.data_index_offsets[cell + batch];
            const Tensor<2, dim, VectorizedArray<number>> *jac =
              mapping_data.jacobians[0].data() + offsets;
            const VectorizedArray<number> * j_value =
              &mapping_data.JxW_values[offsets];
            VectorizedArray<number> *grad_ptr =
              gradients_quad + batch * n_q_points * dim;
            if (matrix_free.get_mapping_info().cell_type[cell + batch] <=
                internal::MatrixFreeFunctions::affine)
              {
                // const SymmetricTensor<2, dim, VectorizedArray<number>>
                //  my_metric = j_value[0] * symmetrize(transpose(jac[0]) *
                //  jac[0]);
                SymmetricTensor<2, dim, VectorizedArray<number>> my_metric;
                for (unsigned int d = 0; d < dim; ++d)
                  for (unsigned int f = d; f < dim; ++f)
                    {
                      VectorizedArray<number> sum = jac[0][0][d] * jac[0][0][f];
                      for (unsigned int e = 1; e < dim; ++e)
                        sum += jac[0][e][d] * jac[0][e][f];
                      my_metric[d][f] = sum * j_value[0];
                    }

                for (unsigned int q = 0; q < n_q_points; ++q, grad_ptr += dim)
                  {
                    Tensor<1, dim, VectorizedArray<number>> grad;
                    for (unsigned int d = 0; d < dim; ++d)
                      grad[d] = grad_ptr[d];
                    Tensor<1, dim, VectorizedArray<number>> result =
                      my_metric * grad;
                    const number weight = quadrature_weights[q];
                    for (unsigned int d = 0; d < dim; ++d)
                      grad_ptr[d] = weight * result[d];
                  }
              }
            else
              {
                for (unsigned int q = 0; q < n_q_points; ++q, grad_ptr += dim)
                  {
                    Tensor<1, dim, VectorizedArray<number>> grad;
                    for (unsigned int d = 0; d < dim; ++d)
                      grad[d] = grad_ptr[d];
                    Tensor<1, dim, VectorizedArray<number>> result =
                      j_value[q] * (transpose(jac[q]) * (jac[q] * grad));
                    for (unsigned int d = 0; d < dim; ++d)
                      grad_ptr[d] = result[d];
                  }
              }
          }

        // integrate
        apply_matrix_vector_product_4<false, false>(
          shape_info.data[0].shape_gradients.data(),
          gradients_quad,
          values_dofs,
          dofs_per_cell,
          n_q_points * dim);

        // distribute local to global
        dof_indices = &manual_dof_indices(cell, 0);
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            if (dof_indices_have_constraints[cell + batch])
              {
                for (unsigned int i = 0; i < dofs_per_cell;
                     ++i, dof_indices += n_lanes)
                  {
#if 1 || DEAL_II_VECTORIZATION_WIDTH_IN_BITS < 512
                    for (unsigned int v = 0; v < n_lanes; ++v)
                      if (dof_indices[v] != numbers::invalid_unsigned_int)
                        dst.local_element(dof_indices[v]) +=
                          values_dofs[batch * dofs_per_cell + i][v];
#else
                    VectorizedArray<number> val;
                    val.gather(dst.begin(), dof_indices);
                    val += values_dofs[batch * dofs_per_cell + i];
                    val.scatter(dof_indices, dst.begin());
#endif
                  }
              }
            else
              for (unsigned int i = 0; i < dofs_per_cell;
                   ++i, dof_indices += n_lanes)
                {
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    dst.local_element(dof_indices[v]) +=
                      values_dofs[batch * dofs_per_cell + i][v];
                }
          }
      }

    matrix_free.release_scratch_data(scratch_data);
  }

  MatrixFree<dim, number> matrix_free;

  AffineConstraints<number> constraints;

  std::vector<unsigned int> constrained_indices;

  Table<2, unsigned int> manual_dof_indices;

  std::vector<unsigned char> dof_indices_have_constraints;
};



template <typename VectorType>
class MGCoarseAMG : public MGCoarseGridBase<VectorType>
{
private:
public:
  MGCoarseAMG(const TrilinosWrappers::PreconditionAMG &amg, const bool is_singular_in)
  {
    amg_preconditioner = &amg;
    is_singular        = is_singular_in;
  }

  void
  operator()(const unsigned int /*level*/,
             VectorType       &dst,
             const VectorType &src) const final
  {
    if constexpr (std::is_same_v<VectorType,
                                 LinearAlgebra::distributed::Vector<TrilinosScalar>>)
      {
        if (is_singular)
          {
            VectorType r(src);
            dealii::VectorTools::subtract_mean_value(r);
            amg_preconditioner->vmult(dst, r);
          }
        else
          amg_preconditioner->vmult(dst, src);
      }
    else
      {
        LinearAlgebra::distributed::Vector<TrilinosScalar> src_;
        LinearAlgebra::distributed::Vector<TrilinosScalar> dst_;

        src_ = src;
        dst_ = dst;

        if (is_singular)
          dealii::VectorTools::subtract_mean_value(src_);
        amg_preconditioner->vmult(dst_, src_);

        dst = dst_;
      }
  }

private:
  const TrilinosWrappers::PreconditionAMG *amg_preconditioner;
  bool                                     is_singular;
};

template <int dim, typename number_operator, typename number = number_operator>
class MultigridPreconditioner
{
  using VectorType       = LinearAlgebra::distributed::Vector<number>;
  using VectorTypeSystem = LinearAlgebra::distributed::Vector<number_operator>;
  using SystemMatrixType = Operator<dim, number_operator>;
  using LevelMatrixType  = Operator<dim, number>;

  using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
  using SmootherType =
    PreconditionChebyshev<LevelMatrixType, VectorType, SmootherPreconditionerType>;
  using PreconditionerType =
    PreconditionMG<dim, VectorType, MGTransferGlobalCoarsening<dim, VectorType>>;

public:
  MultigridPreconditioner(SystemMatrixType &opppppp)
  {
    const auto &dof_handler =
      opppppp.get_matrix_free().get_dof_handler();

    // create coarse grid triangulations
    {
      const unsigned int n_global_levels =  dof_handler.get_triangulation().n_global_levels();
      coarse_grid_triangulations.reserve(n_global_levels);

      for(unsigned int i = 0; i < n_global_levels; ++i)
      {
        const auto serial_grid_generator =
          [i](dealii::Triangulation<dim, dim> &tria_serial) {
              // set up triangulation
              GridGenerator::subdivided_hyper_cube_with_simplices(
                tria_serial, 2, -1.0, 1.0);
              tria_serial.refine_global(i);
              
          };
        const auto serial_grid_partitioner =
          [&](dealii::Triangulation<dim, dim> &tria_serial,
              const MPI_Comm                   comm,
              const unsigned int) {
            dealii::GridTools::partition_triangulation_zorder(
              dealii::Utilities::MPI::n_mpi_processes(comm), tria_serial);
          };

        const unsigned int group_size = 20;

        coarse_grid_triangulations.emplace_back(std::make_unique<parallel::fullydistributed::Triangulation<dim>>(MPI_COMM_WORLD));

        typename dealii::TriangulationDescription::Settings
          triangulation_description_setting =
            dealii::TriangulationDescription::default_setting;
        const auto description = dealii::TriangulationDescription::Utilities::
          create_description_from_triangulation_in_groups<dim, dim>(
            serial_grid_generator,
            serial_grid_partitioner,
            dof_handler.get_triangulation().get_communicator(),
            group_size,
            dealii::Triangulation<dim>::none,
            triangulation_description_setting);

        coarse_grid_triangulations.back()->create_triangulation(description);
      }
    }

    const unsigned int n_h_levels = coarse_grid_triangulations.size() - 1;

    const std::vector<unsigned int> level_degrees =
        MGTransferGlobalCoarseningTools::create_polynomial_coarsening_sequence(
          dof_handler.get_fe().degree,
          MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::decrease_by_one);
    const unsigned int n_p_levels = level_degrees.size();

    const unsigned int minlevel = 0;
    const unsigned int maxlevel =
      n_h_levels + n_p_levels - 1;

    dof_handlers.resize(minlevel, maxlevel);
    mg_matrices.resize(minlevel, maxlevel);
    
    transfers.resize(minlevel, maxlevel);

    unsigned int l = 0;
    // p-MG with linear elements
    for (unsigned int i = 0; i < level_degrees.size(); ++i)
      {
        auto &dof_handler   = dof_handlers[l];
        {
          const FE_SimplexP<dim>  fe(level_degrees[i]);
          dof_handler.reinit(*coarse_grid_triangulations[0]);
          dof_handler.distribute_dofs(fe);
        }
        ++l;
      }
    // h-MG
    for (unsigned int i = 1; i < coarse_grid_triangulations.size(); ++i)
      {
        auto &dof_handler   = dof_handlers[l];
        {
          const FE_SimplexP<dim> fe(level_degrees[level_degrees.size()-1]);
         
          dof_handler.reinit(*coarse_grid_triangulations[i]);
          dof_handler.distribute_dofs(fe);
        }
        ++l;
      }
    

    // init levels
    for (unsigned int level = minlevel; level <= maxlevel; ++level)
      {
        const unsigned int fe_degree     = dof_handlers[level].get_fe().degree;
        Quadrature<dim>      quadrature      = QGaussSimplex<dim>(fe_degree + 1);
        MappingFE<dim>       mapping(FE_SimplexP<dim>(1));

        {
          AffineConstraints<number> constraint;
          // set up constraints, then renumber dofs, and set up constraints again
          if (false)
            {
              const IndexSet locally_relevant_dofs =
                DoFTools::extract_locally_relevant_dofs(dof_handlers[level]);
              constraint.reinit(dof_handlers[level].locally_owned_dofs(),
                                locally_relevant_dofs);
              VectorTools::interpolate_boundary_values(
                mapping,
                dof_handlers[level],
                0,
                Functions::ZeroFunction<dim, number>(1),
                constraint);
              constraint.close();
              typename MatrixFree<dim, number>::AdditionalData data;
              DoFRenumbering::matrix_free_data_locality(dof_handlers[level],
                                                        constraint,
                                                        data);
            }
          const IndexSet locally_relevant_dofs =
            DoFTools::extract_locally_relevant_dofs(dof_handlers[level]);
          constraint.reinit(dof_handlers[level].locally_owned_dofs(),
                            locally_relevant_dofs);
          VectorTools::interpolate_boundary_values(
            mapping, dof_handlers[level], 0, Functions::ZeroFunction<dim, number>(1), constraint);
          constraint.close();

          mg_matrices[level].reinit(MappingFE<dim>(FE_SimplexP<dim>(1)), dof_handlers[level], quadrature, constraint, numbers::invalid_unsigned_int, true);
        }      
      }

    // init transfer
    for (unsigned int level = minlevel; level < maxlevel; ++level)
      transfers[level + 1].reinit(dof_handlers[level + 1], dof_handlers[level]);

    transfer = MGTransferGlobalCoarsening<dim, VectorType>(
      transfers, [&](const auto l, auto &vec) {
        mg_matrices[l].get_matrix_free().initialize_dof_vector(vec);
      });

    // Setup smoother for every level
    smoother_data.resize(minlevel, maxlevel);

    for (unsigned int level = minlevel; level <= maxlevel; ++level)
      {
        if (level > 0)
          {
            smoother_data[level].smoothing_range     = 20.;
            smoother_data[level].degree              = 5;
            smoother_data[level].eig_cg_n_iterations = 10;
          }
        else
          {
            smoother_data[0].smoothing_range     = 1e-3;
            smoother_data[0].degree              = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
          }
        smoother_data[level].preconditioner =
          std::make_shared<SmootherPreconditionerType>();
        mg_matrices[level].compute_inverse_diagonal(
          smoother_data[level].preconditioner->get_vector());
      }

    mg_smoother.initialize(mg_matrices, smoother_data);

    // Setup corase grid AMG
    mg_matrices[minlevel].get_system_matrix(coarse_system_matrix);
    TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;    

    amg_data.elliptic              = true;
    amg_data.higher_order_elements = false;
    amg_data.w_cycle               = false;
    amg_data.aggregation_threshold = 0.2;
    amg_data.smoother_sweeps       = 5;
    amg_data.n_cycles              = 1;
    amg_data.smoother_type         = "Chebyshev";

    precondition_amg.initialize(coarse_system_matrix, amg_data);
  }

  unsigned int
  solve(SystemMatrixType       &oppp,
        VectorTypeSystem       &x,
        const VectorTypeSystem &b)
  {
    // Coarse grid solver
    std::unique_ptr<MGCoarseGridBase<VectorType>> mg_coarse;
    mg_coarse = std::make_unique<MGCoarseAMG<VectorType>>(precondition_amg,
                                                          false);

    // Setup levels and transfers
    mg::Matrix<VectorType> mg_matrix(mg_matrices);
    Multigrid<VectorType>  mg(mg_matrix, *mg_coarse, transfer, mg_smoother, mg_smoother);

    PreconditionerType preconditioner(
      oppp.get_matrix_free().get_dof_handler(), mg, transfer);

    SolverControl              control(100000, 1e-10 * b.l2_norm());
    SolverCG<VectorTypeSystem> solver_cg(control);

    double min_time = 10000000000.;
    double time_total = 0.;
    for(unsigned int t = 0; t < 100; ++t)
    {
      x = 0.;
      Timer time;
      solver_cg.solve(oppp, x, b, preconditioner);
      const double run_time = time.wall_time();
      min_time = std::min(min_time, run_time);
      time_total += run_time;
    }
    const double run_time = min_time;
    
    ConditionalOStream pcout(std::cout,
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
        0);
    unsigned int n    = control.last_step();
    pcout << "n_dofs " << oppp.get_matrix_free().get_dof_handler().n_dofs() << "  time "
                << time_total / 100 << "   1e5 DoFs/s "
                << 1e-5 * oppp.get_matrix_free().get_dof_handler().n_dofs() * 100 / time_total << " in " << n << " iterations" << std::endl;

    
    double l2_0 = control.initial_value();
    double l2_n = control.last_value();
    double rho = std::pow(l2_n / l2_0, 1.0 / n);
    double n10 = -10.0 * std::log(10.0) / std::log(rho);  
    unsigned int const N_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    double const t_10 = run_time * double(n10) / double(n);
  
    double const tau_10 = t_10 * (double)N_mpi_processes / oppp.get_matrix_free().get_dof_handler().n_dofs();
    const double E_10 = 1.0 / tau_10;

    pcout << "n_10 " << n10 << "  t_10 "
                << t_10 << "   tau_10 "
                << tau_10 << " in E_10 " << E_10 * 1e-5 << " 1e5 * DoFs/s/core" << std::endl;


    return control.last_step();
  }

private:
  MGLevelObject<LevelMatrixType>                                    mg_matrices;
  MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType> mg_smoother;
  MGLevelObject<typename SmootherType::AdditionalData>              smoother_data;

  MGLevelObject<DoFHandler<dim>>                     dof_handlers;
  MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers;
  MGTransferGlobalCoarsening<dim, VectorType>        transfer;

  std::vector<std::unique_ptr<parallel::fullydistributed::Triangulation<dim>>> coarse_grid_triangulations;

  TrilinosWrappers::SparseMatrix    coarse_system_matrix;
  TrilinosWrappers::PreconditionAMG precondition_amg;
};




template <int dim, typename Number>
void
do_test(const unsigned int fe_degree)
{
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  pcout << "Running in " << dim << "D with degree " << fe_degree << std::endl;
  FE_SimplexP<dim> fe(fe_degree);
  MappingFE<dim>     mapping(FE_SimplexP<dim>(1));
  QGaussSimplex<dim> quad(fe_degree + 1);

  AffineConstraints<Number> constraint;

  for (unsigned int refinement = 0;  refinement < 3; ++refinement) //refinement < n_refinements &&
    {
      const auto serial_grid_generator =
        [&](dealii::Triangulation<dim, dim> &tria_serial) {
            {
              // set up triangulation
              GridGenerator::subdivided_hyper_cube_with_simplices(
                tria_serial, 2, -1.0, 1.0);

              tria_serial.refine_global(refinement);
            }
        };
      const auto serial_grid_partitioner =
        [&](dealii::Triangulation<dim, dim> &tria_serial,
            const MPI_Comm                   comm,
            const unsigned int) {
          dealii::GridTools::partition_triangulation_zorder(
            dealii::Utilities::MPI::n_mpi_processes(comm), tria_serial);
        };

      const unsigned int group_size = 20;

      parallel::fullydistributed::Triangulation<dim> tria(MPI_COMM_WORLD);
      typename dealii::TriangulationDescription::Settings
        triangulation_description_setting =
          dealii::TriangulationDescription::default_setting;
      const auto description = dealii::TriangulationDescription::Utilities::
        create_description_from_triangulation_in_groups<dim, dim>(
          serial_grid_generator,
          serial_grid_partitioner,
          tria.get_communicator(),
          group_size,
          dealii::Triangulation<dim>::none,
          triangulation_description_setting);

      tria.create_triangulation(description);

      DoFHandler<dim> dof_handler(tria);
      dof_handler.distribute_dofs(fe);

      constraint.clear();
            // set up constraints, then renumber dofs, and set up constraints again
      if (false)
        {
          const IndexSet locally_relevant_dofs =
            DoFTools::extract_locally_relevant_dofs(dof_handler);
          constraint.reinit(dof_handler.locally_owned_dofs(),
                            locally_relevant_dofs);
          VectorTools::interpolate_boundary_values(
            mapping,
            dof_handler,
            0,
            AnalyticalSolution<dim>(),
            constraint);
          constraint.close();
          typename MatrixFree<dim, Number>::AdditionalData data;
          DoFRenumbering::matrix_free_data_locality(dof_handler,
                                                    constraint,
                                                    data);
        }
      const IndexSet locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler);
      constraint.reinit(dof_handler.locally_owned_dofs(),
                        locally_relevant_dofs);
      VectorTools::interpolate_boundary_values(
        mapping, dof_handler, 0, AnalyticalSolution<dim>(), constraint);
      constraint.close();

      Operator<dim, Number> op;
      // set up operator
      op.reinit(mapping,
                dof_handler,
                quad,
                constraint,
                numbers::invalid_unsigned_int,
                true);

      
      
      LinearAlgebra::distributed::Vector<Number> vec1, vec2, vec3, x, b;
      op.initialize_dof_vector(vec1);
      op.initialize_dof_vector(vec2);
      op.initialize_dof_vector(vec3);
      op.initialize_dof_vector(x);
      op.initialize_dof_vector(b);

      for (Number &a : vec1)
        a = static_cast<double>(rand()) / RAND_MAX;

      MPI_Barrier(MPI_COMM_WORLD);
      for (unsigned int r = 0; r < 1; ++r)
        {
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_START(("matvec_p" + std::to_string(fe_degree) + "_s" +
                               std::to_string(dof_handler.n_dofs()))
                                .c_str());
#endif
          Timer time;
          for (unsigned int t = 0; t < 1; ++t)
            op.vmult_reference(vec2, vec1);
          const double run_time = time.wall_time();
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_STOP(("matvec_p" + std::to_string(fe_degree) + "_s" +
                              std::to_string(dof_handler.n_dofs()))
                               .c_str());
#endif
          pcout << "n_dofs " << dof_handler.n_dofs() << "  time "
                << run_time / 1 << "  GDoFs/s "
                << 1e-9 * dof_handler.n_dofs() * 1 / run_time << std::endl;
        }

      for (unsigned int r = 0; r < 1; ++r)
        {
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_START(("matvec_manual_p" + std::to_string(fe_degree) +
                               "_s" + std::to_string(dof_handler.n_dofs()))
                                .c_str());
#endif
          Timer time;
          for (unsigned int t = 0; t < 1; ++t)
            op.vmult(vec3, vec1);
          const double run_time = time.wall_time();
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_STOP(("matvec_manual_p" + std::to_string(fe_degree) +
                              "_s" + std::to_string(dof_handler.n_dofs()))
                               .c_str());
#endif
          pcout << "n_dofs " << dof_handler.n_dofs() << "  time manual "
                << run_time / 1 << "  GDoFs/s "
                << 1e-9 * dof_handler.n_dofs() * 1 / run_time << std::endl;
        }

      vec3 -= vec2;
      pcout << "Verification: " << vec3.l2_norm() / vec2.l2_norm() << std::endl;
      pcout << std::endl;

      unsigned int n_iterations = 0;
      op.rhs(b);

      MultigridPreconditioner<dim, Number, float> multigrid(op);

      MPI_Barrier(MPI_COMM_WORLD);
      for (unsigned int r = 0; r < 1; ++r)
        {
          x = 0.;
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_START(("mg_solve_p" + std::to_string(fe_degree) + "_s" +
                               std::to_string(dof_handler.n_dofs()))
                                .c_str());
#endif
          for (unsigned int t = 0; t < 1; ++t)
            n_iterations = multigrid.solve(op, x, b);
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_STOP(("mg_solve_p" + std::to_string(fe_degree) + "_s" +
                              std::to_string(dof_handler.n_dofs()))
                               .c_str());
#endif
        }
      
        AnalyticalSolution<dim> exact_solution;
        VectorTools::interpolate(mapping, dof_handler, exact_solution, vec1);
        x -= vec1;
        pcout << "relative L2 error: " << x.l2_norm() / vec1.l2_norm() << std::endl;
        pcout << std::endl;
    }
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif

  int degree = 1;
  int dim    = 2;
  if (argc > 1)
    dim = std::atoi(argv[1]);
  if (argc > 2)
    degree = std::atoi(argv[2]);

  if (dim == 2)
    do_test<2, double>(degree);
  else
    do_test<3, double>(degree);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}

