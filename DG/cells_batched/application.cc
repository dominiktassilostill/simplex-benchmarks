
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

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

using namespace dealii;



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

template <int dim_, int n_components = dim_, typename Number = double>
class Operator : public Subscriptor
{
public:
  using value_type = Number;
  using number     = Number;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  static const int dim = dim_;

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
    (void)ones_on_diagonal;
    this->constraints.copy_from(constraints);

    typename MatrixFree<dim, number>::AdditionalData data;
    data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points);
    data.mapping_update_flags_inner_faces =
      (update_gradients | update_JxW_values | update_normal_vectors);
    data.mapping_update_flags_boundary_faces =
      (update_gradients | update_JxW_values | update_normal_vectors |
       update_quadrature_points);
    data.mg_level = mg_level;

    matrix_free.reinit(mapping, dof_handler, constraints, quad, data);
    penalty_factor = 1.0 * (this->matrix_free.get_dof_handler().get_fe().degree + 1) *
           (this->matrix_free.get_dof_handler().get_fe().degree + dim) / dim;
    {
      unsigned int n_cells = matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
      array_penalty_parameter.resize(n_cells);

      dealii::Mapping<dim> const &       mapping = *matrix_free.get_mapping_info().mapping;
      dealii::FiniteElement<dim> const & fe      = matrix_free.get_dof_handler().get_fe();
      unsigned int const                 degree  = fe.degree;

      auto const reference_cells =
        matrix_free.get_dof_handler().get_triangulation().get_reference_cells();

      auto const quadrature = reference_cells[0].template get_gauss_type_quadrature<dim>(degree + 1);
      dealii::FEValues<dim> fe_values(mapping, fe, quadrature, dealii::update_JxW_values);

      auto const face_quadrature =
        reference_cells[0].face_reference_cell(0).template get_gauss_type_quadrature<dim - 1>(degree +
                                                                                              1);
      dealii::FEFaceValues<dim> fe_face_values(mapping, fe, face_quadrature, dealii::update_JxW_values);

      for(unsigned int i = 0; i < n_cells; ++i)
      {
        for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(i); ++v)
        {
          typename dealii::DoFHandler<dim>::cell_iterator cell =
            matrix_free.get_cell_iterator(i, v);
          fe_values.reinit(cell);

          // calculate cell volume
          Number volume = 0;
          for(unsigned int q = 0; q < quadrature.size(); ++q)
          {
            volume += fe_values.JxW(q);
          }

          // calculate surface area
          Number surface_area = 0;
          for(unsigned int const f : cell->face_indices())
          {
            fe_face_values.reinit(cell, f);
            Number const factor =
              (cell->at_boundary(f) and not(cell->has_periodic_neighbor(f))) ? 1. : 0.5;
            for(unsigned int q = 0; q < face_quadrature.size(); ++q)
            {
              surface_area += fe_face_values.JxW(q) * factor;
            }
          }

          array_penalty_parameter[i][v] = surface_area / volume;
        }
      }
    }


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

    constrained_indices.clear();
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
  vmult(VectorType &dst, const VectorType &src) const
  {
    this->matrix_free.loop(
      &Operator::do_cell_integral_range,
      &Operator::do_face_integral_range,
      &Operator::do_boundary,
      this,
      dst,
      src,
      true,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients);
  }


  virtual void
  vmult_manual(VectorType &dst, const VectorType &src) const
  {
    this->matrix_free.loop(
      &Operator::do_cell_integral_manual,
      &Operator::do_face_integral_manual,
      &Operator::do_boundary_integral_manual,
      this,
      dst,
      src,
      true,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients);
  }


  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    vmult(dst, src);
  }

private:
  number
  get_penalty_factor() const
  {
    return penalty_factor;
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
  do_face_integral_range(
    const MatrixFree<dim, number>               &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FEFaceIntegrator integrator_inner(matrix_free, range, true);
    FEFaceIntegrator integrator_outer(matrix_free, range, false);


    for (unsigned int face = range.first; face < range.second; ++face)
      {
        integrator_inner.reinit(face);
        integrator_inner.gather_evaluate(src,
                                         EvaluationFlags::values |
                                           EvaluationFlags::gradients);
        integrator_outer.reinit(face);
        integrator_outer.gather_evaluate(src,
                                         EvaluationFlags::values |
                                           EvaluationFlags::gradients);



        for (unsigned int q = 0; q < integrator_inner.n_q_points; ++q)
          {
             const VectorizedArray<number> sigma =
             std::max(integrator_inner.read_cell_data(array_penalty_parameter),
                   integrator_outer.read_cell_data(array_penalty_parameter)) * get_penalty_factor();

            const VectorizedArray<number> solution_jump =
              (integrator_inner.get_value(q) - integrator_outer.get_value(q));
            const VectorizedArray<number> averaged_normal_derivative =
              (integrator_inner.get_normal_derivative(q) +
               integrator_outer.get_normal_derivative(q)) *
              number(0.5);
            const VectorizedArray<number> test_by_value =
              solution_jump * sigma - averaged_normal_derivative;


            integrator_inner.submit_value(test_by_value, q);
            integrator_outer.submit_value(-test_by_value, q);

            integrator_inner.submit_normal_derivative(-solution_jump *
                                                        number(0.5),
                                                      q);
            integrator_outer.submit_normal_derivative(-solution_jump *
                                                        number(0.5),
                                                      q);
          }

        integrator_inner.integrate_scatter(EvaluationFlags::values |
                                             EvaluationFlags::gradients,
                                           dst);
        integrator_outer.integrate_scatter(EvaluationFlags::values |
                                             EvaluationFlags::gradients,
                                           dst);
      }
  }

  void
  do_boundary(const MatrixFree<dim, number>               &matrix_free,
              VectorType                                  &dst,
              const VectorType                            &src,
              const std::pair<unsigned int, unsigned int> &range) const
  {
    do_boundary_real(matrix_free, dst, src, range);
  }

  void
  do_boundary_real(const MatrixFree<dim, number>               &matrix_free,
                   VectorType                                  &dst,
                   const VectorType                            &src,
                   const std::pair<unsigned int, unsigned int> &range) const
  {
    FEFaceIntegrator integrator_inner(matrix_free, range);

    for (unsigned int face = range.first; face < range.second; ++face)
      {
        integrator_inner.reinit(face);
        integrator_inner.gather_evaluate(src,
                                         EvaluationFlags::values |
                                           EvaluationFlags::gradients);



        for (unsigned int q = 0; q < integrator_inner.n_q_points; ++q)
          {
            const VectorizedArray<number> sigma =
          integrator_inner.read_cell_data(array_penalty_parameter)
                   * get_penalty_factor();
            const VectorizedArray<number> u_inner =
              integrator_inner.get_value(q);
            const VectorizedArray<number> u_outer = -u_inner;
            const VectorizedArray<number> normal_derivative_inner =
              integrator_inner.get_normal_derivative(q);
            const VectorizedArray<number> normal_derivative_outer =
              normal_derivative_inner;
            const VectorizedArray<number> solution_jump = (u_inner - u_outer);
            const VectorizedArray<number> average_normal_derivative =
              (normal_derivative_inner + normal_derivative_outer) * number(0.5);
            const VectorizedArray<number> test_by_value =
              solution_jump * sigma - average_normal_derivative;

            integrator_inner.submit_normal_derivative(-solution_jump *
                                                        number(0.5),
                                                      q);
            integrator_inner.submit_value(test_by_value, q);
          }

        integrator_inner.integrate_scatter(EvaluationFlags::values |
                                             EvaluationFlags::gradients,
                                           dst);
      }
  }

  void
  do_cell_integral_manual(
    const MatrixFree<dim, number>               &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    const internal::MatrixFreeFunctions::DoFInfo &dof_info =
      matrix_free.get_dof_info();
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

    const internal::MatrixFreeFunctions::DoFInfo::DoFAccessIndex ind =
      internal::MatrixFreeFunctions::DoFInfo::dof_access_cell;

    const std::vector<unsigned int> &dof_indices_cont =
      dof_info.dof_indices_contiguous[ind];

    internal::VectorReader<number, VectorizedArray<number>> reader;
    internal::VectorDistributorLocalToGlobal<number, VectorizedArray<number>>
      writer;
    std::bool_constant<internal::is_vectorizable<VectorType, number>::value>
      vector_selector;

    for (unsigned int cell = range.first; cell < range.second;
         cell += batch_size)
      {
        // read dof values
        const unsigned int my_batch_size =
          cell + batch_size <= range.second ? batch_size : range.second - cell;
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            const unsigned int n_active_lanes =
              matrix_free.n_active_entries_per_cell_batch(cell + batch);
            const bool use_vectorized_path = n_active_lanes == n_lanes;

            if (use_vectorized_path)
              {
                reader.process_dofs_vectorized_transpose(
                  dofs_per_cell,
                  dof_indices_cont.data() + (cell + batch) * n_lanes,
                  src,
                  &values_dofs[batch * dofs_per_cell],
                  vector_selector);
              }
            else
              {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  values_dofs[batch * dofs_per_cell + i] = {};

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  for (unsigned int v = 0; v < n_active_lanes; ++v)
                    reader.process_dof(
                      dof_indices_cont[(cell + batch) * n_lanes + v] + i,
                      src,
                      values_dofs[batch * dofs_per_cell + i][v]);
              }
          }

        // interpolate
        if (my_batch_size == 4)
          apply_matrix_vector_product_4<true, false>(
            shape_info.data[0].shape_gradients.data(),
            values_dofs,
            gradients_quad,
            dofs_per_cell,
            n_q_points * dim);
        else if (my_batch_size == 3)
          apply_matrix_vector_product_3<true, false>(
            shape_info.data[0].shape_gradients.data(),
            values_dofs,
            gradients_quad,
            dofs_per_cell,
            n_q_points * dim);
        else if (my_batch_size == 2)
          apply_matrix_vector_product_2<true, false>(
            shape_info.data[0].shape_gradients.data(),
            values_dofs,
            gradients_quad,
            dofs_per_cell,
            n_q_points * dim);
        else
          dealii::internal::apply_matrix_vector_product<
            dealii::internal::EvaluatorVariant::evaluate_general,
            dealii::internal::EvaluatorQuantity::value,
            /*transpose_matrix*/ true,
            /*add*/ false,
            /*consider_strides*/ false>(
            shape_info.data[0].shape_gradients.data(),
            values_dofs,
            gradients_quad,
            dofs_per_cell,
            n_q_points * dim,
            1,
            1);

        // quadrature point operation
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            const unsigned int offsets =
              mapping_data.data_index_offsets[cell + batch];
            const Tensor<2, dim, VectorizedArray<number>> *jac =
              mapping_data.jacobians[0].data() + offsets;
            const VectorizedArray<number> *j_value =
              &mapping_data.JxW_values[offsets];
            if (matrix_free.get_mapping_info().cell_type[cell + batch] <=
                internal::MatrixFreeFunctions::affine)
              {
                SymmetricTensor<2, dim, VectorizedArray<number>> my_metric;
                for (unsigned int d = 0; d < dim; ++d)
                  for (unsigned int f = d; f < dim; ++f)
                    {
                      VectorizedArray<number> sum = jac[0][0][d] * jac[0][0][f];
                      for (unsigned int e = 1; e < dim; ++e)
                        sum += jac[0][e][d] * jac[0][e][f];
                      my_metric[d][f] = sum * j_value[0];
                    }
                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    Tensor<1, dim, VectorizedArray<number>> grad;
                    for (unsigned int d = 0; d < dim; ++d)
                      grad[d] =
                        gradients_quad[(batch * n_q_points + q) * dim + d];
                    Tensor<1, dim, VectorizedArray<number>> result =
                      my_metric * grad;
                    const number weight = quadrature_weights[q];
                    for (unsigned int d = 0; d < dim; ++d)
                      gradients_quad[(batch * n_q_points + q) * dim + d] =
                        weight * result[d];
                  }
              }
            else
              {
                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    Tensor<1, dim, VectorizedArray<number>> grad;
                    for (unsigned int d = 0; d < dim; ++d)
                      grad[d] =
                        gradients_quad[(batch * n_q_points + q) * dim + d];
                    Tensor<1, dim, VectorizedArray<number>> result =
                      (j_value[q]) * transpose(jac[q]) * (jac[q] * grad);
                    for (unsigned int d = 0; d < dim; ++d)
                      gradients_quad[(batch * n_q_points + q) * dim + d] =
                        result[d];
                  }
              }
          }


        // integrate
        if (my_batch_size == 4)
          apply_matrix_vector_product_4<false, false>(
            shape_info.data[0].shape_gradients.data(),
            gradients_quad,
            values_dofs,
            dofs_per_cell,
            n_q_points * dim);
        else if (my_batch_size == 3)
          apply_matrix_vector_product_3<false, false>(
            shape_info.data[0].shape_gradients.data(),
            gradients_quad,
            values_dofs,
            dofs_per_cell,
            n_q_points * dim);
        else if (my_batch_size == 2)
          apply_matrix_vector_product_2<false, false>(
            shape_info.data[0].shape_gradients.data(),
            gradients_quad,
            values_dofs,
            dofs_per_cell,
            n_q_points * dim);
        else
          dealii::internal::apply_matrix_vector_product<
            dealii::internal::EvaluatorVariant::evaluate_general,
            dealii::internal::EvaluatorQuantity::value,
            /*transpose_matrix*/ false,
            /*add*/ false,
            /*consider_strides*/ false>(
            shape_info.data[0].shape_gradients.data(),
            gradients_quad,
            values_dofs,
            dofs_per_cell,
            n_q_points * dim,
            1,
            1);

        // distribute local to global
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            const unsigned int n_active_lanes =
              matrix_free.n_active_entries_per_cell_batch(cell + batch);
            const bool use_vectorized_path = n_active_lanes == n_lanes;

            if (use_vectorized_path)
              {
                writer.process_dofs_vectorized_transpose(
                  dofs_per_cell,
                  dof_indices_cont.data() + (cell + batch) * n_lanes,
                  dst,
                  &values_dofs[batch * dofs_per_cell],
                  vector_selector);
              }
            else
              {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  for (unsigned int v = 0; v < n_active_lanes; ++v)
                    writer.process_dof(
                      dof_indices_cont[(cell + batch) * n_lanes + v] + i,
                      dst,
                      values_dofs[batch * dofs_per_cell + i][v]);
              }
          }
      }

    matrix_free.release_scratch_data(scratch_data);
  }

  void
  do_face_integral_manual(
    const MatrixFree<dim, number>               &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    constexpr int                          NUM_FACES        = 4;
    constexpr int                          NUM_ORIENTATIONS = 6;
    std::vector<std::vector<unsigned int>> face_batches(
      NUM_FACES * NUM_FACES * NUM_ORIENTATIONS * NUM_ORIENTATIONS);

    for (unsigned int face = range.first; face < range.second; ++face)
      {
        const auto          face_info       = matrix_free.get_face_info(face);
        const unsigned char face_number_int = face_info.interior_face_no;
        const unsigned char face_number_ext = face_info.exterior_face_no;
        const unsigned char face_orientation_int =
          (true == (face_info.face_orientation >= 8)) ?
            (face_info.face_orientation % 8) :
            0;
        const unsigned char face_orientation_ext =
          (false == (face_info.face_orientation >= 8)) ?
            (face_info.face_orientation % 8) :
            0;

        face_batches[face_number_int * NUM_FACES * NUM_ORIENTATIONS *
                       NUM_ORIENTATIONS +
                     face_number_ext * NUM_ORIENTATIONS * NUM_ORIENTATIONS +
                     face_orientation_int * NUM_ORIENTATIONS +
                     face_orientation_ext]
          .push_back(face);
      }

    for (auto &face_batch : face_batches)
      if (!face_batch.empty())
        compute_batched_face_integrals(matrix_free, dst, src, face_batch);
  }

  void
  compute_batched_face_integrals(
    const MatrixFree<dim, number>   &matrix_free,
    VectorType                      &dst,
    const VectorType                &src,
    const std::vector<unsigned int> &face_indices) const
  {
    const internal::MatrixFreeFunctions::DoFInfo &dof_info =
      matrix_free.get_dof_info();
    AlignedVector<VectorizedArray<number>> *scratch_data =
      matrix_free.acquire_scratch_data();
    const internal::MatrixFreeFunctions::ShapeInfo<number> &shape_info =
      matrix_free.get_shape_info();
    const unsigned int dofs_per_cell  = shape_info.dofs_per_component_on_cell;
    constexpr unsigned int batch_size = 4;

    constexpr unsigned int n_lanes = VectorizedArray<number>::size();

    const auto   &mapping_data = matrix_free.get_mapping_info().face_data[0];
    const number *quadrature_weights =
      mapping_data.descriptor[0].quadrature_weights.data();

    const auto          face_info = matrix_free.get_face_info(face_indices[0]);
    const unsigned char face_number_int = face_info.interior_face_no;
    const unsigned char face_number_ext = face_info.exterior_face_no;
    const unsigned char face_orientation_int =
      (true == (face_info.face_orientation >= 8)) ?
        (face_info.face_orientation % 8) :
        0;
    const unsigned char face_orientation_ext =
      (false == (face_info.face_orientation >= 8)) ?
        (face_info.face_orientation % 8) :
        0;

    const unsigned int n_q_points =
      shape_info.n_q_points_faces[face_number_int];

    scratch_data->resize_fast(2 * batch_size *
                              (dim * n_q_points + n_q_points + dofs_per_cell));
    VectorizedArray<number> *values_dofs_int = scratch_data->begin();
    VectorizedArray<number> *values_dofs_ext =
      scratch_data->begin() + batch_size * dofs_per_cell;

    VectorizedArray<number> *values_quad_int =
      scratch_data->begin() + 2 * batch_size * dofs_per_cell;
    VectorizedArray<number> *values_quad_ext = scratch_data->begin() +
                                               2 * batch_size * dofs_per_cell +
                                               batch_size * n_q_points;

    VectorizedArray<number> *gradients_quad_int =
      scratch_data->begin() + 2 * batch_size * dofs_per_cell +
      2 * batch_size * n_q_points;
    VectorizedArray<number> *gradients_quad_ext =
      scratch_data->begin() + 2 * batch_size * dofs_per_cell +
      2 * batch_size * n_q_points + batch_size * n_q_points * dim;

    const unsigned int n_faces = face_indices.size();

    const auto       &shape_data = shape_info.data.front();
    const auto *const shape_values_inner =
      &shape_data.shape_values_face(face_number_int, face_orientation_int, 0);
    const auto *const shape_values_outer =
      &shape_data.shape_values_face(face_number_ext, face_orientation_ext, 0);
    const auto *const shape_gradients_inner =
      &shape_data.shape_gradients_face(face_number_int,
                                       face_orientation_int,
                                       0);
    const auto *const shape_gradients_outer =
      &shape_data.shape_gradients_face(face_number_ext,
                                       face_orientation_ext,
                                       0);

    const internal::MatrixFreeFunctions::DoFInfo::DoFAccessIndex ind_int =
      internal::MatrixFreeFunctions::DoFInfo::dof_access_face_interior;
    const internal::MatrixFreeFunctions::DoFInfo::DoFAccessIndex ind_ext =
      internal::MatrixFreeFunctions::DoFInfo::dof_access_face_exterior;

    const std::vector<unsigned int> &dof_indices_cont_int =
      dof_info.dof_indices_contiguous[ind_int];
    const std::vector<unsigned int> &dof_indices_cont_ext =
      dof_info.dof_indices_contiguous[ind_ext];
    internal::VectorReader<number, VectorizedArray<number>> reader;
    internal::VectorDistributorLocalToGlobal<number, VectorizedArray<number>>
      writer;
    std::bool_constant<internal::is_vectorizable<VectorType, number>::value>
      vector_selector;

    for (unsigned int face = 0; face < n_faces; face += batch_size)
      {
        // read dof values
        const unsigned int my_batch_size =
          face + batch_size <= n_faces ? batch_size : n_faces - face;
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            const unsigned int n_active_lanes =
              matrix_free.n_active_entries_per_face_batch(
                face_indices[face + batch]);

            const bool use_vectorized_path = n_active_lanes == n_lanes;

            if (use_vectorized_path)
              {
                reader.process_dofs_vectorized_transpose(
                  dofs_per_cell,
                  dof_indices_cont_int.data() +
                    face_indices[face + batch] * n_lanes,
                  src,
                  &values_dofs_int[batch * dofs_per_cell],
                  vector_selector);
                reader.process_dofs_vectorized_transpose(
                  dofs_per_cell,
                  dof_indices_cont_ext.data() +
                    face_indices[face + batch] * n_lanes,
                  src,
                  &values_dofs_ext[batch * dofs_per_cell],
                  vector_selector);
              }
            else
              {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    values_dofs_int[batch * dofs_per_cell + i] = {};
                    values_dofs_ext[batch * dofs_per_cell + i] = {};
                  }

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  for (unsigned int v = 0; v < n_active_lanes; ++v)
                    reader.process_dof(
                      dof_indices_cont_int[face_indices[face + batch] *
                                             n_lanes +
                                           v] +
                        i,
                      src,
                      values_dofs_int[batch * dofs_per_cell + i][v]);

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  for (unsigned int v = 0; v < n_active_lanes; ++v)
                    reader.process_dof(
                      dof_indices_cont_ext[face_indices[face + batch] *
                                             n_lanes +
                                           v] +
                        i,
                      src,
                      values_dofs_ext[batch * dofs_per_cell + i][v]);
              }
          }

        if (my_batch_size == 4)
          {
            // interpolate
            apply_matrix_vector_product_4<true, false>(shape_values_inner,
                                                       values_dofs_int,
                                                       values_quad_int,
                                                       dofs_per_cell,
                                                       n_q_points);

            apply_matrix_vector_product_4<true, false>(shape_gradients_inner,
                                                       values_dofs_int,
                                                       gradients_quad_int,
                                                       dofs_per_cell,
                                                       n_q_points * dim);

            apply_matrix_vector_product_4<true, false>(shape_values_outer,
                                                       values_dofs_ext,
                                                       values_quad_ext,
                                                       dofs_per_cell,
                                                       n_q_points);

            apply_matrix_vector_product_4<true, false>(shape_gradients_outer,
                                                       values_dofs_ext,
                                                       gradients_quad_ext,
                                                       dofs_per_cell,
                                                       n_q_points * dim);
          }
        else if (my_batch_size == 3)
          {
            // interpolate
            apply_matrix_vector_product_3<true, false>(shape_values_inner,
                                                       values_dofs_int,
                                                       values_quad_int,
                                                       dofs_per_cell,
                                                       n_q_points);

            apply_matrix_vector_product_3<true, false>(shape_gradients_inner,
                                                       values_dofs_int,
                                                       gradients_quad_int,
                                                       dofs_per_cell,
                                                       n_q_points * dim);

            apply_matrix_vector_product_3<true, false>(shape_values_outer,
                                                       values_dofs_ext,
                                                       values_quad_ext,
                                                       dofs_per_cell,
                                                       n_q_points);

            apply_matrix_vector_product_3<true, false>(shape_gradients_outer,
                                                       values_dofs_ext,
                                                       gradients_quad_ext,
                                                       dofs_per_cell,
                                                       n_q_points * dim);
          }
        else if (my_batch_size == 2)
          {
            // interpolate
            apply_matrix_vector_product_2<true, false>(shape_values_inner,
                                                       values_dofs_int,
                                                       values_quad_int,
                                                       dofs_per_cell,
                                                       n_q_points);

            apply_matrix_vector_product_2<true, false>(shape_gradients_inner,
                                                       values_dofs_int,
                                                       gradients_quad_int,
                                                       dofs_per_cell,
                                                       n_q_points * dim);

            apply_matrix_vector_product_2<true, false>(shape_values_outer,
                                                       values_dofs_ext,
                                                       values_quad_ext,
                                                       dofs_per_cell,
                                                       n_q_points);

            apply_matrix_vector_product_2<true, false>(shape_gradients_outer,
                                                       values_dofs_ext,
                                                       gradients_quad_ext,
                                                       dofs_per_cell,
                                                       n_q_points * dim);
          }
        else
          {
            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_general,
              dealii::internal::EvaluatorQuantity::value,
              /*transpose_matrix*/ true,
              /*add*/ false,
              /*consider_strides*/ false>(shape_values_inner,
                                          values_dofs_int,
                                          values_quad_int,
                                          dofs_per_cell,
                                          n_q_points,
                                          1,
                                          1);

            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_general,
              dealii::internal::EvaluatorQuantity::value,
              /*transpose_matrix*/ true,
              /*add*/ false,
              /*consider_strides*/ false>(shape_gradients_inner,
                                          values_dofs_int,
                                          gradients_quad_int,
                                          dofs_per_cell,
                                          n_q_points * dim,
                                          1,
                                          1);

            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_general,
              dealii::internal::EvaluatorQuantity::value,
              /*transpose_matrix*/ true,
              /*add*/ false,
              /*consider_strides*/ false>(shape_values_outer,
                                          values_dofs_ext,
                                          values_quad_ext,
                                          dofs_per_cell,
                                          n_q_points,
                                          1,
                                          1);

            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_general,
              dealii::internal::EvaluatorQuantity::value,
              /*transpose_matrix*/ true,
              /*add*/ false,
              /*consider_strides*/ false>(shape_gradients_outer,
                                          values_dofs_ext,
                                          gradients_quad_ext,
                                          dofs_per_cell,
                                          n_q_points * dim,
                                          1,
                                          1);
          }

        // quadrature point operation
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            const unsigned int offsets =
              mapping_data.data_index_offsets[face_indices[face + batch]];

            const VectorizedArray<number> *j_value =
              &mapping_data.JxW_values[offsets];

            const Tensor<1, dim, VectorizedArray<number>>
              *normal_x_jacobian_int =
                &mapping_data.normals_times_jacobians[0][offsets];
            const Tensor<1, dim, VectorizedArray<number>>
              *normal_x_jacobian_ext =
                &mapping_data.normals_times_jacobians[1][offsets];

            dealii::VectorizedArray<Number> tau_int;
            dealii::VectorizedArray<Number> tau_ext;
            const auto face_cell = matrix_free.get_face_info(face_indices[face + batch]);
            const std::array<unsigned int, n_lanes> cell_ids_interior = face_cell.cells_interior;
            const std::array<unsigned int, n_lanes> cell_ids_exterior = face_cell.cells_exterior;

            
            for (unsigned int i = 0; i < n_lanes; ++i)
            {
                if (cell_ids_interior[i] != dealii::numbers::invalid_unsigned_int)
                  tau_int[i] = array_penalty_parameter[cell_ids_interior[i] / n_lanes][cell_ids_interior[i] % n_lanes];
                if (cell_ids_exterior[i] != dealii::numbers::invalid_unsigned_int)
                  tau_ext[i] = array_penalty_parameter[cell_ids_exterior[i] / n_lanes][cell_ids_exterior[i] % n_lanes];
            }        

            const dealii::VectorizedArray<Number> sigma =
              get_penalty_factor()  * std::max(tau_int,
                  tau_ext);


            if (matrix_free.get_mapping_info()
                  .face_type[face_indices[face + batch]] <=
                internal::MatrixFreeFunctions::affine)
              {
                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    const VectorizedArray<number> solution_jump =
                      values_quad_int[batch * n_q_points + q] -
                      values_quad_ext[batch * n_q_points + q];

                    VectorizedArray<number> grad_int =
                      gradients_quad_int[batch * n_q_points * dim + q * dim] *
                      normal_x_jacobian_int[0][0];
                    VectorizedArray<number> grad_ext =
                      gradients_quad_ext[batch * n_q_points * dim + q * dim] *
                      normal_x_jacobian_ext[0][0];
                    for (unsigned int d = 1; d < dim; ++d)
                      {
                        grad_int +=
                          gradients_quad_int[batch * n_q_points * dim +
                                             q * dim + d] *
                          normal_x_jacobian_int[0][d];
                        grad_ext +=
                          gradients_quad_ext[batch * n_q_points * dim +
                                             q * dim + d] *
                          normal_x_jacobian_ext[0][d];
                      }
                    const VectorizedArray<number> averaged_normal_derivative =
                      number(0.5) * (grad_int + grad_ext);

                    const VectorizedArray<number> test_by_value =
                      solution_jump * sigma - averaged_normal_derivative;

                    values_quad_int[batch * n_q_points + q] =
                      test_by_value * j_value[0] * quadrature_weights[q];
                    values_quad_ext[batch * n_q_points + q] =
                      -test_by_value * j_value[0] * quadrature_weights[q];

                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        gradients_quad_int[batch * n_q_points * dim + q * dim +
                                           d] =
                          (-solution_jump * number(0.5) * j_value[0] *
                           quadrature_weights[q]) *
                          normal_x_jacobian_int[0][d];
                        gradients_quad_ext[batch * n_q_points * dim + q * dim +
                                           d] =
                          (-solution_jump * number(0.5) * j_value[0] *
                           quadrature_weights[q]) *
                          normal_x_jacobian_ext[0][d];
                      }
                  }
              }
            else
              {
                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    const VectorizedArray<number> solution_jump =
                      values_quad_int[batch * n_q_points + q] -
                      values_quad_ext[batch * n_q_points + q];

                    VectorizedArray<number> grad_int =
                      gradients_quad_int[batch * n_q_points * dim + q * dim] *
                      normal_x_jacobian_int[q][0];
                    VectorizedArray<number> grad_ext =
                      gradients_quad_ext[batch * n_q_points * dim + q * dim] *
                      normal_x_jacobian_ext[q][0];
                    for (unsigned int d = 1; d < dim; ++d)
                      {
                        grad_int +=
                          gradients_quad_int[batch * n_q_points * dim +
                                             q * dim + d] *
                          normal_x_jacobian_int[q][d];
                        grad_ext +=
                          gradients_quad_ext[batch * n_q_points * dim +
                                             q * dim + d] *
                          normal_x_jacobian_ext[q][d];
                      }
                    const VectorizedArray<number> averaged_normal_derivative =
                      number(0.5) * (grad_int + grad_ext);

                    const VectorizedArray<number> test_by_value =
                      solution_jump * sigma - averaged_normal_derivative;

                    values_quad_int[batch * n_q_points + q] =
                      test_by_value * j_value[q];
                    values_quad_ext[batch * n_q_points + q] =
                      -test_by_value * j_value[q];

                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        gradients_quad_int[batch * n_q_points * dim + q * dim +
                                           d] =
                          (-solution_jump * number(0.5) * j_value[q]) *
                          normal_x_jacobian_int[q][d];
                        gradients_quad_ext[batch * n_q_points * dim + q * dim +
                                           d] =
                          (-solution_jump * number(0.5) * j_value[q]) *
                          normal_x_jacobian_ext[q][d];
                      }
                  }
              }
          }

        if (my_batch_size == 4)
          {
            // integrate
            apply_matrix_vector_product_4<false, false>(shape_values_inner,
                                                        values_quad_int,
                                                        values_dofs_int,
                                                        dofs_per_cell,
                                                        n_q_points);

            apply_matrix_vector_product_4<false, true>(shape_gradients_inner,
                                                       gradients_quad_int,
                                                       values_dofs_int,
                                                       dofs_per_cell,
                                                       n_q_points * dim);

            apply_matrix_vector_product_4<false, false>(shape_values_outer,
                                                        values_quad_ext,
                                                        values_dofs_ext,
                                                        dofs_per_cell,
                                                        n_q_points);

            apply_matrix_vector_product_4<false, true>(shape_gradients_outer,
                                                       gradients_quad_ext,
                                                       values_dofs_ext,
                                                       dofs_per_cell,
                                                       n_q_points * dim);
          }
        else if (my_batch_size == 3)
          {
            // integrate
            apply_matrix_vector_product_3<false, false>(shape_values_inner,
                                                        values_quad_int,
                                                        values_dofs_int,
                                                        dofs_per_cell,
                                                        n_q_points);

            apply_matrix_vector_product_3<false, true>(shape_gradients_inner,
                                                       gradients_quad_int,
                                                       values_dofs_int,
                                                       dofs_per_cell,
                                                       n_q_points * dim);

            apply_matrix_vector_product_3<false, false>(shape_values_outer,
                                                        values_quad_ext,
                                                        values_dofs_ext,
                                                        dofs_per_cell,
                                                        n_q_points);

            apply_matrix_vector_product_3<false, true>(shape_gradients_outer,
                                                       gradients_quad_ext,
                                                       values_dofs_ext,
                                                       dofs_per_cell,
                                                       n_q_points * dim);
          }
        else if (my_batch_size == 2)
          {
            apply_matrix_vector_product_2<false, false>(shape_values_inner,
                                                        values_quad_int,
                                                        values_dofs_int,
                                                        dofs_per_cell,
                                                        n_q_points);

            apply_matrix_vector_product_2<false, true>(shape_gradients_inner,
                                                       gradients_quad_int,
                                                       values_dofs_int,
                                                       dofs_per_cell,
                                                       n_q_points * dim);

            apply_matrix_vector_product_2<false, false>(shape_values_outer,
                                                        values_quad_ext,
                                                        values_dofs_ext,
                                                        dofs_per_cell,
                                                        n_q_points);

            apply_matrix_vector_product_2<false, true>(shape_gradients_outer,
                                                       gradients_quad_ext,
                                                       values_dofs_ext,
                                                       dofs_per_cell,
                                                       n_q_points * dim);
          }
        else
          {
            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_general,
              dealii::internal::EvaluatorQuantity::value,
              /*transpose_matrix*/ false,
              /*add*/ false,
              /*consider_strides*/ false>(shape_values_inner,
                                          values_quad_int,
                                          values_dofs_int,
                                          dofs_per_cell,
                                          n_q_points,
                                          1,
                                          1);

            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_general,
              dealii::internal::EvaluatorQuantity::value,
              /*transpose_matrix*/ false,
              /*add*/ true,
              /*consider_strides*/ false>(shape_gradients_inner,
                                          gradients_quad_int,
                                          values_dofs_int,
                                          dofs_per_cell,
                                          n_q_points * dim,
                                          1,
                                          1);

            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_general,
              dealii::internal::EvaluatorQuantity::value,
              /*transpose_matrix*/ false,
              /*add*/ false,
              /*consider_strides*/ false>(shape_values_outer,
                                          values_quad_ext,
                                          values_dofs_ext,
                                          dofs_per_cell,
                                          n_q_points,
                                          1,
                                          1);

            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_general,
              dealii::internal::EvaluatorQuantity::value,
              /*transpose_matrix*/ false,
              /*add*/ true,
              /*consider_strides*/ false>(shape_gradients_outer,
                                          gradients_quad_ext,
                                          values_dofs_ext,
                                          dofs_per_cell,
                                          n_q_points * dim,
                                          1,
                                          1);
          }


        // distribute local to global
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            const unsigned int n_active_lanes =
              matrix_free.n_active_entries_per_face_batch(
                face_indices[face + batch]);
            const bool use_vectorized_path = n_active_lanes == n_lanes;

            if (use_vectorized_path)
              {
                writer.process_dofs_vectorized_transpose(
                  dofs_per_cell,
                  dof_indices_cont_int.data() +
                    face_indices[face + batch] * n_lanes,
                  dst,
                  &values_dofs_int[batch * dofs_per_cell],
                  vector_selector);
                writer.process_dofs_vectorized_transpose(
                  dofs_per_cell,
                  dof_indices_cont_ext.data() +
                    face_indices[face + batch] * n_lanes,
                  dst,
                  &values_dofs_ext[batch * dofs_per_cell],
                  vector_selector);
              }
            else
              {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  for (unsigned int v = 0; v < n_active_lanes; ++v)
                    writer.process_dof(
                      dof_indices_cont_int[face_indices[face + batch] *
                                             n_lanes +
                                           v] +
                        i,
                      dst,
                      values_dofs_int[batch * dofs_per_cell + i][v]);

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  for (unsigned int v = 0; v < n_active_lanes; ++v)
                    writer.process_dof(
                      dof_indices_cont_ext[face_indices[face + batch] *
                                             n_lanes +
                                           v] +
                        i,
                      dst,
                      values_dofs_ext[batch * dofs_per_cell + i][v]);
              }
          }
      }

    matrix_free.release_scratch_data(scratch_data);
  }

  void
  do_boundary_integral_manual(
    const MatrixFree<dim, number>               &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    constexpr int                          NUM_FACES        = 4;
    constexpr int                          NUM_ORIENTATIONS = 6;
    std::vector<std::vector<unsigned int>> face_batches(
      NUM_FACES * NUM_ORIENTATIONS);

    for (unsigned int face = range.first; face < range.second; ++face)
      {
        const auto          face_info       = matrix_free.get_face_info(face);
        const unsigned char face_number_int = face_info.interior_face_no;
        const unsigned char face_orientation_int =
          (true == (face_info.face_orientation >= 8)) ?
            (face_info.face_orientation % 8) :
            0;
        face_batches[face_number_int * NUM_ORIENTATIONS + face_orientation_int].push_back(face);
      }

    for (auto &face_batch : face_batches)
      if (!face_batch.empty())
        compute_batched_boundary_integrals(matrix_free, dst, src, face_batch);
  }

  void
  compute_batched_boundary_integrals(
    const MatrixFree<dim, number>   &matrix_free,
    VectorType                      &dst,
    const VectorType                &src,
    const std::vector<unsigned int> &face_indices) const
  {
    const internal::MatrixFreeFunctions::DoFInfo &dof_info =
      matrix_free.get_dof_info();
    AlignedVector<VectorizedArray<number>> *scratch_data =
      matrix_free.acquire_scratch_data();
    const internal::MatrixFreeFunctions::ShapeInfo<number> &shape_info =
      matrix_free.get_shape_info();
    const unsigned int dofs_per_cell  = shape_info.dofs_per_component_on_cell;
    constexpr unsigned int batch_size = 4;

    constexpr unsigned int n_lanes = VectorizedArray<number>::size();

    const auto   &mapping_data = matrix_free.get_mapping_info().face_data[0];
    const number *quadrature_weights =
      mapping_data.descriptor[0].quadrature_weights.data();

    const auto          face_info = matrix_free.get_face_info(face_indices[0]);
    const unsigned char face_number_int = face_info.interior_face_no;
    const unsigned char face_orientation_int =
      (true == (face_info.face_orientation >= 8)) ?
        (face_info.face_orientation % 8) :
        0;

    const unsigned int n_q_points =
      shape_info.n_q_points_faces[face_number_int];

    scratch_data->resize_fast(batch_size *
                              (dim * n_q_points + n_q_points + dofs_per_cell));
    VectorizedArray<number> *values_dofs_int = scratch_data->begin();

    VectorizedArray<number> *values_quad_int =
      scratch_data->begin() + batch_size * dofs_per_cell;

    VectorizedArray<number> *gradients_quad_int =
      scratch_data->begin() + batch_size * dofs_per_cell +
      batch_size * n_q_points;

    const unsigned int n_faces = face_indices.size();

    const auto       &shape_data = shape_info.data.front();
    const auto *const shape_values_inner =
      &shape_data.shape_values_face(face_number_int, face_orientation_int, 0);
    const auto *const shape_gradients_inner =
      &shape_data.shape_gradients_face(face_number_int,
                                       face_orientation_int,
                                       0);

    const internal::MatrixFreeFunctions::DoFInfo::DoFAccessIndex ind_int =
      internal::MatrixFreeFunctions::DoFInfo::dof_access_face_interior;
    
    const std::vector<unsigned int> &dof_indices_cont_int =
      dof_info.dof_indices_contiguous[ind_int];

    internal::VectorReader<number, VectorizedArray<number>> reader;
    internal::VectorDistributorLocalToGlobal<number, VectorizedArray<number>>
      writer;
    std::bool_constant<internal::is_vectorizable<VectorType, number>::value>
      vector_selector;

    for (unsigned int face = 0; face < n_faces; face += batch_size)
      {
        // read dof values
        const unsigned int my_batch_size =
          face + batch_size <= n_faces ? batch_size : n_faces - face;
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            const unsigned int n_active_lanes =
              matrix_free.n_active_entries_per_face_batch(
                face_indices[face + batch]);

            const bool use_vectorized_path = n_active_lanes == n_lanes;

            if (use_vectorized_path)
              {
                reader.process_dofs_vectorized_transpose(
                  dofs_per_cell,
                  dof_indices_cont_int.data() +
                    face_indices[face + batch] * n_lanes,
                  src,
                  &values_dofs_int[batch * dofs_per_cell],
                  vector_selector);
              }
            else
              {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    values_dofs_int[batch * dofs_per_cell + i] = {};
                  }

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  for (unsigned int v = 0; v < n_active_lanes; ++v)
                    reader.process_dof(
                      dof_indices_cont_int[face_indices[face + batch] *
                                             n_lanes +
                                           v] +
                        i,
                      src,
                      values_dofs_int[batch * dofs_per_cell + i][v]);
              }
          }

        if (my_batch_size == 4)
          {
            // interpolate
            apply_matrix_vector_product_4<true, false>(shape_values_inner,
                                                       values_dofs_int,
                                                       values_quad_int,
                                                       dofs_per_cell,
                                                       n_q_points);

            apply_matrix_vector_product_4<true, false>(shape_gradients_inner,
                                                       values_dofs_int,
                                                       gradients_quad_int,
                                                       dofs_per_cell,
                                                       n_q_points * dim);
          }
        else if (my_batch_size == 3)
          {
            // interpolate
            apply_matrix_vector_product_3<true, false>(shape_values_inner,
                                                       values_dofs_int,
                                                       values_quad_int,
                                                       dofs_per_cell,
                                                       n_q_points);

            apply_matrix_vector_product_3<true, false>(shape_gradients_inner,
                                                       values_dofs_int,
                                                       gradients_quad_int,
                                                       dofs_per_cell,
                                                       n_q_points * dim);
          }
        else if (my_batch_size == 2)
          {
            // interpolate
            apply_matrix_vector_product_2<true, false>(shape_values_inner,
                                                       values_dofs_int,
                                                       values_quad_int,
                                                       dofs_per_cell,
                                                       n_q_points);

            apply_matrix_vector_product_2<true, false>(shape_gradients_inner,
                                                       values_dofs_int,
                                                       gradients_quad_int,
                                                       dofs_per_cell,
                                                       n_q_points * dim);
          }
        else
          {
            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_general,
              dealii::internal::EvaluatorQuantity::value,
              /*transpose_matrix*/ true,
              /*add*/ false,
              /*consider_strides*/ false>(shape_values_inner,
                                          values_dofs_int,
                                          values_quad_int,
                                          dofs_per_cell,
                                          n_q_points,
                                          1,
                                          1);

            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_general,
              dealii::internal::EvaluatorQuantity::value,
              /*transpose_matrix*/ true,
              /*add*/ false,
              /*consider_strides*/ false>(shape_gradients_inner,
                                          values_dofs_int,
                                          gradients_quad_int,
                                          dofs_per_cell,
                                          n_q_points * dim,
                                          1,
                                          1);
          }

        // quadrature point operation
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            const unsigned int offsets =
              mapping_data.data_index_offsets[face_indices[face + batch]];

            const VectorizedArray<number> *j_value =
              &mapping_data.JxW_values[offsets];

            const Tensor<1, dim, VectorizedArray<number>>
              *normal_x_jacobian_int =
                &mapping_data.normals_times_jacobians[0][offsets];

            dealii::VectorizedArray<Number> tau_int;
            const auto face_cell = matrix_free.get_face_info(face_indices[face + batch]);
            const std::array<unsigned int, n_lanes> cell_ids_interior = face_cell.cells_interior;
            
            for (unsigned int i = 0; i < n_lanes; ++i)
            {
                if (cell_ids_interior[i] != dealii::numbers::invalid_unsigned_int)
                  tau_int[i] = array_penalty_parameter[cell_ids_interior[i] / n_lanes][cell_ids_interior[i] % n_lanes];
            }        

            const dealii::VectorizedArray<Number> sigma =
              get_penalty_factor()  * tau_int;

            if (matrix_free.get_mapping_info()
                  .face_type[face_indices[face + batch]] <=
                internal::MatrixFreeFunctions::affine)
              {
                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    VectorizedArray<number> grad_int =
                      gradients_quad_int[batch * n_q_points * dim + q * dim] *
                      normal_x_jacobian_int[0][0];
                    for (unsigned int d = 1; d < dim; ++d)
                      {
                        grad_int +=
                          gradients_quad_int[batch * n_q_points * dim +
                                             q * dim + d] *
                          normal_x_jacobian_int[0][d];
                      }

                    const VectorizedArray<number> test_by_value =
                      number(2.0) * values_quad_int[batch * n_q_points + q] * sigma - grad_int;

                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        gradients_quad_int[batch * n_q_points * dim + q * dim +
                                           d] =
                          (-values_quad_int[batch * n_q_points + q] * j_value[0] *
                           quadrature_weights[q]) *
                          normal_x_jacobian_int[0][d];
                      }
                    
                    values_quad_int[batch * n_q_points + q] =
                      test_by_value * j_value[0] * quadrature_weights[q];
                  }
              }
            else
              {
                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    VectorizedArray<number> grad_int =
                      gradients_quad_int[batch * n_q_points * dim + q * dim] *
                      normal_x_jacobian_int[q][0];

                    for (unsigned int d = 1; d < dim; ++d)
                      {
                        grad_int +=
                          gradients_quad_int[batch * n_q_points * dim +
                                             q * dim + d] *
                          normal_x_jacobian_int[q][d];
                      }

                    const VectorizedArray<number> test_by_value =
                      number(2.0) * values_quad_int[batch * n_q_points + q] * sigma - grad_int;

                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        gradients_quad_int[batch * n_q_points * dim + q * dim +
                                           d] =
                          (-values_quad_int[batch * n_q_points + q] * j_value[q]) *
                          normal_x_jacobian_int[q][d];
                      }
                    
                    values_quad_int[batch * n_q_points + q] =
                      test_by_value * j_value[q];
                  }
              }
          }

        if (my_batch_size == 4)
          {
            // integrate
            apply_matrix_vector_product_4<false, false>(shape_values_inner,
                                                        values_quad_int,
                                                        values_dofs_int,
                                                        dofs_per_cell,
                                                        n_q_points);

            apply_matrix_vector_product_4<false, true>(shape_gradients_inner,
                                                       gradients_quad_int,
                                                       values_dofs_int,
                                                       dofs_per_cell,
                                                       n_q_points * dim);
          }
        else if (my_batch_size == 3)
          {
            // integrate
            apply_matrix_vector_product_3<false, false>(shape_values_inner,
                                                        values_quad_int,
                                                        values_dofs_int,
                                                        dofs_per_cell,
                                                        n_q_points);

            apply_matrix_vector_product_3<false, true>(shape_gradients_inner,
                                                       gradients_quad_int,
                                                       values_dofs_int,
                                                       dofs_per_cell,
                                                       n_q_points * dim);
          }
        else if (my_batch_size == 2)
          {
            apply_matrix_vector_product_2<false, false>(shape_values_inner,
                                                        values_quad_int,
                                                        values_dofs_int,
                                                        dofs_per_cell,
                                                        n_q_points);

            apply_matrix_vector_product_2<false, true>(shape_gradients_inner,
                                                       gradients_quad_int,
                                                       values_dofs_int,
                                                       dofs_per_cell,
                                                       n_q_points * dim);
          }
        else
          {
            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_general,
              dealii::internal::EvaluatorQuantity::value,
              /*transpose_matrix*/ false,
              /*add*/ false,
              /*consider_strides*/ false>(shape_values_inner,
                                          values_quad_int,
                                          values_dofs_int,
                                          dofs_per_cell,
                                          n_q_points,
                                          1,
                                          1);

            dealii::internal::apply_matrix_vector_product<
              dealii::internal::EvaluatorVariant::evaluate_general,
              dealii::internal::EvaluatorQuantity::value,
              /*transpose_matrix*/ false,
              /*add*/ true,
              /*consider_strides*/ false>(shape_gradients_inner,
                                          gradients_quad_int,
                                          values_dofs_int,
                                          dofs_per_cell,
                                          n_q_points * dim,
                                          1,
                                          1);
          }


        // distribute local to global
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            const unsigned int n_active_lanes =
              matrix_free.n_active_entries_per_face_batch(
                face_indices[face + batch]);
            const bool use_vectorized_path = n_active_lanes == n_lanes;

            if (use_vectorized_path)
              {
                writer.process_dofs_vectorized_transpose(
                  dofs_per_cell,
                  dof_indices_cont_int.data() +
                    face_indices[face + batch] * n_lanes,
                  dst,
                  &values_dofs_int[batch * dofs_per_cell],
                  vector_selector);
              }
            else
              {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  for (unsigned int v = 0; v < n_active_lanes; ++v)
                    writer.process_dof(
                      dof_indices_cont_int[face_indices[face + batch] *
                                             n_lanes +
                                           v] +
                        i,
                      dst,
                      values_dofs_int[batch * dofs_per_cell + i][v]);
              }
          }
      }

    matrix_free.release_scratch_data(scratch_data);
  }

  MatrixFree<dim, number> matrix_free;

  AffineConstraints<number> constraints;

  std::vector<unsigned int> constrained_indices;

  Number penalty_factor;

  dealii::AlignedVector<dealii::VectorizedArray<Number>> array_penalty_parameter;
};



template <int dim, typename Number>
void
do_test(const unsigned int fe_degree)
{
  const bool use_manifold = false;
  const bool grid_in      = true;
  const bool reorder_grid = false;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  pcout << "Running in " << dim << "D with degree " << fe_degree << std::endl;
  FE_SimplexDGP<dim> fe(fe_degree);
  MappingFE<dim>     mapping(FE_SimplexDGP<dim>(use_manifold ? 2 : 1));
  QGaussSimplex<dim> quad(fe_degree + 1);

  AffineConstraints<double> constraint;

  // compute possibilites of number of cells
  std::vector<unsigned int> refinements;
  std::vector<unsigned int> n_subdivisions;
  std::vector<unsigned int> n_2(1, 40);
  std::vector<unsigned int> n_3(1, 135);
  std::vector<unsigned int> n_5(1, 625);
  for (unsigned int i = 0; i < 10; ++i)
    {
      n_2.emplace_back(n_2[i] * 8);
      n_3.emplace_back(n_3[i] * 8);
      n_5.emplace_back(n_5[i] * 8);
    }

  unsigned int n_current = 0;
  unsigned int n_dofs = 0;
  for (unsigned int k = 0;  n_dofs < 100000000; ++k) //k < n_2.size() * 3 &&
    {
      unsigned int index_2 = n_2.size() - 1;
      unsigned int index_3 = n_3.size() - 1;
      unsigned int index_5 = n_5.size() - 1;

      // get the next bigger n from n_2, n_3 and n_5
      for (unsigned int i = 1; i < n_2.size() + 1; ++i)
        {
          if (n_2[n_2.size() - i] > n_current)
            index_2 = n_2.size() - i;
          if (n_3[n_3.size() - i] > n_current)
            index_3 = n_3.size() - i;
          if (n_5[n_5.size() - i] > n_current)
            index_5 = n_5.size() - i;
        }
      // check which is the next biggest element
      if (n_2[index_2] < n_3[index_3] && n_2[index_2] < n_5[index_5])
        {
          refinements.emplace_back(index_2);
          n_subdivisions.emplace_back(2);
          n_current = n_2[index_2];
        }
      else if (n_3[index_3] < n_2[index_2] && n_3[index_3] < n_5[index_5])
        {
          refinements.emplace_back(index_3);
          n_subdivisions.emplace_back(3);
          n_current = n_3[index_3];
        }
      else
        {
          refinements.emplace_back(index_5);
          n_subdivisions.emplace_back(5);
          n_current = n_5[index_5];
        }
      if (fe_degree == 1)
        n_dofs = n_current * 4;
      else if (fe_degree == 2)
        n_dofs = n_current * 10;
      else
        n_dofs = n_current * 20;
    }

  Point<dim> center;
  for (unsigned int d = 0; d < dim; ++d)
    {
      if (d == 0)
        center[d] = 5.0;
      else
        center[d] = 0.0;
    }
  const SphericalManifold<dim> manifold(center);

  unsigned int n_refinements;
  if (fe_degree == 3)
    n_refinements = 18;
  else if (fe_degree == 2)
    n_refinements = 20;
  else
    n_refinements = 21;

  if (grid_in){
    n_refinements = 1;
    refinements.resize(1);
}

  for (unsigned int refinement = 0;  refinement < refinements.size(); ++refinement) //refinement < n_refinements &&
    {
      const auto serial_grid_generator =
        [&refinement,
         &refinements,
         &n_subdivisions,
         &manifold,
         use_manifold,
         grid_in,
         reorder_grid](dealii::Triangulation<dim, dim> &tria_serial) {
          if (grid_in)
            {
              if (reorder_grid)
                {
                  dealii::Triangulation<dim, dim> tria_in;
                  dealii::GridIn<dim>             grid_in;
                  grid_in.attach_triangulation(tria_in);
                  std::ifstream input_file("lung.vtk");
                  grid_in.read_vtk(input_file);


                  dealii::DynamicSparsityPattern cell_connectivity;
                  dealii::GridTools::get_vertex_connectivity_of_cells(
                    tria_in, cell_connectivity);
                  std::vector<long unsigned int> cell_numbering;
                  cell_numbering.resize(tria_in.n_cells(0));
                  SparsityTools::reorder_hierarchical(cell_connectivity,
                                                      cell_numbering);
                  std::vector<long unsigned int> cell_numbering_inverse =
                    Utilities::invert_permutation(cell_numbering);

                  std::vector<Point<dim>>    vertices(tria_in.n_vertices());
                  std::vector<CellData<dim>> cells(tria_in.n_cells(0));

                  for (const auto &cell : tria_in.active_cell_iterators())
                    for (const unsigned int v : cell->vertex_indices())
                      vertices[cell->vertex_index(v)] = cell->vertex(v);


                  for (unsigned int idx = 0; idx < tria_in.n_cells(0); ++idx)
                    {
                      const unsigned int coarse_cell_idx =
                        cell_numbering_inverse[idx];
                      typename Triangulation<dim, dim>::cell_iterator
                        coarse_cell(&tria_in, 0, coarse_cell_idx);


                      CellData<dim> tet(0);

                      for (const unsigned int v : coarse_cell->vertex_indices())
                        tet.vertices.emplace_back(coarse_cell->vertex_index(v));
                      cells[idx] = tet; //
                    }

                  tria_serial.create_triangulation(vertices,
                                                   cells,
                                                   SubCellData());
                }
              else
                {
                  dealii::GridIn<dim> grid_in;
                  grid_in.attach_triangulation(tria_serial);
                  std::ifstream input_file("lung.vtk");
                  grid_in.read_vtk(input_file);
                }

              for (auto &cell : tria_serial.active_cell_iterators())
                for (auto const &f : cell->face_indices())
                  if (cell->face(f)->at_boundary())
                    cell->face(f)->set_boundary_id(0);
            }
          else
            {
              // set up triangulation
              GridGenerator::subdivided_hyper_cube_with_simplices(
                tria_serial, n_subdivisions[refinement]);
              if (use_manifold)
                {
                  tria_serial.set_all_manifold_ids(0);
                  tria_serial.set_manifold(0, manifold);
                }

              tria_serial.refine_global(refinements[refinement]);
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

      if (use_manifold)
        tria.set_manifold(0, manifold);

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

      if (use_manifold)
        {
          tria.set_all_manifold_ids(0);
          tria.set_manifold(0, manifold);
        }

      DoFHandler<dim> dof_handler(tria);
      dof_handler.distribute_dofs(fe);

      constraint.close();

      Operator<dim, 1, Number> op;
      // set up operator
      op.reinit(mapping,
                dof_handler,
                quad,
                constraint,
                numbers::invalid_unsigned_int,
                true);
      LinearAlgebra::distributed::Vector<Number> vec1, vec2, vec3;
      op.initialize_dof_vector(vec1);
      op.initialize_dof_vector(vec2);
      op.initialize_dof_vector(vec3);

      for (Number &a : vec1)
        a = static_cast<double>(rand()) / RAND_MAX;

      MPI_Barrier(MPI_COMM_WORLD);
      for (unsigned int r = 0; r < 5; ++r)
        {
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_START(("matvec_p" + std::to_string(fe_degree) + "_s" +
                               std::to_string(dof_handler.n_dofs()))
                                .c_str());
#endif
          Timer time;
          for (unsigned int t = 0; t < 100; ++t)
            op.vmult(vec2, vec1);
          const double run_time = time.wall_time();
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_STOP(("matvec_p" + std::to_string(fe_degree) + "_s" +
                              std::to_string(dof_handler.n_dofs()))
                               .c_str());
#endif
          pcout << "n_dofs " << dof_handler.n_dofs() << "  time "
                << run_time / 100 << "  GDoFs/s "
                << 1e-9 * dof_handler.n_dofs() * 100 / run_time << std::endl;
        }

      for (unsigned int r = 0; r < 5; ++r)
        {
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_START(("matvec_manual_p" + std::to_string(fe_degree) +
                               "_s" + std::to_string(dof_handler.n_dofs()))
                                .c_str());
#endif
          Timer time;
          for (unsigned int t = 0; t < 100; ++t)
            op.vmult_manual(vec3, vec1);
          const double run_time = time.wall_time();
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_STOP(("matvec_manual_p" + std::to_string(fe_degree) +
                              "_s" + std::to_string(dof_handler.n_dofs()))
                               .c_str());
#endif
          pcout << "n_dofs " << dof_handler.n_dofs() << "  time manual "
                << run_time / 100 << "  GDoFs/s "
                << 1e-9 * dof_handler.n_dofs() * 100 / run_time << std::endl;
        }

      vec3 -= vec2;
      pcout << "Verification: " << vec3.l2_norm() / vec2.l2_norm() << std::endl;
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

