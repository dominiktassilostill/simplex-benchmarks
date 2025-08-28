
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix_ez.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

using namespace dealii;



template <bool transpose_matrix, typename Number, typename Number2>
void
apply_matrix_vector_product_6(const Number2 *matrix,
                            const Number  *in,
                            Number        *out,
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


  const Number *in0 = in, *in1 = in0 + mm, *in2 = in1 + mm, *in3 = in2 + mm, *in4 = in3 + mm, *in5 = in4 + mm;
  Number       *out0 = out, *out1 = out0 + nn, *out2 = out1 + nn, *out3 = out2 + nn, *out4 = out3 + nn, *out5 = out4 + nn;

  int nn_regular = (nn / 4) * 4;
  for (int col = 0; col < nn_regular; col += 4)
    {
      Number res[24];
      if (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + col;
          const Number   a = in0[0], b = in1[0], c = in2[0], d = in3[0], e = in4[0], f = in5[0];
          Number         m = matrix_ptr[0];
          res[0]           = m * a;
          res[4]           = m * b;
          res[8]           = m * c;
          res[12]          = m * d;
          res[16]          = m * e;
          res[20]          = m * f;
          m                = matrix_ptr[1];
          res[1]           = m * a;
          res[5]           = m * b;
          res[9]           = m * c;
          res[13]          = m * d;
          res[17]          = m * e;
          res[21]          = m * f;
          m                = matrix_ptr[2];
          res[2]           = m * a;
          res[6]           = m * b;
          res[10]          = m * c;
          res[14]          = m * d;
          res[18]          = m * e;
          res[22]          = m * f;
          m                = matrix_ptr[3];
          res[3]           = m * a;
          res[7]           = m * b;
          res[11]          = m * c;
          res[15]          = m * d;
          res[19]          = m * e;
          res[23]          = m * f;
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              const Number a = in0[i], b = in1[i], c = in2[i], d = in3[i], e = in4[i], f = in5[i];
              m = matrix_ptr[0];
              res[0] += m * a;
              res[4] += m * b;
              res[8] += m * c;
              res[12] += m * d;
              res[16]         += m * e;
          res[20]          += m * f;
              m = matrix_ptr[1];
              res[1] += m * a;
              res[5] += m * b;
              res[9] += m * c;
              res[13] += m * d;
              res[17]          += m * e;
          res[21]          += m * f;
              m = matrix_ptr[2];
              res[2] += m * a;
              res[6] += m * b;
              res[10] += m * c;
              res[14] += m * d;
               res[18]          += m * e;
          res[22]          += m * f;
              m = matrix_ptr[3];
              res[3] += m * a;
              res[7] += m * b;
              res[11] += m * c;
              res[15] += m * d;
              res[19]          += m * e;
          res[23]          += m * f;
            }
        }
      else
        {
          const Number2 *matrix_0 = matrix + col * n_columns;
          const Number2 *matrix_1 = matrix + (col + 1) * n_columns;
          const Number2 *matrix_2 = matrix + (col + 2) * n_columns;
          const Number2 *matrix_3 = matrix + (col + 3) * n_columns;

          const Number a = in0[0], b = in1[0], c = in2[0], d = in3[0], e = in4[0], f = in5[0];
          Number       m = matrix_0[0];
          res[0]         = m * a;
          res[4]         = m * b;
          res[8]         = m * c;
          res[12]        = m * d;
           res[16]          = m * e;
          res[20]          = m * f;
          m              = matrix_1[0];
          res[1]         = m * a;
          res[5]         = m * b;
          res[9]         = m * c;
          res[13]        = m * d;
           res[17]          = m * e;
          res[21]          = m * f;
          m              = matrix_2[0];
          res[2]         = m * a;
          res[6]         = m * b;
          res[10]        = m * c;
          res[14]        = m * d;
           res[18]          = m * e;
          res[22]          = m * f;
          m              = matrix_3[0];
          res[3]         = m * a;
          res[7]         = m * b;
          res[11]        = m * c;
          res[15]        = m * d;
           res[19]          = m * e;
          res[23]          = m * f;
          for (int i = 1; i < mm; ++i)
            {
              const Number a = in0[i], b = in1[i], c = in2[i], d = in3[i], e = in4[i], f = in5[i];
              m = matrix_0[i];
              res[0] += m * a;
              res[4] += m * b;
              res[8] += m * c;
              res[12]+= m * d;
               res[16]         += m * e;
          res[20]          += m * f;
              m = matrix_1[i];
              res[1] += m * a;
              res[5] += m * b;
              res[9] += m * c;
              res[13]+= m * d;
               res[17]          += m * e;
          res[21]          += m * f;
              m = matrix_2[i];
              res[2] += m * a;
              res[6] += m * b;
              res[10] += m * c;
              res[14]+= m * d;
               res[18]         += m * e;
          res[22]          += m * f;
              m = matrix_3[i];
              res[3] += m * a;
              res[7] += m * b;
              res[11] += m * c;
              res[15]+= m * d;
               res[19]         += m * e;
          res[23]          += m * f;
            }
        }
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
      out3[0] = res[12];
      out3[1] = res[13];
      out3[2] = res[14];
      out3[3] = res[15];
      out4[0] = res[16];
      out4[1] = res[17];
      out4[2] = res[18];
      out4[3] = res[19];
      out5[0] = res[20];
      out5[1] = res[21];
      out5[2] = res[22];
      out5[3] = res[23];
      out0 += 4;
      out1 += 4;
      out2 +=4;
      out3 +=4;
        out4 +=4;
      out5 +=4;
    }
  if (nn - nn_regular == 3)
    {
      Number res0, res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, res11, res12, res13, res14, res15, res16, res17;
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
           res12                      = matrix_ptr[0] * in4[0];
          res13                     = matrix_ptr[1] * in4[0];
          res14                     = matrix_ptr[2] * in4[0];
           res15                      = matrix_ptr[0] * in5[0];
          res16                     = matrix_ptr[1] * in5[0];
          res17                     = matrix_ptr[2] * in5[0];
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
              res10+= matrix_ptr[1] * in3[i];
              res11+= matrix_ptr[2] * in3[i];
               res12 += matrix_ptr[0] * in4[i];
              res13+= matrix_ptr[1] * in4[i];
              res14+= matrix_ptr[2] * in4[i];
               res15 += matrix_ptr[0] * in5[i];
              res16+= matrix_ptr[1] * in5[i];
              res17+= matrix_ptr[2] * in5[i];
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
          res9 = matrix_0[0] * in3[0];
          res10= matrix_1[0] * in3[0];
          res11= matrix_2[0] * in3[0];
          res12 = matrix_0[0] * in4[0];
          res13= matrix_1[0] * in4[0];
          res14= matrix_2[0] * in4[0];
          res15 = matrix_0[0] * in5[0];
          res16= matrix_1[0] * in5[0];
          res17= matrix_2[0] * in5[0];
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
              res10+= matrix_1[i] * in3[i];
              res11+= matrix_2[i] * in3[i];
              res12 += matrix_0[i] * in4[i];
              res13+= matrix_1[i] * in4[i];
              res14+= matrix_2[i] * in4[i];
              res15 += matrix_0[i] * in5[i];
              res16+= matrix_1[i] * in5[i];
              res17+= matrix_2[i] * in5[i];
            }
        }
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
      out4[0] = res12;
      out4[1] = res13;
      out4[2] = res14;
      out5[0] = res15;
      out5[1] = res16;
      out5[2] = res17;
    }
  else if (nn - nn_regular == 2)
    {
      Number res0, res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, res11;
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
           res8                      = matrix_ptr[0] * in4[0];
          res9                      = matrix_ptr[1] * in4[0];
           res10                      = matrix_ptr[0] * in5[0];
          res11                      = matrix_ptr[1] * in5[0];
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
               res8+= matrix_ptr[0] * in4[i];
              res9 += matrix_ptr[1] * in4[i];
               res10 += matrix_ptr[0] * in5[i];
              res11 += matrix_ptr[1] * in5[i];
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
          res8 = matrix_0[0] * in4[0];
          res9 = matrix_1[0] * in4[0];
          res10 = matrix_0[0] * in5[0];
          res11 = matrix_1[0] * in5[0];
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
                res8 += matrix_0[i] * in4[i];
              res9 += matrix_1[i] * in4[i];
                res10 += matrix_0[i] * in5[i];
              res11 += matrix_1[i] * in5[i];
            }
        }
      out0[0] = res0;
      out0[1] = res1;
      out1[0] = res2;
      out1[1] = res3;
      out2[0] = res4;
      out2[1] = res5;
      out3[0] = res6;
      out3[1] = res7;
       out4[0] = res8;
      out4[1] = res9;
       out5[0] = res10;
      out5[1] = res11;
    }
  else if (nn - nn_regular == 1)
    {
      Number res0, res1, res2, res3, res4, res5;
      if (transpose_matrix == true)
        {
          const Number2 *matrix_ptr = matrix + nn_regular;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[0] * in1[0];
          res2                      = matrix_ptr[0] * in2[0];
          res3                      = matrix_ptr[0] * in3[0];
          res4                      = matrix_ptr[0] * in4[0];
          res5                      = matrix_ptr[0] * in5[0];
          matrix_ptr += n_columns;
          for (int i = 1; i < mm; ++i, matrix_ptr += n_columns)
            {
              res0 += matrix_ptr[0] * in0[i];
              res1 += matrix_ptr[0] * in1[i];
              res2 += matrix_ptr[0] * in2[i];
              res3 += matrix_ptr[0] * in3[i];
               res4 += matrix_ptr[0] * in4[i];
                res5 += matrix_ptr[0] * in5[i];
            }
        }
      else
        {
          const Number2 *matrix_ptr = matrix + nn_regular * n_columns;
          res0                      = matrix_ptr[0] * in0[0];
          res1                      = matrix_ptr[0] * in1[0];
          res2                      = matrix_ptr[0] * in2[0];
          res3                      = matrix_ptr[0] * in3[0];
           res4                      = matrix_ptr[0] * in4[0];
            res5                      = matrix_ptr[0] * in5[0];
          for (int i = 1; i < mm; ++i)
            {
              res0 += matrix_ptr[i] * in0[i];
              res1 += matrix_ptr[i] * in1[i];
              res2 += matrix_ptr[i] * in2[i];
              res3 += matrix_ptr[i] * in3[i];
              res4 += matrix_ptr[i] * in4[i];
              res5 += matrix_ptr[i] * in5[i];
            }
        }
      out0[0] = res0;
      out1[0] = res1;
      out2[0] = res2;
      out3[0] = res3;
      out4[0] = res4;
      out5[0] = res5;
    }
}

template <bool transpose_matrix, typename Number, typename Number2>
void
apply_matrix_vector_product(const Number2 *matrix,
                            const Number  *in0,
                            const Number  *in1,
                            const Number  *in2,
                            Number        *out0,
                            Number        *out1,
                            Number        *out2,
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
      out0 += 4;
      out1 += 4;
      out2 +=4;
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
      out0[0] = res0;
      out0[1] = res1;
      out1[0] = res2;
      out1[1] = res3;
      out2[0] = res4;
      out2[1] = res5;
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
      out0[0] = res0;
      out1[0] = res1;
      out2[0] = res2;
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
    data.mapping_update_flags = update_gradients;
    data.mg_level             = mg_level;

    matrix_free.reinit(mapping, dof_handler, constraints, quad, data);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
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
                << " " << matrix_free.get_dof_info(0).dof_indices.size() << " "
                << dof_handler.get_triangulation().n_active_cells() << " "
                << dof_handler.get_triangulation().n_global_active_cells()
                << " " << dof_handler.n_dofs() << " "
                << static_cast<double>(dof_handler.n_dofs()) /
                     dof_handler.get_triangulation().n_global_active_cells()
                << " " << std::endl;

    constrained_indices.clear();

    if (ones_on_diagonal)
      for (auto i : this->matrix_free.get_constrained_dofs())
        constrained_indices.push_back(i);

    constexpr unsigned int n_lanes = VectorizedArray<number>::size();
    const unsigned int n_dofs_per_cell_per_componenet = matrix_free.get_dof_handler().get_fe().base_element(0).dofs_per_cell;
    if (matrix_free.get_dof_handler().get_fe().degree == 3)
    {
      manual_dof_indices.reinit(
        matrix_free.n_cell_batches(),
        n_components * n_dofs_per_cell_per_componenet * n_lanes,
        true);
      manual_dof_indices.fill(numbers::invalid_unsigned_int);
      
      std::vector<types::global_dof_index> dof_indices(
        matrix_free.get_dof_handler().get_fe().dofs_per_cell);

      dof_indices_have_constraints.clear();
      dof_indices_have_constraints.resize(matrix_free.n_cell_batches());

      const internal::MatrixFreeFunctions::ShapeInfo<number> &shape_info =
        matrix_free.get_shape_info();
      const auto l_n = shape_info.lexicographic_numbering;

      for (unsigned int c = 0; c < matrix_free.n_cell_batches(); ++c)
        {
          bool has_constraints =
            matrix_free.n_active_entries_per_cell_batch(c) < n_lanes;
          for (unsigned int v = 0;
              v < matrix_free.n_active_entries_per_cell_batch(c);
              ++v)
            {
              matrix_free.get_cell_iterator(c, v)->get_dof_indices(dof_indices);
              for (unsigned int i = 0; i < n_components * n_dofs_per_cell_per_componenet; ++i)
                {
                  if (!constraints.is_constrained(dof_indices[l_n[i]])) 
                    manual_dof_indices(c, i * n_lanes + v) =
                      matrix_free.get_dof_info()
                        .vector_partitioner->global_to_local(dof_indices[l_n[i]]); 
                  else
                    has_constraints = true;
              }
            }
          dof_indices_have_constraints[c] = has_constraints;
        }
    }
    else
    {
      manual_dof_indices.reinit(
        matrix_free.n_cell_batches(),
        n_dofs_per_cell_per_componenet * n_lanes,
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
              for (unsigned int i = 0; i < n_dofs_per_cell_per_componenet; ++i)
                if (!constraints.is_constrained(dof_indices[n_components * i]))
                  manual_dof_indices(c, i * n_lanes + v) =
                    matrix_free.get_dof_info()
                      .vector_partitioner->global_to_local(dof_indices[n_components * i]);
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
  vmult(VectorType &dst, const VectorType &src) const
  {
    this->matrix_free.cell_loop(
      &Operator::do_cell_integral_range, this, dst, src, true);

    for (unsigned int i = 0; i < constrained_indices.size(); ++i)
      dst.local_element(constrained_indices[i]) =
        src.local_element(constrained_indices[i]);
  }

  virtual void
  vmult_masked_gather(VectorType &dst, const VectorType &src) const
  {
    this->matrix_free.cell_loop(
      &Operator::do_cell_integral_masked_gather, this, dst, src, true);

    for (unsigned int i = 0; i < constrained_indices.size(); ++i)
      dst.local_element(constrained_indices[i]) =
        src.local_element(constrained_indices[i]);
  }

  virtual void
  vmult_masked_gather_6(VectorType &dst, const VectorType &src) const
  {
    this->matrix_free.cell_loop(
      &Operator::do_cell_integral_masked_gather_6, this, dst, src, true);

    for (unsigned int i = 0; i < constrained_indices.size(); ++i)
      dst.local_element(constrained_indices[i]) =
        src.local_element(constrained_indices[i]);
  }

  virtual void
  vmult_masked_gather_cubic(VectorType &dst, const VectorType &src) const
  {
    this->matrix_free.cell_loop(
      &Operator::do_cell_integral_masked_gather_cubic, this, dst, src, true);

    for (unsigned int i = 0; i < constrained_indices.size(); ++i)
      dst.local_element(constrained_indices[i]) =
        src.local_element(constrained_indices[i]);
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

private:
  void
  do_cell_integral_global(FECellIntegrator &integrator,
                          VectorType       &dst,
                          const VectorType &src) const
  {
    integrator.gather_evaluate(src, EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_symmetric_gradient(integrator.get_symmetric_gradient(q), q);

    integrator.integrate_scatter(EvaluationFlags::gradients, dst);
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
        do_cell_integral_global(integrator, dst, src);
      }
  }

  void
  do_cell_integral_masked_gather(
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

    const unsigned int     n_q_points = shape_info.n_q_points;
    constexpr unsigned int n_lanes    = VectorizedArray<number>::size();

    const auto   &mapping_data = matrix_free.get_mapping_info().cell_data[0];
    const number *quadrature_weights =
      mapping_data.descriptor[0].quadrature_weights.data();

    scratch_data->resize_fast(n_components * (dim * n_q_points + dofs_per_cell));
    VectorizedArray<number> *values_dofs = scratch_data->begin();
    VectorizedArray<number> *gradients_quad =
      scratch_data->begin() + dofs_per_cell * n_components;

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        // read dof values
        const unsigned int *dof_indices = &manual_dof_indices(cell, 0);
          {
            const number *src_ptr = src.begin();
            if (dof_indices_have_constraints[cell])
              {
                for (unsigned int i = 0; i < dofs_per_cell;
                     ++i, dof_indices += n_lanes)
                  {
                    values_dofs[i] = {};
                    values_dofs[dofs_per_cell + i] = {};
                    values_dofs[2 * dofs_per_cell + i] = {};

                    for (unsigned int v = 0; v < n_lanes; ++v)
                      if (dof_indices[v] != numbers::invalid_unsigned_int)
                      {
                        values_dofs[i][v] =
                          src_ptr[dof_indices[v]];

                        values_dofs[dofs_per_cell + i][v] =
                          src_ptr[dof_indices[v] + 1];

                        values_dofs[2 * dofs_per_cell + i][v] =
                          src_ptr[dof_indices[v] + 2];
                      }
                  }
              }
            else
              for (unsigned int i = 0; i < dofs_per_cell;
                   ++i, dof_indices += n_lanes)
                {
                  values_dofs[i] = {};
                  values_dofs[dofs_per_cell + i] = {};
                  values_dofs[2 * dofs_per_cell + i] = {};
                  
                  for (unsigned int v = 0; v < n_lanes; ++v)
                  {
                    values_dofs[i][v] =
                      src_ptr[dof_indices[v]];

                    values_dofs[dofs_per_cell + i][v] =
                      src_ptr[dof_indices[v] + 1];

                    values_dofs[2 * dofs_per_cell + i][v] =
                      src_ptr[dof_indices[v] + 2];
                  }
                }
          }

        apply_matrix_vector_product<true>(
              shape_info.data[0].shape_gradients.data(),
              values_dofs,
              values_dofs + dofs_per_cell,
              values_dofs + 2 * dofs_per_cell,
              gradients_quad,
              gradients_quad + n_q_points * dim,
              gradients_quad + 2 * n_q_points * dim,
              dofs_per_cell,
              n_q_points * dim);

        // quadrature point operation
          {
            const unsigned int offsets =
              mapping_data.data_index_offsets[cell];
            const Tensor<2, dim, VectorizedArray<number>> *jac =
              mapping_data.jacobians[0].data() + offsets;
            const VectorizedArray<number> * j_value =
              &mapping_data.JxW_values[offsets];
            VectorizedArray<number> *grad_ptr =
              gradients_quad;
            if (matrix_free.get_mapping_info().cell_type[cell] <=
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
                for (unsigned int comp = 0; comp < n_components; ++comp)
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
                for (unsigned int comp = 0; comp < n_components; ++comp)
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

      
        apply_matrix_vector_product<false>(
              shape_info.data[0].shape_gradients.data(),
              gradients_quad,
              gradients_quad + n_q_points * dim,
              gradients_quad + 2 * n_q_points * dim,
              values_dofs,
              values_dofs + dofs_per_cell,
              values_dofs + 2 * dofs_per_cell,
              dofs_per_cell,
              n_q_points * dim);

        // distribute local to global
        dof_indices = &manual_dof_indices(cell, 0);
          {
            if (dof_indices_have_constraints[cell])
              {
                for (unsigned int i = 0; i < dofs_per_cell;
                     ++i, dof_indices += n_lanes)
                  {
                    for (unsigned int v = 0; v < n_lanes; ++v)
                      if (dof_indices[v] != numbers::invalid_unsigned_int)
                      {
                        dst.local_element(dof_indices[v]) +=
                          values_dofs[i][v];
                        dst.local_element(dof_indices[v] + 1) +=
                          values_dofs[dofs_per_cell + i][v];
                        dst.local_element(dof_indices[v] + 2) +=
                          values_dofs[2 * dofs_per_cell +i][v];
                      }
                  }
              }
            else
              for (unsigned int i = 0; i < dofs_per_cell;
                   ++i, dof_indices += n_lanes)
                {
                  for (unsigned int v = 0; v < n_lanes; ++v)
                  {
                    dst.local_element(dof_indices[v]) +=
                      values_dofs[i][v];
                    dst.local_element(dof_indices[v] + 1) +=
                      values_dofs[dofs_per_cell + i][v];
                    dst.local_element(dof_indices[v] + 2) +=
                      values_dofs[2 * dofs_per_cell + i][v];
                  }
                }
          }
      }

    matrix_free.release_scratch_data(scratch_data);
  }

  void
  do_cell_integral_masked_gather_cubic(
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

    const unsigned int     n_q_points = shape_info.n_q_points;
    constexpr unsigned int n_lanes    = VectorizedArray<number>::size();

    const auto   &mapping_data = matrix_free.get_mapping_info().cell_data[0];
    const number *quadrature_weights =
      mapping_data.descriptor[0].quadrature_weights.data();

    scratch_data->resize_fast(n_components * (dim * n_q_points + dofs_per_cell));
    VectorizedArray<number> *values_dofs = scratch_data->begin();
    VectorizedArray<number> *gradients_quad =
      scratch_data->begin() + dofs_per_cell * n_components;

    const number *src_ptr = src.begin();

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {        
        // read dof values
        const unsigned int *dof_indices = &manual_dof_indices(cell, 0); 

        if (dof_indices_have_constraints[cell])
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                values_dofs[i] = {};
                values_dofs[dofs_per_cell + i] = {};
                values_dofs[2 * dofs_per_cell + i] = {};

                for (unsigned int v = 0; v < n_lanes; ++v)
                  if (dof_indices[i * n_lanes + v] != numbers::invalid_unsigned_int)
                  {                       
                    values_dofs[i][v] =
                      src_ptr[dof_indices[i * n_lanes + v]];
                    
                    values_dofs[dofs_per_cell + i][v] =
                      src_ptr[dof_indices[(dofs_per_cell + i) * n_lanes + v]];
                    
                    values_dofs[2 * dofs_per_cell + i][v] =
                      src_ptr[dof_indices[(2 * dofs_per_cell + i) * n_lanes + v]];
                                  
                  }
              }
          }
        else
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              values_dofs[i] = {};
              values_dofs[dofs_per_cell + i] = {};
              values_dofs[2 * dofs_per_cell + i] = {};
              
              for (unsigned int v = 0; v < n_lanes; ++v)
              {
                values_dofs[i][v] =
                      src_ptr[dof_indices[i * n_lanes + v]];
                    
                    values_dofs[dofs_per_cell + i][v] =
                      src_ptr[dof_indices[(dofs_per_cell + i) * n_lanes + v]];
                    
                    values_dofs[2 * dofs_per_cell + i][v] =
                      src_ptr[dof_indices[(2 * dofs_per_cell + i) * n_lanes + v]];
              }
            }
                

        apply_matrix_vector_product<true>(
              shape_info.data[0].shape_gradients.data(),
              values_dofs,
              values_dofs + dofs_per_cell,
              values_dofs + 2 * dofs_per_cell,
              gradients_quad,
              gradients_quad + n_q_points * dim,
              gradients_quad + 2 * n_q_points * dim,
              dofs_per_cell,
              n_q_points * dim);

        // do quadrature point operations
        const unsigned int offsets =
          mapping_data.data_index_offsets[cell];
        const Tensor<2, dim, VectorizedArray<number>> *jac =
          mapping_data.jacobians[0].data() + offsets;
        const VectorizedArray<number> * j_value =
          &mapping_data.JxW_values[offsets];
        VectorizedArray<number> *grad_ptr =
          gradients_quad;
        if (matrix_free.get_mapping_info().cell_type[cell] <=
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
            for (unsigned int comp = 0; comp < n_components; ++comp)
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
            for (unsigned int comp = 0; comp < n_components; ++comp)
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
          

         apply_matrix_vector_product<false>(
              shape_info.data[0].shape_gradients.data(),
              gradients_quad,
              gradients_quad + n_q_points * dim,
              gradients_quad + 2 * n_q_points * dim,
              values_dofs,
              values_dofs + dofs_per_cell,
              values_dofs + 2 * dofs_per_cell,
              dofs_per_cell,
              n_q_points * dim);

          
        if (dof_indices_have_constraints[cell])
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int v = 0; v < n_lanes; ++v)
                  if (dof_indices[i * n_lanes + v]  != numbers::invalid_unsigned_int)
                  {
                    dst.local_element(dof_indices[i * n_lanes + v]) +=
                      values_dofs[i][v];
                    dst.local_element(dof_indices[(dofs_per_cell + i) * n_lanes + v]) +=
                      values_dofs[dofs_per_cell + i][v];
                    dst.local_element(dof_indices[(2 * dofs_per_cell + i) * n_lanes + v]) +=
                      values_dofs[2 * dofs_per_cell +i][v];
                  }
              }
          }
        else
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int v = 0; v < n_lanes; ++v)
              {
                dst.local_element(dof_indices[i * n_lanes + v]) +=
                  values_dofs[i][v];
                dst.local_element(dof_indices[(dofs_per_cell + i) * n_lanes + v]) +=
                  values_dofs[dofs_per_cell + i][v];
                dst.local_element(dof_indices[(2 * dofs_per_cell + i) * n_lanes + v]) +=
                  values_dofs[2 * dofs_per_cell + i][v];
              }
            }
          
      }
    matrix_free.release_scratch_data(scratch_data);
  }


  void
  do_cell_integral_masked_gather_6(
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
    
    constexpr unsigned int batch_size = 2;

    const unsigned int     n_q_points = shape_info.n_q_points;
    constexpr unsigned int n_lanes    = VectorizedArray<number>::size();

    const auto   &mapping_data = matrix_free.get_mapping_info().cell_data[0];
    const number *quadrature_weights =
      mapping_data.descriptor[0].quadrature_weights.data();

    scratch_data->resize_fast(batch_size * n_components * (dim * n_q_points + dofs_per_cell));
    VectorizedArray<number> *values_dofs = scratch_data->begin();
    VectorizedArray<number> *gradients_quad =
      scratch_data->begin() + batch_size * dofs_per_cell * n_components;

    const number *src_ptr = src.begin();

    for (unsigned int cell = range.first; cell < range.second; cell += batch_size)
      {
        // read dof values
        const unsigned int my_batch_size =
          cell + batch_size <= range.second ? batch_size : range.second - cell;

        const unsigned int *dof_indices = &manual_dof_indices(cell, 0);
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            const unsigned int offset = batch * dofs_per_cell * n_components;
            if (dof_indices_have_constraints[cell + batch])
              {
                for (unsigned int i = 0; i < dofs_per_cell;
                     ++i, dof_indices += n_lanes)
                  {
                    values_dofs[offset + i] = {};
                    values_dofs[offset + dofs_per_cell + i] = {};
                    values_dofs[offset + 2 * dofs_per_cell + i] = {};

                    for (unsigned int v = 0; v < n_lanes; ++v)
                      if (dof_indices[v] != numbers::invalid_unsigned_int)
                      {
                        values_dofs[offset + i][v] =
                          src_ptr[dof_indices[v]];

                        values_dofs[offset + dofs_per_cell + i][v] =
                          src_ptr[dof_indices[v] + 1];

                        values_dofs[offset + 2 * dofs_per_cell + i][v] =
                          src_ptr[dof_indices[v] + 2];
                      }
                  }
              }
            else
              for (unsigned int i = 0; i < dofs_per_cell;
                   ++i, dof_indices += n_lanes)
                {
                  values_dofs[offset + i] = {};
                  values_dofs[offset + dofs_per_cell + i] = {};
                  values_dofs[offset + 2 * dofs_per_cell + i] = {};
                  
                  for (unsigned int v = 0; v < n_lanes; ++v)
                  {
                    values_dofs[offset + i][v] =
                      src_ptr[dof_indices[v]];

                    values_dofs[offset + dofs_per_cell + i][v] =
                      src_ptr[dof_indices[v] + 1];

                    values_dofs[offset + 2 * dofs_per_cell + i][v] =
                      src_ptr[dof_indices[v] + 2];
                  }
                }
          }

         apply_matrix_vector_product_6<true>(
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
              gradients_quad + batch *  n_components * n_q_points * dim;
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
                for (unsigned int comp = 0; comp < n_components; ++comp)
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
                for (unsigned int comp = 0; comp < n_components; ++comp)
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

      
        apply_matrix_vector_product_6<false>(
              shape_info.data[0].shape_gradients.data(),
              gradients_quad,
              values_dofs,
              dofs_per_cell,
              n_q_points * dim);

        // distribute local to global
        dof_indices = &manual_dof_indices(cell, 0);
        for (unsigned int batch = 0; batch < my_batch_size; ++batch)
          {
            const unsigned int offset = batch * dofs_per_cell * n_components;
            if (dof_indices_have_constraints[cell + batch])
              {
                for (unsigned int i = 0; i < dofs_per_cell;
                     ++i, dof_indices += n_lanes)
                  {
                    for (unsigned int v = 0; v < n_lanes; ++v)
                      if (dof_indices[v] != numbers::invalid_unsigned_int)
                      {
                        dst.local_element(dof_indices[v]) +=
                          values_dofs[offset + i][v];
                        
                        dst.local_element(dof_indices[v] + 1) +=
                          values_dofs[offset + dofs_per_cell + i][v];
                        
                        dst.local_element(dof_indices[v] + 2) +=
                          values_dofs[offset + 2 * dofs_per_cell +i][v];
                      }
                  }
              }
            else
              for (unsigned int i = 0; i < dofs_per_cell;
                   ++i, dof_indices += n_lanes)
                {
                  for (unsigned int v = 0; v < n_lanes; ++v)
                  {
                    dst.local_element(dof_indices[v]) +=
                      values_dofs[offset + i][v];
                    
                    dst.local_element(dof_indices[v] + 1) +=
                      values_dofs[offset + dofs_per_cell + i][v];
                    
                    dst.local_element(dof_indices[v] + 2) +=
                      values_dofs[offset + 2 * dofs_per_cell + i][v];
                  }
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

void
print_memory_stats(MPI_Comm communicator, const std::string &region_name = {})
{
  Utilities::System::MemoryStats stats;
  Utilities::System::get_memory_stats(stats);
  Utilities::MPI::MinMaxAvg memory =
    Utilities::MPI::min_max_avg(stats.VmRSS / 1024, communicator);
  if (Utilities::MPI::this_mpi_process(communicator) == 0)
    std::cout << "   Memory stats " << region_name << " [MB]: " << memory.min
              << " " << memory.avg << " " << memory.max << std::endl;
}

template <int dim, typename Number>
void
do_test(const unsigned int fe_degree, const unsigned int refine)
{
  const bool use_manifold = false;
  const bool grid_in      = false;
  const bool reorder_grid = false;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  pcout << "Running in " << dim << "D with degree " << fe_degree << std::endl;
  //FE_SimplexP<dim>   fe(fe_degree);
  FESystem<dim> fe(FE_SimplexP<dim>(fe_degree), dim);
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
  double n_dofs = 0;
  for (unsigned int k = 0;  n_dofs < 100000000.0; ++k)
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
        n_dofs = n_current * 0.2;
      else if (fe_degree == 2)
        n_dofs = n_current * 1.4;
      else
        n_dofs = n_current * 4.6;
      n_dofs *= 3;
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

  if (grid_in)
    refinements.resize(1);


  for (unsigned int refinement = 0;  refinement < refinements.size(); ++refinement)
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
      print_memory_stats(tria.get_communicator(), "dofs");

      // set up constraints, then renumber dofs, and set up constraints again
      if (true)
        {
          const IndexSet locally_relevant_dofs =
            DoFTools::extract_locally_relevant_dofs(dof_handler);
          constraint.reinit(dof_handler.locally_owned_dofs(),
                            locally_relevant_dofs);
          VectorTools::interpolate_boundary_values(
            mapping,
            dof_handler,
            0,
            Functions::ZeroFunction<dim>(dim),
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
        mapping, dof_handler, 0, Functions::ZeroFunction<dim>(dim), constraint);
      constraint.close();

      Operator<dim, dim, Number> op;
      // set up operator
      op.reinit(mapping,
                dof_handler,
                quad,
                constraint,
                numbers::invalid_unsigned_int,
                true);
      print_memory_stats(tria.get_communicator(), "matrix-free");
      {
        const double mem_mf =
          static_cast<double>(op.get_matrix_free().memory_consumption()) /
          tria.n_locally_owned_active_cells();
        Utilities::MPI::MinMaxAvg memory =
          Utilities::MPI::min_max_avg(mem_mf, tria.get_communicator());
        if (Utilities::MPI::this_mpi_process(tria.get_communicator()) == 0)
          std::cout << "   Memory matrix-free (byte per cell): " << memory.min
                    << " " << memory.avg << " " << memory.max << std::endl;
      }

      LinearAlgebra::distributed::Vector<Number> vec1, vec2, vec4, vec5;
      op.initialize_dof_vector(vec1);
      op.initialize_dof_vector(vec2);
      op.initialize_dof_vector(vec4);
      op.initialize_dof_vector(vec5);
      for (Number &a : vec1)
        a = static_cast<double>(rand()) / RAND_MAX;

      for (unsigned int r = 0; r < 5; ++r)
        {
          Timer time;
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_START(("matvec_p" + std::to_string(fe_degree) + "_s" +
                               std::to_string(dof_handler.n_dofs()))
                                .c_str());
#endif
          for (unsigned int t = 0; t < 100; ++t)
            op.vmult(vec2, vec1);
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_STOP(("matvec_p" + std::to_string(fe_degree) + "_s" +
                              std::to_string(dof_handler.n_dofs()))
                               .c_str());
#endif
          const double run_time = time.wall_time();
          pcout << "n_dofs mf basic  " << dof_handler.n_dofs() << "  time "
                << run_time / 100 << "  GDoFs/s "
                << 1e-9 * dof_handler.n_dofs() * 100 / run_time << std::endl;
        }
      if (fe_degree == 3)
      {
        for (unsigned int r = 0; r < 5; ++r)
        {
          Timer time;
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_START(("matvec_gather_p" + std::to_string(fe_degree) +
                               "_s" + std::to_string(dof_handler.n_dofs()))
                                .c_str());
#endif
          for (unsigned int t = 0; t < 100; ++t)
            op.vmult_masked_gather_cubic(vec5, vec1);
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_STOP(("matvec_gather_p" + std::to_string(fe_degree) +
                              "_s" + std::to_string(dof_handler.n_dofs()))
                               .c_str());
#endif
          const double run_time = time.wall_time();
          pcout << "n_dofs mf gather " << dof_handler.n_dofs() << "  time "
                << run_time / 100 << "  GDoFs/s "
                << 1e-9 * dof_handler.n_dofs() * 100 / run_time << std::endl;
        }
        vec4 = vec2;
      }
      else
      {
        for (unsigned int r = 0; r < 5; ++r)
        {
          Timer time;
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_START(("matvec_gather_p" + std::to_string(fe_degree) +
                               "_s" + std::to_string(dof_handler.n_dofs()))
                                .c_str());
#endif
          for (unsigned int t = 0; t < 100; ++t)
            op.vmult_masked_gather(vec5, vec1);
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_STOP(("matvec_gather_p" + std::to_string(fe_degree) +
                              "_s" + std::to_string(dof_handler.n_dofs()))
                               .c_str());
#endif
          const double run_time = time.wall_time();
          pcout << "n_dofs mf gather " << dof_handler.n_dofs() << "  time "
                << run_time / 100 << "  GDoFs/s "
                << 1e-9 * dof_handler.n_dofs() * 100 / run_time << std::endl;
        }
        if (false)
        for (unsigned int r = 0; r < 5; ++r)
        {
          Timer time;
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_START(("matvec_gather_6_p" + std::to_string(fe_degree) +
                               "_s" + std::to_string(dof_handler.n_dofs()))
                                .c_str());
#endif
          for (unsigned int t = 0; t < 100; ++t)
            op.vmult_masked_gather_6(vec4, vec1);
#ifdef LIKWID_PERFMON
          LIKWID_MARKER_STOP(("matvec_gather_6_p" + std::to_string(fe_degree) +
                              "_s" + std::to_string(dof_handler.n_dofs()))
                               .c_str());
#endif
          const double run_time = time.wall_time();
          pcout << "n_dofs mf gather 6 " << dof_handler.n_dofs() << "  time "
                << run_time / 100 << "  GDoFs/s "
                << 1e-9 * dof_handler.n_dofs() * 100 / run_time << std::endl;
        }
      }
      
      pcout << std::endl;

      vec4 -= vec2;
      vec5 -= vec2;
      pcout << "   Error MF variants: " 
            << vec5.l2_norm() / vec2.l2_norm() << " " << vec4.l2_norm() / vec2.l2_norm() << std::endl
            << std::endl;

      print_memory_stats(tria.get_communicator(), "end experiment");

      const bool do_matrix = false;
      if (do_matrix)
      {
        DynamicSparsityPattern dsp(locally_relevant_dofs);

        DoFTools::make_sparsity_pattern(dof_handler, dsp, constraint, true);
        SparsityTools::distribute_sparsity_pattern(
          dsp,
          dof_handler.locally_owned_dofs(),
          dof_handler.get_communicator(),
          locally_relevant_dofs);

        TrilinosWrappers::SparseMatrix matrix;
        matrix.reinit(dof_handler.locally_owned_dofs(),
                      dof_handler.locally_owned_dofs(),
                      dsp,
                      dof_handler.get_communicator());

        pcout << "Matrix nnz/dof: "
              << (double)matrix.n_nonzero_elements() / matrix.m() << std::endl;

        for (unsigned int r = 0; r < 5; ++r)
          {
            Timer time;
#ifdef LIKWID_PERFMON
            LIKWID_MARKER_START(("spmv_p" + std::to_string(fe_degree) + "_s" +
                                std::to_string(dof_handler.n_dofs()))
                                  .c_str());
#endif
            for (unsigned int t = 0; t < 100; ++t)
              matrix.vmult(vec2, vec1);
#ifdef LIKWID_PERFMON
            LIKWID_MARKER_STOP(("spmv_p" + std::to_string(fe_degree) + "_s" +
                                std::to_string(dof_handler.n_dofs()))
                                .c_str());
#endif
            const double run_time = time.wall_time();
            pcout << "n_dofs spmv " << dof_handler.n_dofs() << "  time "
                  << run_time / 100 << "  GDoFs/s "
                  << 1e-9 * dof_handler.n_dofs() * 100 / run_time << std::endl;
          }
        pcout << std::endl;
        print_memory_stats(tria.get_communicator(), "matrix");
      }
    }
}

int
main(int argc, char **argv)
{
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  int degree = 2;
  int dim    = 3;
  int refine = 4;
  if (argc > 1)
    dim = std::atoi(argv[1]);
  if (argc > 2)
    degree = std::atoi(argv[2]);
  if (argc > 3)
    refine = std::atoi(argv[3]);

  if (dim == 2)
    do_test<2, double>(degree, refine);
  else
    do_test<3, double>(degree, refine);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}

