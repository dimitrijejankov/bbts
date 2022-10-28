#include <iostream>
#include <memory>
#include <vector>

#include <algorithm> // fill
#include <numeric>   // iota

#include "xla_kernel.h"

// This example includes the constant ones. This is to make sure that extra buffers are needed.
//  def f(x,y):
//    zz = jnp.ones((2,100,100,2))
//    w = jnp.einsum("ijkl,il->ji", zz, x)
//    e = jnp.dot(w, y)
//    return e
std::string hlo04 =
  "HloModule jit_f, entry_computation_layout={(f32[2,2]{1,0},f32[2,2]{1,0})->f32[100,2]{1,0}}\n"
  "\n"
  "region_0.5 {\n"
  "  Arg_0.6 = f32[] parameter(0)\n"
  "  Arg_1.7 = f32[] parameter(1)\n"
  "  ROOT add.8 = f32[] add(Arg_0.6, Arg_1.7)\n"
  "}\n"
  "\n"
  "_einsum.9 {\n"
  "  Arg_1.11 = f32[2,2]{1,0} parameter(1)\n"
  "  Arg_0.10 = f32[2,100,100,2]{3,2,1,0} parameter(0)\n"
  "  constant.12 = f32[] constant(0)\n"
  "  reduce.13 = f32[2,100,2]{2,1,0} reduce(Arg_0.10, constant.12), dimensions={2}, to_apply=region_0.5\n"
  "  dot.14 = f32[2,100]{1,0} dot(Arg_1.11, reduce.13), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={2}\n"
  "  ROOT transpose.15 = f32[100,2]{0,1} transpose(dot.14), dimensions={1,0}\n"
  "}\n"
  "\n"
  "ENTRY main.18 {\n"
  "  constant.3 = f32[] constant(1)\n"
  "  broadcast.4 = f32[2,100,100,2]{3,2,1,0} broadcast(constant.3), dimensions={}\n"
  "  Arg_0.1 = f32[2,2]{1,0} parameter(0)\n"
  "  call.16 = f32[100,2]{0,1} call(broadcast.4, Arg_0.1), to_apply=_einsum.9\n"
  "  Arg_1.2 = f32[2,2]{1,0} parameter(1)\n"
  "  ROOT dot.17 = f32[100,2]{1,0} dot(call.16, Arg_1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}\n"
  "}\n";

// This kernel was actually generated and includes reblocking
// It computes
//   out_ij = sum_e (x_ie - y_je)^2
// with however x and y are partitioned.
std::string hlo05 =
  "HloModule jit_f, entry_computation_layout={(f32[4,2]{1,0},f32[4,2]{1,0},f32[4,2]{1,0},f32[4,1]{1,0},f32[4,2]{1,0},f32[4,2]{1,0},f32[4,2]{1,0},f32[4,1]{1,0},f32[3,2]{1,0},f32[3,2]{1,0},f32[3,2]{1,0},f32[3,1]{1,0},f32[4,2]{1,0},f32[4,2]{1,0},f32[4,2]{1,0},f32[4,1]{1,0})->f32[11,4]{1,0}}\n"
  "\n"
  "region_0.33 {\n"
  "  Arg_0.34 = f32[] parameter(0)\n"
  "  Arg_1.35 = f32[] parameter(1)\n"
  "  ROOT add.36 = f32[] add(Arg_0.34, Arg_1.35)\n"
  "}\n"
  "\n"
  "ENTRY main.38 {\n"
  "  Arg_0.1 = f32[4,2]{1,0} parameter(0)\n"
  "  Arg_1.2 = f32[4,2]{1,0} parameter(1)\n"
  "  Arg_2.3 = f32[4,2]{1,0} parameter(2)\n"
  "  Arg_3.4 = f32[4,1]{1,0} parameter(3)\n"
  "  concatenate.18 = f32[4,7]{1,0} concatenate(Arg_0.1, Arg_1.2, Arg_2.3, Arg_3.4), dimensions={1}\n"
  "  Arg_4.5 = f32[4,2]{1,0} parameter(4)\n"
  "  Arg_5.6 = f32[4,2]{1,0} parameter(5)\n"
  "  Arg_6.7 = f32[4,2]{1,0} parameter(6)\n"
  "  Arg_7.8 = f32[4,1]{1,0} parameter(7)\n"
  "  concatenate.19 = f32[4,7]{1,0} concatenate(Arg_4.5, Arg_5.6, Arg_6.7, Arg_7.8), dimensions={1}\n"
  "  Arg_8.9 = f32[3,2]{1,0} parameter(8)\n"
  "  Arg_9.10 = f32[3,2]{1,0} parameter(9)\n"
  "  Arg_10.11 = f32[3,2]{1,0} parameter(10)\n"
  "  Arg_11.12 = f32[3,1]{1,0} parameter(11)\n"
  "  concatenate.20 = f32[3,7]{1,0} concatenate(Arg_8.9, Arg_9.10, Arg_10.11, Arg_11.12), dimensions={1}\n"
  "  concatenate.21 = f32[11,7]{1,0} concatenate(concatenate.18, concatenate.19, concatenate.20), dimensions={0}\n"
  "  reshape.23 = f32[11,1,7]{2,1,0} reshape(concatenate.21)\n"
  "  broadcast.25 = f32[11,1,7]{2,1,0} broadcast(reshape.23), dimensions={0,1,2}\n"
  "  reshape.26 = f32[11,7]{1,0} reshape(broadcast.25)\n"
  "  broadcast.27 = f32[11,4,7]{2,1,0} broadcast(reshape.26), dimensions={0,2}\n"
  "  Arg_12.13 = f32[4,2]{1,0} parameter(12)\n"
  "  Arg_13.14 = f32[4,2]{1,0} parameter(13)\n"
  "  Arg_14.15 = f32[4,2]{1,0} parameter(14)\n"
  "  Arg_15.16 = f32[4,1]{1,0} parameter(15)\n"
  "  concatenate.22 = f32[4,7]{1,0} concatenate(Arg_12.13, Arg_13.14, Arg_14.15, Arg_15.16), dimensions={1}\n"
  "  reshape.24 = f32[1,4,7]{2,1,0} reshape(concatenate.22)\n"
  "  broadcast.28 = f32[1,4,7]{2,1,0} broadcast(reshape.24), dimensions={0,1,2}\n"
  "  reshape.29 = f32[4,7]{1,0} reshape(broadcast.28)\n"
  "  broadcast.30 = f32[11,4,7]{2,1,0} broadcast(reshape.29), dimensions={1,2}\n"
  "  subtract.31 = f32[11,4,7]{2,1,0} subtract(broadcast.27, broadcast.30)\n"
  "  multiply.32 = f32[11,4,7]{2,1,0} multiply(subtract.31, subtract.31)\n"
  "  constant.17 = f32[] constant(0)\n"
  "  ROOT reduce.37 = f32[11,4]{1,0} reduce(multiply.32, constant.17), dimensions={2}, to_apply=region_0.33\n"
  "}\n"
  "\n"
  "\n";

int main() {
  tos::CpuKernelBla k04(hlo04);
  k04.test();

  tos::CpuKernelBla k05(hlo05);
  k05.test();
}

