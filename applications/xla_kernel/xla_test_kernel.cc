#include "xla_test_kernel.h"

// the hlo string of the function we want to execute
const std::string bbts::xla_test_kernel::hlo =
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

bbts::xla_test_kernel::xla_test_kernel() : xla_kernel_base(hlo) {

  // set the names
  impl_name = "xla_test_kernel_cpu";
  ud_name = "xla_test_kernel";
}