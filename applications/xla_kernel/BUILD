cc_library(
    name = "xla_kernel_common_lib",
    srcs = ["xla_kernel_base.cc",  "xla_lib_init.cc",  "xla_types.cc", "xla_test_kernel.cc"],
    hdrs = ["xla_kernel_base.h", "xla_types.h", "xla_test_kernel.h"],
    deps = ["@org_tensorflow//tensorflow/compiler/xla:tos_cpu_kernel", "//main:bbts_lib"],
    alwayslink = 1,
)

cc_binary(
    name = "xla_kernel_lib.so",
    deps = [":xla_kernel_common_lib"],
    linkshared = True
)

cc_binary(
    name = "xla_kernel_test",
    srcs = ["main.cc"],
    deps = [":xla_kernel_common_lib"],
)