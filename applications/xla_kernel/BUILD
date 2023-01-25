cc_library(
    name = "xla_kernel_common_lib",
    srcs = ["xla_lib_init.cc", "xla_test_kernel.cc"],
    hdrs = ["xla_test_kernel.h"],
    deps = ["//main:bbts_lib"],
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