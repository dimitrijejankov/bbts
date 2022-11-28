cc_library(
    name = "tensor_headers",
    hdrs = ["builtin_formats.h", "tensor_factory.h", "tensor.h", "xla_types.h"],
    deps = ["@org_tensorflow//tensorflow/compiler/xla:tos_cpu_kernel"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tensor",
    srcs = ["builtin_formats.cc", "tensor_factory.cc", "xla_types.cc"],
    deps = ["//main/tensor:tensor_headers"],
    alwayslink = 1,
    visibility = ["//visibility:public"]
)