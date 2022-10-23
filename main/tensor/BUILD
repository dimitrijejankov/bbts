cc_library(
    name = "tensor_headers",
    hdrs = ["builtin_formats.h", "tensor_factory.h", "tensor.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tensor",
    srcs = ["builtin_formats.cc", "tensor_factory.cc"],
    deps = ["//main/tensor:tensor_headers"],
    alwayslink = 1,
    visibility = ["//visibility:public"]
)