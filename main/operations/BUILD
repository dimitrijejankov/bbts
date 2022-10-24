cc_library(
    name = "operations_headers",
    hdrs = ["broadcast_op.h", "move_op.h", "partial_reduce_op.h", "reduce_op.h"],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "operations",
    srcs = ["broadcast_op.cc", "move_op.cc", "partial_reduce_op.cc", "reduce_op.cc"],
    deps = ["//main/operations:operations_headers",
            "//main/communication:communication_headers",
            "//main/tensor:tensor_headers",
            "//main/server:server_headers",
            "//main/ud_functions:ud_functions_headers",
            "//main/commands:commands_headers",
            "//main/utils:utils_headers",
            "//main/storage:storage_headers"],
    alwayslink = 1,
    visibility = ["//visibility:public"]
)