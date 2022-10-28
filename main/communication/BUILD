cc_library(
    name = "communication_headers",
    hdrs = ["communicator.h"],
    deps = ["@libmpi_headers"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "communication",
    srcs = ["communicator.cc"],
    deps = ["//main/communication:communication_headers",
            "//main/tensor:tensor_headers",
            "//main/server:server_headers",
            "//main/commands:commands_headers",
            "//main/utils:utils_headers",
            "//main/ud_functions:ud_functions_headers",
            "@libmpi"],
    alwayslink = 1,
    visibility = ["//visibility:public"]
)
