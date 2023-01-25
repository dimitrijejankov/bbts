cc_library(
    name = "server_headers",
    hdrs = ["coordinator.h", "coordinator_ops.h", "logger.h", "node_config.h", "node.h", "static_config.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "server",
    srcs = ["coordinator.cc", "node.cc"],
    deps = ["//main/server:server_headers", 
            "//main/utils:utils_headers", 
            "//main/commands:commands_headers", 
            "//main/ud_functions:ud_functions_headers", 
            "//main/storage:storage_headers"],
    alwayslink = 1,
    visibility = ["//visibility:public"],
)