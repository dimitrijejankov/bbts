cc_library(
    name = "storage_headers",
    hdrs = ["block_allocator.h", "memory_storage.h", "nvme_storage.h", "storage.h"],
    deps = ["//main/communication:communication_headers", 
            "//main/commands:commands_headers"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "storage",
    srcs = ["memory_storage.cc", "nvme_storage.cc"],
    deps = ["//main/storage:storage_headers",
            "//main/tensor:tensor_headers",
            "//main/server:server_headers",
            "//main/ud_functions:ud_functions_headers",
            "//main/utils:utils_headers"],
    alwayslink = 1,
    visibility = ["//visibility:public"]
)