cc_binary(
    name = "node",
    srcs = ["node_cli.cc"],
    deps = ["//main:bbts_lib",
            "//third_party:cli_headers",],
    linkopts = ["-ldl -lpthread -lstdc++fs"],
)