cc_library(
    name = "ffnn_common_lib",
    srcs = [
        "ffnn_activation_mult.cc",  
        "ffnn_back_mult.cc",  
        "ffnn_matrix_hadamard.cc",  
        "ffnn_types.cc",         
        "ffnn_uniform_sparse_data.cc",  
        "ffnn_weighted_sum.cc",               
        "ffnn_add.cc",              
        "ffnn_lib_init.cc",   
        "ffnn_mult.cc",            
        "ffnn_uniform_data.cc",  
        "ffnn_uniform_weights.cc",      
        "ffnn_weighted_sum_sparse_dense.cc"
    ],
    hdrs = ["ffnn_activation_mult.h", 
            "ffnn_back_mult.h",        
            "ffnn_mult.h",   
            "ffnn_uniform_data.h",         
            "ffnn_uniform_weights.h",  
            "ffnn_weighted_sum_sparse_dense.h",
            "ffnn_add.h",              
            "ffnn_matrix_hadamard.h",  
            "ffnn_types.h",  
            "ffnn_uniform_sparse_data.h",  
            "ffnn_weighted_sum.h"],
    deps = ["//main:bbts_lib", "//third_party:cli_headers"],
    alwayslink = 1,
)

cc_binary(
    name = "ffnn_lib.so",
    deps = [":ffnn_common_lib"],
    linkshared = True
)

cc_binary(
    name = "generate_ffnn",
    srcs = ["generate_ffnn.cc"],
    deps = [":ffnn_common_lib"],
)