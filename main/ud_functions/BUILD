cc_library(
    name = "ud_functions_headers",
    hdrs = ["builtin_functions.h",  
            "gpu_scheduler.h",  
            "gpu_scheduler_impl.h",  
            "null_gpu_scheduler.h",  
            "udf_manager.h",  
            "ud_function.h",
            "xla_kernel_base.h",
            "impls/dense_matrix_hadamard.h",  
            "impls/dense_matrix_relu_dif.h",  
            "impls/dense_matrix_sigmoid.h",       
            "impls/dense_uniform.h",
            "impls/dense_matrix_add.h",  
            "impls/dense_matrix_mult.h",      
            "impls/dense_matrix_relu.h",      
            "impls/dense_matrix_weighted_sum.h"],
    deps = ["//main/tensor:tensor_headers", "@org_tensorflow//tensorflow/compiler/xla:tos_cpu_kernel"],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "ud_functions",
    srcs = ["builtin_functions.cc",  
            "gpu_scheduler_impl.cc",  
            "udf_manager.cc", 
            "ud_function.cc", 
            "xla_kernel_base.cc",
            "impls/dense_matrix_add.cc",       
            "impls/dense_matrix_mult.cc",   
            "impls/dense_matrix_relu_dif.cc",  
            "impls/dense_matrix_weighted_sum.cc",
            "impls/dense_matrix_hadamard.cc", 
            "impls/dense_matrix_relu.cc",  
            "impls/dense_matrix_sigmoid.cc",   
            "impls/dense_uniform.cc"],
    deps = ["//main/ud_functions:ud_functions_headers",
            "//main/tensor:tensor_headers",
            "//main/commands:commands_headers",
            "//main/utils:utils_headers",
            "@libmkl",
            "@libmkl_headers"],
    alwayslink = 1,
    visibility = ["//visibility:public"]
)
