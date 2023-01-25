cc_library(
    name = "commands_headers",
    hdrs = ["abstract_command.h",  
            "command_notifier.h",  
            "command_utils.h",        
            "cost_model.h",      
            "reorder_buffer.h",       
            "two_layer_compiler.h",
            "command.h",           
            "command_loader.h",   
            "command_runner.h",    
            "compile_source_file.h",  
            "parsed_command.h", 
            "reservation_station.h",
            "command_handlers/command_handler_apply.h",  
            "command_handlers/command_handler.h",
            "command_handlers/command_handler_reduce.h",
            "command_handlers/command_handler_delete.h", 
            "command_handlers/command_handler_move.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "commands",
    srcs = ["command_notifier.cc",  
            "command_runner.cc",  
            "reorder_buffer.cc",  
            "reservation_station.cc",
            "command_handlers/command_handler_apply.cc",  
            "command_handlers/command_handler_reduce.cc",
            "command_handlers/command_handler_delete.cc", 
            "command_handlers/command_handler_move.cc"],
    deps = ["//main/commands:commands_headers", 
            "//main/server:server_headers",
            "//main/utils:utils_headers",
            "//main/ud_functions:ud_functions_headers",
            "//main/operations:operations_headers",
            "//main/storage:storage_headers"],
    alwayslink = 1,
    visibility = ["//visibility:public"]

)

