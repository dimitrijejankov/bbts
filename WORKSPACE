new_local_repository(
    name = "libmkl",
    path = "/opt/intel/oneapi/mkl/latest/",
    build_file = "//third_party/mkl:libmkl.BUILD"
)

new_local_repository(
    name = "libmkl_headers",
    path = "/opt/intel/oneapi/mkl/latest/",
    build_file = "//third_party/mkl:libmkl_headers.BUILD"
)

new_local_repository(
    name = "libmpi",
    path = "/usr/lib/x86_64-linux-gnu/openmpi/",
    build_file = "//third_party/openmpi:libmpi.BUILD"
)

new_local_repository(
    name = "libmpi_headers",
    path = "/usr/lib/x86_64-linux-gnu/openmpi/",
    build_file = "//third_party/openmpi:libmpi_headers.BUILD"
)

local_repository(
    name = "org_tensorflow",
    path = "third_party/tensorflow",
)

load("//third_party/ducc:workspace.bzl", ducc = "repo")
ducc()

# Initialize TensorFlow's external dependencies.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")
tf_workspace0()
