file(REMOVE_RECURSE
  "libbbts-common.a"
  "libbbts-common.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA CXX)
  include(CMakeFiles/bbts-common.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
