#
# dog-cat-classification

set(TEST_TARGET dog-cat-classification)
add_library(${TEST_TARGET} STATIC dog-cat-classification.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)