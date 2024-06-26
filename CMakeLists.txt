cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(yolov8_libtorch VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for use with tools like clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------- OpenCV --------------
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})

# -------------- libtorch --------------
list(APPEND CMAKE_PREFIX_PATH "/home/jh/libtorch/")
set(Torch_DIR "/home/jh/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "Torch library status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")
message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}")

# -------------- YOLOv8_LT --------------

# Add executable and link libraries
add_executable(yolov8_libtorch_inference 
"${CMAKE_CURRENT_SOURCE_DIR}/main.cc"
"${CMAKE_CURRENT_SOURCE_DIR}/src/YOLOv8_LT.cpp"
)

target_include_directories(yolov8_libtorch_inference PUBLIC
    ${TORCH_INCLUDE_DIRS}
)

target_include_directories(yolov8_libtorch_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(yolov8_libtorch_inference 
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

set_property(TARGET yolov8_libtorch_inference PROPERTY CXX_STANDARD 17)


# -------------- Google Test --------------
enable_testing()
find_package(GTest REQUIRED)

set(GTEST_INCLUDE_DIRS /usr/local/include)
set(GTEST_LIBRARIES /usr/local/lib/libgtest.a /usr/local/lib/libgtest_main.a)

message(STATUS "Google Test found: ${GTEST_INCLUDE_DIRS}")

add_executable(runTests 
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_YOLOv8_LT.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/YOLOv8_LT.cpp"
)

target_include_directories(runTests PUBLIC
    ${TORCH_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(runTests 
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    ${GTEST_LIBRARIES}
    pthread
)

add_test(NAME runTests COMMAND runTests)


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET yolov8_libtorch
#                      POST_BUILD
#                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      ${TORCH_DLLS}
#                      $<TARGET_FILE_DIR:yolov8_libtorch>)
# endif (MSVC)