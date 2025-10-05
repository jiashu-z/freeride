# 
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 
# ---[ Check cmake version.
CMAKE_MINIMUM_REQUIRED(VERSION 3.10.0 FATAL_ERROR)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_COMPILER /root/miniconda3/envs/freeride/bin/nvcc)

INCLUDE(GNUInstallDirs)

# ---[ Project specIFication.
SET(PROJECT_NAME imageResizeWatermark)
PROJECT(${PROJECT_NAME} LANGUAGES CUDA CXX)

include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

#if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#endif()
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Protobuf_DIR "/root/.local/lib/cmake/protobuf")
include(common.cmake)

message("${CMAKE_CURRENT_BINARY_DIR}")

get_filename_component(task_proto "task.proto" ABSOLUTE)
get_filename_component(task_proto_path "${task_proto}" PATH)
get_filename_component(scheduler_proto "scheduler.proto" ABSOLUTE)
get_filename_component(scheduler_proto_path "${scheduler_proto}" PATH)

message("${task_proto}")
message("${task_proto_path}")
message("${scheduler_proto}")
message("${scheduler_proto_path}")

# Generated sources
set(task_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/task.pb.cc")
set(task_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/task.pb.h")
set(task_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/task.grpc.pb.cc")
set(task_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/task.grpc.pb.h")
set(scheduler_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/scheduler.pb.cc")
set(scheduler_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/scheduler.pb.h")
set(scheduler_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/scheduler.grpc.pb.cc")
set(scheduler_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/scheduler.grpc.pb.h")
add_custom_command(
      OUTPUT "${task_proto_srcs}" "${task_proto_hdrs}" "${task_grpc_srcs}" "${task_grpc_hdrs}" "${scheduler_proto_srcs}" "${scheduler_proto_hdrs}" "${scheduler_grpc_srcs}" "${scheduler_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${task_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${task_proto}" "${scheduler_proto}"
      DEPENDS "${task_proto}" "${scheduler_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
link_directories(${CMAKE_CUDA_INCLUDE_DIRS})

if (UNIX)
    find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(NVJPEG_LIBRARY nvjpeg ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(NPPIG_LIBRARY nppig ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(NPPIAL_LIBRARY nppial ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(NPPC_LIBRARY nppc ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(CULIBOS culibos ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif (UNIX)

if (MSVC OR WIN32 OR MSYS)
    MESSAGE( STATUS "CMAKE_CUDA_COMPILER:         " ${CMAKE_CUDA_COMPILER} )
    set(CUDA_LIBRARY_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64")
    set(CUDART_LIBRARY ${CUDA_LIBRARY_PATH}/cudart.lib)
    set(NVJPEG_LIBRARY ${CUDA_LIBRARY_PATH}/nvjpeg.lib)
    set(NPPIG_LIBRARY ${CUDA_LIBRARY_PATH}/nppig.lib)
    set(NPPIAL_LIBRARY ${CUDA_LIBRARY_PATH}/nppial.lib)
    set(NPPC_LIBRARY ${CUDA_LIBRARY_PATH}/nppc.lib)
endif()

# ---[ Use the default installation path if not set.
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# ---[ Build type
IF(NOT CMAKE_BUILD_TYPE) 
    SET(CMAKE_BUILD_TYPE Develop)
ENDIF(NOT CMAKE_BUILD_TYPE)

SET(EXAMPLES_DESCRIPTOR_SOURCES "imageResizeWatermark.cpp")
ADD_EXECUTABLE(${PROJECT_NAME} ${EXAMPLES_DESCRIPTOR_SOURCES})
SET_SOURCE_FILES_PROPERTIES(${EXAMPLES_DESCRIPTOR_SOURCES} PROPERTIES LANGUAGE CUDA)
TARGET_COMPILE_FEATURES(${PROJECT_NAME} PUBLIC cxx_std_11)
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
if (UNIX)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC ${CUDART_LIBRARY} ${NVJPEG_LIBRARY} ${NPPIG_LIBRARY} ${NPPC_LIBRARY} ${NPPIAL_LIBRARY} ${CULIBOS})
endif (UNIX)

if(MSVC OR WIN32 OR MSYS)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC ${CUDART_LIBRARY} ${NVJPEG_LIBRARY} ${NPPIG_LIBRARY} ${NPPC_LIBRARY} ${NPPIAL_LIBRARY})
endif(MSVC OR WIN32 OR MSYS)

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET ${PROJECT_NAME} 
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif(APPLE)

INSTALL(TARGETS ${PROJECT_NAME} DESTINATION bin)

# task_grpc_proto
add_library(task_grpc_proto
  ${task_grpc_srcs}
  ${task_grpc_hdrs}
  ${task_proto_srcs}
  ${task_proto_hdrs})
target_link_libraries(task_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

# scheduler_grpc_proto
add_library(scheduler_grpc_proto
  ${scheduler_grpc_srcs}
  ${scheduler_grpc_hdrs}
  ${scheduler_proto_srcs}
  ${scheduler_proto_hdrs})
target_link_libraries(scheduler_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

add_executable(image_resize_watermark_side_task
image_resize_watermark_side_task.cu
)
set_target_properties(image_resize_watermark_side_task PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(image_resize_watermark_side_task
PRIVATE include
PRIVATE /home/jiashu/env/miniconda3/envs/bubblebandit/include/cub
)
target_link_libraries(image_resize_watermark_side_task
PRIVATE argparse
PRIVATE task_grpc_proto
PRIVATE scheduler_grpc_proto
)
if (UNIX)
    TARGET_LINK_LIBRARIES(image_resize_watermark_side_task PUBLIC ${CUDART_LIBRARY} ${NVJPEG_LIBRARY} ${NPPIG_LIBRARY} ${NPPC_LIBRARY} ${NPPIAL_LIBRARY} ${CULIBOS})
endif (UNIX)

if(MSVC OR WIN32 OR MSYS)
    TARGET_LINK_LIBRARIES(image_resize_watermark_side_task PUBLIC ${CUDART_LIBRARY} ${NVJPEG_LIBRARY} ${NPPIG_LIBRARY} ${NPPC_LIBRARY} ${NPPIAL_LIBRARY})
endif(MSVC OR WIN32 OR MSYS)

add_executable(playground
playground.cpp)
