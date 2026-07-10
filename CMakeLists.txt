cmake_minimum_required(VERSION 3.10)
project(HailoRoiDetector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# 1. 构建类型与默认配置
# =============================================================================
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# 多配置生成器（如 Visual Studio、Ninja Multi-Config）支持
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "" FORCE)

# =============================================================================
# 2. 编译器选项 —— Debug 模式（GDB 调试优化）
# =============================================================================
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fno-omit-frame-pointer -Wall -Wextra ")
set(CMAKE_C_FLAGS_DEBUG   "-g -O0 -fno-omit-frame-pointer -Wall -Wextra ")

# 生成更丰富的调试信息（GCC/Clang）
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb3")
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} -ggdb3")
    # 调试宏定义
    add_compile_definitions($<$<CONFIG:Debug>:DEBUG>)
endif()

# =============================================================================
# 3. 编译器选项 —— Release 模式（编译速度 + 运行速度优化）
# =============================================================================
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -Wextra")
set(CMAKE_C_FLAGS_RELEASE   "-O3 -DNDEBUG -Wall -Wextra")

# 6. 构建信息输出
# =============================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "  Project: ${PROJECT_NAME}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "========================================")
message(STATUS "")

# ============================================================
# 1. 3rdparty: utility_tool
# ============================================================
add_subdirectory(3rdparty/utility_tool)
include_directories(3rdparty/utility_tool/include)

# ============================================================
# 1. OpenCV
# ============================================================
find_package(OpenCV 4.12 QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    find_package(OpenCV REQUIRED)
    message(WARNING "OpenCV >= 4.12 not found, using ${OpenCV_VERSION}")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

# ============================================================
# 1. OpenMP
# ============================================================
find_package(OpenMP REQUIRED)

# ============================================================
# 1. OpenCL (optional)
# ============================================================
find_package(OpenCL QUIET)
if(NOT OpenCL_FOUND)
    find_library(
    OpenCL_LIBRARY
    NAMES OpenCL libOpenCL.so.1
    PATHS /usr/lib /usr/local/lib /lib /usr/lib/aarch64-linux-gnu
          /usr/lib/x86_64-linux-gnu
    NO_DEFAULT_PATH)
    find_path(
    OpenCL_INCLUDE_DIR
    NAMES CL/cl.h
    PATHS /usr/include /usr/local/include)
    if(OpenCL_LIBRARY AND OpenCL_INCLUDE_DIR)
        set(OpenCL_FOUND TRUE)
        set(OpenCL_LIBRARIES ${OpenCL_LIBRARY})
        set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIR})
    endif()
endif()

if(OpenCL_FOUND)
    include_directories(${OpenCL_INCLUDE_DIRS})
    message(STATUS "OpenCL: ENABLED")
else()
    message(STATUS "OpenCL: DISABLED")
endif()

# ============================================================
# 1. librga：先系统，后 3rdparty
# ============================================================
find_library(
  RGA_LIBRARY
  NAMES rga
  PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu)
if(NOT RGA_LIBRARY)
    message(STATUS "System librga not found, falling back to 3rdparty")
    find_library(
    RGA_LIBRARY
    NAMES rga
    PATHS ${CMAKE_SOURCE_DIR}/3rdparty/librga/lib/aarch64-linux-gnu
    NO_DEFAULT_PATH)
endif()

find_path(
  RGA_INCLUDE_DIR
  NAMES im2d.hpp
  PATHS /usr/include/rga /usr/local/include/rga)
if(NOT RGA_INCLUDE_DIR)
    find_path(
    RGA_INCLUDE_DIR
    NAMES im2d.hpp
    PATHS ${CMAKE_SOURCE_DIR}/3rdparty/librga/include/rga
    NO_DEFAULT_PATH)
endif()

if(RGA_LIBRARY AND RGA_INCLUDE_DIR)
    include_directories(${RGA_INCLUDE_DIR})
    message(STATUS "librga include: ${RGA_INCLUDE_DIR}")
    message(STATUS "librga lib: ${RGA_LIBRARY}")
    set(RGA_FOUND TRUE)
else()
    message(WARNING "librga not found (system or 3rdparty)")
    set(RGA_FOUND FALSE)
endif()

# ============================================================
# 1. rockchip_mpp：先系统，后 3rdparty
# ============================================================
find_library(
  MPP_LIBRARY
  NAMES rockchip_mpp
  PATHS /usr/local/lib /usr/lib /usr/lib/aarch64-linux-gnu)
if(NOT MPP_LIBRARY)
    message(STATUS "System rockchip_mpp not found, falling back to 3rdparty")
    find_library(
    MPP_LIBRARY
    NAMES rockchip_mpp
    PATHS ${CMAKE_SOURCE_DIR}/3rdparty/mpp/lib/aarch64-linux-gnu
    NO_DEFAULT_PATH)
endif()

find_path(
  MPP_INCLUDE_DIR
  NAMES rockchip/mpp_frame.h
  PATHS /usr/local/include /usr/include)
if(NOT MPP_INCLUDE_DIR)
    find_path(
    MPP_INCLUDE_DIR
    NAMES rockchip/mpp_frame.h
    PATHS ${CMAKE_SOURCE_DIR}/3rdparty/mpp/include
    NO_DEFAULT_PATH)
endif()

if(MPP_LIBRARY AND MPP_INCLUDE_DIR)
    include_directories(${MPP_INCLUDE_DIR})
    message(STATUS "rockchip_mpp include: ${MPP_INCLUDE_DIR}")
    message(STATUS "rockchip_mpp lib: ${MPP_LIBRARY}")
    set(MPP_FOUND TRUE)
else()
    message(WARNING "rockchip_mpp not found (system or 3rdparty)")
    set(MPP_FOUND FALSE)
endif()

# ============================================================
# 1. HailoRT：先系统，后 3rdparty
# ============================================================
find_library(
  HAILO_RT_LIB
  NAMES hailort
  PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu)
if(NOT HAILO_RT_LIB)
    message(STATUS "System HailoRT not found, falling back to 3rdparty")
    find_library(
    HAILO_RT_LIB
    NAMES hailort
    PATHS ${CMAKE_SOURCE_DIR}/3rdparty/hailort/lib
    NO_DEFAULT_PATH)
endif()

find_path(
  HAILO_RT_INCLUDE
  NAMES hailo/hailort.h
  PATHS /usr/include /usr/local/include)
if(NOT HAILO_RT_INCLUDE)
    find_path(
    HAILO_RT_INCLUDE
    NAMES hailo/hailort.h
    PATHS ${CMAKE_SOURCE_DIR}/3rdparty/hailort/include
    NO_DEFAULT_PATH)
endif()

if(HAILO_RT_LIB AND HAILO_RT_INCLUDE)
    set(HAILO_RT_FOUND TRUE)
    include_directories(${HAILO_RT_INCLUDE})
    message(STATUS "HailoRT include: ${HAILO_RT_INCLUDE}")
    message(STATUS "HailoRT lib: ${HAILO_RT_LIB}")
else()
    set(HAILO_RT_FOUND FALSE)
    message(WARNING "HailoRT not found (system or 3rdparty)")
endif()

# ============================================================
# 1. 公共库
# ============================================================
set(COMMON_LIBS ${OpenCV_LIBS} utility_tool OpenMP::OpenMP_CXX)
if(OpenCL_FOUND)
    list(APPEND COMMON_LIBS ${OpenCL_LIBRARIES})
endif()

# ============================================================
# 1. Include 目录
# ============================================================
include_directories(${CMAKE_SOURCE_DIR}/include)

# ============================================================
# 1. 源文件
# ============================================================
set(SOURCES src/camera.cpp src/rga_engine.cpp src/yolo_detector.cpp
            src/rga_visualizer.cpp src/display_thread.cpp)

# ============================================================
# 1. 目标：只要库找到就编译，不判断平台
# ============================================================
if(HAILO_RT_FOUND
   AND MPP_FOUND
   AND RGA_FOUND)
    add_library(${PROJECT_NAME} ${SOURCES})
    target_link_libraries(${PROJECT_NAME} ${COMMON_LIBS} ${HAILO_RT_LIB}
                        ${MPP_LIBRARY} ${RGA_LIBRARY})

    add_executable(roi_detect_hailo_main src/main.cpp)
    target_link_libraries(roi_detect_hailo_main ${PROJECT_NAME})

    target_compile_options(roi_detect_hailo_main PRIVATE ${OpenMP_CXX_FLAGS})
    message(STATUS "Target roi_detect_hailo_main: ENABLED")

    # test
    add_executable(test_full_image ./src/test/test_full_image.cpp)
    target_link_libraries(test_full_image ${COMMON_LIBS} ${HAILO_RT_LIB}
                        ${MPP_LIBRARY} ${RGA_LIBRARY})

    add_executable(test_tiling_batch ./src/test/test_tiling_batch.cpp)
    target_link_libraries(test_tiling_batch ${COMMON_LIBS} ${HAILO_RT_LIB}
                        ${MPP_LIBRARY} ${RGA_LIBRARY})

    add_executable(test_camera ./src/test/test_camera.cpp)
    target_link_libraries(test_camera ${PROJECT_NAME})

    add_executable(test_camera_rga ./src/test/test_camera_rga.cpp)
    target_link_libraries(test_camera_rga ${PROJECT_NAME})
else()
    message(
    STATUS "Target roi_detect_hailo_main: DISABLED (missing dependencies)")
    if(NOT HAILO_RT_FOUND)
        message(STATUS "  - HailoRT missing")
    endif()
    if(NOT MPP_FOUND)
        message(STATUS "  - rockchip_mpp missing")
    endif()
    if(NOT RGA_FOUND)
        message(STATUS "  - librga missing")
    endif()
endif()
