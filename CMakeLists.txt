cmake_minimum_required(VERSION 3.18)
project(EdgePredictEngine LANGUAGES CXX CUDA)

# =============================================================================
# EdgePredict Engine v4 - Clean Architecture Build
# =============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Allow CUDA architecture override from command line
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "120")  # RTX 5060 Ti (Blackwell SM 12.0)
endif()

# CUDA flags for device lambdas and relaxed constexpr
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")

# =============================================================================
# Dependencies
# =============================================================================

# Eigen3 for linear algebra (OPTIONAL - not actually used in source code)
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIRS}")
else()
    message(STATUS "Eigen3 not found (not required - engine uses custom Vec3)")
endif()

# HDF5 for data export (optional)
find_package(HDF5 COMPONENTS CXX)

# JSON - using nlohmann/json (header-only)
# Expected to be in include/json.hpp
# Download from: https://github.com/nlohmann/json/releases

# OpenCASCADE for CAD geometry
# Try to find package first
find_package(OpenCASCADE QUIET)
if(OpenCASCADE_FOUND)
    message(STATUS "OpenCASCADE found")
    add_definitions(-DHAS_OPENCASCADE)
else()
    # Try manual fallback for systems without CMake config
    if(WIN32)
        set(OpenCASCADE_INCLUDE_DIR "C:/OpenCASCADE/include/opencascade")
        if(EXISTS "${OpenCASCADE_INCLUDE_DIR}")
            set(OpenCASCADE_FOUND TRUE)
            add_definitions(-DHAS_OPENCASCADE)
            link_directories("C:/OpenCASCADE/win64/vc14/lib")
        else()
            message(STATUS "OpenCASCADE NOT found - STEP/IGES loading disabled, STL still works")
        endif()
    else()
        set(OpenCASCADE_INCLUDE_DIR "/usr/include/opencascade")
        if(EXISTS "${OpenCASCADE_INCLUDE_DIR}")
            set(OpenCASCADE_FOUND TRUE)
            add_definitions(-DHAS_OPENCASCADE)
            link_directories("/usr/lib/x86_64-linux-gnu")
        else()
            message(STATUS "OpenCASCADE NOT found - STEP/IGES loading disabled, STL still works")
        endif()
    endif()
endif()

# =============================================================================
# Include Directories
# =============================================================================

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

if(Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIRS})
endif()

if(OpenCASCADE_FOUND)
    include_directories(${OpenCASCADE_INCLUDE_DIR})
endif()

if(HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIRS})
    add_definitions(-DHAS_HDF5)
endif()

# =============================================================================
# Source Files
# =============================================================================

set(CPP_SOURCES
    src/main.cpp
    src/Config.cpp
    src/SimulationEngine.cpp
    src/GeometryLoader.cpp
    src/GCodeInterpreter.cpp
    src/VTKExporter.cpp
    src/HDF5Exporter.cpp
    src/StdoutIPCExporter.cpp
    src/OptimizationManager.cpp
    src/TurningStrategy.cpp
    src/MillingStrategy.cpp
    src/DrillingStrategy.cpp
    src/MachiningStrategyFactory.cpp
    src/SurfaceRoughnessPredictor.cpp
    src/ResidualStressAnalyzer.cpp
)

set(CUDA_SOURCES
    src/SPHSolver.cu
    src/FEMSolver.cu
    src/ContactSolver.cu
    src/CFDSolverGPU.cu
    src/AdiabaticShearModel.cu
    src/ToolCoatingModel.cu
)

# =============================================================================
# Executable
# =============================================================================

add_executable(edgepredict-engine ${CPP_SOURCES} ${CUDA_SOURCES})

# =============================================================================
# Compile Options
# =============================================================================

# NVTX for profiling
set(USE_NVTX ON)
if(USE_NVTX)
    add_definitions(-DUSE_NVTX)
    # NVTX headers are in the CUDA include directory, which is already included
    if(UNIX)
        target_link_libraries(edgepredict-engine PRIVATE nvToolsExt)
    endif()
endif()

if(WIN32)
    add_definitions(-DNOMINMAX)
endif()

# Output name
set_target_properties(edgepredict-engine PROPERTIES OUTPUT_NAME "edgepredict-engine")

# CUDA separable compilation
set_target_properties(edgepredict-engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Hardware specific optimization flags
if(MSVC)
    target_compile_options(edgepredict-engine PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:/O2 /arch:AVX2 /W3 /Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcompiler=/O2 -Xcompiler=/Zc:preprocessor>
    )
else()
    target_compile_options(edgepredict-engine PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcompiler="-O3">
    )
endif()

# =============================================================================
# Link Libraries
# =============================================================================

# Eigen3 (optional)
if(Eigen3_FOUND)
    target_link_libraries(edgepredict-engine PRIVATE Eigen3::Eigen)
endif()

# CUDA runtime
target_link_libraries(edgepredict-engine PRIVATE
    ${CUDA_LIBRARIES}
)

# OpenCASCADE libraries (optional - for STEP/IGES loading)
if(OpenCASCADE_FOUND)
    target_link_libraries(edgepredict-engine PRIVATE
        TKernel TKMath TKG2d TKG3d TKGeomBase
        TKBRep TKGeomAlgo TKTopAlgo TKMesh
        TKSTEP TKIGES TKXSBase TKSTL TKShHealing
    )
endif()

# HDF5 (if found)
if(HDF5_FOUND)
    target_link_libraries(edgepredict-engine PRIVATE ${HDF5_LIBRARIES})
endif()

# =============================================================================
# Installation
# =============================================================================

install(TARGETS edgepredict-engine DESTINATION bin)

# =============================================================================
# Print Configuration Summary
# =============================================================================

message(STATUS "=== EdgePredict Engine v4 Build Configuration ===")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Eigen3: ${Eigen3_FOUND}")
message(STATUS "HDF5: ${HDF5_FOUND}")
message(STATUS "OpenCASCADE: ${OpenCASCADE_FOUND}")
