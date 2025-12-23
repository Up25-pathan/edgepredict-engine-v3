# Use Intel OneAPI Base Toolkit (Contains icpx compiler for SYCL)
FROM intel/oneapi-basekit:latest

# Install System Dependencies (OpenCASCADE, Eigen, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencascade-dev \
    libeigen3-dev \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Working Directory
WORKDIR /app

# Copy Source Code
COPY . /app

# Create Build Directory
RUN mkdir -p build

# --------------------------------------------------------------------------
# COMPILE THE ENGINE
# --------------------------------------------------------------------------
# Critical Update: Added OptimizationManager.cpp and EnergyMonitor.cpp
# --------------------------------------------------------------------------
RUN icpx -fsycl -std=c++17 -O3 \
    -I./include \
    -I/usr/include/eigen3 \
    -I/usr/include/opencascade \
    src/main.cpp \
    src/simulation.cpp \
    src/physics_models.cpp \
    src/navier_stokes.cpp \
    src/MillingStrategy.cpp \
    src/TurningStrategy.cpp \
    src/RotationalFEA.cpp \
    src/SPHWorkpieceModel.cpp \
    src/OptimizationManager.cpp \
    src/EnergyMonitor.cpp \
    -lTKernel -lTKMath -lTKG2d -lTKG3d -lTKGeomBase \
    -lTKBRep -lTKGeomAlgo -lTKTopAlgo -lTKMesh \
    -lTKSTEP -lTKIGES -lTKXSBase -lTKSTL -lTKShHealing \
    -o build/edgepredict-engine

# Default Command
CMD ["./build/edgepredict-engine", "input.json"]