# Use Intel OneAPI Base Toolkit (Contains icpx compiler for SYCL)
FROM intel/oneapi-basekit:latest

# Install System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libocct-data-exchange-dev \
    libocct-foundation-dev \
    libocct-modeling-algorithms-dev \
    libocct-visualization-dev \
    libocct-ocaf-dev \
    libeigen3-dev \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Working Directory
WORKDIR /app

# Copy Source Code
COPY . /app

# Create Build Directory
RUN mkdir -p build

# COMPILE THE ENGINE
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
    src/GCodeInterpreter.cpp \
    -lTKernel -lTKMath -lTKG2d -lTKG3d -lTKGeomBase \
    -lTKBRep -lTKGeomAlgo -lTKTopAlgo -lTKMesh \
    -lTKSTEP -lTKIGES -lTKXSBase -lTKSTL -lTKShHealing \
    -o build/edgepredict-engine

# ENTRYPOINT ensures the executable is always run
ENTRYPOINT ["./build/edgepredict-engine"]

# Default command (can be overridden by user)
CMD ["input.json"]