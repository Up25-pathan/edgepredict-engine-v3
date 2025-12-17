# ═══════════════════════════════════════════════════════════════
# Stage 1: Build Environment
# ═══════════════════════════════════════════════════════════════
FROM gcc:10-bullseye AS build
LABEL maintainer="EdgePredict Engineering"
LABEL version="12.0-HPC-Production"

# 1. Install Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libeigen3-dev \
    libtbb-dev \
    libocct-data-exchange-dev \
    libocct-foundation-dev \
    libocct-modeling-algorithms-dev \
    libocct-modeling-data-dev \
    libocct-ocaf-dev \
    libocct-visualization-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app
COPY . .

# 2. Compile (Notice: No GeometricAnalyzer)
RUN CXX_FLAGS="-std=c++17 -O3 -fopenmp -I./include -I/usr/include/eigen3 -I/usr/include/opencascade" && \
    g++ $CXX_FLAGS -c src/main.cpp -o main.o && \
    g++ $CXX_FLAGS -c src/simulation.cpp -o simulation.o && \
    g++ $CXX_FLAGS -c src/physics_models.cpp -o physics_models.o && \
    g++ $CXX_FLAGS -c src/navier_stokes.cpp -o navier_stokes.o && \
    g++ $CXX_FLAGS -c src/particle_system.cpp -o particle_system.o && \
    g++ $CXX_FLAGS -c src/MillingStrategy.cpp -o MillingStrategy.o && \
    g++ $CXX_FLAGS -c src/TurningStrategy.cpp -o TurningStrategy.o && \
    g++ $CXX_FLAGS -c src/RotationalFEA.cpp -o RotationalFEA.o && \
    g++ $CXX_FLAGS -c src/SPHWorkpieceModel.cpp -o SPHWorkpieceModel.o

# 3. Link
RUN g++ -std=c++17 -O3 -fopenmp \
    main.o simulation.o physics_models.o navier_stokes.o particle_system.o \
    MillingStrategy.o TurningStrategy.o RotationalFEA.o SPHWorkpieceModel.o \
    -lTKernel -lTKMath -lTKG2d -lTKG3d -lTKGeomBase \
    -lTKBRep -lTKGeomAlgo \
    -lTKTopAlgo -lTKMesh -lTKSTEP -lTKIGES -lTKXSBase -lTKSTL \
    -lTKShHealing \
    -o /usr/local/bin/edgepredict-engine

# ═══════════════════════════════════════════════════════════════
# Stage 2: Runtime
# ═══════════════════════════════════════════════════════════════
FROM debian:bullseye-slim

RUN apt-get update && apt-get install -y \
    libgomp1 \
    libtbb2 \
    libocct-data-exchange-7.5 \
    libocct-foundation-7.5 \
    libocct-modeling-algorithms-7.5 \
    libocct-modeling-data-7.5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/bin/edgepredict-engine /usr/local/bin/edgepredict-engine

RUN mkdir -p /data
WORKDIR /data

ENTRYPOINT ["/usr/local/bin/edgepredict-engine"]
