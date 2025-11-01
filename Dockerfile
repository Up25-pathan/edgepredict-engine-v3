# ═══════════════════════════════════════════════════════════════
# Stage 1: Build Environment
# ═══════════════════════════════════════════════════════════════
FROM gcc:10-bullseye AS build
LABEL maintainer="EdgePredict Engineering"
LABEL version="3.2-CAD-Source"

# 1. Install build tools and OpenCASCADE from Debian repos
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

# 2. Set up the engine build directory
WORKDIR /usr/src/app

# Copy the entire project source code into the container
COPY . .

# 3. Compile the EdgePredict Engine
# Point to system OpenCASCADE library
RUN CXX_FLAGS="-std=c++17 -O3 -fopenmp -I./include -I/usr/include/eigen3 -I/usr/include/opencascade" && \
    g++ $CXX_FLAGS -c src/main.cpp -o main.o && \
    g++ $CXX_FLAGS -c src/simulation.cpp -o simulation.o && \
    g++ $CXX_FLAGS -c src/physics_models.cpp -o physics_models.o && \
    g++ $CXX_FLAGS -c src/navier_stokes.cpp -o navier_stokes.o && \
    g++ $CXX_FLAGS -c src/particle_system.cpp -o particle_system.o

# 4. Link the EdgePredict Engine
RUN g++ -std=c++17 -O3 -fopenmp \
    main.o simulation.o physics_models.o navier_stokes.o particle_system.o \
    -lTKernel -lTKMath -lTKG2d -lTKG3d -lTKGeomBase -lTKBRep -lTKGeomAlgo \
    -lTKTopAlgo -lTKMesh -lTKSTEP -lTKIGES -lTKXSBase -lTKSTL \
    -o /usr/local/bin/edgepredict-engine

# ═══════════════════════════════════════════════════════════════
# Stage 2: Production Environment
# ═══════════════════════════════════════════════════════════════
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libtbb2 \
    libocct-data-exchange-7.5 \
    libocct-foundation-7.5 \
    libocct-modeling-algorithms-7.5 \
    libocct-modeling-data-7.5 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the compiled engine executable from the build stage
COPY --from=build /usr/local/bin/edgepredict-engine /usr/local/bin/edgepredict-engine

# Create the working directory for simulation data
RUN mkdir -p /data
WORKDIR /data

# Set the entrypoint for the container
ENTRYPOINT ["/usr/local/bin/edgepredict-engine"]
