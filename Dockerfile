# ═══════════════════════════════════════════════════════════════
# Stage 1: Build Environment
# ═══════════════════════════════════════════════════════════════
#
# Use the GCC 10 image. This stage will be large and slow, but
# it only runs once.
FROM gcc:10-bullseye AS build
LABEL maintainer="EdgePredict Engineering"
LABEL version="3.2-CAD-Source"

# 1. Install build tools for OpenCASCADE and the engine
# --- FIX: Added ALL build dependencies for OCCT, including OpenGL ---
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libeigen3-dev \
    libtbb-dev \
    libfreetype-dev \
    tcl-dev \
    tk-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Download OpenCASCADE source code
# We use a specific, stable tag (7.5.0) to match the runtime
WORKDIR /usr/src/
RUN git clone --depth 1 --branch V7_5_0 https://github.com/Open-Cascade-SAS/OCCT.git

# 3. Configure and build OpenCASCADE
# We will install it to /usr/local/occt to keep it separate.
# This will take a long time (30-60 minutes).
WORKDIR /usr/src/OCCT
RUN cmake . -DCMAKE_INSTALL_PREFIX=/usr/local/occt -DBUILD_MODULE_Draw=OFF
RUN make -j$(nproc) install

# 4. Set up the engine build directory
WORKDIR /usr/src/app

# Copy the entire project source code into the container
COPY . .

# 5. Compile the EdgePredict Engine
# We now point CXX_FLAGS to our self-built OpenCASCADE library
RUN CXX_FLAGS="-std=c++17 -O3 -fopenmp -I./include -I/usr/include/eigen3 -I/usr/local/occt/include/opencascade" && \
    g++ $CXX_FLAGS -c src/main.cpp -o main.o && \
    g++ $CXX_FLAGS -c src/simulation.cpp -o simulation.o && \
    g++ $CXX_FLAGS -c src/physics_models.cpp -o physics_models.o && \
    g++ $CXX_FLAGS -c src/navier_stokes.cpp -o navier_stokes.o && \
    g++ $CXX_FLAGS -c src/particle_system.cpp -o particle_system.o

# 6. Link the EdgePredict Engine
# We tell the linker to look in our self-built library path
RUN g++ -std=c++17 -O3 -fopenmp \
    main.o simulation.o physics_models.o navier_stokes.o particle_system.o \
    -L/usr/local/occt/lib \
    -lTKernel -lTKMath -lTKG2d -lTKG3d -lTKGeomBase -lTKBRep -lTKGeomAlgo \
    -lTKTopAlgo -lTKMesh -lTKSTEP -lTKIges -lTKXSBase -lTKStl \
    -o /usr/local/bin/edgepredict-engine

# ═══════════════════════════════════════════════════════════════
# Stage 2: Production Environment
# ═══════════════════════════════════════════════════════════════
#
# Use the matching "bullseye-slim" image for runtime
FROM debian:bullseye-slim

# Install runtime dependencies for OCCT and OpenMP
# --- FIX: Added runtime libraries for OpenGL ---
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libtbb2 \
    libfreetype6 \
    tcl \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled OpenCASCADE libraries from the build stage
COPY --from=build /usr/local/occt/lib/*.so* /usr/local/lib/

# Copy only the compiled engine executable from the build stage
COPY --from=build /usr/local/bin/edgepredict-engine /usr/local/bin/edgepredict-engine

# Run ldconfig to make the system aware of the new libraries
RUN ldconfig

# Create the working directory for simulation data
RUN mkdir -p /data
WORKDIR /data

# Set the entrypoint for the container
ENTRYPOINT ["/usr/local/bin/edgepredict-engine"]
