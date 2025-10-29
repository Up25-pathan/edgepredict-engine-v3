# EdgePredict Engine v2.0 - Production Build
# Tier 1 + 2 CFD Integration
FROM gcc:latest

# Install dependencies
RUN apt-get update && apt-get install -y \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set build directory
WORKDIR /usr/src/app

# Copy project files
COPY . .

# Compile with all modules
# Order matters: compile implementation files first, then link
RUN g++ -I./include -I/usr/include/eigen3 \
    -std=c++17 -O3 -fopenmp \
    -c src/main.cpp -o main.o && \
    g++ -I./include -I/usr/include/eigen3 \
    -std=c++17 -O3 -fopenmp \
    -c src/physics_models.cpp -o physics_models.o && \
    g++ -I./include -I/usr/include/eigen3 \
    -std=c++17 -O3 -fopenmp \
    -c src/navier_stokes.cpp -o navier_stokes.o && \
    g++ -I./include -I/usr/include/eigen3 \
    -std=c++17 -O3 -fopenmp \
    -c src/particle_system.cpp -o particle_system.o && \
    g++ -I./include -I/usr/include/eigen3 \
    -std=c++17 -O3 -fopenmp \
    -c src/simulation.cpp -o simulation.o && \
    g++ -std=c++17 -O3 -fopenmp \
    main.o physics_models.o navier_stokes.o particle_system.o simulation.o \
    -o edgepredict-engine

# Create working directory for input/output
RUN mkdir -p /data

# Set the working directory to /data
WORKDIR /data

# Set the entrypoint
ENTRYPOINT ["/usr/src/app/edgepredict-engine"]