/**
 * @file MultiGPUManager.cu
 * @brief Multi-GPU particle distribution implementation
 */

#include "MultiGPUManager.cuh"
#include <iostream>
#include <algorithm>
#include <cstring>

namespace edgepredict {

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void findGhostParticlesKernel(
    const float* positions, int numParticles,
    int* ghostFlags, int* ghostCount,
    float ghostMin, float ghostMax, int axis) {
    
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= numParticles) return;
    
    float pos = positions[pid * 3 + axis];
    
    // Check if particle is in ghost region
    bool isGhost = (pos >= ghostMin && pos <= ghostMax);
    ghostFlags[pid] = isGhost ? 1 : 0;
    
    if (isGhost) {
        atomicAdd(ghostCount, 1);
    }
}

__global__ void compactGhostParticlesKernel(
    const float* positions, const float* velocities,
    const int* ghostFlags, const int* ghostOffsets,
    float* ghostPositions, float* ghostVelocities,
    int numParticles) {
    
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= numParticles) return;
    
    if (ghostFlags[pid]) {
        int outIdx = ghostOffsets[pid];
        
        ghostPositions[outIdx * 3 + 0] = positions[pid * 3 + 0];
        ghostPositions[outIdx * 3 + 1] = positions[pid * 3 + 1];
        ghostPositions[outIdx * 3 + 2] = positions[pid * 3 + 2];
        
        ghostVelocities[outIdx * 3 + 0] = velocities[pid * 3 + 0];
        ghostVelocities[outIdx * 3 + 1] = velocities[pid * 3 + 1];
        ghostVelocities[outIdx * 3 + 2] = velocities[pid * 3 + 2];
    }
}

__global__ void findMigratingParticlesKernel(
    const float* positions, int numParticles,
    int* migrateFlags, int* migrateTarget,
    float partitionMin, float partitionMax, int axis,
    int partitionIdx, int numPartitions) {
    
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= numParticles) return;
    
    float pos = positions[pid * 3 + axis];
    
    migrateFlags[pid] = 0;
    migrateTarget[pid] = partitionIdx;
    
    if (pos < partitionMin && partitionIdx > 0) {
        migrateFlags[pid] = 1;
        migrateTarget[pid] = partitionIdx - 1;
    } else if (pos > partitionMax && partitionIdx < numPartitions - 1) {
        migrateFlags[pid] = 1;
        migrateTarget[pid] = partitionIdx + 1;
    }
}

// Helper kernel for particle distribution
__global__ void assignPartitionsKernel(
    const float* positions, int* partitionIds,
    int numParticles, int numPartitions,
    float domainMin, float domainMax, int axis) {
    
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= numParticles) return;
    
    float pos = positions[pid * 3 + axis];
    float range = domainMax - domainMin;
    float partitionSize = range / numPartitions;
    
    int partition = static_cast<int>((pos - domainMin) / partitionSize);
    partition = max(0, min(numPartitions - 1, partition));
    
    partitionIds[pid] = partition;
}

// ============================================================================
// MultiGPUManager Implementation
// ============================================================================

MultiGPUManager::MultiGPUManager() = default;

MultiGPUManager::~MultiGPUManager() {
    for (auto& p : m_partitions) {
        freePartitionMemory(p);
    }
}

int MultiGPUManager::queryDevices() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "[MultiGPU] No CUDA devices found" << std::endl;
        return 0;
    }
    
    m_devices.resize(deviceCount);
    
    std::cout << "[MultiGPU] Found " << deviceCount << " GPU(s):" << std::endl;
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        m_devices[i].deviceId = i;
        m_devices[i].name = prop.name;
        m_devices[i].totalMemory = prop.totalGlobalMem;
        m_devices[i].multiProcessorCount = prop.multiProcessorCount;
        
        size_t free, total;
        cudaSetDevice(i);
        cudaMemGetInfo(&free, &total);
        m_devices[i].freeMemory = free;
        
        std::cout << "  GPU " << i << ": " << prop.name 
                  << " (" << (prop.totalGlobalMem / 1024 / 1024) << " MB, "
                  << prop.multiProcessorCount << " SMs)" << std::endl;
    }
    
    return deviceCount;
}

bool MultiGPUManager::initialize(int numGPUs) {
    int available = queryDevices();
    
    if (available == 0) {
        std::cerr << "[MultiGPU] No GPUs available" << std::endl;
        return false;
    }
    
    m_numGPUs = (numGPUs <= 0) ? available : std::min(numGPUs, available);
    
    std::cout << "[MultiGPU] Initializing with " << m_numGPUs << " GPU(s)" << std::endl;
    
    // Setup peer-to-peer access
    setupPeerAccess();
    
    // Create partitions
    createPartitions();
    
    return true;
}

void MultiGPUManager::setupPeerAccess() {
    for (int i = 0; i < m_numGPUs; ++i) {
        for (int j = 0; j < m_numGPUs; ++j) {
            if (i == j) continue;
            
            int canAccess = 0;
            cudaDeviceCanAccessPeer(&canAccess, i, j);
            
            if (canAccess) {
                cudaSetDevice(i);
                cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                if (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled) {
                    m_devices[i].canAccessPeer[j] = true;
                    std::cout << "[MultiGPU] P2P access enabled: GPU " << i << " -> GPU " << j << std::endl;
                }
            }
        }
    }
}

void MultiGPUManager::setDomainBounds(const Vec3& minBound, const Vec3& maxBound) {
    m_domainMin = minBound;
    m_domainMax = maxBound;
    
    // Recreate partitions with new bounds
    if (!m_partitions.empty()) {
        createPartitions();
    }
}

void MultiGPUManager::createPartitions() {
    m_partitions.resize(m_numGPUs);
    
    float domainRange = 0;
    switch (m_decompAxis) {
        case 0: domainRange = m_domainMax.x - m_domainMin.x; break;
        case 1: domainRange = m_domainMax.y - m_domainMin.y; break;
        case 2: domainRange = m_domainMax.z - m_domainMin.z; break;
    }
    
    float partitionSize = domainRange / m_numGPUs;
    
    for (int i = 0; i < m_numGPUs; ++i) {
        m_partitions[i].deviceId = m_devices[i].deviceId;
        
        // Set bounds based on decomposition axis
        m_partitions[i].minBound = m_domainMin;
        m_partitions[i].maxBound = m_domainMax;
        
        float partMin = (m_decompAxis == 0 ? m_domainMin.x : 
                        (m_decompAxis == 1 ? m_domainMin.y : m_domainMin.z));
        
        float pMin = partMin + i * partitionSize;
        float pMax = partMin + (i + 1) * partitionSize;
        
        switch (m_decompAxis) {
            case 0:
                m_partitions[i].minBound.x = pMin;
                m_partitions[i].maxBound.x = pMax;
                break;
            case 1:
                m_partitions[i].minBound.y = pMin;
                m_partitions[i].maxBound.y = pMax;
                break;
            case 2:
                m_partitions[i].minBound.z = pMin;
                m_partitions[i].maxBound.z = pMax;
                break;
        }
        
        m_partitions[i].ghostWidth = m_ghostWidth;
        
        std::cout << "[MultiGPU] Partition " << i << ": ["
                  << pMin << ", " << pMax << "]" << std::endl;
    }
}

void MultiGPUManager::allocatePartitionMemory(GPUPartition& p, int maxParticles) {
    cudaSetDevice(p.deviceId);
    
    p.maxLocalParticles = maxParticles;
    p.maxGhostParticles = maxParticles / 10;  // 10% for ghost
    
    size_t particleSize = maxParticles * 3 * sizeof(float);
    size_t ghostSize = p.maxGhostParticles * 3 * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&p.d_positions, particleSize));
    CUDA_CHECK(cudaMalloc(&p.d_velocities, particleSize));
    CUDA_CHECK(cudaMalloc(&p.d_densities, maxParticles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p.d_pressures, maxParticles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p.d_temperatures, maxParticles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p.d_status, maxParticles * sizeof(int)));
    
    CUDA_CHECK(cudaMalloc(&p.d_ghostPositions, ghostSize));
    CUDA_CHECK(cudaMalloc(&p.d_ghostVelocities, ghostSize));
    
    std::cout << "[MultiGPU] GPU " << p.deviceId << ": Allocated " 
              << (particleSize * 2 + maxParticles * 4 * sizeof(float) + ghostSize * 2) / 1024 / 1024
              << " MB for " << maxParticles << " particles" << std::endl;
}

void MultiGPUManager::freePartitionMemory(GPUPartition& p) {
    if (p.d_positions) { cudaSetDevice(p.deviceId); cudaFree(p.d_positions); }
    if (p.d_velocities) cudaFree(p.d_velocities);
    if (p.d_densities) cudaFree(p.d_densities);
    if (p.d_pressures) cudaFree(p.d_pressures);
    if (p.d_temperatures) cudaFree(p.d_temperatures);
    if (p.d_status) cudaFree(p.d_status);
    if (p.d_ghostPositions) cudaFree(p.d_ghostPositions);
    if (p.d_ghostVelocities) cudaFree(p.d_ghostVelocities);
}

void MultiGPUManager::distributeParticles(const SPHParticle* particles, int numParticles) {
    std::cout << "[MultiGPU] Distributing " << numParticles << " particles across " 
              << m_numGPUs << " GPUs" << std::endl;
    
    // Count particles per partition
    std::vector<int> counts(m_numGPUs, 0);
    std::vector<std::vector<int>> particleIndices(m_numGPUs);
    
    for (int i = 0; i < numParticles; ++i) {
        Vec3 pos(particles[i].x, particles[i].y, particles[i].z);
        int partition = getPartitionForPosition(pos);
        counts[partition]++;
        particleIndices[partition].push_back(i);
    }
    
    // Allocate memory on each GPU
    for (int g = 0; g < m_numGPUs; ++g) {
        int maxLocal = static_cast<int>(counts[g] * 1.5);  // 50% headroom
        maxLocal = std::max(maxLocal, 10000);  // Minimum
        allocatePartitionMemory(m_partitions[g], maxLocal);
    }
    
    // Copy particles to each GPU
    for (int g = 0; g < m_numGPUs; ++g) {
        std::vector<float> positions(counts[g] * 3);
        std::vector<float> velocities(counts[g] * 3);
        
        for (int i = 0; i < counts[g]; ++i) {
            int srcIdx = particleIndices[g][i];
            positions[i * 3 + 0] = static_cast<float>(particles[srcIdx].x);
            positions[i * 3 + 1] = static_cast<float>(particles[srcIdx].y);
            positions[i * 3 + 2] = static_cast<float>(particles[srcIdx].z);
            velocities[i * 3 + 0] = static_cast<float>(particles[srcIdx].vx);
            velocities[i * 3 + 1] = static_cast<float>(particles[srcIdx].vy);
            velocities[i * 3 + 2] = static_cast<float>(particles[srcIdx].vz);
        }
        
        cudaSetDevice(m_partitions[g].deviceId);
        CUDA_CHECK(cudaMemcpy(m_partitions[g].d_positions, positions.data(),
                              counts[g] * 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_partitions[g].d_velocities, velocities.data(),
                              counts[g] * 3 * sizeof(float), cudaMemcpyHostToDevice));
        
        m_partitions[g].numLocalParticles = counts[g];
        
        std::cout << "[MultiGPU] GPU " << g << ": " << counts[g] << " particles" << std::endl;
    }
}

int MultiGPUManager::getPartitionForPosition(const Vec3& pos) const {
    float coord = 0;
    float range = 0;
    float minVal = 0;
    
    switch (m_decompAxis) {
        case 0: coord = pos.x; minVal = m_domainMin.x; range = m_domainMax.x - m_domainMin.x; break;
        case 1: coord = pos.y; minVal = m_domainMin.y; range = m_domainMax.y - m_domainMin.y; break;
        case 2: coord = pos.z; minVal = m_domainMin.z; range = m_domainMax.z - m_domainMin.z; break;
    }
    
    float partitionSize = range / m_numGPUs;
    int partition = static_cast<int>((coord - minVal) / partitionSize);
    return std::max(0, std::min(m_numGPUs - 1, partition));
}

void MultiGPUManager::exchangeGhostParticles() {
    // For each pair of neighboring partitions, exchange ghost particles
    for (int g = 0; g < m_numGPUs - 1; ++g) {
        GPUPartition& left = m_partitions[g];
        GPUPartition& right = m_partitions[g + 1];
        
        // Find ghost particles in left partition (upper ghost zone)
        // This would need proper implementation with temporary buffers
        
        // If P2P access is available, use direct copy
        if (m_devices[left.deviceId].canAccessPeer[right.deviceId]) {
            // Direct GPU-to-GPU copy would go here
        } else {
            // Stage through host
        }
    }
    
    // For now, this is a placeholder
    // Full implementation would:
    // 1. Find particles in ghost zones
    // 2. Compact them into buffers
    // 3. Copy between GPUs
    // 4. Append to ghost arrays
}

void MultiGPUManager::gatherParticles(SPHParticle* particles, int& numParticles) {
    numParticles = 0;
    
    for (int g = 0; g < m_numGPUs; ++g) {
        GPUPartition& p = m_partitions[g];
        
        std::vector<float> positions(p.numLocalParticles * 3);
        std::vector<float> velocities(p.numLocalParticles * 3);
        
        cudaSetDevice(p.deviceId);
        CUDA_CHECK(cudaMemcpy(positions.data(), p.d_positions,
                              p.numLocalParticles * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(velocities.data(), p.d_velocities,
                              p.numLocalParticles * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < p.numLocalParticles; ++i) {
            int idx = numParticles + i;
            particles[idx].x = positions[i * 3 + 0];
            particles[idx].y = positions[i * 3 + 1];
            particles[idx].z = positions[i * 3 + 2];
            particles[idx].vx = velocities[i * 3 + 0];
            particles[idx].vy = velocities[i * 3 + 1];
            particles[idx].vz = velocities[i * 3 + 2];
        }
        
        numParticles += p.numLocalParticles;
    }
}

void MultiGPUManager::migrateParticles() {
    // Find and move particles that crossed partition boundaries
    // This is a complex operation requiring:
    // 1. Find particles outside current partition
    // 2. Determine target partition
    // 3. Remove from current
    // 4. Send to target
    // 5. Insert at target
    
    // Placeholder - would use findMigratingParticlesKernel
}

int MultiGPUManager::getTotalParticleCount() const {
    int total = 0;
    for (const auto& p : m_partitions) {
        total += p.numLocalParticles;
    }
    return total;
}

void MultiGPUManager::synchronize() {
    for (int g = 0; g < m_numGPUs; ++g) {
        cudaSetDevice(m_partitions[g].deviceId);
        cudaDeviceSynchronize();
    }
}

float MultiGPUManager::checkLoadBalance() const {
    if (m_numGPUs <= 1) return 1.0f;
    
    int minParticles = m_partitions[0].numLocalParticles;
    int maxParticles = minParticles;
    
    for (int g = 1; g < m_numGPUs; ++g) {
        minParticles = std::min(minParticles, m_partitions[g].numLocalParticles);
        maxParticles = std::max(maxParticles, m_partitions[g].numLocalParticles);
    }
    
    if (minParticles == 0) return 999.f;
    return static_cast<float>(maxParticles) / minParticles;
}

void MultiGPUManager::rebalanceIfNeeded(float threshold) {
    float imbalance = checkLoadBalance();
    
    if (imbalance > threshold) {
        std::cout << "[MultiGPU] Load imbalance detected (" << imbalance 
                  << "), rebalancing..." << std::endl;
        
        // Would adjust partition boundaries and migrate particles
        // This is complex and often done periodically rather than every step
    }
}

} // namespace edgepredict
