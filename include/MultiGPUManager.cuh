 #pragma once
/**
 * @file MultiGPUManager.cuh
 * @brief Multi-GPU particle distribution manager
 * 
 * Distributes SPH particles across multiple GPUs:
 * - Spatial domain decomposition
 * - Ghost particle exchange between GPUs
 * - Load balancing
 * 
 * Enables simulating larger workpieces (8+ million particles)
 */

#include "Types.h"
#include "CudaUtils.cuh"
#include "SPHSolver.cuh"
#include <vector>
#include <memory>

namespace edgepredict {

/**
 * @brief GPU device information
 */
struct GPUDeviceInfo {
    int deviceId = 0;
    std::string name;
    size_t totalMemory = 0;      // bytes
    size_t freeMemory = 0;
    int multiProcessorCount = 0;
    bool canAccessPeer[8] = {false};  // P2P access to other GPUs
};

/**
 * @brief Domain partition for a single GPU
 */
struct GPUPartition {
    int deviceId = 0;
    
    // Spatial bounds (world coordinates)
    Vec3 minBound;
    Vec3 maxBound;
    
    // Ghost region size (for particle exchange)
    float ghostWidth = 0;
    
    // Particles owned by this partition
    int numLocalParticles = 0;
    int maxLocalParticles = 0;
    
    // Ghost particles received from neighbors
    int numGhostParticles = 0;
    int maxGhostParticles = 0;
    
    // Device pointers
    float* d_positions = nullptr;
    float* d_velocities = nullptr;
    float* d_densities = nullptr;
    float* d_pressures = nullptr;
    float* d_temperatures = nullptr;
    int* d_status = nullptr;
    
    // Ghost buffers
    float* d_ghostPositions = nullptr;
    float* d_ghostVelocities = nullptr;
    
    // Neighbor list
    int* d_neighborOffsets = nullptr;
    int* d_neighborList = nullptr;
};

/**
 * @brief Multi-GPU manager for particle simulations
 */
class MultiGPUManager {
public:
    MultiGPUManager();
    ~MultiGPUManager();
    
    // No copy
    MultiGPUManager(const MultiGPUManager&) = delete;
    MultiGPUManager& operator=(const MultiGPUManager&) = delete;
    
    /**
     * @brief Query available GPUs
     */
    int queryDevices();
    
    /**
     * @brief Get device info
     */
    const GPUDeviceInfo& getDeviceInfo(int idx) const { return m_devices[idx]; }
    
    /**
     * @brief Initialize with specified number of GPUs
     * @param numGPUs Number of GPUs to use (0 = use all)
     */
    bool initialize(int numGPUs = 0);
    
    /**
     * @brief Set simulation domain bounds
     */
    void setDomainBounds(const Vec3& minBound, const Vec3& maxBound);
    
    /**
     * @brief Set ghost region width
     */
    void setGhostWidth(float width) { m_ghostWidth = width; }
    
    /**
     * @brief Distribute particles across GPUs
     * @param particles Initial particle array
     * @param numParticles Number of particles
     */
    void distributeParticles(const SPHParticle* particles, int numParticles);
    
    /**
     * @brief Exchange ghost particles between GPUs
     * 
     * Must be called before neighbor search
     */
    void exchangeGhostParticles();
    
    /**
     * @brief Gather particles back to host
     * @param particles Output array
     * @param numParticles Output count
     */
    void gatherParticles(SPHParticle* particles, int& numParticles);
    
    /**
     * @brief Migrate particles that crossed partition boundaries
     */
    void migrateParticles();
    
    /**
     * @brief Get partition for device
     */
    GPUPartition& getPartition(int deviceIdx) { return m_partitions[deviceIdx]; }
    const GPUPartition& getPartition(int deviceIdx) const { return m_partitions[deviceIdx]; }
    
    /**
     * @brief Get number of active GPUs
     */
    int getNumGPUs() const { return m_numGPUs; }
    
    /**
     * @brief Get total particle count across all GPUs
     */
    int getTotalParticleCount() const;
    
    /**
     * @brief Synchronize all GPUs
     */
    void synchronize();
    
    /**
     * @brief Execute kernel on all GPUs
     * @param kernel Function to execute on each GPU
     */
    template<typename Func>
    void executeOnAllGPUs(Func kernel);
    
    /**
     * @brief Check load balance
     * @return Imbalance ratio (max/min particles)
     */
    float checkLoadBalance() const;
    
    /**
     * @brief Rebalance partitions if needed
     */
    void rebalanceIfNeeded(float threshold = 1.5f);

private:
    // Device management
    int m_numGPUs = 0;
    int m_maxGPUs = 8;
    std::vector<GPUDeviceInfo> m_devices;
    std::vector<GPUPartition> m_partitions;
    
    // Domain
    Vec3 m_domainMin;
    Vec3 m_domainMax;
    float m_ghostWidth = 0.001f;  // 1mm default
    
    // Decomposition axis (0=X, 1=Y, 2=Z)
    int m_decompAxis = 2;  // Default: Z-axis (vertical slabs)
    
    // Helper methods
    void createPartitions();
    void allocatePartitionMemory(GPUPartition& p, int maxParticles);
    void freePartitionMemory(GPUPartition& p);
    int getPartitionForPosition(const Vec3& pos) const;
    void setupPeerAccess();
};

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief Find particles in ghost region
 */
__global__ void findGhostParticlesKernel(
    const float* positions, int numParticles,
    int* ghostFlags, int* ghostCount,
    float ghostMin, float ghostMax, int axis);

/**
 * @brief Compact ghost particles for transfer
 */
__global__ void compactGhostParticlesKernel(
    const float* positions, const float* velocities,
    const int* ghostFlags, const int* ghostOffsets,
    float* ghostPositions, float* ghostVelocities,
    int numParticles);

/**
 * @brief Find particles that migrated out of partition
 */
__global__ void findMigratingParticlesKernel(
    const float* positions, int numParticles,
    int* migrateFlags, int* migrateTarget,
    float partitionMin, float partitionMax, int axis,
    int partitionIdx, int numPartitions);

// ============================================================================
// Template Implementation
// ============================================================================

template<typename Func>
void MultiGPUManager::executeOnAllGPUs(Func kernel) {
    // Launch on all GPUs in parallel
    for (int i = 0; i < m_numGPUs; ++i) {
        cudaSetDevice(m_partitions[i].deviceId);
        kernel(m_partitions[i]);
    }
    
    // Synchronize all
    synchronize();
}

} // namespace edgepredict
