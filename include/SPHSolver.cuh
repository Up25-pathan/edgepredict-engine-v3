#pragma once
/**
 * @file SPHSolver.cuh
 * @brief Smoothed Particle Hydrodynamics solver for workpiece simulation
 * 
 * Key improvements over v3:
 * - Correct Tait EOS for metals (not water)
 * - Particle deletion for material removal
 * - Chip separation logic
 * - Velocity Verlet integration (not Euler)
 * - Device memory with explicit transfers (not Unified Memory)
 */

#include "Types.h"
#include "IPhysicsSolver.h"
#include "CudaUtils.cuh"
#include "AdiabaticShearModel.cuh"
#include <vector>

namespace edgepredict {

/**
 * @brief SPH kernel configuration
 */
struct SPHKernelConfig {
    double h;               // Smoothing radius (m)
    double h2;              // h^2 (precomputed)
    double h3;              // h^3 (precomputed)
    double h6;              // h^6 (precomputed)
    double h9;              // h^9 (precomputed)
    double poly6Coeff;      // Poly6 kernel coefficient
    double spikyGradCoeff;  // Spiky gradient coefficient
    double viscLapCoeff;    // Viscosity Laplacian coefficient
    double restDensity;     // Rest density (kg/m³)
    double gasStiffness;    // Gas stiffness constant
    double viscosity;       // Kinematic viscosity
    double particleMass;    // Mass per particle
    
    EP_HOST_DEVICE SPHKernelConfig() 
        : h(0.0001), h2(0), h3(0), h6(0), h9(0),
          poly6Coeff(0), spikyGradCoeff(0), viscLapCoeff(0),
          restDensity(4430), gasStiffness(3000), viscosity(0.001),
          particleMass(1e-9) {}
};

// SPHParticle is now defined in Types.h for global availability

/**
 * @brief SPH simulation result data
 */
struct SPHResults {
    double maxDensity = 0;
    double maxPressure = 0;
    double maxVelocity = 0;
    double maxTemperature = 0;
    double totalKineticEnergy = 0;
    int activeParticleCount = 0;
    int chipParticleCount = 0;
    int removedParticleCount = 0;
};

/**
 * @brief GPU-accelerated SPH solver for workpiece simulation
 */
class SPHSolver : public IPhysicsSolver, public IMetricsProvider {
public:
    SPHSolver();
    ~SPHSolver() override;
    
    // No copy
    SPHSolver(const SPHSolver&) = delete;
    SPHSolver& operator=(const SPHSolver&) = delete;
    
    // IPhysicsSolver interface
    std::string getName() const override { return "SPHSolver"; }
    bool initialize(const Config& config) override;
    void step(double dt) override;
    double getStableTimeStep() const override;
    double getCurrentTime() const override { return m_currentTime; }
    bool isInitialized() const override { return m_isInitialized; }
    void reset() override;
    void getBounds(double& minX, double& minY, double& minZ, 
                   double& maxX, double& maxY, double& maxZ) const override;
    
    // IMetricsProvider interface
    double getMaxStress() const override { return m_results.maxPressure; }
    double getMaxTemperature() const override { return m_results.maxTemperature; }
    double getTotalKineticEnergy() const override { return m_results.totalKineticEnergy; }
    void syncMetrics() override { updateResults(); }
    
    // SPH-specific methods
    
    /**
     * @brief Initialize particles in a box region
     * @param minCorner Minimum corner of box
     * @param maxCorner Maximum corner of box
     * @param spacing Particle spacing
     */
    void initializeParticleBox(const Vec3& minCorner, const Vec3& maxCorner, double spacing);
    
    /**
     * @brief Get particle data (copies from GPU)
     */
    std::vector<SPHParticle> getParticles();
    
    /**
     * @brief Get total particle count
     */
    int getParticleCount() const { return m_numParticles; }
    
    /**
     * @brief Get current simulation results
     */
    const SPHResults& getResults() const { return m_results; }
    
    /**
     * @brief Apply external force to particles in region
     */
    void applyExternalForce(const Vec3& center, double radius, const Vec3& force);
    
    /**
     * @brief Mark particles for removal (cutting)
     * @param plane Plane normal and distance (ax + by + cz + d = 0)
     */
    void cutParticles(const Vec3& planeNormal, double planeDist);
    
    /**
     * @brief Access device particle pointer (for interaction kernel)
     */
    SPHParticle* getDeviceParticles() { return d_particles; }
    int* getDeviceParticleCount() { return &m_numParticles; }
    
    /**
     * @brief Set tool position for LOD zone calculation
     */
    void setToolPosition(const Vec3& pos) { m_toolPosition = pos; }
    
    /**
     * @brief Get host particle array (for AMR modifications)
     * Note: Call copyFromDevice() first if you need current GPU data
     */
    std::vector<SPHParticle>& getHostParticles() { 
        copyFromDevice();
        return h_particles; 
    }
    
    /**
     * @brief Update device particles from host array (after AMR modifications)
     */
    void updateParticlesFromHost() {
        m_numParticles = static_cast<int>(h_particles.size());
        copyToDevice();
    }

private:
    void allocateMemory(int capacity);
    void freeMemory();
    void copyToDevice();
    void copyFromDevice();
    void buildSpatialHash();
    void computeDensityPressure();
    void computeForces();
    void integrate(double dt);
    void updateResults();
    
    // Host data
    std::vector<SPHParticle> h_particles;
    
    // Device data
    SPHParticle* d_particles = nullptr;
    int* d_cellStart = nullptr;    // Spatial hash: cell start indices
    int* d_cellEnd = nullptr;      // Spatial hash: cell end indices
    int* d_particleOrder = nullptr;// Sorted particle indices
    
    // Pinned host memory for faster GPU transfers
    SPHParticle* h_pinnedParticles = nullptr;
    
    // Configuration
    SPHKernelConfig m_config;
    int m_maxParticles = 100000;
    int m_numParticles = 0;
    int m_hashTableSize = 1000000;
    double m_cellSize = 0.0001;
    
    // Domain bounds
    Vec3 m_domainMin;
    Vec3 m_domainMax;
    
    // State
    bool m_isInitialized = false;
    double m_currentTime = 0.0;
    double m_stableTimeStep = 1e-6;
    int m_currentStep = 0;           // For LOD scheduling
    const Config* m_mainConfig = nullptr; // Persistent config reference
    SPHResults m_results;
    
    // LOD configuration
    bool m_lodEnabled = true;
    double m_lodActiveRadius = 0.002;
    double m_lodNearRadius = 0.01;
    int m_lodNearSkipSteps = 5;
    int m_lodFarSkipSteps = 20;
    Vec3 m_toolPosition;             // Updated by engine for LOD zones
    
    // Damage/Chip Separation configuration
    bool m_damageEnabled = true;
    double m_jc_D1 = 0.05;
    double m_jc_D2 = 3.44;
    double m_jc_D3 = -2.12;
    double m_jc_D4 = 0.002;
    double m_jc_D5 = 0.61;
    double m_damageThreshold = 1.0;
    double m_refStrainRate = 1.0;
    double m_refTemp = 25.0;         // Reference temperature (°C)
    double m_meltTemp = 1660.0;      // Melting temperature (°C) for Ti-6Al-4V
    
    // CUDA streams for async execution
    cudaStream_t m_computeStream = nullptr;
    
    // Advanced Physics Models
    AdiabaticShearModel m_adiabaticShearModel;
    
public:
    // Stream accessor for external kernels (contact solver)
    cudaStream_t getComputeStream() const { return m_computeStream; }
};

} // namespace edgepredict
