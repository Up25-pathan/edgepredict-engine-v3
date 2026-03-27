/**
 * @file AMRManager.cuh
 * @brief Adaptive Mesh Refinement manager for SPH particles
 * 
 * This class handles dynamic particle splitting (refinement) and merging (coarsening)
 * based on distance to the cutting zone. Near the tool, particles are split for
 * higher resolution. Far away, they are merged for efficiency.
 * 
 * Key concepts:
 * - Refinement: 1 particle → 8 child particles (mass conserved)
 * - Coarsening: 8 neighbor particles → 1 parent particle (when far from tool)
 * - Conservation: Total mass and momentum are preserved during split/merge
 */

#pragma once

#include "Types.h"
#include "SPHSolver.cuh"
#include "CudaUtils.cuh"
#include <vector>

namespace edgepredict {

/**
 * @brief AMR configuration parameters
 */
struct AMRConfig {
    bool enabled = false;                // Enable/disable AMR
    double refineRadius = 0.003;         // 3mm - refine particles within this distance
    double coarsenRadius = 0.015;        // 15mm - coarsen particles beyond this distance
    double minParticleSize = 0.00005;    // 50 microns - minimum size (highest refinement)
    double maxParticleSize = 0.001;      // 1mm - maximum size (coarsest)
    int refineIntervalSteps = 50;        // How often to check for refinement
    int maxParticles = 5000000;          // Hard limit on particle count
};

/**
 * @brief Particle refinement level tracking
 */
struct ParticleLevel {
    int32_t particleId;
    int8_t level;          // 0 = coarsest, higher = finer
    int8_t maxLevel = 5;   // Maximum refinement level
    double effectiveRadius; // Particle's effective smoothing radius
};

/**
 * @brief Adaptive Mesh Refinement Manager
 * 
 * Manages dynamic particle resolution based on tool proximity.
 * Uses a cell-based approach for efficient neighbor finding during merge operations.
 */
class AMRManager {
public:
    AMRManager();
    ~AMRManager();
    
    /**
     * @brief Initialize with configuration
     */
    void initialize(const AMRConfig& config);
    
    /**
     * @brief Set tool position for distance calculations
     */
    void setToolPosition(const Vec3& pos) { m_toolPosition = pos; }
    
    /**
     * @brief Update refinement - split/merge particles as needed
     * @param solver SPH solver to modify
     * @param currentStep Current simulation step
     * 
     * This should be called periodically (not every step) to:
     * 1. Find particles near tool that need splitting
     * 2. Find particles far from tool that can be merged
     * 3. Execute split/merge operations
     * 4. Update solver's particle arrays
     */
    void updateRefinement(SPHSolver& solver, int currentStep);
    
    /**
     * @brief Get number of particles split in last update
     */
    int getLastSplitCount() const { return m_lastSplitCount; }
    
    /**
     * @brief Get number of particles merged in last update
     */
    int getLastMergeCount() const { return m_lastMergeCount; }
    
    /**
     * @brief Check if AMR is enabled
     */
    bool isEnabled() const { return m_config.enabled; }

private:
    /**
     * @brief Split a single particle into 8 children
     * Children are placed in a 2x2x2 pattern around parent center
     * Mass is divided equally, momentum is preserved
     */
    void splitParticle(std::vector<SPHParticle>& particles, int parentIdx, 
                       std::vector<SPHParticle>& newParticles);
    
    /**
     * @brief Merge 8 neighboring particles into 1
     * Combined mass and momentum-weighted velocity
     */
    void mergeParticles(std::vector<SPHParticle>& particles,
                        const std::vector<int>& mergeIndices,
                        SPHParticle& resultParticle);
    
    /**
     * @brief Find candidate particles for splitting (near tool, not too refined)
     */
    std::vector<int> findSplitCandidates(const std::vector<SPHParticle>& particles);
    
    /**
     * @brief Find groups of particles that can be merged (far from tool, neighbors)
     */
    std::vector<std::vector<int>> findMergeGroups(const std::vector<SPHParticle>& particles);
    
    AMRConfig m_config;
    Vec3 m_toolPosition;
    int m_lastSplitCount = 0;
    int m_lastMergeCount = 0;
    int m_lastUpdateStep = 0;
    
    // Particle level tracking (maps particle ID to refinement level)
    std::vector<int8_t> m_particleLevels;
};

} // namespace edgepredict
