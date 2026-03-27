#pragma once
/**
 * @file ContactSolver.cuh
 * @brief Tool-workpiece contact detection and force computation
 * 
 * Key improvements over v3:
 * - Spatial hash acceleration (not O(N×M) brute force)
 * - Proper Coulomb friction (not velocity damping)
 * - Heat generation at contact
 */

#include "Types.h"
#include "SPHSolver.cuh"
#include "FEMSolver.cuh"
#include "CudaUtils.cuh"

namespace edgepredict {

/**
 * @brief Contact detection configuration
 */
struct ContactConfig {
    double contactRadius = 0.0002;      // Detection radius (m)
    double contactStiffness = 1e7;      // Contact stiffness (N/m)
    double frictionCoefficient = 0.3;   // Coulomb friction
    double heatPartition = 0.5;         // Fraction of heat to tool (Taylor-Quinney)
    double plasticWorkFraction = 0.9;   // Fraction of plastic work converted to heat
};

/**
 * @brief Contact event for post-processing
 */
struct ContactEvent {
    int particleIndex;
    int nodeIndex;
    Vec3 position;
    Vec3 normal;
    Vec3 force;
    double penetration;
    double heatGenerated;
};

/**
 * @brief Contact solver for SPH-FEM interaction
 * 
 * Uses spatial hashing for efficient contact detection.
 * Handles force and heat transfer between workpiece and tool.
 */
class ContactSolver {
public:
    ContactSolver();
    ~ContactSolver();
    
    ContactSolver(const ContactSolver&) = delete;
    ContactSolver& operator=(const ContactSolver&) = delete;
    
    /**
     * @brief Initialize with SPH and FEM solvers
     */
    void initialize(SPHSolver* sph, FEMSolver* fem, const ContactConfig& config);
    
    /**
     * @brief Detect and resolve contacts
     * @param dt Time step
     */
    void resolveContacts(double dt);
    
    /**
     * @brief Get number of active contacts this step
     */
    int getContactCount() const { return m_numContacts; }
    
    /**
     * @brief Get total heat generated this step
     */
    double getHeatGenerated() const { return m_totalHeatGenerated; }
    
    /**
     * @brief Get total contact force magnitude this step
     */
    double getTotalContactForce() const { return m_totalContactForce; }
    
    /**
     * @brief Get contact events for visualization
     */
    std::vector<ContactEvent> getContactEvents() const;
    
    /**
     * @brief Set contact configuration
     */
    void setConfig(const ContactConfig& config) { m_config = config; }

private:
    void buildSpatialHash();
    void launchContactKernel(double dt);
    void transferResults();
    
    // References to solvers
    SPHSolver* m_sph = nullptr;
    FEMSolver* m_fem = nullptr;
    
    // Configuration
    ContactConfig m_config;
    
    // Spatial hash for tool nodes
    int* d_toolCellStart = nullptr;
    int* d_toolCellEnd = nullptr;
    int* d_toolNodeOrder = nullptr;
    int m_hashTableSize = 100000;
    double m_cellSize = 0.0002;
    
    // Contact results (device)
    int* d_numContacts = nullptr;
    double* d_totalHeat = nullptr;
    double* d_totalForce = nullptr;
    
    // Contact results (host)
    int m_numContacts = 0;
    double m_totalHeatGenerated = 0;
    double m_totalContactForce = 0;
    
    // Domain bounds
    Vec3 m_domainMin;
    Vec3 m_domainMax;
    
    bool m_isInitialized = false;
};

// ============================================================================
// Free function for launching contact kernel (called by solvers)
// ============================================================================

/**
 * @brief Launch contact detection and resolution kernel
 * 
 * This is a convenience function that can be called directly
 * without instantiating the full ContactSolver class.
 */
void launchContactInteraction(
    SPHParticle* particles, int numParticles,
    FEMNodeGPU* nodes, int numNodes,
    double contactRadius, double contactStiffness,
    double friction, double heatPartition, double dt
);

} // namespace edgepredict
