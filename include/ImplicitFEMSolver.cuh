/**
 * @file ImplicitFEMSolver.cuh
 * @brief Implicit FEM solver using Conjugate Gradient method
 * 
 * Unlike the explicit FEM solver which uses tiny time steps due to stability limits,
 * this implicit solver can take very large time steps (0.001s vs 0.00000001s) by
 * solving a linear system at each step.
 * 
 * Mathematical formulation:
 *   (M + dt²*K) * Δx = dt * (F - K*x_current)
 * 
 * where:
 *   M = Mass matrix (diagonal, so easy)
 *   K = Stiffness matrix (sparse, assembled from springs)
 *   F = External force vector
 *   Δx = Displacement increment
 * 
 * We use the Conjugate Gradient method to solve Ax = b without forming A explicitly.
 */

#pragma once

#include "Types.h"
#include "IPhysicsSolver.h"
#include "CudaUtils.cuh"
#include <vector>

namespace edgepredict {

// Forward declaration
struct FEMNodeGPU;
struct FEMSpring;
class Config;
class Mesh;

/**
 * @brief Configuration for implicit solver
 */
struct ImplicitSolverConfig {
    bool enabled = false;
    int maxIterations = 500;        // Max CG iterations
    double tolerance = 1e-8;        // Convergence tolerance
    double alpha = 0.5;             // Newmark-beta parameter (0.5 = unconditionally stable)
    double beta = 0.25;             // Newmark-beta parameter (0.25 = average acceleration)
};

/**
 * @brief Sparse matrix row in CSR format
 */
struct SparseRow {
    std::vector<int> colIndices;
    std::vector<double> values;
};

/**
 * @brief GPU-accelerated Implicit FEM Solver
 * 
 * Uses Conjugate Gradient to solve the implicit system, allowing
 * time steps 100-1000x larger than explicit methods.
 */
class ImplicitFEMSolver : public IPhysicsSolver, public IMetricsProvider {
public:
    ImplicitFEMSolver();
    ~ImplicitFEMSolver() override;
    
    // IPhysicsSolver interface
    std::string getName() const override { return "ImplicitFEMSolver"; }
    bool initialize(const Config& config) override;
    void step(double dt) override;
    double getStableTimeStep() const override;
    double getCurrentTime() const override { return m_currentTime; }
    bool isInitialized() const override { return m_isInitialized; }
    void reset() override {}
    void getBounds(double& minX, double& minY, double& minZ, 
                   double& maxX, double& maxY, double& maxZ) const override;
    
    // IMetricsProvider interface
    double getMaxTemperature() const override { return m_maxTemperature; }
    double getMaxStress() const override { return m_maxStress; }
    double getTotalKineticEnergy() const override { return 0.0; }
    void syncMetrics() override {}
    
    /**
     * @brief Initialize from mesh geometry
     */
    void initializeFromMesh(const Mesh& mesh);
    
    /**
     * @brief Set external force on a node
     */
    void setExternalForce(int nodeId, const Vec3& force);
    
    /**
     * @brief Get node displacement
     */
    Vec3 getDisplacement(int nodeId) const;
    
    /**
     * @brief Get convergence info from last solve
     */
    int getLastIterationCount() const { return m_lastIterations; }
    double getLastResidual() const { return m_lastResidual; }

private:
    /**
     * @brief Assemble the stiffness matrix K (sparse)
     */
    void assembleStiffnessMatrix();
    
    /**
     * @brief Solve (M + dt²K)x = b using Conjugate Gradient
     */
    void conjugateGradientSolve(double dt);
    
    /**
     * @brief Matrix-vector product: y = (M + dt²K) * x (GPU)
     */
    void matrixVectorProduct(const double* x, double* y, double dt);
    
    /**
     * @brief Compute residual r = b - Ax
     */
    double computeResidual();
    
    // Host data (for assembly)
    std::vector<FEMNodeGPU> h_nodes;
    std::vector<FEMSpring> h_springs;
    
    // Device data
    FEMNodeGPU* d_nodes = nullptr;
    double* d_displacement = nullptr;  // Current displacement
    double* d_velocity = nullptr;
    double* d_force = nullptr;         // External force
    double* d_rhs = nullptr;           // Right-hand side of linear system
    
    // CG vectors (device)
    double* d_p = nullptr;             // Search direction
    double* d_r = nullptr;             // Residual
    double* d_Ap = nullptr;            // A * p
    double* d_temp = nullptr;          // Temporary storage
    
    // Sparse stiffness matrix (CSR format on device)
    int* d_rowPtr = nullptr;
    int* d_colIdx = nullptr;
    double* d_values = nullptr;
    int m_nnz = 0;  // Number of non-zeros
    
    // Configuration
    ImplicitSolverConfig m_config;
    double m_youngsModulus = 200e9;
    double m_damping = 0.1;
    int m_numNodes = 0;
    int m_numSprings = 0;
    
    // Metrics
    double m_maxTemperature = 25.0;
    double m_maxStress = 0.0;
    double m_currentTime = 0.0;
    int m_lastIterations = 0;
    double m_lastResidual = 0.0;
    
    bool m_isInitialized = false;
};

} // namespace edgepredict
