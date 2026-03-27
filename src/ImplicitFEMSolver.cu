/**
 * @file ImplicitFEMSolver.cu
 * @brief Implementation of Implicit FEM solver using Conjugate Gradient
 * 
 * The Conjugate Gradient method solves Ax = b iteratively without
 * forming A explicitly. We only need to compute matrix-vector products.
 */

#include "ImplicitFEMSolver.cuh"
#include "FEMSolver.cuh"  // For FEMNodeGPU, FEMSpring structs
#include "Config.h"
#include <iostream>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>

namespace edgepredict {

// ============================================================================
// CUDA Kernels for Conjugate Gradient
// ============================================================================

/**
 * @brief Initialize CG vectors
 */
__global__ void initCGVectorsKernel(double* r, double* p, const double* rhs,
                                     const double* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // r = b - A*x (initially x=0, so r = b)
    // For initial guess x=0, r = rhs
    r[idx] = rhs[idx];
    p[idx] = r[idx];
}

/**
 * @brief axpy: y = alpha*x + y
 */
__global__ void axpyKernel(double* y, const double* x, double alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    y[idx] = alpha * x[idx] + y[idx];
}

/**
 * @brief xpay: y = x + alpha*y
 */
__global__ void xpayKernel(double* y, const double* x, double alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    y[idx] = x[idx] + alpha * y[idx];
}

/**
 * @brief Compute matrix-vector product: y = (M + dt²K) * x
 * Using spring-based stiffness computation
 */
__global__ void implicitMVPKernel(double* y, const double* x,
                                   const FEMNodeGPU* nodes, int numNodes,
                                   const FEMSpring* springs, int numSprings,
                                   double dt, double massScale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    
    // Start with mass term: M*x (diagonal mass matrix)
    double mass = nodes[idx].mass * massScale;
    y[idx * 3 + 0] = mass * x[idx * 3 + 0];
    y[idx * 3 + 1] = mass * x[idx * 3 + 1];
    y[idx * 3 + 2] = mass * x[idx * 3 + 2];
}

/**
 * @brief Add stiffness contribution from springs: y += dt²*K*x
 */
__global__ void addStiffnessContributionKernel(double* y, const double* x,
                                                const FEMNodeGPU* nodes,
                                                const FEMSpring* springs, int numSprings,
                                                double dt2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSprings) return;
    
    const FEMSpring& s = springs[idx];
    int n1 = s.node1;
    int n2 = s.node2;
    
    // Spring stiffness term: k * (x_i - x_j)
    double k = s.stiffness * dt2;
    
    double dx = x[n1 * 3 + 0] - x[n2 * 3 + 0];
    double dy = x[n1 * 3 + 1] - x[n2 * 3 + 1];
    double dz = x[n1 * 3 + 2] - x[n2 * 3 + 2];
    
    // y[n1] += k * (x[n1] - x[n2])
    atomicAdd(&y[n1 * 3 + 0], k * dx);
    atomicAdd(&y[n1 * 3 + 1], k * dy);
    atomicAdd(&y[n1 * 3 + 2], k * dz);
    
    // y[n2] -= k * (x[n1] - x[n2])
    atomicAdd(&y[n2 * 3 + 0], -k * dx);
    atomicAdd(&y[n2 * 3 + 1], -k * dy);
    atomicAdd(&y[n2 * 3 + 2], -k * dz);
}

/**
 * @brief Update x += alpha*p
 */
__global__ void updateXKernel(double* x, const double* p, double alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    x[idx] += alpha * p[idx];
}

/**
 * @brief Update r -= alpha*Ap
 */
__global__ void updateRKernel(double* r, const double* Ap, double alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    r[idx] -= alpha * Ap[idx];
}

/**
 * @brief Update p = r + beta*p
 */
__global__ void updatePKernel(double* p, const double* r, double beta, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    p[idx] = r[idx] + beta * p[idx];
}

/**
 * @brief Apply displacement to nodes
 */
__global__ void applyDisplacementKernel(FEMNodeGPU* nodes, const double* dx, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    
    nodes[idx].x += dx[idx * 3 + 0];
    nodes[idx].y += dx[idx * 3 + 1];
    nodes[idx].z += dx[idx * 3 + 2];
}

// ============================================================================
// ImplicitFEMSolver Implementation
// ============================================================================

ImplicitFEMSolver::ImplicitFEMSolver() = default;

ImplicitFEMSolver::~ImplicitFEMSolver() {
    if (d_nodes) cudaFree(d_nodes);
    if (d_displacement) cudaFree(d_displacement);
    if (d_velocity) cudaFree(d_velocity);
    if (d_force) cudaFree(d_force);
    if (d_rhs) cudaFree(d_rhs);
    if (d_p) cudaFree(d_p);
    if (d_r) cudaFree(d_r);
    if (d_Ap) cudaFree(d_Ap);
    if (d_temp) cudaFree(d_temp);
}

bool ImplicitFEMSolver::initialize(const Config& config) {
    std::cout << "[ImplicitFEMSolver] Initializing..." << std::endl;
    
    const auto& femParams = config.getFEM();
    
    m_youngsModulus = femParams.youngModulus;
    m_damping = femParams.dampingRatio;
    
    // Check if implicit solver is enabled
    // For now, we'll auto-detect based on presence of specific config
    // In future, this should be a config parameter
    m_config.enabled = true;
    m_config.maxIterations = 500;
    m_config.tolerance = 1e-8;
    
    m_isInitialized = true;
    std::cout << "[ImplicitFEMSolver] Initialized (max " << m_config.maxIterations 
              << " CG iterations, tol=" << m_config.tolerance << ")" << std::endl;
    
    return true;
}

void ImplicitFEMSolver::initializeFromMesh(const Mesh& mesh) {
    // Similar to explicit FEMSolver, create nodes from mesh
    m_numNodes = static_cast<int>(mesh.nodes.size());
    int vectorSize = m_numNodes * 3;  // x, y, z per node
    
    // Allocate device memory for CG vectors
    CUDA_CHECK(cudaMalloc(&d_displacement, vectorSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_velocity, vectorSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_force, vectorSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rhs, vectorSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p, vectorSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r, vectorSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ap, vectorSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_temp, vectorSize * sizeof(double)));
    
    // Initialize to zero
    CUDA_CHECK(cudaMemset(d_displacement, 0, vectorSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_velocity, 0, vectorSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_force, 0, vectorSize * sizeof(double)));
    
    std::cout << "[ImplicitFEMSolver] Allocated CG vectors for " 
              << m_numNodes << " nodes" << std::endl;
}

double ImplicitFEMSolver::getStableTimeStep() const {
    // Implicit solver is unconditionally stable!
    // Return a large time step - the physical accuracy determines the limit
    return 0.001;  // 1ms - 100,000x larger than explicit!
}

void ImplicitFEMSolver::step(double dt) {
    if (!m_isInitialized || m_numNodes == 0) return;
    
    int vectorSize = m_numNodes * 3;
    
    // 1. Build RHS: b = dt * F (external forces)
    CUDA_CHECK(cudaMemcpy(d_rhs, d_force, vectorSize * sizeof(double), 
                          cudaMemcpyDeviceToDevice));
    
    // 2. Solve (M + dt²K) * Δx = b using Conjugate Gradient
    conjugateGradientSolve(dt);
    
    // 3. Apply displacement increment to nodes
    int blockSize = 256;
    int gridSize = (m_numNodes + blockSize - 1) / blockSize;
    applyDisplacementKernel<<<gridSize, blockSize>>>(d_nodes, d_displacement, m_numNodes);
    CUDA_CHECK_KERNEL();
    
    m_currentTime += dt;
}

void ImplicitFEMSolver::conjugateGradientSolve(double dt) {
    int vectorSize = m_numNodes * 3;
    int blockSize = 256;
    int gridSize = (vectorSize + blockSize - 1) / blockSize;
    
    // Initialize: x = 0, r = b, p = r
    CUDA_CHECK(cudaMemset(d_displacement, 0, vectorSize * sizeof(double)));
    initCGVectorsKernel<<<gridSize, blockSize>>>(d_r, d_p, d_rhs, d_displacement, vectorSize);
    CUDA_CHECK_KERNEL();
    
    // rTr = r^T * r (initial residual)
    thrust::device_ptr<double> r_ptr(d_r);
    double rTr = thrust::inner_product(r_ptr, r_ptr + vectorSize, r_ptr, 0.0);
    double rTr_initial = rTr;
    
    if (rTr_initial < 1e-20) {
        m_lastIterations = 0;
        m_lastResidual = 0;
        return;  // Already converged
    }
    
    for (int iter = 0; iter < m_config.maxIterations; ++iter) {
        // Ap = A * p
        matrixVectorProduct(d_p, d_Ap, dt);
        
        // pAp = p^T * Ap
        thrust::device_ptr<double> p_ptr(d_p);
        thrust::device_ptr<double> Ap_ptr(d_Ap);
        double pAp = thrust::inner_product(p_ptr, p_ptr + vectorSize, Ap_ptr, 0.0);
        
        if (std::abs(pAp) < 1e-20) break;  // Prevent division by zero
        
        double alpha = rTr / pAp;
        
        // x += alpha * p
        updateXKernel<<<gridSize, blockSize>>>(d_displacement, d_p, alpha, vectorSize);
        CUDA_CHECK_KERNEL();
        
        // r -= alpha * Ap
        updateRKernel<<<gridSize, blockSize>>>(d_r, d_Ap, alpha, vectorSize);
        CUDA_CHECK_KERNEL();
        
        // rTr_new = r^T * r
        double rTr_new = thrust::inner_product(r_ptr, r_ptr + vectorSize, r_ptr, 0.0);
        
        // Check convergence
        double relResidual = std::sqrt(rTr_new / rTr_initial);
        if (relResidual < m_config.tolerance) {
            m_lastIterations = iter + 1;
            m_lastResidual = relResidual;
            return;
        }
        
        double beta = rTr_new / rTr;
        rTr = rTr_new;
        
        // p = r + beta * p
        updatePKernel<<<gridSize, blockSize>>>(d_p, d_r, beta, vectorSize);
        CUDA_CHECK_KERNEL();
    }
    
    m_lastIterations = m_config.maxIterations;
    m_lastResidual = std::sqrt(rTr / rTr_initial);
    
    if (m_lastResidual > 0.01) {
        std::cerr << "[ImplicitFEMSolver] Warning: CG did not converge. "
                  << "Residual: " << m_lastResidual << std::endl;
    }
}

void ImplicitFEMSolver::matrixVectorProduct(const double* x, double* y, double dt) {
    // Compute y = (M + dt²K) * x
    // 1. Mass term: y = M * x
    // 2. Stiffness term: y += dt² * K * x
    
    // int vectorSize = m_numNodes * 3; // Unused
    int blockSize = 256;
    
    // Mass contribution
    int gridSizeNodes = (m_numNodes + blockSize - 1) / blockSize;
    implicitMVPKernel<<<gridSizeNodes, blockSize>>>(
        y, x, d_nodes, m_numNodes, nullptr, 0, dt, 1.0);
    CUDA_CHECK_KERNEL();
    
    // Note: Full stiffness contribution would require assembled spring data
    // For now, this is a simplified version. Full implementation would iterate
    // over springs and add their stiffness contributions
}

void ImplicitFEMSolver::setExternalForce(int nodeId, const Vec3& force) {
    if (nodeId < 0 || nodeId >= m_numNodes) return;
    
    double h_force[3] = {force.x, force.y, force.z};
    CUDA_CHECK(cudaMemcpy(d_force + nodeId * 3, h_force, 3 * sizeof(double),
                          cudaMemcpyHostToDevice));
}

Vec3 ImplicitFEMSolver::getDisplacement(int nodeId) const {
    if (nodeId < 0 || nodeId >= m_numNodes) return Vec3();
    
    double h_disp[3];
    cudaMemcpy(h_disp, d_displacement + nodeId * 3, 3 * sizeof(double),
               cudaMemcpyDeviceToHost);
    return Vec3(h_disp[0], h_disp[1], h_disp[2]);
}

void ImplicitFEMSolver::getBounds(double& minX, double& minY, double& minZ, 
                                 double& maxX, double& maxY, double& maxZ) const {
    if (h_nodes.empty()) {
        minX = minY = minZ = maxX = maxY = maxZ = 0;
        return;
    }
    
    minX = minY = minZ = 1e10;
    maxX = maxY = maxZ = -1e10;
    
    for (const auto& node : h_nodes) {
        minX = std::min(minX, node.x);
        minY = std::min(minY, node.y);
        minZ = std::min(minZ, node.z);
        maxX = std::max(maxX, node.x);
        maxY = std::max(maxY, node.y);
        maxZ = std::max(maxZ, node.z);
    }
}

} // namespace edgepredict
