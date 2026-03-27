/**
 * @file CFDSolver.cpp
 * @brief CFD solver implementation for coolant simulation
 * 
 * Implements proper Navier-Stokes on staggered MAC grid:
 * - Semi-Lagrangian advection (stable for any CFL)
 * - Implicit viscosity diffusion
 * - Conjugate gradient pressure projection
 * - Temperature advection and diffusion
 */

#include "CFDSolver.h"
#include "SPHSolver.cuh"
#include "FEMSolver.cuh"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>

namespace edgepredict {

// ============================================================================
// Helper functions
// ============================================================================

inline double clamp(double v, double lo, double hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

inline double lerp(double a, double b, double t) {
    return a + t * (b - a);
}

inline double trilerp(double c000, double c100, double c010, double c110,
                       double c001, double c101, double c011, double c111,
                       double tx, double ty, double tz) {
    double c00 = lerp(c000, c100, tx);
    double c10 = lerp(c010, c110, tx);
    double c01 = lerp(c001, c101, tx);
    double c11 = lerp(c011, c111, tx);
    double c0 = lerp(c00, c10, ty);
    double c1 = lerp(c01, c11, ty);
    return lerp(c0, c1, tz);
}

// ============================================================================
// CFDSolver Implementation
// ============================================================================

CFDSolver::CFDSolver() = default;

CFDSolver::~CFDSolver() = default;

bool CFDSolver::initialize(const Config& config) {
    std::cout << "[CFDSolver] Initializing..." << std::endl;
    
    const auto& cfdParams = config.getCFD();
    
    // Set grid from config
    m_nx = cfdParams.gridX;
    m_ny = cfdParams.gridY;
    m_nz = cfdParams.gridZ;
    m_cellSize = cfdParams.cellSize;
    
    // Fluid properties from config
    m_fluid.name = cfdParams.coolantType;
    m_fluid.inletVelocity = cfdParams.inletVelocity;
    m_fluid.inletTemperature = cfdParams.inletTemperature;
    
    // Default coolant properties based on type
    if (m_fluid.name == "Water" || m_fluid.name == "water") {
        m_fluid.density = 998.0;
        m_fluid.dynamicViscosity = 0.001;
        m_fluid.specificHeat = 4186.0;
        m_fluid.thermalConductivity = 0.6;
    } else if (m_fluid.name == "Oil" || m_fluid.name == "oil" || m_fluid.name == "Cutting-Oil") {
        m_fluid.density = 870.0;
        m_fluid.dynamicViscosity = 0.03;
        m_fluid.specificHeat = 1900.0;
        m_fluid.thermalConductivity = 0.15;
    } else if (m_fluid.name == "MWF" || m_fluid.name == "Emulsion") {
        m_fluid.density = 1000.0;
        m_fluid.dynamicViscosity = 0.002;
        m_fluid.specificHeat = 3500.0;
        m_fluid.thermalConductivity = 0.45;
    }
    // Default: Water-glycol mixture (already set in struct)
    
    // Allocate grid arrays
    int numCells = m_nx * m_ny * m_nz;
    int numU = (m_nx + 1) * m_ny * m_nz;
    int numV = m_nx * (m_ny + 1) * m_nz;
    int numW = m_nx * m_ny * (m_nz + 1);
    
    m_u.resize(numU, 0.0);
    m_v.resize(numV, 0.0);
    m_w.resize(numW, 0.0);
    m_uTemp.resize(numU, 0.0);
    m_vTemp.resize(numV, 0.0);
    m_wTemp.resize(numW, 0.0);
    m_pressure.resize(numCells, 0.0);
    m_temperature.resize(numCells, m_fluid.inletTemperature);
    m_tempNew.resize(numCells, m_fluid.inletTemperature);
    m_divergence.resize(numCells, 0.0);
    m_cellType.resize(numCells, FluidCellType::FLUID);
    
    // Conjugate gradient vectors
    m_r.resize(numCells, 0.0);
    m_d.resize(numCells, 0.0);
    m_q.resize(numCells, 0.0);
    
    // Set grid origin (default centered around tool)
    m_origin = Vec3(-m_nx * m_cellSize / 2, -m_ny * m_cellSize / 2, 0);
    
    m_isInitialized = true;
    
    std::cout << "[CFDSolver] Grid: " << m_nx << "x" << m_ny << "x" << m_nz 
              << " = " << numCells << " cells" << std::endl;
    std::cout << "[CFDSolver] Coolant: " << m_fluid.name 
              << ", inlet velocity: " << m_fluid.inletVelocity << " m/s" << std::endl;
    
    return true;
}

void CFDSolver::setGridParams(const CFDGridParams& params) {
    m_gridParams = params;
    m_nx = params.nx;
    m_ny = params.ny;
    m_nz = params.nz;
    m_cellSize = params.cellSize;
    m_origin = params.origin;
}

void CFDSolver::step(double dt) {
    if (!m_isInitialized) return;
    
    // 1. Apply boundary conditions
    applyBoundaryConditions();
    
    // 2. Advect velocity (semi-Lagrangian)
    advectVelocity(dt);
    
    // 3. Diffuse velocity (implicit)
    diffuseVelocity(dt);
    
    // 4. Add external forces (gravity, etc.)
    addExternalForces(dt);
    
    // 5. Project to make divergence-free
    projectPressure(dt);
    
    // 6. Advect temperature
    advectTemperature(dt);
    
    // 7. Diffuse temperature
    diffuseTemperature(dt);
    
    // 8. Coupling with particles/tool
    couplingStep(dt);
    
    m_currentTime += dt;
    
    // Update results
    updateResults();
}

void CFDSolver::advectVelocity(double dt) {
    // Semi-Lagrangian advection: trace back and interpolate
    
    // Advect U
    for (int k = 0; k < m_nz; ++k) {
        for (int j = 0; j < m_ny; ++j) {
            for (int i = 0; i <= m_nx; ++i) {
                // Position of U face
                double x = m_origin.x + i * m_cellSize;
                double y = m_origin.y + (j + 0.5) * m_cellSize;
                double z = m_origin.z + (k + 0.5) * m_cellSize;
                
                // Velocity at this position
                double u = interpolateU(x, y, z);
                double v = interpolateV(x, y, z);
                double w = interpolateW(x, y, z);
                
                // Trace back
                double x0 = x - u * dt;
                double y0 = y - v * dt;
                double z0 = z - w * dt;
                
                // Interpolate at backtrace position
                m_uTemp[idxU(i, j, k)] = interpolateU(x0, y0, z0);
            }
        }
    }
    
    // Advect V
    for (int k = 0; k < m_nz; ++k) {
        for (int j = 0; j <= m_ny; ++j) {
            for (int i = 0; i < m_nx; ++i) {
                double x = m_origin.x + (i + 0.5) * m_cellSize;
                double y = m_origin.y + j * m_cellSize;
                double z = m_origin.z + (k + 0.5) * m_cellSize;
                
                double u = interpolateU(x, y, z);
                double v = interpolateV(x, y, z);
                double w = interpolateW(x, y, z);
                
                double x0 = x - u * dt;
                double y0 = y - v * dt;
                double z0 = z - w * dt;
                
                m_vTemp[idxV(i, j, k)] = interpolateV(x0, y0, z0);
            }
        }
    }
    
    // Advect W
    for (int k = 0; k <= m_nz; ++k) {
        for (int j = 0; j < m_ny; ++j) {
            for (int i = 0; i < m_nx; ++i) {
                double x = m_origin.x + (i + 0.5) * m_cellSize;
                double y = m_origin.y + (j + 0.5) * m_cellSize;
                double z = m_origin.z + k * m_cellSize;
                
                double u = interpolateU(x, y, z);
                double v = interpolateV(x, y, z);
                double w = interpolateW(x, y, z);
                
                double x0 = x - u * dt;
                double y0 = y - v * dt;
                double z0 = z - w * dt;
                
                m_wTemp[idxW(i, j, k)] = interpolateW(x0, y0, z0);
            }
        }
    }
    
    // Swap
    std::swap(m_u, m_uTemp);
    std::swap(m_v, m_vTemp);
    std::swap(m_w, m_wTemp);
}

void CFDSolver::diffuseVelocity(double dt) {
    // Implicit diffusion using Gauss-Seidel iteration
    double nu = m_fluid.dynamicViscosity / m_fluid.density;
    double alpha = dt * nu / (m_cellSize * m_cellSize);
    
    // For simplicity, use a few Gauss-Seidel iterations
    int numIters = 5;
    
    for (int iter = 0; iter < numIters; ++iter) {
        // Diffuse U
        for (int k = 1; k < m_nz - 1; ++k) {
            for (int j = 1; j < m_ny - 1; ++j) {
                for (int i = 1; i < m_nx; ++i) {
                    double sum = m_u[idxU(i-1,j,k)] + m_u[idxU(i+1,j,k)] +
                                 m_u[idxU(i,j-1,k)] + m_u[idxU(i,j+1,k)] +
                                 m_u[idxU(i,j,k-1)] + m_u[idxU(i,j,k+1)];
                    m_u[idxU(i,j,k)] = (m_uTemp[idxU(i,j,k)] + alpha * sum) / (1 + 6 * alpha);
                }
            }
        }
        
        // Diffuse V (similar pattern)
        for (int k = 1; k < m_nz - 1; ++k) {
            for (int j = 1; j < m_ny; ++j) {
                for (int i = 1; i < m_nx - 1; ++i) {
                    double sum = m_v[idxV(i-1,j,k)] + m_v[idxV(i+1,j,k)] +
                                 m_v[idxV(i,j-1,k)] + m_v[idxV(i,j+1,k)] +
                                 m_v[idxV(i,j,k-1)] + m_v[idxV(i,j,k+1)];
                    m_v[idxV(i,j,k)] = (m_vTemp[idxV(i,j,k)] + alpha * sum) / (1 + 6 * alpha);
                }
            }
        }
        
        // Diffuse W
        for (int k = 1; k < m_nz; ++k) {
            for (int j = 1; j < m_ny - 1; ++j) {
                for (int i = 1; i < m_nx - 1; ++i) {
                    double sum = m_w[idxW(i-1,j,k)] + m_w[idxW(i+1,j,k)] +
                                 m_w[idxW(i,j-1,k)] + m_w[idxW(i,j+1,k)] +
                                 m_w[idxW(i,j,k-1)] + m_w[idxW(i,j,k+1)];
                    m_w[idxW(i,j,k)] = (m_wTemp[idxW(i,j,k)] + alpha * sum) / (1 + 6 * alpha);
                }
            }
        }
    }
}

void CFDSolver::addExternalForces(double dt) {
    (void)dt;
    // Add gravity (if significant)
    // For coolant, gravity is usually small compared to flow velocity
    // double g = -9.81;
    // for (int i = 0; i < m_v.size(); ++i) m_v[i] += g * dt;
    
    // Could add other body forces here
}

void CFDSolver::projectPressure(double dt) {
    // Compute divergence
    double scale = 1.0 / m_cellSize;
    
    for (int k = 0; k < m_nz; ++k) {
        for (int j = 0; j < m_ny; ++j) {
            for (int i = 0; i < m_nx; ++i) {
                if (m_cellType[idx(i,j,k)] == FluidCellType::SOLID) continue;
                
                double divU = (m_u[idxU(i+1,j,k)] - m_u[idxU(i,j,k)]) * scale;
                double divV = (m_v[idxV(i,j+1,k)] - m_v[idxV(i,j,k)]) * scale;
                double divW = (m_w[idxW(i,j,k+1)] - m_w[idxW(i,j,k)]) * scale;
                
                m_divergence[idx(i,j,k)] = -(divU + divV + divW);
            }
        }
    }
    
    // Solve pressure Poisson equation using conjugate gradient
    // ∇²p = divergence / dt
    
    // Initialize
    std::fill(m_pressure.begin(), m_pressure.end(), 0.0);
    std::copy(m_divergence.begin(), m_divergence.end(), m_r.begin());
    std::copy(m_r.begin(), m_r.end(), m_d.begin());
    
    double deltaNew = 0;
    for (int i = 0; i < (int)m_r.size(); ++i) {
        deltaNew += m_r[i] * m_r[i];
    }
    
    double tolerance = m_gridParams.tolerance * m_gridParams.tolerance * deltaNew;
    int maxIter = m_gridParams.maxIterations;
    
    for (int iter = 0; iter < maxIter; ++iter) {
        // q = A * d (Laplacian)
        for (int k = 1; k < m_nz - 1; ++k) {
            for (int j = 1; j < m_ny - 1; ++j) {
                for (int i = 1; i < m_nx - 1; ++i) {
                    int id = idx(i,j,k);
                    if (m_cellType[id] == FluidCellType::SOLID) {
                        m_q[id] = 0;
                        continue;
                    }
                    
                    double lap = (m_d[idx(i-1,j,k)] + m_d[idx(i+1,j,k)] +
                                  m_d[idx(i,j-1,k)] + m_d[idx(i,j+1,k)] +
                                  m_d[idx(i,j,k-1)] + m_d[idx(i,j,k+1)] -
                                  6 * m_d[id]) / (m_cellSize * m_cellSize);
                    m_q[id] = lap;
                }
            }
        }
        
        // alpha = deltaNew / (d · q)
        double dq = 0;
        for (int i = 0; i < (int)m_d.size(); ++i) {
            dq += m_d[i] * m_q[i];
        }
        if (std::abs(dq) < 1e-30) break;
        
        double alpha = deltaNew / dq;
        
        // p = p + alpha * d
        // r = r - alpha * q
        double deltaOld = deltaNew;
        deltaNew = 0;
        for (int i = 0; i < (int)m_pressure.size(); ++i) {
            m_pressure[i] += alpha * m_d[i];
            m_r[i] -= alpha * m_q[i];
            deltaNew += m_r[i] * m_r[i];
        }
        
        if (deltaNew < tolerance) break;
        
        // d = r + (deltaNew / deltaOld) * d
        double beta = deltaNew / deltaOld;
        for (int i = 0; i < (int)m_d.size(); ++i) {
            m_d[i] = m_r[i] + beta * m_d[i];
        }
    }
    
    // Apply pressure gradient to make velocity divergence-free
    double invDx = dt / (m_fluid.density * m_cellSize);
    
    for (int k = 0; k < m_nz; ++k) {
        for (int j = 0; j < m_ny; ++j) {
            for (int i = 1; i < m_nx; ++i) {
                m_u[idxU(i,j,k)] -= (m_pressure[idx(i,j,k)] - m_pressure[idx(i-1,j,k)]) * invDx;
            }
        }
    }
    
    for (int k = 0; k < m_nz; ++k) {
        for (int j = 1; j < m_ny; ++j) {
            for (int i = 0; i < m_nx; ++i) {
                m_v[idxV(i,j,k)] -= (m_pressure[idx(i,j,k)] - m_pressure[idx(i,j-1,k)]) * invDx;
            }
        }
    }
    
    for (int k = 1; k < m_nz; ++k) {
        for (int j = 0; j < m_ny; ++j) {
            for (int i = 0; i < m_nx; ++i) {
                m_w[idxW(i,j,k)] -= (m_pressure[idx(i,j,k)] - m_pressure[idx(i,j,k-1)]) * invDx;
            }
        }
    }
}

void CFDSolver::advectTemperature(double dt) {
    // Semi-Lagrangian advection for temperature
    for (int k = 0; k < m_nz; ++k) {
        for (int j = 0; j < m_ny; ++j) {
            for (int i = 0; i < m_nx; ++i) {
                if (m_cellType[idx(i,j,k)] == FluidCellType::SOLID) {
                    m_tempNew[idx(i,j,k)] = m_temperature[idx(i,j,k)];
                    continue;
                }
                
                double x = m_origin.x + (i + 0.5) * m_cellSize;
                double y = m_origin.y + (j + 0.5) * m_cellSize;
                double z = m_origin.z + (k + 0.5) * m_cellSize;
                
                double u = interpolateU(x, y, z);
                double v = interpolateV(x, y, z);
                double w = interpolateW(x, y, z);
                
                double x0 = x - u * dt;
                double y0 = y - v * dt;
                double z0 = z - w * dt;
                
                m_tempNew[idx(i,j,k)] = interpolateScalar(m_temperature, x0, y0, z0);
            }
        }
    }
    
    std::swap(m_temperature, m_tempNew);
}

void CFDSolver::diffuseTemperature(double dt) {
    // Implicit diffusion
    double alpha = m_fluid.thermalConductivity / (m_fluid.density * m_fluid.specificHeat);
    double coeff = dt * alpha / (m_cellSize * m_cellSize);
    
    // Gauss-Seidel iterations
    for (int iter = 0; iter < 5; ++iter) {
        for (int k = 1; k < m_nz - 1; ++k) {
            for (int j = 1; j < m_ny - 1; ++j) {
                for (int i = 1; i < m_nx - 1; ++i) {
                    if (m_cellType[idx(i,j,k)] == FluidCellType::SOLID) continue;
                    
                    double sum = m_temperature[idx(i-1,j,k)] + m_temperature[idx(i+1,j,k)] +
                                 m_temperature[idx(i,j-1,k)] + m_temperature[idx(i,j+1,k)] +
                                 m_temperature[idx(i,j,k-1)] + m_temperature[idx(i,j,k+1)];
                    
                    m_temperature[idx(i,j,k)] = (m_tempNew[idx(i,j,k)] + coeff * sum) / (1 + 6 * coeff);
                }
            }
        }
    }
}

void CFDSolver::applyBoundaryConditions() {
    // Set inlet velocity
    for (int k = 0; k < m_nz; ++k) {
        for (int j = 0; j < m_ny; ++j) {
            if (m_cellType[idx(0,j,k)] == FluidCellType::INFLOW) {
                m_u[idxU(0,j,k)] = m_fluid.inletVelocity;
                m_temperature[idx(0,j,k)] = m_fluid.inletTemperature;
            }
        }
    }
    
    // Set outlet (zero gradient)
    for (int k = 0; k < m_nz; ++k) {
        for (int j = 0; j < m_ny; ++j) {
            if (m_cellType[idx(m_nx-1,j,k)] == FluidCellType::OUTFLOW) {
                m_u[idxU(m_nx,j,k)] = m_u[idxU(m_nx-1,j,k)];
            }
        }
    }
    
    // No-slip at solid boundaries
    for (int k = 0; k < m_nz; ++k) {
        for (int j = 0; j < m_ny; ++j) {
            for (int i = 0; i < m_nx; ++i) {
                if (m_cellType[idx(i,j,k)] == FluidCellType::SOLID) {
                    if (i > 0) m_u[idxU(i,j,k)] = 0;
                    if (i < m_nx) m_u[idxU(i+1,j,k)] = 0;
                    if (j > 0) m_v[idxV(i,j,k)] = 0;
                    if (j < m_ny) m_v[idxV(i,j+1,k)] = 0;
                    if (k > 0) m_w[idxW(i,j,k)] = 0;
                    if (k < m_nz) m_w[idxW(i,j,k+1)] = 0;
                }
            }
        }
    }
}

void CFDSolver::couplingStep(double dt) {
    // Coupling with SPH particles
    if (m_sph) {
        // Apply fluid forces to particles
        auto particles = m_sph->getParticles();
        for (auto& p : particles) {
            if (p.status == ParticleStatus::CHIP) {
                // Drag force on chips
                Vec3 fluidVel = getVelocityAt(p.x, p.y, p.z);
                double relVx = fluidVel.x - p.vx;
                double relVy = fluidVel.y - p.vy;
                double relVz = fluidVel.z - p.vz;
                double relV = std::sqrt(relVx * relVx + relVy * relVy + relVz * relVz);
                
                // Drag coefficient (sphere approximation)
                double Cd = 0.44;
                double r = 0.0001;  // Particle radius estimate
                double A = 3.14159 * r * r;
                double dragMag = 0.5 * m_fluid.density * Cd * A * relV * relV;
                
                if (relV > 1e-12) {
                    p.fx += dragMag * relVx / relV;
                    p.fy += dragMag * relVy / relV;
                    p.fz += dragMag * relVz / relV;
                }
            }
        }
    }
    
    // Heat transfer from tool (FEM) to coolant
    if (m_fem) {
        auto nodes = m_fem->getNodes();
        
        double totalHeatRemoval = 0;
        int numNodes = 0;
        
        for (const auto& node : nodes) {
            if (node.inContact) continue;  // Skip nodes in workpiece contact
            
            // Find fluid cell at node location
            int i = static_cast<int>((node.x - m_origin.x) / m_cellSize);
            int j = static_cast<int>((node.y - m_origin.y) / m_cellSize);
            int k = static_cast<int>((node.z - m_origin.z) / m_cellSize);
            
            if (i < 0 || i >= m_nx || j < 0 || j >= m_ny || k < 0 || k >= m_nz) continue;
            
            // Heat transfer: q = h * A * (T_tool - T_fluid)
            double h = 5000;  // Heat transfer coefficient W/(m²·K)
            double A = m_cellSize * m_cellSize;
            double dT = node.temperature - m_temperature[idx(i,j,k)];
            double q = h * A * dT;
            
            // Heat removal from coolant perspective
            totalHeatRemoval += q;
            
            // Heat added to fluid
            m_temperature[idx(i,j,k)] += q * dt / (m_fluid.density * m_fluid.specificHeat * 
                                                    m_cellSize * m_cellSize * m_cellSize);
            
            numNodes++;
        }
        
        m_results.totalHeatRemoval = totalHeatRemoval;
        if (numNodes > 0) {
            m_results.averageHTC = 5000;  // Fixed for now
        }
    }
}

void CFDSolver::setSolidGeometry(const Mesh& toolMesh, const Vec3& workpieceMin, const Vec3& workpieceMax) {
    // Mark cells as solid based on geometry
    for (int k = 0; k < m_nz; ++k) {
        for (int j = 0; j < m_ny; ++j) {
            for (int i = 0; i < m_nx; ++i) {
                double x = m_origin.x + (i + 0.5) * m_cellSize;
                double y = m_origin.y + (j + 0.5) * m_cellSize;
                double z = m_origin.z + (k + 0.5) * m_cellSize;
                
                // Check workpiece bounding box
                if (x >= workpieceMin.x && x <= workpieceMax.x &&
                    y >= workpieceMin.y && y <= workpieceMax.y &&
                    z >= workpieceMin.z && z <= workpieceMax.z) {
                    m_cellType[idx(i,j,k)] = FluidCellType::SOLID;
                    continue;
                }
                
                // Check tool mesh (simplified - compute bounding box from nodes)
                if (toolMesh.nodes.size() > 0) {
                    Vec3 toolMin = toolMesh.nodes[0].position;
                    Vec3 toolMax = toolMesh.nodes[0].position;
                    for (const auto& node : toolMesh.nodes) {
                        toolMin.x = std::min(toolMin.x, node.position.x);
                        toolMin.y = std::min(toolMin.y, node.position.y);
                        toolMin.z = std::min(toolMin.z, node.position.z);
                        toolMax.x = std::max(toolMax.x, node.position.x);
                        toolMax.y = std::max(toolMax.y, node.position.y);
                        toolMax.z = std::max(toolMax.z, node.position.z);
                    }
                    
                    if (x >= toolMin.x && x <= toolMax.x &&
                        y >= toolMin.y && y <= toolMax.y &&
                        z >= toolMin.z && z <= toolMax.z) {
                        m_cellType[idx(i,j,k)] = FluidCellType::SOLID;
                    }
                }
            }
        }
    }
}

void CFDSolver::setInletOutlet(const Vec3& inletMin, const Vec3& inletMax,
                                const Vec3& outletMin, const Vec3& outletMax) {
    for (int k = 0; k < m_nz; ++k) {
        for (int j = 0; j < m_ny; ++j) {
            for (int i = 0; i < m_nx; ++i) {
                double x = m_origin.x + (i + 0.5) * m_cellSize;
                double y = m_origin.y + (j + 0.5) * m_cellSize;
                double z = m_origin.z + (k + 0.5) * m_cellSize;
                
                if (x >= inletMin.x && x <= inletMax.x &&
                    y >= inletMin.y && y <= inletMax.y &&
                    z >= inletMin.z && z <= inletMax.z) {
                    m_cellType[idx(i,j,k)] = FluidCellType::INFLOW;
                }
                
                if (x >= outletMin.x && x <= outletMax.x &&
                    y >= outletMin.y && y <= outletMax.y &&
                    z >= outletMin.z && z <= outletMax.z) {
                    m_cellType[idx(i,j,k)] = FluidCellType::OUTFLOW;
                }
            }
        }
    }
}

// Interpolation methods
double CFDSolver::interpolateU(double x, double y, double z) const {
    // U is at face centers (i, j+0.5, k+0.5)
    double fx = (x - m_origin.x) / m_cellSize;
    double fy = (y - m_origin.y) / m_cellSize - 0.5;
    double fz = (z - m_origin.z) / m_cellSize - 0.5;
    
    int i0 = clamp((int)fx, 0, m_nx);
    int j0 = clamp((int)fy, 0, m_ny - 1);
    int k0 = clamp((int)fz, 0, m_nz - 1);
    int i1 = std::min(i0 + 1, m_nx);
    int j1 = std::min(j0 + 1, m_ny - 1);
    int k1 = std::min(k0 + 1, m_nz - 1);
    
    double tx = clamp(fx - i0, 0.0, 1.0);
    double ty = clamp(fy - j0, 0.0, 1.0);
    double tz = clamp(fz - k0, 0.0, 1.0);
    
    return trilerp(
        m_u[idxU(i0,j0,k0)], m_u[idxU(i1,j0,k0)], m_u[idxU(i0,j1,k0)], m_u[idxU(i1,j1,k0)],
        m_u[idxU(i0,j0,k1)], m_u[idxU(i1,j0,k1)], m_u[idxU(i0,j1,k1)], m_u[idxU(i1,j1,k1)],
        tx, ty, tz
    );
}

double CFDSolver::interpolateV(double x, double y, double z) const {
    double fx = (x - m_origin.x) / m_cellSize - 0.5;
    double fy = (y - m_origin.y) / m_cellSize;
    double fz = (z - m_origin.z) / m_cellSize - 0.5;
    
    int i0 = clamp((int)fx, 0, m_nx - 1);
    int j0 = clamp((int)fy, 0, m_ny);
    int k0 = clamp((int)fz, 0, m_nz - 1);
    int i1 = std::min(i0 + 1, m_nx - 1);
    int j1 = std::min(j0 + 1, m_ny);
    int k1 = std::min(k0 + 1, m_nz - 1);
    
    double tx = clamp(fx - i0, 0.0, 1.0);
    double ty = clamp(fy - j0, 0.0, 1.0);
    double tz = clamp(fz - k0, 0.0, 1.0);
    
    return trilerp(
        m_v[idxV(i0,j0,k0)], m_v[idxV(i1,j0,k0)], m_v[idxV(i0,j1,k0)], m_v[idxV(i1,j1,k0)],
        m_v[idxV(i0,j0,k1)], m_v[idxV(i1,j0,k1)], m_v[idxV(i0,j1,k1)], m_v[idxV(i1,j1,k1)],
        tx, ty, tz
    );
}

double CFDSolver::interpolateW(double x, double y, double z) const {
    double fx = (x - m_origin.x) / m_cellSize - 0.5;
    double fy = (y - m_origin.y) / m_cellSize - 0.5;
    double fz = (z - m_origin.z) / m_cellSize;
    
    int i0 = clamp((int)fx, 0, m_nx - 1);
    int j0 = clamp((int)fy, 0, m_ny - 1);
    int k0 = clamp((int)fz, 0, m_nz);
    int i1 = std::min(i0 + 1, m_nx - 1);
    int j1 = std::min(j0 + 1, m_ny - 1);
    int k1 = std::min(k0 + 1, m_nz);
    
    double tx = clamp(fx - i0, 0.0, 1.0);
    double ty = clamp(fy - j0, 0.0, 1.0);
    double tz = clamp(fz - k0, 0.0, 1.0);
    
    return trilerp(
        m_w[idxW(i0,j0,k0)], m_w[idxW(i1,j0,k0)], m_w[idxW(i0,j1,k0)], m_w[idxW(i1,j1,k0)],
        m_w[idxW(i0,j0,k1)], m_w[idxW(i1,j0,k1)], m_w[idxW(i0,j1,k1)], m_w[idxW(i1,j1,k1)],
        tx, ty, tz
    );
}

double CFDSolver::interpolateScalar(const std::vector<double>& field, double x, double y, double z) const {
    double fx = (x - m_origin.x) / m_cellSize - 0.5;
    double fy = (y - m_origin.y) / m_cellSize - 0.5;
    double fz = (z - m_origin.z) / m_cellSize - 0.5;
    
    int i0 = clamp((int)fx, 0, m_nx - 1);
    int j0 = clamp((int)fy, 0, m_ny - 1);
    int k0 = clamp((int)fz, 0, m_nz - 1);
    int i1 = std::min(i0 + 1, m_nx - 1);
    int j1 = std::min(j0 + 1, m_ny - 1);
    int k1 = std::min(k0 + 1, m_nz - 1);
    
    double tx = clamp(fx - i0, 0.0, 1.0);
    double ty = clamp(fy - j0, 0.0, 1.0);
    double tz = clamp(fz - k0, 0.0, 1.0);
    
    return trilerp(
        field[idx(i0,j0,k0)], field[idx(i1,j0,k0)], field[idx(i0,j1,k0)], field[idx(i1,j1,k0)],
        field[idx(i0,j0,k1)], field[idx(i1,j0,k1)], field[idx(i0,j1,k1)], field[idx(i1,j1,k1)],
        tx, ty, tz
    );
}

Vec3 CFDSolver::getVelocityAt(double x, double y, double z) const {
    return Vec3(interpolateU(x, y, z), interpolateV(x, y, z), interpolateW(x, y, z));
}

void CFDSolver::applyHeatFlux(double x, double y, double z, double heatFlux) {
    int i = static_cast<int>((x - m_origin.x) / m_cellSize);
    int j = static_cast<int>((y - m_origin.y) / m_cellSize);
    int k = static_cast<int>((z - m_origin.z) / m_cellSize);
    
    if (i >= 0 && i < m_nx && j >= 0 && j < m_ny && k >= 0 && k < m_nz) {
        double volume = m_cellSize * m_cellSize * m_cellSize;
        double dT = heatFlux / (m_fluid.density * m_fluid.specificHeat * volume);
        m_temperature[idx(i,j,k)] += dT;
    }
}

double CFDSolver::getTemperatureAt(double x, double y, double z) const {
    return interpolateScalar(m_temperature, x, y, z);
}

double CFDSolver::getStableTimeStep() const {
    // CFL condition: dt < cell_size / max_velocity
    double maxV = std::max(m_results.maxVelocity, 0.001);
    double dt_advection = 0.5 * m_cellSize / maxV;
    
    // Diffusion stability
    double nu = m_fluid.dynamicViscosity / m_fluid.density;
    double dt_diffusion = 0.25 * m_cellSize * m_cellSize / nu;
    
    return std::min(dt_advection, dt_diffusion);
}

void CFDSolver::updateResults() {
    m_results.maxVelocity = 0;
    m_results.maxPressure = 0;
    m_results.maxTemperature = 0;
    m_results.minTemperature = 1000;
    
    for (int k = 0; k < m_nz; ++k) {
        for (int j = 0; j < m_ny; ++j) {
            for (int i = 0; i < m_nx; ++i) {
                if (m_cellType[idx(i,j,k)] == FluidCellType::SOLID) continue;
                
                double u = 0.5 * (m_u[idxU(i,j,k)] + m_u[idxU(i+1,j,k)]);
                double v = 0.5 * (m_v[idxV(i,j,k)] + m_v[idxV(i,j+1,k)]);
                double w = 0.5 * (m_w[idxW(i,j,k)] + m_w[idxW(i,j,k+1)]);
                double vel = std::sqrt(u*u + v*v + w*w);
                
                m_results.maxVelocity = std::max(m_results.maxVelocity, vel);
                m_results.maxPressure = std::max(m_results.maxPressure, std::abs(m_pressure[idx(i,j,k)]));
                m_results.maxTemperature = std::max(m_results.maxTemperature, m_temperature[idx(i,j,k)]);
                m_results.minTemperature = std::min(m_results.minTemperature, m_temperature[idx(i,j,k)]);
            }
        }
    }
}

double CFDSolver::getTotalKineticEnergy() const {
    double ke = 0;
    double cellMass = m_fluid.density * m_cellSize * m_cellSize * m_cellSize;
    
    for (int k = 0; k < m_nz; ++k) {
        for (int j = 0; j < m_ny; ++j) {
            for (int i = 0; i < m_nx; ++i) {
                if (m_cellType[idx(i,j,k)] == FluidCellType::SOLID) continue;
                
                double u = 0.5 * (m_u[idxU(i,j,k)] + m_u[idxU(i+1,j,k)]);
                double v = 0.5 * (m_v[idxV(i,j,k)] + m_v[idxV(i,j+1,k)]);
                double w = 0.5 * (m_w[idxW(i,j,k)] + m_w[idxW(i,j,k+1)]);
                
                ke += 0.5 * cellMass * (u*u + v*v + w*w);
            }
        }
    }
    
    return ke;
}

void CFDSolver::reset() {
    std::fill(m_u.begin(), m_u.end(), 0.0);
    std::fill(m_v.begin(), m_v.end(), 0.0);
    std::fill(m_w.begin(), m_w.end(), 0.0);
    std::fill(m_pressure.begin(), m_pressure.end(), 0.0);
    std::fill(m_temperature.begin(), m_temperature.end(), m_fluid.inletTemperature);
    m_currentTime = 0;
    m_results = CFDResults();
}

void CFDSolver::getBounds(double& minX, double& minY, double& minZ, 
                         double& maxX, double& maxY, double& maxZ) const {
    minX = m_origin.x;
    minY = m_origin.y;
    minZ = m_origin.z;
    maxX = m_origin.x + m_nx * m_cellSize;
    maxY = m_origin.y + m_ny * m_cellSize;
    maxZ = m_origin.z + m_nz * m_cellSize;
}

void CFDSolver::exportToVTK(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[CFDSolver] Failed to open: " << filename << std::endl;
        return;
    }
    
    file << "# vtk DataFile Version 3.0\n";
    file << "EdgePredict CFD\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << m_nx << " " << m_ny << " " << m_nz << "\n";
    file << "ORIGIN " << m_origin.x << " " << m_origin.y << " " << m_origin.z << "\n";
    file << "SPACING " << m_cellSize << " " << m_cellSize << " " << m_cellSize << "\n";
    
    int numCells = m_nx * m_ny * m_nz;
    file << "\nPOINT_DATA " << numCells << "\n";
    
    // Velocity
    file << "VECTORS Velocity double\n";
    for (int k = 0; k < m_nz; ++k) {
        for (int j = 0; j < m_ny; ++j) {
            for (int i = 0; i < m_nx; ++i) {
                double u = 0.5 * (m_u[idxU(i,j,k)] + m_u[idxU(std::min(i+1,m_nx),j,k)]);
                double v = 0.5 * (m_v[idxV(i,j,k)] + m_v[idxV(i,std::min(j+1,m_ny),k)]);
                double w = 0.5 * (m_w[idxW(i,j,k)] + m_w[idxW(i,j,std::min(k+1,m_nz))]);
                file << u << " " << v << " " << w << "\n";
            }
        }
    }
    
    // Temperature
    file << "\nSCALARS Temperature double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int k = 0; k < m_nz; ++k) {
        for (int j = 0; j < m_ny; ++j) {
            for (int i = 0; i < m_nx; ++i) {
                file << m_temperature[idx(i,j,k)] << "\n";
            }
        }
    }
    
    // Pressure
    file << "\nSCALARS Pressure double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int k = 0; k < m_nz; ++k) {
        for (int j = 0; j < m_ny; ++j) {
            for (int i = 0; i < m_nx; ++i) {
                file << m_pressure[idx(i,j,k)] << "\n";
            }
        }
    }
    
    file.close();
}

} // namespace edgepredict
