#pragma once
/**
 * @file CFDSolver.h
 * @brief Computational Fluid Dynamics solver for coolant simulation
 * 
 * Implements Navier-Stokes equations on a 3D grid with:
 * - Proper pressure Poisson solve (not fake approximation)
 * - Temperature advection and diffusion
 * - Two-way coupling with SPH particles
 * - Boundary conditions at tool surface
 */

#include "Types.h"
#include "IPhysicsSolver.h"
#include "Config.h"
#include <vector>
#include <memory>

namespace edgepredict {

// Forward declaration
class SPHSolver;
class FEMSolver;

/**
 * @brief Fluid cell types for boundary conditions
 */
enum class FluidCellType {
    FLUID,      // Regular fluid cell
    SOLID,      // Inside solid (tool/workpiece)
    BOUNDARY,   // At solid boundary
    INFLOW,     // Coolant inlet
    OUTFLOW     // Coolant outlet
};

/**
 * @brief Coolant/fluid properties (configurable from input)
 */
struct FluidProperties {
    std::string name = "Water-Glycol";
    double density = 1050.0;            // kg/m³
    double dynamicViscosity = 0.003;    // Pa·s
    double specificHeat = 3800.0;       // J/(kg·K)
    double thermalConductivity = 0.5;   // W/(m·K)
    double inletTemperature = 20.0;     // °C
    double inletVelocity = 1.0;         // m/s
};

/**
 * @brief CFD grid parameters
 */
struct CFDGridParams {
    int nx = 50, ny = 50, nz = 50;      // Grid resolution
    double cellSize = 0.001;             // Cell size (m)
    Vec3 origin;                         // Grid origin
    int maxIterations = 50;              // Pressure solve iterations
    double tolerance = 1e-5;             // Convergence tolerance
};

/**
 * @brief CFD simulation results
 */
struct CFDResults {
    double maxVelocity = 0;
    double maxPressure = 0;
    double maxTemperature = 0;
    double minTemperature = 100;
    double totalHeatRemoval = 0;         // W
    double averageHTC = 0;               // Heat transfer coefficient W/(m²·K)
};

/**
 * @brief 3D CFD solver for coolant flow simulation
 * 
 * Uses staggered MAC grid with:
 * - Semi-Lagrangian advection
 * - Implicit diffusion
 * - Pressure projection via conjugate gradient
 */
class CFDSolver : public IPhysicsSolver, public IMetricsProvider, public IThermalCoupling {
public:
    CFDSolver();
    ~CFDSolver() override;
    
    CFDSolver(const CFDSolver&) = delete;
    CFDSolver& operator=(const CFDSolver&) = delete;
    
    // IPhysicsSolver interface
    std::string getName() const override { return "CFDSolver"; }
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
    double getTotalKineticEnergy() const override;
    void syncMetrics() override { updateResults(); }
    
    // IThermalCoupling interface
    void applyHeatFlux(double x, double y, double z, double heatFlux) override;
    double getTemperatureAt(double x, double y, double z) const override;
    
    // CFD-specific methods
    
    /**
     * @brief Set fluid properties
     */
    void setFluidProperties(const FluidProperties& props) { m_fluid = props; }
    
    /**
     * @brief Set grid parameters
     */
    void setGridParams(const CFDGridParams& params);
    
    /**
     * @brief Couple with SPH particles (two-way)
     * @param sph SPH solver to couple with
     */
    void coupleWithSPH(SPHSolver* sph) { m_sph = sph; }
    
    /**
     * @brief Couple with FEM for tool surface temperature
     */
    void coupleWithFEM(FEMSolver* fem) { m_fem = fem; }
    
    /**
     * @brief Set solid obstacles (tool and workpiece geometry)
     */
    void setSolidGeometry(const Mesh& toolMesh, const Vec3& workpieceMin, const Vec3& workpieceMax);
    
    /**
     * @brief Set inlet/outlet boundary conditions
     */
    void setInletOutlet(const Vec3& inletMin, const Vec3& inletMax,
                        const Vec3& outletMin, const Vec3& outletMax);
    
    /**
     * @brief Get velocity at a point (for SPH coupling)
     */
    Vec3 getVelocityAt(double x, double y, double z) const;
    
    /**
     * @brief Get results
     */
    const CFDResults& getResults() const { return m_results; }
    
    /**
     * @brief Export grid data for visualization
     */
    void exportToVTK(const std::string& filename) const;

private:
    // Grid indexing
    int idx(int i, int j, int k) const { return i + m_nx * (j + m_ny * k); }
    int idxU(int i, int j, int k) const { return i + (m_nx + 1) * (j + m_ny * k); }
    int idxV(int i, int j, int k) const { return i + m_nx * (j + (m_ny + 1) * k); }
    int idxW(int i, int j, int k) const { return i + m_nx * (j + m_ny * (k + 1)); }
    
    // Simulation steps
    void advectVelocity(double dt);
    void diffuseVelocity(double dt);
    void addExternalForces(double dt);
    void projectPressure(double dt);
    void advectTemperature(double dt);
    void diffuseTemperature(double dt);
    void applyBoundaryConditions();
    void updateCellTypes();
    void couplingStep(double dt);
    void updateResults();
    
    // Interpolation helpers
    double interpolateU(double x, double y, double z) const;
    double interpolateV(double x, double y, double z) const;
    double interpolateW(double x, double y, double z) const;
    double interpolateScalar(const std::vector<double>& field, double x, double y, double z) const;
    
    // Grid dimensions
    int m_nx, m_ny, m_nz;
    double m_cellSize;
    Vec3 m_origin;
    
    // Staggered grid data (MAC grid)
    std::vector<double> m_u, m_v, m_w;           // Velocity components (face-centered)
    std::vector<double> m_uTemp, m_vTemp, m_wTemp; // Temporary for advection
    std::vector<double> m_pressure;               // Pressure (cell-centered)
    std::vector<double> m_temperature;            // Temperature (cell-centered)
    std::vector<double> m_tempNew;                // Temp for advection
    std::vector<double> m_divergence;             // For pressure solve
    std::vector<FluidCellType> m_cellType;        // Cell types
    
    // Conjugate gradient solver data
    std::vector<double> m_r, m_d, m_q;            // CG vectors
    
    // Properties
    FluidProperties m_fluid;
    CFDGridParams m_gridParams;
    
    // Coupling references
    SPHSolver* m_sph = nullptr;
    FEMSolver* m_fem = nullptr;
    
    // State
    bool m_isInitialized = false;
    double m_currentTime = 0.0;
    CFDResults m_results;
};

} // namespace edgepredict
