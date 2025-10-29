#ifndef NAVIER_STOKES_H
#define NAVIER_STOKES_H

#include <Eigen/Dense>
#include <vector>
#include "json.hpp"

using json = nlohmann::json;

// ═══════════════════════════════════════════════════════════
// SIMPLIFIED NAVIER-STOKES SOLVER FOR CHIP FLOW
// Uses Semi-Lagrangian method for stability
// ═══════════════════════════════════════════════════════════

struct FluidCell {
    Eigen::Vector3d velocity;      // m/s
    double pressure;                // Pa
    double temperature;             // °C
    double density;                 // kg/m³
    bool is_solid;                  // Boundary condition
};

class NavierStokesSolver {
public:
    NavierStokesSolver(const json& config);
    
    // Initialize fluid grid around tool
    void initialize_grid(int nx, int ny, int nz, double cell_size);
    
    // Time step update
    void update(double dt);
    
    // Boundary conditions
    void set_inlet_velocity(const Eigen::Vector3d& velocity);
    void set_tool_boundary(const std::vector<Eigen::Vector3d>& tool_nodes);
    
    // Apply heat source from cutting (FEA coupling)
    void apply_heat_source(const Eigen::Vector3d& position, double heat_flux);
    
    // Query results
    double get_pressure_at(const Eigen::Vector3d& position) const;
    Eigen::Vector3d get_velocity_at(const Eigen::Vector3d& position) const;
    double get_temperature_at(const Eigen::Vector3d& position) const;
    
    // Get all cells for visualization
    std::vector<FluidCell> get_all_cells() const;
    
    // Export pressure field on rake face
    std::vector<double> get_rake_face_pressure(const Eigen::Vector3d& rake_normal) const;

private:
    // Grid parameters
    int nx, ny, nz;
    double dx; // cell size
    std::vector<std::vector<std::vector<FluidCell>>> grid;
    
    // Fluid properties
    double viscosity;      // Pa·s
    double base_density;   // kg/m³
    double specific_heat;  // J/(kg·K)
    double thermal_conductivity;
    
    // Solver methods
    void advect_velocity(double dt);
    void apply_external_forces(double dt);
    void project_velocity();
    void advect_temperature(double dt);
    void diffuse_heat(double dt);
    
    // Helper functions
    int get_index_x(double x) const;
    int get_index_y(double y) const;
    int get_index_z(double z) const;
    bool is_valid_index(int i, int j, int k) const;
    
    Eigen::Vector3d interpolate_velocity(const Eigen::Vector3d& pos) const;
    double interpolate_scalar(const Eigen::Vector3d& pos, 
                              const std::vector<std::vector<std::vector<double>>>& field) const;
};

#endif // NAVIER_STOKES_H