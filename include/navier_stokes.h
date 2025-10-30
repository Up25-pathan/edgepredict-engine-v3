#ifndef NAVIER_STOKES_H
#define NAVIER_STOKES_H

// #include "simulation.h" // REMOVED direct include to avoid potential circular dependency
#include <vector>
#include <Eigen/Dense>
#include "json.hpp"
// Include physics_models.h to access ThermalModel for heat flux calculation
#include "physics_models.h"

using json = nlohmann::json;

// --- FORWARD DECLARATION FIX ---
// Tell the compiler that Mesh is a struct type defined elsewhere.
// This is sufficient because we only use it as a reference (Mesh&) in this header.
struct Mesh;
// --- END FORWARD DECLARATION FIX ---

// Forward declare ThermalModel (already included via physics_models.h)
// class ThermalModel;

class NavierStokesSolver {
public:
    NavierStokesSolver(const json& config);
    ~NavierStokesSolver();

    void initialize_grid(int nx, int ny, int nz, double cell_size);
    void set_inlet_velocity(const Eigen::Vector3d& vel);

    // --- API CHANGE: Accept Mesh reference directly ---
    // NEW combined function to handle boundary conditions and sources by reading the mesh
    void update_boundaries_and_sources(const Mesh& mesh, double ambient_temp, double strain_rate, const ThermalModel& thermal_model);
    // --- END API CHANGE ---


    void update(double dt); // Time-stepping function remains the same

    // Functions to get results (signatures might depend on your implementation)
    double get_temperature_at(const Eigen::Vector3d& pos) const;
    double get_pressure_at(const Eigen::Vector3d& pos) const;
    Eigen::Vector3d get_velocity_at(const Eigen::Vector3d& pos) const;
    std::vector<double> get_rake_face_pressure(const Eigen::Vector3d& normal) const;


private:
    // Internal state of the solver
    json config_data;
    int grid_nx, grid_ny, grid_nz;
    double grid_cell_size;
    Eigen::Vector3d inlet_velocity;

    // Example internal grid data (adjust based on your solver method - FDM, FVM, etc.)
    std::vector<double> pressure_grid;
    std::vector<Eigen::Vector3d> velocity_grid;
    std::vector<double> temperature_grid;
    // Example: Add a vector to store heat source terms for the temperature equation
    std::vector<double> heat_source_grid;
    // Example: Add a vector/marker for boundary cells
    std::vector<bool> is_boundary_cell;

    // Helper methods for internal calculations
    int get_grid_index(int i, int j, int k) const;
    Eigen::Vector3d get_grid_position(int i, int j, int k) const;
    // Helper to find the grid cell index containing a given world position
    int find_cell_index_for_position(const Eigen::Vector3d& pos) const;
    // ... other private methods for solving NS equations ...
};

#endif // NAVIER_STOKES_H