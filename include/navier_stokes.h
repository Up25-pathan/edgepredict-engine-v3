#ifndef NAVIER_STOKES_H
#define NAVIER_STOKES_H

#include <vector>
#include <Eigen/Dense>
#include "json.hpp"
#include "simulation.h" 
#include "physics_models.h" 

using json = nlohmann::json;

class NavierStokesSolver {
public:
    NavierStokesSolver(const json& config);
    ~NavierStokesSolver();

    void initialize_grid(int nx, int ny, int nz, double cell_size);
    void set_inlet_velocity(const Eigen::Vector3d& vel);
    void update_boundaries_and_sources(const Mesh& mesh, double ambient_temp, double strain_rate, const ThermalModel& thermal_model);
    void update(double dt);

    // Getters for SPH interaction
    double get_temperature_at(const Eigen::Vector3d& pos) const;
    double get_pressure_at(const Eigen::Vector3d& pos) const;
    Eigen::Vector3d get_velocity_at(const Eigen::Vector3d& pos) const;
    
    // --- NEW: Physics Coupling ---
    // Returns the pressure gradient (Force Direction) at a point
    Eigen::Vector3d get_pressure_gradient_at(const Eigen::Vector3d& pos) const;
    
    std::vector<double> get_rake_face_pressure(const Eigen::Vector3d& normal) const;

private:
    int get_grid_index(int i, int j, int k) const;
    Eigen::Vector3d get_grid_position(int i, int j, int k) const;
    int find_cell_index_for_position(const Eigen::Vector3d& pos) const;

    // Trilinear interpolation helper
    template <typename T>
    T trilinear_interp(const Eigen::Vector3d& pos, const std::vector<T>& grid) const;

    // Physics Steps
    void advect(double dt);
    void diffuse(double dt);
    void project(double dt);

    json config_data;
    
    int grid_nx, grid_ny, grid_nz;
    double grid_cell_size;
    Eigen::Vector3d inlet_velocity;

    // Fluid Properties
    double density;
    double viscosity;
    double thermal_diffusivity;

    // Grids (Double buffered)
    std::vector<double> pressure_grid;
    std::vector<double> pressure_grid_next;
    
    std::vector<Eigen::Vector3d> velocity_grid;
    std::vector<Eigen::Vector3d> velocity_grid_next;
    
    std::vector<double> temperature_grid;
    std::vector<double> temperature_grid_next;
    
    std::vector<double> heat_source_grid;
    std::vector<double> divergence_grid;
    std::vector<bool> is_boundary_cell; 
};

#endif // NAVIER_STOKES_H