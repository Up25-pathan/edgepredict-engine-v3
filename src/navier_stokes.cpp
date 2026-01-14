#include "navier_stokes.h"
#include "simulation.h" 
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <algorithm>

NavierStokesSolver::NavierStokesSolver(const json& config) : config_data(config) {
    grid_nx = grid_ny = grid_nz = 0;
    grid_cell_size = 0.0;
    inlet_velocity = Eigen::Vector3d::Zero();
    
    // Load fluid properties with defaults
    if (config.contains("cfd_parameters")) {
        density = config["cfd_parameters"].value("density", 1000.0); 
        viscosity = config["cfd_parameters"].value("viscosity", 0.01); 
        thermal_diffusivity = config["cfd_parameters"].value("thermal_diffusivity", 0.01); 
    } else {
        density = 1000.0;
        viscosity = 0.01;
        thermal_diffusivity = 0.01;
    }
    std::cout << "  NavierStokesSolver constructed." << std::endl;
}

NavierStokesSolver::~NavierStokesSolver() {}

void NavierStokesSolver::initialize_grid(int nx, int ny, int nz, double cell_size) {
    grid_nx = nx; grid_ny = ny; grid_nz = nz;
    grid_cell_size = cell_size;
    int total_cells = nx * ny * nz;

    // Resize Grids
    pressure_grid.resize(total_cells, 0.0);
    velocity_grid.resize(total_cells, Eigen::Vector3d::Zero());
    temperature_grid.resize(total_cells, config_data["physics_parameters"].value("ambient_temperature_C", 25.0));
    heat_source_grid.resize(total_cells, 0.0);
    is_boundary_cell.resize(total_cells, false);

    // Resize Next-Step Buffers
    pressure_grid_next.resize(total_cells, 0.0);
    velocity_grid_next.resize(total_cells, Eigen::Vector3d::Zero());
    temperature_grid_next.resize(total_cells, config_data["physics_parameters"].value("ambient_temperature_C", 25.0));
    divergence_grid.resize(total_cells, 0.0);

    std::cout << "  Navier-Stokes grid initialized (" << nx << "x" << ny << "x" << nz << ")" << std::endl;
}

void NavierStokesSolver::set_inlet_velocity(const Eigen::Vector3d& vel) {
    inlet_velocity = vel;
}

void NavierStokesSolver::update_boundaries_and_sources(const Mesh& mesh, double ambient_temp, double strain_rate, const ThermalModel& thermal_model) {
    // Reset sources and boundaries
    std::fill(heat_source_grid.begin(), heat_source_grid.end(), 0.0);
    std::fill(is_boundary_cell.begin(), is_boundary_cell.end(), false);

    // Map Mesh to Fluid Grid (Voxelization)
    for (size_t i = 0; i < mesh.nodes.size(); ++i) {
        const Node& node = mesh.nodes[i];
        int cell_idx = find_cell_index_for_position(node.position);
        
        if (cell_idx >= 0 && cell_idx < (int)is_boundary_cell.size()) {
             is_boundary_cell[cell_idx] = true;
             velocity_grid[cell_idx] = Eigen::Vector3d::Zero(); // No slip condition
             temperature_grid[cell_idx] = node.temperature; 
        }
        
        // --- FIX: Added cast (int) to NodeStatus::OK ---
        if (node.temperature > ambient_temp + 10.0 && node.status == (int)NodeStatus::OK) {
            double heat_flux_density = thermal_model.calculate_heat_generation(node.stress, strain_rate);
            if (cell_idx >= 0 && cell_idx < (int)heat_source_grid.size()) {
                 double cell_volume = grid_cell_size * grid_cell_size * grid_cell_size;
                 heat_source_grid[cell_idx] += heat_flux_density * cell_volume;
            }
        }
    }

    // Set Inlet Boundary Condition
    for(int k=0; k<grid_nz; ++k) {
        for(int j=0; j<grid_ny; ++j) {
            int inlet_idx = get_grid_index(0, j, k);
             if (inlet_idx >= 0 && inlet_idx < (int)velocity_grid.size() && !is_boundary_cell[inlet_idx]) {
                 velocity_grid[inlet_idx] = inlet_velocity;
                 temperature_grid[inlet_idx] = ambient_temp; 
            }
        }
    }
}

template <typename T>
T NavierStokesSolver::trilinear_interp(const Eigen::Vector3d& pos, const std::vector<T>& grid) const {
    if (grid_cell_size <= 0) return T();
    Eigen::Vector3d grid_pos = pos / grid_cell_size;
    int i = static_cast<int>(grid_pos.x());
    int j = static_cast<int>(grid_pos.y());
    int k = static_cast<int>(grid_pos.z());

    // Boundary checks
    if (i < 0 || i >= grid_nx - 1 || j < 0 || j >= grid_ny - 1 || k < 0 || k >= grid_nz - 1) {
        // Return nearest valid or zero
        return T(); 
    }

    double tx = grid_pos.x() - i;
    double ty = grid_pos.y() - j;
    double tz = grid_pos.z() - k;

    T c000 = grid[get_grid_index(i, j, k)];
    T c100 = grid[get_grid_index(i + 1, j, k)];
    T c010 = grid[get_grid_index(i, j + 1, k)];
    T c110 = grid[get_grid_index(i + 1, j + 1, k)];
    T c001 = grid[get_grid_index(i, j, k + 1)];
    T c101 = grid[get_grid_index(i + 1, j, k + 1)];
    T c011 = grid[get_grid_index(i, j + 1, k + 1)];
    T c111 = grid[get_grid_index(i + 1, j + 1, k + 1)];

    T c00 = c000 * (1 - tx) + c100 * tx;
    T c01 = c001 * (1 - tx) + c101 * tx;
    T c10 = c010 * (1 - tx) + c110 * tx;
    T c11 = c011 * (1 - tx) + c111 * tx;
    T c0 = c00 * (1 - ty) + c10 * ty;
    T c1 = c01 * (1 - ty) + c11 * ty;
    return c0 * (1 - tz) + c1 * tz;
}

double NavierStokesSolver::get_temperature_at(const Eigen::Vector3d& pos) const {
    int idx = find_cell_index_for_position(pos);
    if (idx < 0 || idx >= (int)temperature_grid.size()) 
        return config_data["physics_parameters"].value("ambient_temperature_C", 25.0);
    return trilinear_interp(pos, temperature_grid);
}

double NavierStokesSolver::get_pressure_at(const Eigen::Vector3d& pos) const {
    int idx = find_cell_index_for_position(pos);
    if (idx < 0 || idx >= (int)pressure_grid.size()) return 0.0;
    return trilinear_interp(pos, pressure_grid);
}

Eigen::Vector3d NavierStokesSolver::get_velocity_at(const Eigen::Vector3d& pos) const {
    int idx = find_cell_index_for_position(pos);
    if (idx < 0 || idx >= (int)velocity_grid.size()) return Eigen::Vector3d::Zero();
    return trilinear_interp(pos, velocity_grid);
}

Eigen::Vector3d NavierStokesSolver::get_pressure_gradient_at(const Eigen::Vector3d& pos) const {
    double h = grid_cell_size;
    if (h <= 0) return Eigen::Vector3d::Zero();

    // Central Difference: Grad(P) = (P(x+h) - P(x-h)) / 2h
    double dx = 0.5 * h; 
    
    double p_x1 = get_pressure_at(pos + Eigen::Vector3d(dx, 0, 0));
    double p_x0 = get_pressure_at(pos - Eigen::Vector3d(dx, 0, 0));
    
    double p_y1 = get_pressure_at(pos + Eigen::Vector3d(0, dx, 0));
    double p_y0 = get_pressure_at(pos - Eigen::Vector3d(0, dx, 0));
    
    double p_z1 = get_pressure_at(pos + Eigen::Vector3d(0, 0, dx));
    double p_z0 = get_pressure_at(pos - Eigen::Vector3d(0, 0, dx));

    return Eigen::Vector3d(
        (p_x1 - p_x0) / (2.0 * dx),
        (p_y1 - p_y0) / (2.0 * dx),
        (p_z1 - p_z0) / (2.0 * dx)
    );
}

std::vector<double> NavierStokesSolver::get_rake_face_pressure(const Eigen::Vector3d& normal) const {
    std::vector<double> pressures;
    for(int k=0; k<grid_nz; ++k) {
        for(int j=0; j<grid_ny; ++j) {
            for(int i=0; i<grid_nx; ++i) {
                int idx = get_grid_index(i, j, k);
                if (idx >= 0 && idx < (int)is_boundary_cell.size() && is_boundary_cell[idx]) {
                    pressures.push_back(pressure_grid[idx]);
                }
            }
        }
    }
    return pressures;
}

int NavierStokesSolver::get_grid_index(int i, int j, int k) const {
    if (i < 0 || i >= grid_nx || j < 0 || j >= grid_ny || k < 0 || k >= grid_nz) {
        // Clamp to edge
        i = std::max(0, std::min(i, grid_nx - 1));
        j = std::max(0, std::min(j, grid_ny - 1));
        k = std::max(0, std::min(k, grid_nz - 1));
    }
    return i + j * grid_nx + k * grid_nx * grid_ny;
}

Eigen::Vector3d NavierStokesSolver::get_grid_position(int i, int j, int k) const {
    return Eigen::Vector3d((i + 0.5) * grid_cell_size, (j + 0.5) * grid_cell_size, (k + 0.5) * grid_cell_size);
}

int NavierStokesSolver::find_cell_index_for_position(const Eigen::Vector3d& pos) const {
     if (grid_cell_size <= 0) return -1;
     int i = static_cast<int>(pos.x() / grid_cell_size);
     int j = static_cast<int>(pos.y() / grid_cell_size);
     int k = static_cast<int>(pos.z() / grid_cell_size);
    if (i < 0 || i >= grid_nx || j < 0 || j >= grid_ny || k < 0 || k >= grid_nz) return -1;
     return get_grid_index(i, j, k);
}

void NavierStokesSolver::advect(double dt) {
    #pragma omp parallel for collapse(3)
    for(int k=1; k<grid_nz-1; ++k) {
        for(int j=1; j<grid_ny-1; ++j) {
            for(int i=1; i<grid_nx-1; ++i) {
                int idx = get_grid_index(i, j, k);
                if (is_boundary_cell[idx]) continue;
                Eigen::Vector3d pos = get_grid_position(i, j, k);
                Eigen::Vector3d prev_pos = pos - velocity_grid[idx] * dt;
                
                // Semi-Lagrangian Advection
                velocity_grid_next[idx] = trilinear_interp(prev_pos, velocity_grid);
                temperature_grid_next[idx] = trilinear_interp(prev_pos, temperature_grid);
            }
        }
    }
}

void NavierStokesSolver::diffuse(double dt) {
    double h2 = grid_cell_size * grid_cell_size;
    if (h2 < 1e-12) return;
    
    #pragma omp parallel for collapse(3)
    for(int k=1; k<grid_nz-1; ++k) {
        for(int j=1; j<grid_ny-1; ++j) {
            for(int i=1; i<grid_nx-1; ++i) {
                int idx = get_grid_index(i, j, k);
                if (is_boundary_cell[idx]) continue;

                Eigen::Vector3d laplacian_vel = (
                    velocity_grid_next[get_grid_index(i+1, j, k)] + velocity_grid_next[get_grid_index(i-1, j, k)] +
                    velocity_grid_next[get_grid_index(i, j+1, k)] + velocity_grid_next[get_grid_index(i, j-1, k)] +
                    velocity_grid_next[get_grid_index(i, j, k+1)] + velocity_grid_next[get_grid_index(i, j, k-1)] -
                    velocity_grid_next[idx] * 6.0
                ) / h2;
                
                velocity_grid_next[idx] += viscosity * laplacian_vel * dt;

                double T = temperature_grid_next[idx];
                double laplacian_temp = (
                    temperature_grid_next[get_grid_index(i+1, j, k)] + temperature_grid_next[get_grid_index(i-1, j, k)] +
                    temperature_grid_next[get_grid_index(i, j+1, k)] + temperature_grid_next[get_grid_index(i, j-1, k)] +
                    temperature_grid_next[get_grid_index(i, j, k+1)] + temperature_grid_next[get_grid_index(i, j, k-1)] -
                    T * 6.0
                ) / h2;
                
                double source = 0.0;
                double cp = 1005.0; // Air approx
                if (density * cp > 1e-9) {
                     source = heat_source_grid[idx] / (density * cp * grid_cell_size*grid_cell_size*grid_cell_size);
                }
                
                temperature_grid_next[idx] += dt * (thermal_diffusivity * laplacian_temp + source);
                // Clamp temp
                temperature_grid_next[idx] = std::max(20.0, std::min(2000.0, temperature_grid_next[idx]));
            }
        }
    }
}

void NavierStokesSolver::project(double dt) {
    double h = grid_cell_size;
    
    // 1. Compute Divergence
    #pragma omp parallel for collapse(3)
    for(int k=1; k<grid_nz-1; ++k) {
        for(int j=1; j<grid_ny-1; ++j) {
            for(int i=1; i<grid_nx-1; ++i) {
                int idx = get_grid_index(i, j, k);
                if (is_boundary_cell[idx]) { divergence_grid[idx] = 0.0; continue; }
                
                divergence_grid[idx] = (
                    velocity_grid_next[get_grid_index(i+1, j, k)].x() - velocity_grid_next[get_grid_index(i-1, j, k)].x() +
                    velocity_grid_next[get_grid_index(i, j+1, k)].y() - velocity_grid_next[get_grid_index(i, j-1, k)].y() +
                    velocity_grid_next[get_grid_index(i, j, k+1)].z() - velocity_grid_next[get_grid_index(i, j, k-1)].z()
                ) / (2.0 * h);
            }
        }
    }
    
    // 2. Compute Pressure (Jacobi Iteration or simple direct approx)
    // Simplified Pressure Update: P = -rho * div / dt
    #pragma omp parallel for
    for(size_t idx=0; idx < pressure_grid.size(); ++idx) {
        if (is_boundary_cell[idx]) pressure_grid_next[idx] = 0.0;
        else pressure_grid_next[idx] = -density * divergence_grid[idx] / dt; 
    }
    
    // 3. Subtract Gradient
    #pragma omp parallel for collapse(3)
    for(int k=1; k<grid_nz-1; ++k) {
        for(int j=1; j<grid_ny-1; ++j) {
            for(int i=1; i<grid_nx-1; ++i) {
                int idx = get_grid_index(i, j, k);
                if (is_boundary_cell[idx]) continue;
                
                Eigen::Vector3d grad_p(
                    pressure_grid_next[get_grid_index(i+1, j, k)] - pressure_grid_next[get_grid_index(i-1, j, k)],
                    pressure_grid_next[get_grid_index(i, j+1, k)] - pressure_grid_next[get_grid_index(i, j-1, k)],
                    pressure_grid_next[get_grid_index(i, j, k+1)] - pressure_grid_next[get_grid_index(i, j, k-1)]
                );
                
                velocity_grid_next[idx] -= (dt / density) * (grad_p / (2.0 * h));
            }
        }
    }
}

void NavierStokesSolver::update(double dt) {
    advect(dt);
    diffuse(dt);
    project(dt);
    
    // Swap buffers
    std::swap(velocity_grid, velocity_grid_next);
    std::swap(pressure_grid, pressure_grid_next);
    std::swap(temperature_grid, temperature_grid_next);
}