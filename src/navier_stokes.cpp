#include "navier_stokes.h"
#include <algorithm>
#include <cmath>
#include <fstream>

NavierStokesSolver::NavierStokesSolver(const json& config) {
    // Material properties for chip material (acts as fluid at high temp)
    const auto& props = config["material_properties"];
    base_density = props["density_kg_m3"].get<double>();
    specific_heat = props["specific_heat_J_kgC"].get<double>();
    thermal_conductivity = props.value("thermal_conductivity_W_mK", 20.0);
    
    // Effective viscosity for chip flow (calibrated empirically)
    // At high temps, metal acts like very viscous fluid
    viscosity = 1000.0; // Pa·s (very viscous, like honey but 10000x more)
}

void NavierStokesSolver::initialize_grid(int nx_in, int ny_in, int nz_in, double cell_size) {
    nx = nx_in;
    ny = ny_in;
    nz = nz_in;
    dx = cell_size;
    
    // Get ambient temperature from config (default 20°C)
    const auto& config = json::parse(std::ifstream("input.json"));
    double ambient_temp = config["physics_parameters"].value("ambient_temperature_C", 20.0);
    
    // Allocate 3D grid
    grid.resize(nx);
    for (int i = 0; i < nx; ++i) {
        grid[i].resize(ny);
        for (int j = 0; j < ny; ++j) {
            grid[i][j].resize(nz);
            
            // Initialize all cells with proper temperature
            for (int k = 0; k < nz; ++k) {
                grid[i][j][k].velocity = Eigen::Vector3d(0, 0, 0);
                grid[i][j][k].pressure = 101325.0; // 1 atm
                grid[i][j][k].temperature = ambient_temp;
                grid[i][j][k].density = base_density;
                grid[i][j][k].is_solid = false;
            }
        }
    }
}

void NavierStokesSolver::set_inlet_velocity(const Eigen::Vector3d& velocity) {
    // Set inlet boundary (e.g., x = 0 face)
    for (int j = 0; j < ny; ++j) {
        for (int k = 0; k < nz; ++k) {
            grid[0][j][k].velocity = velocity;
        }
    }
}

void NavierStokesSolver::set_tool_boundary(const std::vector<Eigen::Vector3d>& tool_nodes) {
    // Mark cells occupied by tool as solid
    for (const auto& node_pos : tool_nodes) {
        int i = get_index_x(node_pos.x());
        int j = get_index_y(node_pos.y());
        int k = get_index_z(node_pos.z());
        
        if (is_valid_index(i, j, k)) {
            grid[i][j][k].is_solid = true;
            grid[i][j][k].velocity = Eigen::Vector3d(0, 0, 0); // No-slip condition
        }
    }
}

void NavierStokesSolver::apply_heat_source(const Eigen::Vector3d& position, double heat_flux) {
    int i = get_index_x(position.x());
    int j = get_index_y(position.y());
    int k = get_index_z(position.z());
    
    if (is_valid_index(i, j, k) && !grid[i][j][k].is_solid) {
        // Add heat: ΔT = Q × dt / (ρ × V × c)
        double cell_volume = dx * dx * dx;
        double temp_increase = heat_flux / (grid[i][j][k].density * cell_volume * specific_heat);
        grid[i][j][k].temperature += temp_increase;
    }
}

void NavierStokesSolver::update(double dt) {
    // Enhanced Navier-Stokes time step with improved heat treatment
    
    // 1. Update velocity field
    advect_velocity(dt);
    apply_external_forces(dt);
    project_velocity();      // Enforce incompressibility
    
    // 2. Enhanced thermal update
    // First handle advection
    advect_temperature(dt);
    
    // Then add heat generation from friction/chip interaction
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int k = 1; k < nz - 1; ++k) {
                if (!grid[i][j][k].is_solid) {
                    // Heat from viscous dissipation
                    double vel_mag = grid[i][j][k].velocity.norm();
                    double visc_heat = 0.5 * viscosity * vel_mag * vel_mag;
                    
                    // Temperature increase from viscous heating
                    grid[i][j][k].temperature += (visc_heat * dt) / (grid[i][j][k].density * specific_heat);
                }
            }
        }
    }
    
    // Finally handle diffusion
    diffuse_heat(dt);
}

void NavierStokesSolver::advect_velocity(double dt) {
    // Semi-Lagrangian advection (unconditionally stable)
    // For each cell, trace particle backward and interpolate
    
    std::vector<std::vector<std::vector<Eigen::Vector3d>>> new_velocities(nx);
    for (int i = 0; i < nx; ++i) {
        new_velocities[i].resize(ny);
        for (int j = 0; j < ny; ++j) {
            new_velocities[i][j].resize(nz);
        }
    }
    
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int k = 1; k < nz - 1; ++k) {
                if (grid[i][j][k].is_solid) continue;
                
                // Current position
                Eigen::Vector3d pos(i * dx, j * dx, k * dx);
                
                // Trace backward
                Eigen::Vector3d prev_pos = pos - grid[i][j][k].velocity * dt;
                
                // Interpolate velocity at previous position
                new_velocities[i][j][k] = interpolate_velocity(prev_pos);
            }
        }
    }
    
    // Copy back
    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int k = 1; k < nz - 1; ++k) {
                if (!grid[i][j][k].is_solid) {
                    grid[i][j][k].velocity = new_velocities[i][j][k];
                }
            }
        }
    }
}

void NavierStokesSolver::apply_external_forces(double dt) {
    // Add gravity and other body forces
    Eigen::Vector3d gravity(0, 0, -9.81);
    
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                if (!grid[i][j][k].is_solid) {
                    grid[i][j][k].velocity += gravity * dt;
                }
            }
        }
    }
}

void NavierStokesSolver::project_velocity() {
    // Simplified pressure projection to enforce incompressibility
    // Solve Poisson equation: ∇²p = ρ/dt × ∇·v
    
    // For demo: Use Jacobi iteration (simple but slow)
    std::vector<std::vector<std::vector<double>>> divergence(nx);
    std::vector<std::vector<std::vector<double>>> pressure_new(nx);
    
    for (int i = 0; i < nx; ++i) {
        divergence[i].resize(ny);
        pressure_new[i].resize(ny);
        for (int j = 0; j < ny; ++j) {
            divergence[i][j].resize(nz, 0.0);
            pressure_new[i][j].resize(nz, 101325.0);
        }
    }
    
    // Calculate divergence
    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int k = 1; k < nz - 1; ++k) {
                if (grid[i][j][k].is_solid) continue;
                
                double dvx_dx = (grid[i+1][j][k].velocity.x() - grid[i-1][j][k].velocity.x()) / (2*dx);
                double dvy_dy = (grid[i][j+1][k].velocity.y() - grid[i][j-1][k].velocity.y()) / (2*dx);
                double dvz_dz = (grid[i][j][k+1].velocity.z() - grid[i][j][k-1].velocity.z()) / (2*dx);
                
                divergence[i][j][k] = dvx_dx + dvy_dy + dvz_dz;
            }
        }
    }
    
    // Jacobi iterations for pressure (simplified - 10 iterations)
    for (int iter = 0; iter < 10; ++iter) {
        for (int i = 1; i < nx - 1; ++i) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int k = 1; k < nz - 1; ++k) {
                    if (grid[i][j][k].is_solid) continue;
                    
                    double sum_neighbors = 
                        grid[i+1][j][k].pressure + grid[i-1][j][k].pressure +
                        grid[i][j+1][k].pressure + grid[i][j-1][k].pressure +
                        grid[i][j][k+1].pressure + grid[i][j][k-1].pressure;
                    
                    pressure_new[i][j][k] = (sum_neighbors - dx*dx * divergence[i][j][k]) / 6.0;
                }
            }
        }
        
        // Copy back
        for (int i = 1; i < nx - 1; ++i) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int k = 1; k < nz - 1; ++k) {
                    grid[i][j][k].pressure = pressure_new[i][j][k];
                }
            }
        }
    }
    
    // Correct velocity: v_new = v_old - dt/ρ × ∇p
    double dt = 0.0001; // Use simulation time step
    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int k = 1; k < nz - 1; ++k) {
                if (grid[i][j][k].is_solid) continue;
                
                Eigen::Vector3d grad_p;
                grad_p.x() = (grid[i+1][j][k].pressure - grid[i-1][j][k].pressure) / (2*dx);
                grad_p.y() = (grid[i][j+1][k].pressure - grid[i][j-1][k].pressure) / (2*dx);
                grad_p.z() = (grid[i][j][k+1].pressure - grid[i][j][k-1].pressure) / (2*dx);
                
                grid[i][j][k].velocity -= (dt / grid[i][j][k].density) * grad_p;
            }
        }
    }
}

void NavierStokesSolver::advect_temperature(double dt) {
    // Similar to velocity advection
    std::vector<std::vector<std::vector<double>>> new_temps(nx);
    for (int i = 0; i < nx; ++i) {
        new_temps[i].resize(ny);
        for (int j = 0; j < ny; ++j) {
            new_temps[i][j].resize(nz);
        }
    }
    
    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int k = 1; k < nz - 1; ++k) {
                if (grid[i][j][k].is_solid) continue;
                
                Eigen::Vector3d pos(i * dx, j * dx, k * dx);
                Eigen::Vector3d prev_pos = pos - grid[i][j][k].velocity * dt;
                
                // Interpolate temperature (simplified - use nearest neighbor)
                int pi = get_index_x(prev_pos.x());
                int pj = get_index_y(prev_pos.y());
                int pk = get_index_z(prev_pos.z());
                
                if (is_valid_index(pi, pj, pk)) {
                    new_temps[i][j][k] = grid[pi][pj][pk].temperature;
                } else {
                    new_temps[i][j][k] = grid[i][j][k].temperature;
                }
            }
        }
    }
    
    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int k = 1; k < nz - 1; ++k) {
                if (!grid[i][j][k].is_solid) {
                    grid[i][j][k].temperature = new_temps[i][j][k];
                }
            }
        }
    }
}

void NavierStokesSolver::diffuse_heat(double dt) {
    // Heat diffusion: ∂T/∂t = α × ∇²T
    // α = k / (ρ × c) is thermal diffusivity
    
    double alpha = thermal_conductivity / (base_density * specific_heat);
    double diffusion_coeff = alpha * dt / (dx * dx);
    
    std::vector<std::vector<std::vector<double>>> new_temps(nx);
    for (int i = 0; i < nx; ++i) {
        new_temps[i].resize(ny);
        for (int j = 0; j < ny; ++j) {
            new_temps[i][j].resize(nz);
        }
    }
    
    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int k = 1; k < nz - 1; ++k) {
                if (grid[i][j][k].is_solid) continue;
                
                double laplacian = 
                    grid[i+1][j][k].temperature + grid[i-1][j][k].temperature +
                    grid[i][j+1][k].temperature + grid[i][j-1][k].temperature +
                    grid[i][j][k+1].temperature + grid[i][j][k-1].temperature -
                    6.0 * grid[i][j][k].temperature;
                
                new_temps[i][j][k] = grid[i][j][k].temperature + diffusion_coeff * laplacian;
            }
        }
    }
    
    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int k = 1; k < nz - 1; ++k) {
                if (!grid[i][j][k].is_solid) {
                    grid[i][j][k].temperature = new_temps[i][j][k];
                }
            }
        }
    }
}

// Helper functions
int NavierStokesSolver::get_index_x(double x) const {
    return static_cast<int>(x / dx);
}

int NavierStokesSolver::get_index_y(double y) const {
    return static_cast<int>(y / dx);
}

int NavierStokesSolver::get_index_z(double z) const {
    return static_cast<int>(z / dx);
}

bool NavierStokesSolver::is_valid_index(int i, int j, int k) const {
    return i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz;
}

Eigen::Vector3d NavierStokesSolver::interpolate_velocity(const Eigen::Vector3d& pos) const {
    // Trilinear interpolation (simplified - use nearest neighbor for demo)
    int i = get_index_x(pos.x());
    int j = get_index_y(pos.y());
    int k = get_index_z(pos.z());
    
    if (is_valid_index(i, j, k)) {
        return grid[i][j][k].velocity;
    }
    return Eigen::Vector3d(0, 0, 0);
}

std::vector<FluidCell> NavierStokesSolver::get_all_cells() const {
    std::vector<FluidCell> cells;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                if (!grid[i][j][k].is_solid) {
                    cells.push_back(grid[i][j][k]);
                }
            }
        }
    }
    return cells;
}

std::vector<double> NavierStokesSolver::get_rake_face_pressure(const Eigen::Vector3d& rake_normal) const {
    std::vector<double> pressures;
    
    // Extract pressures from cells near rake face (simplified)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                if (!grid[i][j][k].is_solid) {
                    pressures.push_back(grid[i][j][k].pressure);
                }
            }
        }
    }
    
    return pressures;
}

double NavierStokesSolver::get_pressure_at(const Eigen::Vector3d& position) const {
    int i = get_index_x(position.x());
    int j = get_index_y(position.y());
    int k = get_index_z(position.z());
    
    if (is_valid_index(i, j, k)) {
        return grid[i][j][k].pressure;
    }
    return 101325.0; // Atmospheric pressure
}

Eigen::Vector3d NavierStokesSolver::get_velocity_at(const Eigen::Vector3d& position) const {
    int i = get_index_x(position.x());
    int j = get_index_y(position.y());
    int k = get_index_z(position.z());
    
    if (is_valid_index(i, j, k)) {
        return grid[i][j][k].velocity;
    }
    return Eigen::Vector3d(0, 0, 0);
}

double NavierStokesSolver::get_temperature_at(const Eigen::Vector3d& position) const {
    int i = get_index_x(position.x());
    int j = get_index_y(position.y());
    int k = get_index_z(position.z());
    
    if (is_valid_index(i, j, k)) {
        return grid[i][j][k].temperature;
    }
    return 20.0; // Ambient
}