#include "navier_stokes.h"
#include "simulation.h" 
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <algorithm> // For std::swap

// Constructor
NavierStokesSolver::NavierStokesSolver(const json& config) : config_data(config) {
    grid_nx = grid_ny = grid_nz = 0;
    grid_cell_size = 0.0;
    inlet_velocity = Eigen::Vector3d::Zero();
    
    // --- NEW: Read fluid properties from config, with defaults ---
    density = config["cfd_parameters"].value("density", 1000.0); // kg/m^3
    viscosity = config["cfd_parameters"].value("viscosity", 0.01); // Pa*s
    thermal_diffusivity = config["cfd_parameters"].value("thermal_diffusivity", 0.01); // Alpha
    
    std::cout << "  NavierStokesSolver constructed." << std::endl;
}

// Destructor
NavierStokesSolver::~NavierStokesSolver() {}

// Grid Initialization
void NavierStokesSolver::initialize_grid(int nx, int ny, int nz, double cell_size) {
    grid_nx = nx;
    grid_ny = ny;
    grid_nz = nz;
    grid_cell_size = cell_size;
    int total_cells = nx * ny * nz;

    // Resize and initialize grid data vectors
    pressure_grid.resize(total_cells, 0.0);
    velocity_grid.resize(total_cells, Eigen::Vector3d::Zero());
    temperature_grid.resize(total_cells, config_data["physics_parameters"]["ambient_temperature_C"].get<double>());
    heat_source_grid.resize(total_cells, 0.0);
    is_boundary_cell.resize(total_cells, false);

    // --- NEW: Initialize next-step and helper grids ---
    pressure_grid_next.resize(total_cells, 0.0);
    velocity_grid_next.resize(total_cells, Eigen::Vector3d::Zero());
    temperature_grid_next.resize(total_cells, config_data["physics_parameters"]["ambient_temperature_C"].get<double>());
    divergence_grid.resize(total_cells, 0.0);

    std::cout << "  Navier-Stokes grid initialized (" << nx << "x" << ny << "x" << nz << ")" << std::endl;
}

// Set Inlet Velocity
void NavierStokesSolver::set_inlet_velocity(const Eigen::Vector3d& vel) {
    inlet_velocity = vel;
    std::cout << "  Inlet velocity set to (" << vel.x() << ", " << vel.y() << ", " << vel.z() << ") m/s" << std::endl;
}


// --- update_boundaries_and_sources remains unchanged from your provided file ---
void NavierStokesSolver::update_boundaries_and_sources(const Mesh& mesh, double ambient_temp, double strain_rate, const ThermalModel& thermal_model) {
    heat_source_grid.assign(grid_nx * grid_ny * grid_nz, 0.0);
    is_boundary_cell.assign(grid_nx * grid_ny * grid_nz, false);

    for (size_t i = 0; i < mesh.nodes.size(); ++i) {
        const Node& node = mesh.nodes[i];
        const Eigen::Vector3d& node_pos = node.position;

        int cell_idx = find_cell_index_for_position(node_pos);
        if (cell_idx >= 0 && cell_idx < is_boundary_cell.size()) {
             is_boundary_cell[cell_idx] = true;
             velocity_grid[cell_idx] = Eigen::Vector3d::Zero();
             // Set temperature boundary condition for the solid tool
             temperature_grid[cell_idx] = node.temperature; 
        }

        if (node.temperature > ambient_temp + 10.0 && node.status == NodeStatus::OK) {
            double heat_flux_density = thermal_model.calculate_heat_generation(node.stress, strain_rate);
            if (cell_idx >= 0 && cell_idx < heat_source_grid.size()) {
                 double cell_volume = grid_cell_size * grid_cell_size * grid_cell_size;
                 heat_source_grid[cell_idx] += heat_flux_density * cell_volume;
            }
        }
    }

    for(int k=0; k<grid_nz; ++k) {
        for(int j=0; j<grid_ny; ++j) {
            int inlet_idx = get_grid_index(0, j, k);
             if (inlet_idx >= 0 && inlet_idx < velocity_grid.size() && !is_boundary_cell[inlet_idx]) {
                 velocity_grid[inlet_idx] = inlet_velocity;
                 temperature_grid[inlet_idx] = ambient_temp; 
            }
        }
    }
}


// --- NEW: Trilinear interpolation helper ---
template <typename T>
T NavierStokesSolver::trilinear_interp(const Eigen::Vector3d& pos, const std::vector<T>& grid) const {
    if (grid_cell_size <= 0) return T();

    // Calculate grid indices and interpolation weights
    Eigen::Vector3d grid_pos = pos / grid_cell_size;
    int i = static_cast<int>(grid_pos.x());
    int j = static_cast<int>(grid_pos.y());
    int k = static_cast<int>(grid_pos.z());

    double tx = grid_pos.x() - i;
    double ty = grid_pos.y() - j;
    double tz = grid_pos.z() - k;

    // Get values at the 8 corners
    T c000 = grid[get_grid_index(i, j, k)];
    T c100 = grid[get_grid_index(i + 1, j, k)];
    T c010 = grid[get_grid_index(i, j + 1, k)];
    T c110 = grid[get_grid_index(i + 1, j + 1, k)];
    T c001 = grid[get_grid_index(i, j, k + 1)];
    T c101 = grid[get_grid_index(i + 1, j, k + 1)];
    T c011 = grid[get_grid_index(i, j + 1, k + 1)];
    T c111 = grid[get_grid_index(i + 1, j + 1, k + 1)];

    // Interpolate along x
    T c00 = c000 * (1 - tx) + c100 * tx;
    T c01 = c001 * (1 - tx) + c101 * tx;
    T c10 = c010 * (1 - tx) + c110 * tx;
    T c11 = c011 * (1 - tx) + c111 * tx;

    // Interpolate along y
    T c0 = c00 * (1 - ty) + c10 * ty;
    T c1 = c01 * (1 - ty) + c11 * ty;

    // Interpolate along z
    return c0 * (1 - tz) + c1 * tz;
}


// --- UPDATED: Getters now use trilinear interpolation ---
double NavierStokesSolver::get_temperature_at(const Eigen::Vector3d& pos) const {
    int idx = find_cell_index_for_position(pos);
    if (idx < 0 || idx >= temperature_grid.size()) {
        return config_data["physics_parameters"]["ambient_temperature_C"].get<double>();
    }
    // Use interpolation for smoother particle movement
    return trilinear_interp(pos, temperature_grid);
}

double NavierStokesSolver::get_pressure_at(const Eigen::Vector3d& pos) const {
    int idx = find_cell_index_for_position(pos);
    if (idx < 0 || idx >= pressure_grid.size()) {
        return 0.0;
    }
    return trilinear_interp(pos, pressure_grid);
}

Eigen::Vector3d NavierStokesSolver::get_velocity_at(const Eigen::Vector3d& pos) const {
    int idx = find_cell_index_for_position(pos);
    if (idx < 0 || idx >= velocity_grid.size()) {
        return Eigen::Vector3d::Zero();
    }
    // Use interpolation for smoother particle movement
    return trilinear_interp(pos, velocity_grid);
}

// --- UPDATED: Functional implementation ---
std::vector<double> NavierStokesSolver::get_rake_face_pressure(const Eigen::Vector3d& normal) const {
    // This implementation returns pressures from all tool boundary cells
    // A more advanced version could filter by the 'normal'
    static_cast<void>(normal); // Unused for now

    std::vector<double> pressures;
    for(int k=0; k<grid_nz; ++k) {
        for(int j=0; j<grid_ny; ++j) {
            for(int i=0; i<grid_nx; ++i) {
                int idx = get_grid_index(i, j, k);
                if (idx >= 0 && idx < is_boundary_cell.size() && is_boundary_cell[idx]) {
                    pressures.push_back(pressure_grid[idx]);
                }
            }
        }
    }
    return pressures;
}

// --- Grid index and position helpers (Unchanged) ---
int NavierStokesSolver::get_grid_index(int i, int j, int k) const {
    if (i < 0 || i >= grid_nx || j < 0 || j >= grid_ny || k < 0 || k >= grid_nz) {
        // Clamp to boundary for interpolation
        i = std::max(0, std::min(i, grid_nx - 1));
        j = std::max(0, std::min(j, grid_ny - 1));
        k = std::max(0, std::min(k, grid_nz - 1));
    }
    return i + j * grid_nx + k * grid_nx * grid_ny;
}

Eigen::Vector3d NavierStokesSolver::get_grid_position(int i, int j, int k) const {
    return Eigen::Vector3d(
        (i + 0.5) * grid_cell_size,
        (j + 0.5) * grid_cell_size,
        (k + 0.5) * grid_cell_size
    );
}

int NavierStokesSolver::find_cell_index_for_position(const Eigen::Vector3d& pos) const {
     if (grid_cell_size <= 0) return -1;
     int i = static_cast<int>(pos.x() / grid_cell_size);
     int j = static_cast<int>(pos.y() / grid_cell_size);
     int k = static_cast<int>(pos.z() / grid_cell_size);
     
     // Check bounds (return -1 if outside grid)
    if (i < 0 || i >= grid_nx || j < 0 || j >= grid_ny || k < 0 || k >= grid_nz) {
        return -1;
    }
     return get_grid_index(i, j, k);
}


// --- NEW: Semi-Lagrangian Advection Step ---
void NavierStokesSolver::advect(double dt) {
    double h = grid_cell_size;

    #pragma omp parallel for collapse(3)
    for(int k=1; k<grid_nz-1; ++k) {
        for(int j=1; j<grid_ny-1; ++j) {
            for(int i=1; i<grid_nx-1; ++i) {
                int idx = get_grid_index(i, j, k);
                if (is_boundary_cell[idx]) continue;

                // Trace back particle position (Semi-Lagrangian method)
                Eigen::Vector3d pos = get_grid_position(i, j, k);
                Eigen::Vector3d prev_pos = pos - velocity_grid[idx] * dt;

                // Advect velocity and temperature by interpolating at prev_pos
                velocity_grid_next[idx] = trilinear_interp(prev_pos, velocity_grid);
                temperature_grid_next[idx] = trilinear_interp(prev_pos, temperature_grid);
            }
        }
    }
}

// --- NEW: Diffusion Step (Viscosity + Thermal) ---
void NavierStokesSolver::diffuse(double dt) {
    double h = grid_cell_size;
    double h2 = h * h;
    
    // Safety check for division by zero
    if (h2 < 1e-12) return;

    double nu = viscosity;
    double alpha = thermal_diffusivity;

    #pragma omp parallel for collapse(3)
    for(int k=1; k<grid_nz-1; ++k) {
        for(int j=1; j<grid_ny-1; ++j) {
            for(int i=1; i<grid_nx-1; ++i) {
                int idx = get_grid_index(i, j, k);
                if (is_boundary_cell[idx]) continue;

                // --- Velocity Diffusion (Viscosity) ---
                // Calculate Laplacian of velocity (central differencing)
                Eigen::Vector3d laplacian_vel = (
                    velocity_grid_next[get_grid_index(i+1, j, k)] + velocity_grid_next[get_grid_index(i-1, j, k)] +
                    velocity_grid_next[get_grid_index(i, j+1, k)] + velocity_grid_next[get_grid_index(i, j-1, k)] +
                    velocity_grid_next[get_grid_index(i, j, k+1)] + velocity_grid_next[get_grid_index(i, j, k-1)] -
                    velocity_grid_next[idx] * 6.0
                ) / h2;
                
                velocity_grid_next[idx] += nu * laplacian_vel * dt;

                // --- Temperature Diffusion + Heat Source ---
                double T = temperature_grid_next[idx];
                double laplacian_temp = (
                    temperature_grid_next[get_grid_index(i+1, j, k)] + temperature_grid_next[get_grid_index(i-1, j, k)] +
                    temperature_grid_next[get_grid_index(i, j+1, k)] + temperature_grid_next[get_grid_index(i, j-1, k)] +
                    temperature_grid_next[get_grid_index(i, j, k+1)] + temperature_grid_next[get_grid_index(i, j, k-1)] -
                    T * 6.0
                ) / h2;

                // Calculate source term (°C/s)
                double source_term_degC_per_s = 0.0;
                double cell_volume = h * h * h;
                double cp = config_data["cfd_parameters"].value("specific_heat", 1005.0);
                if (density * cp * cell_volume > 1e-9) {
                    source_term_degC_per_s = heat_source_grid[idx] / (density * cp * cell_volume);
                }

                temperature_grid_next[idx] += dt * (alpha * laplacian_temp + source_term_degC_per_s);

                // Clamp temperature
                temperature_grid_next[idx] = std::max(config_data["physics_parameters"]["ambient_temperature_C"].get<double>(), 
                                                      std::min(2000.0, temperature_grid_next[idx]));
            }
        }
    }
}

// --- NEW: Projection Step (Pressure Solve) ---
void NavierStokesSolver::project(double dt) {
    // This is a simplified projection step (Helmholtz decomposition)
    // A full solver would use Jacobi, Gauss-Seidel, or SOR iterations
    
    double h = grid_cell_size;
    if (h < 1e-9) return;
    
    // 1. Calculate divergence of the velocity field
    #pragma omp parallel for collapse(3)
    for(int k=1; k<grid_nz-1; ++k) {
        for(int j=1; j<grid_ny-1; ++j) {
            for(int i=1; i<grid_nx-1; ++i) {
                int idx = get_grid_index(i, j, k);
                if (is_boundary_cell[idx]) {
                    divergence_grid[idx] = 0.0;
                    continue;
                }

                divergence_grid[idx] = (
                    velocity_grid_next[get_grid_index(i+1, j, k)].x() - velocity_grid_next[get_grid_index(i-1, j, k)].x() +
                    velocity_grid_next[get_grid_index(i, j+1, k)].y() - velocity_grid_next[get_grid_index(i, j-1, k)].y() +
                    velocity_grid_next[get_grid_index(i, j, k+1)].z() - velocity_grid_next[get_grid_index(i, j, k-1)].z()
                ) / (2.0 * h);
            }
        }
    }

    // 2. Solve Poisson equation for pressure (simplified: P = -density * divergence)
    // A real solver would iterate here: p_next[idx] = (p[i-1] + p[i+1] + ...) - h^2 * div[idx]
    #pragma omp parallel for
    for(size_t idx=0; idx < pressure_grid.size(); ++idx) {
        if (is_boundary_cell[idx]) {
            pressure_grid_next[idx] = 0.0; // Dirichlet boundary
        } else {
            // Simplified pressure approximation
            pressure_grid_next[idx] = -density * divergence_grid[idx] / (dt); 
        }
    }

    // 3. Correct velocities to be divergence-free
    #pragma omp parallel for collapse(3)
    for(int k=1; k<grid_nz-1; ++k) {
        for(int j=1; j<grid_ny-1; ++j) {
            for(int i=1; i<grid_nx-1; ++i) {
                int idx = get_grid_index(i, j, k);
                if (is_boundary_cell[idx]) continue;

                // Grad(p) = (p[i+1] - p[i-1], ...) / (2h)
                Eigen::Vector3d grad_p(
                    pressure_grid_next[get_grid_index(i+1, j, k)] - pressure_grid_next[get_grid_index(i-1, j, k)],
                    pressure_grid_next[get_grid_index(i, j+1, k)] - pressure_grid_next[get_grid_index(i, j-1, k)],
                    pressure_grid_next[get_grid_index(i, j, k+1)] - pressure_grid_next[get_grid_index(i, j, k-1)]
                );
                
                // V = V - dt * (1/rho) * Grad(p)
                velocity_grid_next[idx] -= (dt / density) * (grad_p / (2.0 * h));
            }
        }
    }
}

// --- UPDATED: Main Time-Stepping Function ---
void NavierStokesSolver::update(double dt) {
    // This is a basic projection method solver
    
    // --- 1. Advection ---
    // Move velocity and temperature through the grid
    advect(dt);

    // --- 2. Diffusion ---
    // Apply viscosity and thermal diffusion, and add heat sources
    diffuse(dt);
    
    // --- 3. Projection ---
    // Enforce mass conservation by solving for pressure and correcting velocity
    project(dt);
    
    // --- 4. Swap buffers for next step ---
    std::swap(velocity_grid, velocity_grid_next);
    std::swap(pressure_grid, pressure_grid_next);
    std::swap(temperature_grid, temperature_grid_next);
}