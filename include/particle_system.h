#ifndef PARTICLE_SYSTEM_H
#define PARTICLE_SYSTEM_H

#include <Eigen/Dense>
#include <vector>
#include "json.hpp"
#include "navier_stokes.h"

using json = nlohmann::json;

// ═══════════════════════════════════════════════════════════
// PARTICLE SYSTEM FOR CHIP FLOW VISUALIZATION
// Lagrangian particles advected by Eulerian fluid grid
// ═══════════════════════════════════════════════════════════

struct Particle {
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    double temperature;
    double pressure;
    double age;               // Time since creation
    bool is_active;
};

class ParticleSystem {
public:
    ParticleSystem(int max_particles);
    
    // Generate new particles at shear zone
    void emit_particles(const Eigen::Vector3d& emission_point, 
                       const Eigen::Vector3d& emission_velocity,
                       double temperature,
                       int count);
    
    // Update particles using fluid velocity field
    void update(double dt, const NavierStokesSolver& fluid_solver);
    
    // Get active particles for rendering
    std::vector<Particle> get_active_particles() const;
    
    // Statistics
    int get_active_count() const;
    double get_average_temperature() const;
    double get_max_pressure() const;
    
    // Export to JSON
    json export_particles() const;

private:
    std::vector<Particle> particles;
    int max_particles;
    
    void remove_old_particles(double max_age);
    void remove_out_of_bounds_particles(double max_distance);
};

#endif // PARTICLE_SYSTEM_H