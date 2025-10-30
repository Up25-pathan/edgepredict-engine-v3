#include "particle_system.h"
#include <algorithm>
#include <random>

ParticleSystem::ParticleSystem(int max_particles) : max_particles(max_particles) {
    particles.reserve(max_particles);
}

void ParticleSystem::emit_particles(const Eigen::Vector3d& emission_point, 
                                    const Eigen::Vector3d& emission_velocity,
                                    double temperature,
                                    int count) 
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.0001, 0.0001);
    
    for (int i = 0; i < count; ++i) {
        if (particles.size() >= max_particles) break;
        
        Particle p;
        
        // Small random offset from emission point
        p.position = emission_point + Eigen::Vector3d(dis(gen), dis(gen), dis(gen));
        
        // Velocity with small random variation
        double vel_variation = 0.9 + dis(gen) * 2.0; // 0.7 to 1.1
        p.velocity = emission_velocity * vel_variation;
        
        p.temperature = temperature + dis(gen) * 50.0; // ±50°C variation
        p.pressure = 1000.0 + dis(gen) * 200.0;        // High pressure at shear zone
        p.age = 0.0;
        p.is_active = true;
        
        particles.push_back(p);
    }
}

void ParticleSystem::update(double dt, const NavierStokesSolver& fluid_solver) {
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        Particle& p = particles[i];
        
        if (!p.is_active) continue;
        
        // Get fluid velocity at particle position
        Eigen::Vector3d fluid_vel = fluid_solver.get_velocity_at(p.position);
        
        // Update particle velocity (drag model: particle follows fluid)
        double drag_coeff = 0.1; // How quickly particle matches fluid velocity
        p.velocity += (fluid_vel - p.velocity) * drag_coeff * dt;
        
        // Gravity
        p.velocity.z() -= 9.81 * dt * 0.1; // Reduced gravity effect
        
        // Update position
        p.position += p.velocity * dt;
        
        // Update temperature (query from fluid field)
        p.temperature = fluid_solver.get_temperature_at(p.position);
        
        // Update pressure
        p.pressure = fluid_solver.get_pressure_at(p.position);
        
        // Age
        p.age += dt;
    }
    
    // Remove old/out-of-bounds particles
    remove_old_particles(0.5);           // Remove particles older than 0.5s
    remove_out_of_bounds_particles(0.02); // Remove particles beyond 20mm
}

void ParticleSystem::remove_old_particles(double max_age) {
    for (auto& p : particles) {
        if (p.age > max_age) {
            p.is_active = false;
        }
    }
    
    // Actually remove inactive particles
    particles.erase(
        std::remove_if(particles.begin(), particles.end(),
            [](const Particle& p) { return !p.is_active; }),
        particles.end()
    );
}

void ParticleSystem::remove_out_of_bounds_particles(double max_distance) {
    for (auto& p : particles) {
        if (p.position.norm() > max_distance) {
            p.is_active = false;
        }
    }
}

std::vector<Particle> ParticleSystem::get_active_particles() const {
    std::vector<Particle> active;
    for (const auto& p : particles) {
        if (p.is_active) {
            active.push_back(p);
        }
    }
    return active;
}

int ParticleSystem::get_active_count() const {
    int count = 0;
    for (const auto& p : particles) {
        if (p.is_active) count++;
    }
    return count;
}

double ParticleSystem::get_average_temperature() const {
    double sum = 0.0;
    int count = 0;
    for (const auto& p : particles) {
        if (p.is_active) {
            sum += p.temperature;
            count++;
        }
    }
    return count > 0 ? sum / count : 0.0;
}

double ParticleSystem::get_max_pressure() const {
    double max_p = 0.0;
    for (const auto& p : particles) {
        if (p.is_active && p.pressure > max_p) {
            max_p = p.pressure;
        }
    }
    return max_p;
}

json ParticleSystem::export_particles() const {
    json particle_array = json::array();
    
    for (const auto& p : particles) {
        if (p.is_active) {
            particle_array.push_back({
                {"position", {p.position.x(), p.position.y(), p.position.z()}},
                {"velocity", {p.velocity.x(), p.velocity.y(), p.velocity.z()}},
                {"temperature", p.temperature},
                {"pressure", p.pressure / 1e6}, // Convert Pa to MPa
                {"age", p.age}
            });
        }
    }
    
    return particle_array;
}