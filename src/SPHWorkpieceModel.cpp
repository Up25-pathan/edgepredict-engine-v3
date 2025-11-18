#include "SPHWorkpieceModel.h"
#include <iostream>
#include <stdexcept>
#include <omp.h> // For OpenMP parallelization

// --- SPH KERNELS ---
// These are standard SPH weighting functions.
// 'h' is the smoothing_radius

// W_poly6 kernel
double poly6_kernel(const Eigen::Vector3d& r_vec, double h) {
    double r = r_vec.norm();
    if (r >= 0.0 && r <= h) {
        double h2 = h * h;
        double h9 = h2 * h2 * h2 * h; // h^9
        double coeff = 315.0 / (64.0 * M_PI * h9);
        double r2 = r * r;
        return coeff * std::pow(h2 - r2, 3.0);
    }
    return 0.0;
}

// W_spiky kernel (gradient)
Eigen::Vector3d spiky_kernel_grad(const Eigen::Vector3d& r_vec, double h) {
    double r = r_vec.norm();
    if (r > 0.0 && r <= h) {
        double h6 = h * h * h * h * h * h;
        double coeff = -45.0 / (M_PI * h6);
        double hr = h - r;
        return (r_vec / r) * (coeff * hr * hr);
    }
    return Eigen::Vector3d::Zero();
}

// -------------------------------------------------

SPHWorkpieceModel::SPHWorkpieceModel(const json& config) : m_config_data(config) {
    // Load material properties for the *workpiece*
    // We re-use the same models, but they now apply to the particles
    m_material_model = std::make_unique<JohnsonCookModel>(config_data);
    m_failure_model = std::make_unique<FailureCriterion>(config_data);

    // Get SPH-specific parameters (we'd add these to input.json)
    m_smoothing_radius = config["sph_parameters"].value("smoothing_radius_m", 0.0001); // 0.1 mm
    m_gas_stiffness = config["sph_parameters"].value("gas_stiffness", 3000.0);
    m_viscosity = config["sph_parameters"].value("viscosity", 0.01);
    
    // Get density from material properties
    double density = config["material_properties"]["density_kg_m3"].get<double>();
    
    // Calculate particle mass based on particle spacing (h/2)
    double particle_spacing = m_smoothing_radius / 2.0;
    double particle_volume = particle_spacing * particle_spacing * particle_spacing;
    m_particle_mass = particle_volume * density;

    std::cout << "[SPH] SPHWorkpieceModel constructed." << std::endl;
}

SPHWorkpieceModel::~SPHWorkpieceModel() {}

void SPHWorkpieceModel::initialize_workpiece(const Eigen::Vector3d& workpiece_min, 
                                           const Eigen::Vector3d& workpiece_max) 
{
    std::cout << "[SPH] Initializing workpiece particle grid..." << std::endl;
    
    double particle_spacing = m_smoothing_radius / 2.0; // Place particles closer than 'h'
    double ambient_temp = m_config_data["physics_parameters"]["ambient_temperature_C"].get<double>();

    // Create a 3D grid of particles
    for (double z = workpiece_min.z(); z < workpiece_max.z(); z += particle_spacing) {
        for (double y = workpiece_min.y(); y < workpiece_max.y(); y += particle_spacing) {
            for (double x = workpiece_min.x(); x < workpiece_max.x(); x += particle_spacing) {
                
                SPHParticle p;
                p.position = Eigen::Vector3d(x, y, z);
                p.velocity = Eigen::Vector3d::Zero();
                p.acceleration = Eigen::Vector3d::Zero();
                p.mass = m_particle_mass;
                p.density = m_config_data["material_properties"]["density_kg_m3"].get<double>();
                p.pressure = 0.0;
                p.temperature = ambient_temp;
                p.strain = 0.0;
                p.plastic_strain = 0.0;
                p.stress = 0.0;
                p.damage = 0.0;
                p.status = ParticleStatus::WORKPIECE_SOLID;
                
                m_particles.push_back(p);
            }
        }
    }
    std::cout << "[SPH] Created " << m_particles.size() << " workpiece particles." << std::endl;
}

void SPHWorkpieceModel::update_step(double dt) {
    // --- This is the core SPH physics loop ---
    
    // 1. Calculate density and pressure for all particles
    calculate_density_and_pressure();
    
    // 2. Calculate internal SPH forces (pressure, viscosity)
    calculate_internal_forces();
    
    // 3. Integrate to get new positions
    integrate(dt);
}

void SPHWorkpieceModel::calculate_density_and_pressure() {
    double h = m_smoothing_radius;
    double initial_density = m_config_data["material_properties"]["density_kg_m3"].get<double>();

    #pragma omp parallel for
    for (size_t i = 0; i < m_particles.size(); ++i) {
        if (m_particles[i].status == ParticleStatus::INACTIVE) continue;

        double total_density = 0.0;
        
        // Find neighbors (simple all-pairs, slow. A grid-search is faster)
        for (size_t j = 0; j < m_particles.size(); ++j) {
            Eigen::Vector3d r_ij = m_particles[i].position - m_particles[j].position;
            if (r_ij.norm() <= h) {
                total_density += m_particles[j].mass * poly6_kernel(r_ij, h);
            }
        }
        
        m_particles[i].density = std::max(total_density, initial_density);
        
        // Equation of State (Tait's equation) to calculate pressure
        m_particles[i].pressure = m_gas_stiffness * (std::pow(m_particles[i].density / initial_density, 7.0) - 1.0);
    }
}

void SPHWorkpieceModel::calculate_internal_forces() {
    double h = m_smoothing_radius;

    #pragma omp parallel for
    for (size_t i = 0; i < m_particles.size(); ++i) {
        if (m_particles[i].status == ParticleStatus::INACTIVE) continue;

        Eigen::Vector3d pressure_force = Eigen::Vector3d::Zero();
        Eigen::Vector3d viscosity_force = Eigen::Vector3d::Zero();

        for (size_t j = 0; j < m_particles.size(); ++j) {
            if (i == j) continue;

            Eigen::Vector3d r_ij = m_particles[i].position - m_particles[j].position;
            if (r_ij.norm() <= h) {
                // Pressure Force
                pressure_force -= m_particles[j].mass * (m_particles[i].pressure / (m_particles[i].density * m_particles[i].density) + 
                                   m_particles[j].pressure / (m_particles[j].density * m_particles[j].density)) * spiky_kernel_grad(r_ij, h);

                // Viscosity Force
                viscosity_force += m_particles[j].mass * (m_particles[j].velocity - m_particles[i].velocity) / m_particles[j].density *
                                   poly6_kernel(r_ij, h); // Simplified viscosity
            }
        }
        
        // F = F_pressure + F_viscosity + F_gravity
        m_particles[i].acceleration = (pressure_force + viscosity_force * m_viscosity) / m_particles[i].density;
        // m_particles[i].acceleration.z() -= 9.81; // Optional gravity
    }
}

void SPHWorkpieceModel::integrate(double dt) {
    #pragma omp parallel for
    for (size_t i = 0; i < m_particles.size(); ++i) {
        if (m_particles[i].status == ParticleStatus::INACTIVE) continue;

        // Simple Euler integration
        m_particles[i].velocity += m_particles[i].acceleration * dt;
        m_particles[i].position += m_particles[i].velocity * dt;

        // Deactivate particles that fly too far
        if (m_particles[i].position.norm() > 1.0) { // 1 meter boundary
             m_particles[i].status = ParticleStatus::INACTIVE;
        }
    }
}

// THIS IS THE R&D "CUTTING" FUNCTION
void SPHWorkpieceModel::interact_with_tool(Mesh& tool_mesh, double dt) {
    if (!m_material_model || !m_failure_model) return;

    double h = m_smoothing_radius;
    double ambient_temp = m_config_data["physics_parameters"]["ambient_temperature_C"].get<double>();
    
    // This is the core R&D interaction
    #pragma omp parallel for
    for (size_t i = 0; i < m_particles.size(); ++i) {
        SPHParticle& p = m_particles[i];
        if (p.status == ParticleStatus::INACTIVE) continue;

        // Find the *closest* active tool node to this particle
        double min_dist_sq = h * h;
        Node* contact_node = nullptr;
        Eigen::Vector3d node_world_pos; // We'll store the node's position

        for (Node& node : tool_mesh.nodes) {
            if (!node.is_active_contact) continue; // Only check "cutting edge" nodes

            // TODO: This is where we use the tool_transform from RotationalFEA
            // For now, we assume the node positions are already in world space.
            // In the real MillingStrategy, we'd pass in the tool_transform.
            // Eigen::Vector3d node_world_pos = tool_transform * node.position;
            
            // This is a placeholder until RotationalFEA is built
            // We'll just use the node's base position for this step
            node_world_pos = node.position; 
            
            double dist_sq = (p.position - node_world_pos).normSqr();
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                contact_node = &node;
            }
        }

        // If a tool node is close enough...
        if (contact_node) {
            double dist = std::sqrt(min_dist_sq);
            
            // --- 1. Apply Force ---
            // Apply a strong repulsion force (like a spring)
            double repulsion_force_mag = 1e9 * (h - dist) * p.mass;
            Eigen::Vector3d force_dir = (p.position - node_world_pos).normalized();
            p.acceleration += force_dir * repulsion_force_mag;

            // --- 2. Calculate Strain & Damage ---
            // This is the "cutting" part. The tool's velocity causes strain.
            // TODO: Get real velocity from RotationalFEA
            double tool_velocity = 2.0; // Placeholder for v = r * omega
            double strain_rate = tool_velocity / (h * 2.0); // Simple estimate
            p.strain += strain_rate * dt;
            
            // Calculate stress on the particle using Johnson-Cook
            p.stress = m_material_model->calculate_stress(p.temperature, p.strain, strain_rate);

            // Calculate damage on the particle
            double damage_inc = m_failure_model->calculate_damage_increment(p.stress, 1.0, p.damage);
            p.damage += damage_inc;
            
            // --- 3. Check for Fracture (This is the R&D part) ---
            if (m_failure_model->check_failure(p.stress, p.temperature, p.damage)) {
                p.status = ParticleStatus::CHIP_FLOWING;
            }
            
            // --- 4. Heat Transfer ---
            // Particle gets hot from plastic deformation
            double heat_gen = p.stress * strain_rate * 0.9; // 90% of work becomes heat
            double cp = m_config_data["material_properties"]["specific_heat_J_kgC"].get<double>();
            double temp_increase = (heat_gen * dt) / (p.density * cp);
            p.temperature += temp_increase;

            // Heat flows from particle *to* the tool node
            double T_diff = p.temperature - contact_node->temperature;
            if (T_diff > 0) {
                double conductivity = 1000.0; // High contact conductivity
                double heat_transfer = T_diff * conductivity * dt;
                
                // This is a simplified, non-physical heat transfer. A real model
                // would be much more complex.
                p.temperature -= heat_transfer * 0.1; 
                #pragma omp atomic
                contact_node->temperature += heat_transfer * 0.01; // Tool heats up
            }
        }
        else {
            // Particle is not near the tool, just cool it down
            double heat_loss = (p.temperature - ambient_temp) * 0.1 * dt;
            p.temperature -= heat_loss;
        }
    }
}

std::vector<const SPHParticle*> SPHWorkpieceModel::get_particles_for_visualization() const {
    std::vector<const SPHParticle*> active_particles;
    for (const auto& p : m_particles) {
        if (p.status != ParticleStatus::INACTIVE) {
            active_particles.push_back(&p);
        }
    }
    return active_particles;
}