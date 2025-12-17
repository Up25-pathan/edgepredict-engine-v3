#include "SPHWorkpieceModel.h"
#include "SpatialHash.h"
#include <iostream>
#include <stdexcept>
#include <omp.h>
#include <cmath>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const double POLY6_COEFF = 315.0 / (64.0 * M_PI);
const double SPIKY_COEFF = -45.0 / (M_PI);

double poly6_kernel(const Eigen::Vector3d& r_vec, double h) {
    double r = r_vec.norm();
    if (r >= 0.0 && r <= h) {
        double h2 = h * h;
        double h9 = h2 * h2 * h2 * h * h * h; 
        double term = h2 - r * r;
        return (POLY6_COEFF / h9) * term * term * term;
    }
    return 0.0;
}

Eigen::Vector3d spiky_kernel_grad(const Eigen::Vector3d& r_vec, double h) {
    double r = r_vec.norm();
    if (r > 1e-9 && r <= h) {
        double h6 = h * h * h * h * h * h;
        double term = h - r;
        return (r_vec / r) * (SPIKY_COEFF / h6) * term * term;
    }
    return Eigen::Vector3d::Zero();
}

SPHWorkpieceModel::SPHWorkpieceModel(const json& config) : m_config_data(config) {
    m_material_model = std::make_unique<JohnsonCookModel>(config);
    m_failure_model = std::make_unique<FailureCriterion>(config);

    m_smoothing_radius = config["sph_parameters"].value("smoothing_radius_m", 0.0001); 
    m_gas_stiffness = config["sph_parameters"].value("gas_stiffness", 3000.0);
    m_viscosity = config["sph_parameters"].value("viscosity", 0.01);
    
    double density = config["material_properties"]["density_kg_m3"].get<double>();
    double particle_spacing = m_smoothing_radius / 2.0;
    m_particle_mass = std::pow(particle_spacing, 3) * density;

    m_grid = std::make_unique<SpatialGrid>(m_smoothing_radius);
    m_active_bounds.setEmpty();
    std::cout << "[SPH] Z-Curve Memory Layout Initialized." << std::endl;
}

SPHWorkpieceModel::~SPHWorkpieceModel() {}

void SPHWorkpieceModel::initialize_workpiece(const Eigen::Vector3d& min_c, const Eigen::Vector3d& max_c) {
    double spacing = m_smoothing_radius / 2.0;
    double density = m_config_data["material_properties"]["density_kg_m3"].get<double>();
    double temp = m_config_data["physics_parameters"]["ambient_temperature_C"].get<double>();

    m_particles.clear();
    int est_count = (int)((max_c.x()-min_c.x())*(max_c.y()-min_c.y())*(max_c.z()-min_c.z()) / std::pow(spacing,3));
    if(est_count > 0) m_particles.reserve(est_count);

    for (double z = min_c.z(); z < max_c.z(); z += spacing) {
        for (double y = min_c.y(); y < max_c.y(); y += spacing) {
            for (double x = min_c.x(); x < max_c.x(); x += spacing) {
                SPHParticle p;
                p.position = Eigen::Vector3d(x, y, z);
                p.velocity = Eigen::Vector3d::Zero();
                p.acceleration = Eigen::Vector3d::Zero();
                p.mass = m_particle_mass;
                p.density = density;
                p.pressure = 0.0;
                p.temperature = temp;
                p.strain = 0.0; 
                p.plastic_strain = 0.0;
                p.stress = 0.0; 
                p.damage = 0.0;
                p.status = ParticleStatus::WORKPIECE_SOLID;
                m_particles.push_back(p);
            }
        }
    }
    m_grid = std::make_unique<SpatialGrid>(m_smoothing_radius, m_particles.size());
    std::cout << "[SPH] Generated " << m_particles.size() << " particles." << std::endl;
}

void SPHWorkpieceModel::update_step(double dt) {
    m_grid->clear(m_particles.size());
    for (size_t i = 0; i < m_particles.size(); ++i) {
        if (m_particles[i].status != ParticleStatus::INACTIVE) {
            m_grid->insert(i, m_particles[i].position);
        }
    }
    calculate_density_and_pressure();
    calculate_internal_forces();
    integrate(dt);
}

void SPHWorkpieceModel::calculate_density_and_pressure() {
    double h = m_smoothing_radius;
    double h2 = h * h;
    double rho0 = m_config_data["material_properties"]["density_kg_m3"].get<double>();

    #pragma omp parallel 
    {
        #pragma omp for
        for (size_t i = 0; i < m_particles.size(); ++i) {
            if (m_particles[i].status == ParticleStatus::INACTIVE) continue;

            if (m_bounds_initialized && !m_active_bounds.contains(m_particles[i].position)) {
                m_particles[i].density = rho0;
                m_particles[i].pressure = 0.0;
                continue;
            }

            double total_density = 0.0;
            m_grid->query_neighbors(m_particles[i].position, [&](int j) {
                Eigen::Vector3d r_vec = m_particles[i].position - m_particles[j].position;
                double r2 = r_vec.squaredNorm();
                if (r2 < h2) {
                    total_density += m_particles[j].mass * poly6_kernel(r_vec, h);
                }
            });

            m_particles[i].density = std::max(total_density, rho0);
            m_particles[i].pressure = m_gas_stiffness * (pow(m_particles[i].density / rho0, 7.0) - 1.0);
        }
    }
}

void SPHWorkpieceModel::calculate_internal_forces() {
    double h = m_smoothing_radius;

    #pragma omp parallel 
    {
        #pragma omp for
        for (size_t i = 0; i < m_particles.size(); ++i) {
            if (m_particles[i].status == ParticleStatus::INACTIVE) continue;

            if (m_bounds_initialized && !m_active_bounds.contains(m_particles[i].position)) {
                m_particles[i].acceleration = Eigen::Vector3d::Zero();
                continue;
            }

            Eigen::Vector3d f_press = Eigen::Vector3d::Zero();
            Eigen::Vector3d f_visc = Eigen::Vector3d::Zero();

            m_grid->query_neighbors(m_particles[i].position, [&](int j) {
                if (i == j) return;
                Eigen::Vector3d r_vec = m_particles[i].position - m_particles[j].position;
                double r = r_vec.norm();
                if (r > 1e-9 && r < h) {
                    double p_term = (m_particles[i].pressure / (m_particles[i].density * m_particles[i].density)) + 
                                    (m_particles[j].pressure / (m_particles[j].density * m_particles[j].density));
                    f_press -= m_particles[j].mass * p_term * spiky_kernel_grad(r_vec, h);
                    f_visc += m_particles[j].mass * (m_particles[j].velocity - m_particles[i].velocity) / m_particles[j].density * poly6_kernel(r_vec, h);
                }
            });

            m_particles[i].acceleration = (f_press + f_visc * m_viscosity) / m_particles[i].density;
        }
    }
}

void SPHWorkpieceModel::integrate(double dt) {
    #pragma omp parallel for
    for (size_t i = 0; i < m_particles.size(); ++i) {
        if (m_particles[i].status == ParticleStatus::INACTIVE) continue;
        
        bool is_moving = m_particles[i].velocity.squaredNorm() > 1e-6;
        bool in_bounds = m_bounds_initialized && m_active_bounds.contains(m_particles[i].position);

        if (!is_moving && !in_bounds) {
            m_particles[i].velocity = Eigen::Vector3d::Zero();
            m_particles[i].acceleration = Eigen::Vector3d::Zero();
            continue;
        }

        m_particles[i].velocity += m_particles[i].acceleration * dt;
        m_particles[i].position += m_particles[i].velocity * dt;
        if (m_particles[i].position.norm() > 10.0) m_particles[i].status = ParticleStatus::INACTIVE;
    }
}

void SPHWorkpieceModel::interact_with_tool(Mesh& tool_mesh, double dt) {
    if (!m_material_model || !m_failure_model) return;

    double h = m_smoothing_radius;
    double k_contact = 1e7; 
    double A_wear = m_config_data["material_properties"]["usui_wear_model"].value("A_constant", 1e-9);
    double B_wear = m_config_data["material_properties"]["usui_wear_model"].value("B_inv_temp_K", 1000.0);

    std::vector<Node*> active_nodes;
    active_nodes.reserve(tool_mesh.nodes.size());
    m_active_bounds.setEmpty();
    
    SpatialGrid tool_grid(h, tool_mesh.nodes.size());
    for (size_t i = 0; i < tool_mesh.nodes.size(); ++i) {
        if (tool_mesh.nodes[i].is_active_contact) {
            active_nodes.push_back(&tool_mesh.nodes[i]);
            tool_grid.insert(i, tool_mesh.nodes[i].position);
            m_active_bounds.extend(tool_mesh.nodes[i].position);
        }
    }

    if (active_nodes.empty()) { m_bounds_initialized = false; return; }

    Eigen::Vector3d margin(h*4.0, h*4.0, h*4.0);
    m_active_bounds.min() -= margin;
    m_active_bounds.max() += margin;
    m_bounds_initialized = true;

    double cp_val = m_config_data["material_properties"]["specific_heat_J_kgC"].get<double>();

    #pragma omp parallel 
    {
        #pragma omp for
        for (size_t i = 0; i < m_particles.size(); ++i) {
            SPHParticle& p = m_particles[i];
            if (p.status == ParticleStatus::INACTIVE) continue;
            if (!m_active_bounds.contains(p.position)) continue;

            Node* contact_node = nullptr;
            double min_dist_sq = h * h;

            tool_grid.query_neighbors(p.position, [&](int node_idx) {
                double d2 = (p.position - tool_mesh.nodes[node_idx].position).squaredNorm();
                if (d2 < min_dist_sq) {
                    min_dist_sq = d2;
                    contact_node = &tool_mesh.nodes[node_idx];
                }
            });

            if (contact_node) {
                double dist = std::sqrt(min_dist_sq);
                double pen = h - dist;
                double f_mag = std::min(k_contact * pen, 20.0);

                Eigen::Vector3d dir = (p.position - contact_node->position).normalized();
                Eigen::Vector3d f_vec = dir * f_mag;

                p.acceleration += f_vec / p.mass;

                #pragma omp atomic
                contact_node->force.x() -= f_vec.x();
                #pragma omp atomic
                contact_node->force.y() -= f_vec.y();
                #pragma omp atomic
                contact_node->force.z() -= f_vec.z();

                Eigen::Vector3d rel_vel = p.velocity - contact_node->velocity;
                double slip = rel_vel.norm();
                double heat = f_mag * 0.3 * slip;
                
                p.temperature += (heat * dt) / (p.mass * cp_val);
                if (p.temperature > contact_node->temperature) {
                    double cond = (p.temperature - contact_node->temperature) * 500.0 * dt;
                    p.temperature -= cond;
                    #pragma omp atomic
                    contact_node->temperature += cond * 0.1;
                }

                // Usui Wear Logic
                double contact_pressure = f_mag / (h * h);
                double T_kelvin = contact_node->temperature + 273.15;
                double wear_rate = A_wear * contact_pressure * slip * std::exp(-B_wear / T_kelvin);
                #pragma omp atomic
                contact_node->accumulated_wear += wear_rate * dt;

                double sr = slip / h;
                p.strain += sr * dt;
                p.stress = m_material_model->calculate_stress(p.temperature, p.strain, sr);
                if (m_failure_model->check_failure(p.stress, p.temperature, p.damage)) {
                    p.status = ParticleStatus::CHIP_FLOWING;
                }
            }
        }
    }
}

std::vector<const SPHParticle*> SPHWorkpieceModel::get_particles_for_visualization() const {
    std::vector<const SPHParticle*> active;
    active.reserve(m_particles.size());
    for (const auto& p : m_particles) {
        if (p.status != ParticleStatus::INACTIVE) active.push_back(&p);
    }
    return active;
}