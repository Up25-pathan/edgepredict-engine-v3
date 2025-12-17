#include "SPHWorkpieceModel.h"
#include "SpatialHash.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

// --- SYCL / GPU HEADER ---
#include <sycl/sycl.hpp>

using namespace sycl;

// Constants for Physics (must be available to device)
constexpr double M_PI_CONST = 3.14159265358979323846;
constexpr double POLY6_COEFF = 315.0 / (64.0 * M_PI_CONST);
constexpr double SPIKY_COEFF = -45.0 / (M_PI_CONST);

// --- GPU KERNEL HELPERS ---
// These run on the Graphics Card. 

inline double gpu_poly6_kernel(const double3& r_vec, double h) {
    double r = sycl::length(r_vec);
    if (r >= 0.0 && r <= h) {
        double h2 = h * h;
        double h9 = h2 * h2 * h2 * h2 * h; // h^9
        double term = h2 - r * r;
        return (POLY6_COEFF / h9) * term * term * term;
    }
    return 0.0;
}

inline double3 gpu_spiky_kernel_grad(const double3& r_vec, double h) {
    double r = sycl::length(r_vec);
    if (r > 1e-9 && r <= h) {
        double h6 = h * h * h * h * h * h;
        double term = h - r;
        // Gradient direction * value
        return (r_vec / r) * (SPIKY_COEFF / h6) * term * term;
    }
    return double3(0.0, 0.0, 0.0);
}

// Z-Curve Helper (Must match SpatialHash.h logic)
inline uint32_t gpu_expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// -------------------------------------------------------

SPHWorkpieceModel::SPHWorkpieceModel(const json& config) 
    : m_config_data(config)
    // Initialize SYCL Queue (Selects GPU if available, else CPU)
    , m_queue(default_selector_v)
{
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

    std::cout << "[SPH] SYCL Hybrid Engine Initialized on device: " 
              << m_queue.get_device().get_info<info::device::name>() << std::endl;
}

SPHWorkpieceModel::~SPHWorkpieceModel() {}

void SPHWorkpieceModel::initialize_workpiece(const Eigen::Vector3d& min_c, const Eigen::Vector3d& max_c) {
    double spacing = m_smoothing_radius / 2.0;
    double density = m_config_data["material_properties"]["density_kg_m3"].get<double>();
    double temp = m_config_data["physics_parameters"]["ambient_temperature_C"].get<double>();

    m_data.clear();
    
    // Heuristic for reservation
    int est = (int)((max_c.x()-min_c.x())*(max_c.y()-min_c.y())*(max_c.z()-min_c.z()) / std::pow(spacing,3));
    if(est > 0) m_data.reserve(est);

    for (double z = min_c.z(); z < max_c.z(); z += spacing) {
        for (double y = min_c.y(); y < max_c.y(); y += spacing) {
            for (double x = min_c.x(); x < max_c.x(); x += spacing) {
                m_data.push_back(Eigen::Vector3d(x, y, z), m_particle_mass, density, temp);
            }
        }
    }
    // Initialize Grid size
    m_grid = std::make_unique<SpatialGrid>(m_smoothing_radius, m_data.count);
    
    std::cout << "[SPH] Generated " << m_data.count << " particles." << std::endl;
}

void SPHWorkpieceModel::update_step(double dt) {
    // 1. CPU: Build Spatial Hash Grid
    // (Fast enough on CPU for now, complex to sort on GPU without specialized libs)
    m_grid->clear(m_data.count);
    for (size_t i = 0; i < m_data.count; ++i) {
        if (m_data.status[i] != (int)ParticleStatus::INACTIVE) {
            m_grid->insert(i, m_data.position[i]);
        }
    }

    // 2. GPU: Density & Pressure
    calculate_density_and_pressure();

    // 3. GPU: Internal Forces (Viscosity, Pressure Gradient)
    calculate_internal_forces();

    // 4. CPU: Tool Interaction 
    // (Must be on CPU because 'Tool Mesh' is a complex object not yet on GPU)
    // Note: Data automatically syncs back from GPU buffers before this runs.
    interact_with_tool(*static_cast<Mesh*>(nullptr), dt); // Placeholder call, actual call happens in Strategy

    // 5. GPU: Integration
    integrate(dt);
}

// -------------------------------------------------------
// GPU KERNEL: DENSITY & PRESSURE
// -------------------------------------------------------
void SPHWorkpieceModel::calculate_density_and_pressure() {
    size_t N = m_data.count;
    if (N == 0) return;

    double h = m_smoothing_radius;
    double h2 = h * h;
    double rho0 = m_config_data["material_properties"]["density_kg_m3"].get<double>();
    double k_gas = m_gas_stiffness;
    
    // Grid Params
    double cell_size = m_grid->cell_size;
    int table_size = m_grid->table_size;

    // Create Buffers (Transfers Data CPU -> GPU)
    // Converting Eigen::Vector3d (double[3]) to sycl::double3 for alignment
    // Ideally we'd use a custom struct, but for now we cast or copy.
    // To be safe and simple, we assume m_data memory layout is compatible or we let SYCL handle raw pointers.
    // Actually, Eigen vectors are just 3 doubles. We can reinterpret_cast in the kernel 
    // or use a buffer of doubles and index *3. Let's use buffer<double> for components.
    // BUT for speed in this example, let's wrap the struct of arrays directly.
    
    buffer<Eigen::Vector3d, 1> b_pos(m_data.position.data(), range<1>(N));
    buffer<double, 1> b_mass(m_data.mass.data(), range<1>(N));
    buffer<double, 1> b_dens(m_data.density.data(), range<1>(N));
    buffer<double, 1> b_press(m_data.pressure.data(), range<1>(N));
    buffer<int, 1> b_stat(m_data.status.data(), range<1>(N));
    
    // Grid Buffers
    buffer<int, 1> b_head(m_grid->head.data(), range<1>(m_grid->head.size()));
    buffer<int, 1> b_next(m_grid->next.data(), range<1>(m_grid->next.size()));

    m_queue.submit([&](handler& cgh) {
        auto pos = b_pos.get_access<access::mode::read>(cgh);
        auto mass = b_mass.get_access<access::mode::read>(cgh);
        auto head = b_head.get_access<access::mode::read>(cgh);
        auto next = b_next.get_access<access::mode::read>(cgh);
        auto status = b_stat.get_access<access::mode::read>(cgh);
        
        auto dens = b_dens.get_access<access::mode::write>(cgh);
        auto press = b_press.get_access<access::mode::write>(cgh);

        cgh.parallel_for(range<1>(N), [=](id<1> idx) {
            int i = idx[0];
            if (status[i] == 0) return;

            double total_density = 0.0;
            
            // Map Eigen::Vector3d to sycl::double3
            double3 pi(pos[i].x(), pos[i].y(), pos[i].z());

            // Z-Curve Neighbor Search
            double offset = 1000.0;
            int cx = (int)((pi.x() + offset) / cell_size);
            int cy = (int)((pi.y() + offset) / cell_size);
            int cz = (int)((pi.z() + offset) / cell_size);

            for (int k = cz - 1; k <= cz + 1; ++k) {
                for (int j = cy - 1; j <= cy + 1; ++j) {
                    for (int gx = cx - 1; gx <= cx + 1; ++gx) {
                        // Reconstruct Z-Curve Hash
                        uint32_t ux = (uint32_t)gx;
                        uint32_t uy = (uint32_t)j;
                        uint32_t uz = (uint32_t)k;
                        
                        uint32_t code = (gpu_expand_bits(ux) | (gpu_expand_bits(uy) << 1) | (gpu_expand_bits(uz) << 2));
                        int bucket = code & (table_size - 1);

                        int p_idx = head[bucket];
                        while (p_idx != -1) {
                            double3 pj(pos[p_idx].x(), pos[p_idx].y(), pos[p_idx].z());
                            double3 r_vec = pi - pj;
                            
                            // Manual length check
                            double r2 = r_vec.x()*r_vec.x() + r_vec.y()*r_vec.y() + r_vec.z()*r_vec.z();
                            
                            if (r2 < h2) {
                                total_density += mass[p_idx] * gpu_poly6_kernel(r_vec, h);
                            }
                            
                            p_idx = next[p_idx]; // Linked list traversal
                        }
                    }
                }
            }

            dens[i] = (total_density > rho0) ? total_density : rho0;
            // Tait's Equation
            press[i] = k_gas * (sycl::pow(dens[i] / rho0, 7.0) - 1.0);
        });
    });
    // Destructors of 'buffer' objects automatically wait and copy data back to CPU
}

// -------------------------------------------------------
// GPU KERNEL: INTERNAL FORCES
// -------------------------------------------------------
void SPHWorkpieceModel::calculate_internal_forces() {
    size_t N = m_data.count;
    if (N == 0) return;

    double h = m_smoothing_radius;
    double mu = m_viscosity;

    // Buffers
    buffer<Eigen::Vector3d, 1> b_pos(m_data.position.data(), range<1>(N));
    buffer<Eigen::Vector3d, 1> b_vel(m_data.velocity.data(), range<1>(N));
    buffer<Eigen::Vector3d, 1> b_acc(m_data.acceleration.data(), range<1>(N)); // WRITE
    buffer<double, 1> b_mass(m_data.mass.data(), range<1>(N));
    buffer<double, 1> b_dens(m_data.density.data(), range<1>(N));
    buffer<double, 1> b_press(m_data.pressure.data(), range<1>(N));
    buffer<int, 1> b_stat(m_data.status.data(), range<1>(N));
    
    buffer<int, 1> b_head(m_grid->head.data(), range<1>(m_grid->head.size()));
    buffer<int, 1> b_next(m_grid->next.data(), range<1>(m_grid->next.size()));
    
    double cell_size = m_grid->cell_size;
    int table_size = m_grid->table_size;

    m_queue.submit([&](handler& cgh) {
        auto pos = b_pos.get_access<access::mode::read>(cgh);
        auto vel = b_vel.get_access<access::mode::read>(cgh);
        auto mass = b_mass.get_access<access::mode::read>(cgh);
        auto dens = b_dens.get_access<access::mode::read>(cgh);
        auto press = b_press.get_access<access::mode::read>(cgh);
        auto status = b_stat.get_access<access::mode::read>(cgh);
        auto head = b_head.get_access<access::mode::read>(cgh);
        auto next = b_next.get_access<access::mode::read>(cgh);
        
        auto acc = b_acc.get_access<access::mode::write>(cgh);

        cgh.parallel_for(range<1>(N), [=](id<1> idx) {
            int i = idx[0];
            if (status[i] == 0) return;

            double3 force_press(0.0, 0.0, 0.0);
            double3 force_visc(0.0, 0.0, 0.0);
            
            double3 pi(pos[i].x(), pos[i].y(), pos[i].z());
            double3 vi(vel[i].x(), vel[i].y(), vel[i].z());

            // Neighbor Search (Duplicated logic for kernel independence)
            double offset = 1000.0;
            int cx = (int)((pi.x() + offset) / cell_size);
            int cy = (int)((pi.y() + offset) / cell_size);
            int cz = (int)((pi.z() + offset) / cell_size);

            for (int k = cz - 1; k <= cz + 1; ++k) {
                for (int j = cy - 1; j <= cy + 1; ++j) {
                    for (int gx = cx - 1; gx <= cx + 1; ++gx) {
                        uint32_t code = (gpu_expand_bits((uint32_t)gx) | (gpu_expand_bits((uint32_t)j) << 1) | (gpu_expand_bits((uint32_t)k) << 2));
                        int bucket = code & (table_size - 1);

                        int p_idx = head[bucket];
                        while (p_idx != -1) {
                            if (p_idx != i) {
                                double3 pj(pos[p_idx].x(), pos[p_idx].y(), pos[p_idx].z());
                                double3 vj(vel[p_idx].x(), vel[p_idx].y(), vel[p_idx].z());
                                double3 r_vec = pi - pj;
                                double r = sycl::length(r_vec);

                                if (r > 1e-9 && r < h) {
                                    // Pressure Force
                                    double p_term = (press[i] / (dens[i]*dens[i])) + (press[p_idx] / (dens[p_idx]*dens[p_idx]));
                                    force_press -= gpu_spiky_kernel_grad(r_vec, h) * mass[p_idx] * p_term;

                                    // Viscosity Force
                                    double3 v_diff = vj - vi;
                                    force_visc += v_diff * (mass[p_idx] / dens[p_idx]) * gpu_poly6_kernel(r_vec, h);
                                }
                            }
                            p_idx = next[p_idx];
                        }
                    }
                }
            }

            double3 total_acc = (force_press + force_visc * mu) / dens[i];
            
            // Write back to Eigen vector (requires manual component assignment)
            acc[i].x() = total_acc.x();
            acc[i].y() = total_acc.y();
            acc[i].z() = total_acc.z();
        });
    });
}

// -------------------------------------------------------
// CPU: TOOL INTERACTION
// -------------------------------------------------------
// Kept on CPU for now because 'tool_mesh' updates involve 
// complex atomic writes to a structure not yet ported to GPU.
// The buffers from previous steps sync back automatically here.
void SPHWorkpieceModel::interact_with_tool(Mesh& tool_mesh, double dt) {
    if (!m_material_model || !m_failure_model) return;
    if (tool_mesh.nodes.empty()) return; // Safety

    double h = m_smoothing_radius;
    double k_contact = 1e7; 
    double A_wear = m_config_data["material_properties"]["usui_wear_model"].value("A_constant", 1e-9);
    double B_wear = m_config_data["material_properties"]["usui_wear_model"].value("B_inv_temp_K", 1000.0);
    double cp_val = m_config_data["material_properties"]["specific_heat_J_kgC"].get<double>();

    // Build CPU Tool Grid (Fast for small tool mesh)
    SpatialGrid tool_grid(h, tool_mesh.nodes.size());
    std::vector<Node*> active_nodes;
    for (size_t i = 0; i < tool_mesh.nodes.size(); ++i) {
        if (tool_mesh.nodes[i].is_active_contact) {
            active_nodes.push_back(&tool_mesh.nodes[i]);
            tool_grid.insert(i, tool_mesh.nodes[i].position);
        }
    }
    if (active_nodes.empty()) return;

    #pragma omp parallel for
    for (size_t i = 0; i < m_data.count; ++i) {
        if (m_data.status[i] == (int)ParticleStatus::INACTIVE) continue;

        // Optimization: Pre-check if particle is even close to tool bounds
        // (Can add AABB check here if needed)

        Node* contact_node = nullptr;
        double min_dist_sq = h * h;

        // Find nearest tool node
        tool_grid.query_neighbors(m_data.position[i], [&](int node_idx) {
            double d2 = (m_data.position[i] - tool_mesh.nodes[node_idx].position).squaredNorm();
            if (d2 < min_dist_sq) {
                min_dist_sq = d2;
                contact_node = &tool_mesh.nodes[node_idx];
            }
        });

        if (contact_node) {
            double dist = std::sqrt(min_dist_sq);
            double pen = h - dist;
            double f_mag = std::min(k_contact * pen, 20.0); // Clamp

            Eigen::Vector3d dir = (m_data.position[i] - contact_node->position).normalized();
            Eigen::Vector3d f_vec = dir * f_mag;

            // Apply to particle
            m_data.acceleration[i] += f_vec / m_data.mass[i];

            // Apply to tool (Atomic)
            #pragma omp atomic
            contact_node->force.x() -= f_vec.x();
            #pragma omp atomic
            contact_node->force.y() -= f_vec.y();
            #pragma omp atomic
            contact_node->force.z() -= f_vec.z();

            // Heat / Wear / Stress Logic
            Eigen::Vector3d rel_vel = m_data.velocity[i] - contact_node->velocity;
            double slip = rel_vel.norm();
            double heat = f_mag * 0.3 * slip;
            
            m_data.temperature[i] += (heat * dt) / (m_data.mass[i] * cp_val);
            
            // Contact conduction
            if (m_data.temperature[i] > contact_node->temperature) {
                double cond = (m_data.temperature[i] - contact_node->temperature) * 500.0 * dt;
                m_data.temperature[i] -= cond;
                #pragma omp atomic
                contact_node->temperature += cond * 0.1;
            }

            // Wear
            double contact_pressure = f_mag / (h * h);
            double T_kelvin = contact_node->temperature + 273.15;
            double wear_rate = A_wear * contact_pressure * slip * std::exp(-B_wear / T_kelvin);
            #pragma omp atomic
            contact_node->accumulated_wear += wear_rate * dt;

            // Plasticity
            double sr = slip / h;
            m_data.strain[i] += sr * dt;
            m_data.stress[i] = m_material_model->calculate_stress(m_data.temperature[i], m_data.strain[i], sr);
            
            if (m_failure_model->check_failure(m_data.stress[i], m_data.temperature[i], m_data.damage[i])) {
                m_data.status[i] = (int)ParticleStatus::CHIP_FLOWING;
            }
        }
    }
}

// -------------------------------------------------------
// GPU KERNEL: INTEGRATION
// -------------------------------------------------------
void SPHWorkpieceModel::integrate(double dt) {
    size_t N = m_data.count;
    if (N == 0) return;

    buffer<Eigen::Vector3d, 1> b_pos(m_data.position.data(), range<1>(N));
    buffer<Eigen::Vector3d, 1> b_vel(m_data.velocity.data(), range<1>(N));
    buffer<Eigen::Vector3d, 1> b_acc(m_data.acceleration.data(), range<1>(N));
    buffer<int, 1> b_stat(m_data.status.data(), range<1>(N));

    m_queue.submit([&](handler& cgh) {
        auto pos = b_pos.get_access<access::mode::read_write>(cgh);
        auto vel = b_vel.get_access<access::mode::read_write>(cgh);
        auto acc = b_acc.get_access<access::mode::read>(cgh);
        auto stat = b_stat.get_access<access::mode::read_write>(cgh);

        cgh.parallel_for(range<1>(N), [=](id<1> idx) {
            int i = idx[0];
            if (stat[i] == 0) return;

            // v += a * dt
            vel[i].x() += acc[i].x() * dt;
            vel[i].y() += acc[i].y() * dt;
            vel[i].z() += acc[i].z() * dt;

            // x += v * dt
            pos[i].x() += vel[i].x() * dt;
            pos[i].y() += vel[i].y() * dt;
            pos[i].z() += vel[i].z() * dt;

            // Bounds check (simple box)
            double dist_sq = pos[i].x()*pos[i].x() + pos[i].y()*pos[i].y() + pos[i].z()*pos[i].z();
            if (dist_sq > 100.0) { // > 10 meters away
                stat[i] = 0; // Inactive
            }
        });
    });
}

// Helper for Visualization
const ParticleData& SPHWorkpieceModel::get_data() const {
    return m_data;
}

// Legacy adapter if needed
std::vector<const SPHParticle*> SPHWorkpieceModel::get_particles_for_visualization() const {
    return {}; 
}