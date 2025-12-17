#ifndef PHYSICS_DATA_H
#define PHYSICS_DATA_H

#include <vector>
#include <Eigen/Dense>

// Enum for particle state
enum class ParticleStatus {
    INACTIVE = 0,
    WORKPIECE_SOLID = 1,
    CHIP_FLOWING = 2
};

// Data-Oriented Design: Struct of Arrays (SoA)
// This layout is cache-friendly for CPU and ready for GPU buffers
struct ParticleData {
    // Kinematics
    std::vector<Eigen::Vector3d> position;
    std::vector<Eigen::Vector3d> velocity;
    std::vector<Eigen::Vector3d> acceleration;
    
    // Properties
    std::vector<double> mass;
    std::vector<double> density;
    std::vector<double> pressure;
    std::vector<double> temperature;
    
    // Material State
    std::vector<double> stress;
    std::vector<double> strain;
    std::vector<double> damage;
    std::vector<double> plastic_strain;
    
    // Status (0=Inactive)
    std::vector<int> status;

    size_t count = 0;

    void reserve(size_t n) {
        position.reserve(n); velocity.reserve(n); acceleration.reserve(n);
        mass.reserve(n); density.reserve(n); pressure.reserve(n); temperature.reserve(n);
        stress.reserve(n); strain.reserve(n); damage.reserve(n); plastic_strain.reserve(n);
        status.reserve(n);
    }

    void push_back(const Eigen::Vector3d& pos, double m, double rho, double temp) {
        position.push_back(pos);
        velocity.push_back(Eigen::Vector3d::Zero());
        acceleration.push_back(Eigen::Vector3d::Zero());
        mass.push_back(m);
        density.push_back(rho);
        pressure.push_back(0.0);
        temperature.push_back(temp);
        stress.push_back(0.0);
        strain.push_back(0.0);
        damage.push_back(0.0);
        plastic_strain.push_back(0.0);
        status.push_back((int)ParticleStatus::WORKPIECE_SOLID);
        count++;
    }

    void clear() {
        count = 0;
        position.clear(); velocity.clear(); acceleration.clear();
        mass.clear(); density.clear(); pressure.clear(); temperature.clear();
        stress.clear(); strain.clear(); damage.clear(); plastic_strain.clear();
        status.clear();
    }
};

#endif // PHYSICS_DATA_H