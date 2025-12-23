#ifndef PHYSICS_DATA_H
#define PHYSICS_DATA_H

#include <vector>
#include <Eigen/Dense>

enum class ParticleStatus {
    INACTIVE = 0,
    WORKPIECE_SOLID = 1,
    CHIP_ACTIVE = 2
};

// Structure of Arrays (SoA) Layout for GPU Performance
struct ParticleData {
    // Capacity tracking
    size_t count = 0;
    size_t capacity = 0;

    // Position
    std::vector<double> pos_x, pos_y, pos_z;
    // Velocity
    std::vector<double> vel_x, vel_y, vel_z;
    // Acceleration
    std::vector<double> acc_x, acc_y, acc_z;

    // Properties
    std::vector<double> mass;
    std::vector<double> density;
    std::vector<double> pressure;
    std::vector<double> temperature;
    std::vector<int> status;

    void reserve(size_t n) {
        capacity = n;
        pos_x.reserve(n); pos_y.reserve(n); pos_z.reserve(n);
        vel_x.reserve(n); vel_y.reserve(n); vel_z.reserve(n);
        acc_x.reserve(n); acc_y.reserve(n); acc_z.reserve(n);
        mass.reserve(n); density.reserve(n); pressure.reserve(n);
        temperature.reserve(n); status.reserve(n);
    }

    void clear() {
        count = 0;
        pos_x.clear(); pos_y.clear(); pos_z.clear();
        vel_x.clear(); vel_y.clear(); vel_z.clear();
        acc_x.clear(); acc_y.clear(); acc_z.clear();
        mass.clear(); density.clear(); pressure.clear();
        temperature.clear(); status.clear();
    }

    void push_back(const Eigen::Vector3d& p, double m, double rho, double temp) {
        if (count >= capacity) return; // Prevent overflow
        count++;
        pos_x.push_back(p.x()); pos_y.push_back(p.y()); pos_z.push_back(p.z());
        vel_x.push_back(0.0); vel_y.push_back(0.0); vel_z.push_back(0.0);
        acc_x.push_back(0.0); acc_y.push_back(0.0); acc_z.push_back(0.0);
        mass.push_back(m);
        density.push_back(rho);
        pressure.push_back(0.0);
        temperature.push_back(temp);
        status.push_back((int)ParticleStatus::WORKPIECE_SOLID);
    }
};

#endif // PHYSICS_DATA_H