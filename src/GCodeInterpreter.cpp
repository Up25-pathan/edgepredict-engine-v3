#include "GCodeInterpreter.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>

GCodeInterpreter::GCodeInterpreter() : m_total_duration(0.0) {}

void GCodeInterpreter::load_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[G-Code] Error: Cannot open file " << filepath << std::endl;
        return;
    }

    m_trajectory.clear();
    std::string line;
    
    // Initial State (Machine Zero)
    double cx = 0.0, cy = 0.0, cz = 0.0;
    double cf = 100.0; // Default feed
    double cs = 0.0;   // Default spindle
    double c_time = 0.0;

    // Add start point
    m_trajectory.push_back({0.0, cx, cy, cz, cf, cs});

    while (std::getline(file, line)) {
        parse_line(line, cx, cy, cz, cf, cs, c_time);
    }

    m_total_duration = c_time;
    std::cout << "[G-Code] Loaded " << m_trajectory.size() << " waypoints. Total duration: " << m_total_duration << "s" << std::endl;
}

void GCodeInterpreter::parse_line(const std::string& line, double& cx, double& cy, double& cz, double& cf, double& cs, double& c_time) {
    if (line.empty() || line[0] == ';' || line[0] == '(') return;

    std::stringstream ss(line);
    std::string token;
    
    double nx = cx, ny = cy, nz = cz;
    bool is_move = false;
    bool is_rapid = false;

    while (ss >> token) {
        char code = token[0];
        try {
            double val = std::stod(token.substr(1));
            switch (code) {
                case 'G': 
                    if (val == 0.0) { is_move = true; is_rapid = true; }
                    if (val == 1.0) { is_move = true; is_rapid = false; }
                    break;
                case 'X': nx = val / 1000.0; break; // Convert mm to meters immediately
                case 'Y': ny = val / 1000.0; break;
                case 'Z': nz = val / 1000.0; break;
                case 'F': cf = val; break;
                case 'S': cs = val; break;
            }
        } catch (...) {}
    }

    if (is_move || (nx != cx || ny != cy || nz != cz)) {
        double dist = std::sqrt(std::pow(nx-cx, 2) + std::pow(ny-cy, 2) + std::pow(nz-cz, 2));
        double feed_m_s = is_rapid ? (5000.0/60.0/1000.0) : (cf/60.0/1000.0); // Assume Rapid = 5000mm/min
        
        if (feed_m_s < 1e-6) feed_m_s = 0.001; 
        
        double dt = dist / feed_m_s;
        c_time += dt;
        
        m_trajectory.push_back({c_time, nx, ny, nz, cf, cs});
        
        cx = nx; cy = ny; cz = nz;
    }
}

MachineState GCodeInterpreter::get_state_at_time(double time_s) {
    MachineState state;
    state.active = false;

    if (m_trajectory.empty()) return state;
    if (time_s >= m_total_duration) {
        auto& last = m_trajectory.back();
        state.target_pos = Eigen::Vector3d(last.x, last.y, last.z);
        state.feed_rate_mm_min = 0;
        state.spindle_speed_rpm = 0;
        return state;
    }

    // Binary search for time interval
    auto it = std::lower_bound(m_trajectory.begin(), m_trajectory.end(), time_s, 
        [](const TrajectoryPoint& p, double val) { return p.time_s < val; });

    if (it == m_trajectory.begin()) it++;
    
    const auto& p1 = *(it - 1);
    const auto& p2 = *it;

    double ratio = (time_s - p1.time_s) / (p2.time_s - p1.time_s);
    if (ratio < 0) ratio = 0;
    if (ratio > 1) ratio = 1;

    // Linear Interpolation
    double x = p1.x + ratio * (p2.x - p1.x);
    double y = p1.y + ratio * (p2.y - p1.y);
    double z = p1.z + ratio * (p2.z - p1.z);

    state.target_pos = Eigen::Vector3d(x, y, z);
    state.feed_rate_mm_min = p2.f; // Use target feed
    state.spindle_speed_rpm = p2.s;
    state.active = true;

    return state;
}