#ifndef GCODE_INTERPRETER_H
#define GCODE_INTERPRETER_H

#include <string>
#include <vector>
#include <Eigen/Dense>

struct MachineState {
    Eigen::Vector3d target_pos; // Absolute position (mm -> m scaled)
    double feed_rate_mm_min;    
    double spindle_speed_rpm;
    bool active;                // Is the machine currently moving?
};

class GCodeInterpreter {
public:
    GCodeInterpreter();
    void load_file(const std::string& filepath);
    
    // Returns interpolated state for specific simulation time
    MachineState get_state_at_time(double time_s);

private:
    struct TrajectoryPoint {
        double time_s;
        double x, y, z;
        double f, s;
    };

    std::vector<TrajectoryPoint> m_trajectory;
    double m_total_duration;
    
    void parse_line(const std::string& line, double& current_x, double& current_y, double& current_z, double& current_f, double& current_s, double& current_time);
};

#endif