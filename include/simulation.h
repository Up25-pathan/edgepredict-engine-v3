#ifndef SIMULATION_H
#define SIMULATION_H

#include "json.hpp"
#include <string>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <opencascade/TopoDS_Shape.hxx>

// --- NEW HEADERS ---
#include "OptimizationManager.h"
#include "EnergyMonitor.h"

using json = nlohmann::json;

// Forward declarations
class IPhysicsStrategy;

struct Node {
    Eigen::Vector3d position;
    Eigen::Vector3d original_position;
    Eigen::Vector3d velocity;
    Eigen::Vector3d acceleration;
    Eigen::Vector3d force;
    double mass;
    double temperature;
    double stress;
    double accumulated_wear; 
    bool is_active_contact; 
    int status; 
};

enum class NodeStatus {
    INACTIVE = 0,
    OK = 1,
    FAILED = 2
};

struct Mesh {
    std::vector<Node> nodes;
    std::vector<int> triangle_indices;
};

class Simulation {
public:
    Simulation();
    ~Simulation();

    void run();
    void load_config(const std::string& path);
    void load_geometry();
    
    // --- Getters for Strategy Access ---
    OptimizationManager* get_optimizer() { return m_optimizer.get(); }
    EnergyMonitor* get_energy_monitor() { return m_energy_monitor.get(); }

    // Public Data
    json config_data;
    Mesh mesh;
    TopoDS_Shape cad_shape;
    json time_series_output;
    json cfd_visualization_data;

private:
    void load_geometry_stl(const std::string& path);
    void load_geometry_cad(const std::string& path, const std::string& format);
    void create_mesh_from_cad(const TopoDS_Shape& shape);
    std::string get_file_ext(const std::string& s);
    
    void write_progress(int step, int total_steps, const json& current_metrics);
    void calculate_tool_life_prediction(const json& final_results);
    void write_output(const json& final_results);

    IPhysicsStrategy* m_strategy;
    
    // --- MANAGERS ---
    std::unique_ptr<OptimizationManager> m_optimizer;
    std::unique_ptr<EnergyMonitor> m_energy_monitor;
    
    double predicted_tool_life_hours;
    double wear_threshold_mm;
};

#endif // SIMULATION_H