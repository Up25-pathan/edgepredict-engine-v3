#ifndef SIMULATION_H
#define SIMULATION_H

#include <vector>
#include <string>
#include <memory>
#include "json.hpp"
#include <Eigen/Dense>

// --- OpenCASCADE Headers ---
#include <opencascade/TopoDS_Shape.hxx>
#include <opencascade/TopoDS.hxx>
#include <opencascade/Poly_Triangulation.hxx>
// (Include other OCC headers as needed from your original file)

using json = nlohmann::json;

// ═══════════════════════════════════════════════════════════
// NODE & MESH STRUCTURES (UPDATED FOR EXPLICIT DYNAMICS)
// ═══════════════════════════════════════════════════════════

enum class NodeStatus {
    OK,
    FAILED,
    FRACTURED
};

struct Node {
    // --- Geometric State ---
    Eigen::Vector3d position;          // Current deformed position
    Eigen::Vector3d original_position; // Reference state (t=0)

    // --- Explicit Dynamics State (NEW) ---
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
    Eigen::Vector3d force = Eigen::Vector3d::Zero(); // Accumulator for internal + external forces
    double mass = 1e-9; // Lumped mass (prevent div by zero default)

    // --- Material & Thermal State ---
    double temperature = 20.0;
    double stress = 0.0;         // Von Mises Stress
    double strain = 0.0;
    double plastic_strain = 0.0;
    double accumulated_wear = 0.0;
    double accumulated_damage = 0.0;
    
    NodeStatus status = NodeStatus::OK;
    
    // Helper for collision detection optimization
    bool is_active_contact = false;
};

struct Mesh {
    std::vector<Node> nodes;
    std::vector<unsigned int> triangle_indices;
};

// ═══════════════════════════════════════════════════════════
// FORWARD DECLARATIONS
// ═══════════════════════════════════════════════════════════

class Simulation;
// Note: FEASolver definition remains here if you are keeping legacy support,
// but RotationalFEA will primarily use the Mesh struct defined above.

// ═══════════════════════════════════════════════════════════
// MAIN SIMULATION CLASS
// ═══════════════════════════════════════════════════════════

class Simulation {
public:
    Simulation();
    ~Simulation();
    void run();

    // Writes progress to stdout/logs for the worker to pick up
    void write_progress(int step, int total_steps, const json& current_metrics);

    // Shared Output Data
    json time_series_output;
    json cfd_visualization_data;
    json config_data; // Made public so strategies can access it

private:
    void load_config(const std::string& path);
    void load_geometry();
    void load_geometry_stl(const std::string& path);
    void load_geometry_cad(const std::string& path, const std::string& format);
    void create_mesh_from_cad(const TopoDS_Shape& shape);
    
    std::string get_file_ext(const std::string& s);

    void write_output(const json& final_results);
    void calculate_tool_life_prediction(const json& final_results);
    
    Mesh mesh;
    TopoDS_Shape cad_shape;
    
    // Strategy Pattern pointer (Polyrmorphism)
    // Note: You'll need to include IPhysicsStrategy.h in your cpp or forward declare here
    class IPhysicsStrategy* m_strategy; 

    double predicted_tool_life_hours;
    double wear_threshold_mm;
};

#endif // SIMULATION_H