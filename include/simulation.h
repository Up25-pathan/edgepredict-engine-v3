#ifndef SIMULATION_H
#define SIMULATION_H

#include <vector>
#include <string>
#include <memory>
#include "json.hpp"
#include <Eigen/Dense>

// --- OpenCASCADE Headers (with full include path) ---
#include <opencascade/STEPControl_Reader.hxx>
#include <opencascade/IGESControl_Reader.hxx>
#include <opencascade/TopoDS_Shape.hxx>
#include <opencascade/TopExp_Explorer.hxx>
#include <opencascade/TopoDS.hxx>
#include <opencascade/TopoDS_Face.hxx>
#include <opencascade/BRep_Tool.hxx>
#include <opencascade/BRepMesh_IncrementalMesh.hxx>
#include <opencascade/Poly_Triangulation.hxx>
#include <opencascade/TColgp_Array1OfPnt.hxx>
#include <opencascade/Poly_Array1OfTriangle.hxx>
#include <opencascade/gp_Pnt.hxx>
#include <map> // For node de-duplication
// --- End OpenCASCADE Headers ---

// Include all physics modules
#include "physics_models.h"
#include "navier_stokes.h"
#include "particle_system.h"

using json = nlohmann::json;

// ═══════════════════════════════════════════════════════════
// NODE & MESH STRUCTURES
// ═══════════════════════════════════════════════════════════

enum class NodeStatus {
    OK,
    FAILED,
    FRACTURED
};

struct Node {
    Eigen::Vector3d position;
    double temperature = 20.0;
    double stress = 0.0;
    double strain = 0.0;
    double plastic_strain = 0.0;
    double accumulated_wear = 0.0;
    double accumulated_damage = 0.0;
    NodeStatus status = NodeStatus::OK;

    // NEW: The engine will mark this true if the node is part of the cutting tip
    bool is_active_contact = false;
};

struct Mesh {
    std::vector<Node> nodes;
    std::vector<unsigned int> triangle_indices;
};

// Forward declaration
class Simulation;

// ═══════════════════════════════════════════════════════════
// FEA SOLVER (Enhanced with CFD Coupling)
// ═══════════════════════════════════════════════════════════

class FEASolver {
public:
    FEASolver(const json& config, Simulation* sim);
    ~FEASolver();
    
    void solve(Mesh& mesh, int num_steps);

    // NEW: The function that gives the engine "eyes"
    void auto_detect_cutting_edge(Mesh& mesh);

private:
    // Core FEA loop
    json solve_time_step(int step_num);
    
    // Node-level updates
    void update_node_mechanics(Node& node, double dt, bool is_in_contact_zone);
    void update_node_thermal(Node& node, double dt, double chip_heat_flux, bool is_in_contact_zone);
    bool check_node_failure(Node& node);
    
    // Configuration
    json config_data;
    double dt;
    Mesh* active_mesh;
    Simulation* parent_sim;
    
    // Physics models
    std::unique_ptr<JohnsonCookModel> material_model;
    std::unique_ptr<UsuiWearModel> wear_model;
    std::unique_ptr<ThermalModel> thermal_model;
    std::unique_ptr<FailureCriterion> failure_model;
    std::unique_ptr<ChipFormationModel> chip_model;
    
    // CFD integration
    bool cfd_enabled;
    std::unique_ptr<NavierStokesSolver> fluid_solver;
    std::unique_ptr<ParticleSystem> particle_system;
    
    // Simulation parameters
    double strain_rate;
    double strain_increment;
    double sliding_velocity;
    double ambient_temperature;

    // REMOVED: Manual contact zone boundaries
    // Eigen::Vector3d contact_zone_min;
    // Eigen::Vector3d contact_zone_max;

    // Statistics
    int total_fractured_nodes;
};

// ═══════════════════════════════════════════════════════════
// MAIN SIMULATION COORDINATOR
// ═══════════════════════════════════════════════════════════

class Simulation {
public:
    Simulation();
    void run();

    // Writes progress to stdout/logs for the worker to pick up
    void write_progress(int step, int total_steps, const json& current_metrics);

    // Output data structures
    json time_series_output;
    json cfd_visualization_data;

private:
    void load_config(const std::string& path);
    
    void load_geometry();
    void load_geometry_stl(const std::string& path);
    void load_geometry_cad(const std::string& path, const std::string& format);

    void write_output();
    void calculate_tool_life_prediction();
    
    Mesh mesh;
    json config_data;
    
    // Tool life prediction
    double predicted_tool_life_hours;
    double wear_threshold_mm;
};

#endif // SIMULATION_H
