#include "MillingStrategy.h"
#include "simulation.h"
#include <iostream>
#include <stdexcept>
#include <vector>

// ═══════════════════════════════════════════════════════════
// MILLING STRATEGY (The "New R&D Machine")
// ═══════════════════════════════════════════════════════════

MillingStrategy::MillingStrategy(Simulation* sim) 
    : m_parent_sim(sim), m_mesh(nullptr) 
{
    if (!m_parent_sim) {
        throw std::runtime_error("MillingStrategy: Simulation parent pointer is null.");
    }
    std::cout << "[Strategy] MillingStrategy (R&D Core) created." << std::endl;
}

MillingStrategy::~MillingStrategy() {}

void MillingStrategy::initialize(Mesh& mesh, const TopoDS_Shape& cad_shape, const json& config) {
    std::cout << "[MillingStrategy] Initializing..." << std::endl;
    m_mesh = &mesh;
    m_config = config;

    if (cad_shape.IsNull()) {
        throw std::runtime_error("MillingStrategy: CAD shape is null. Milling/Drilling requires a STEP/IGES file.");
    }

    // 1. --- "THE BRAIN": ANALYZE GEOMETRY ---
    // The engine "sees" the tool design for the first time.
    std::cout << "[MillingStrategy] Loading GeometricAnalyzer..." << std::endl;
    m_analyzer = std::make_unique<GeometricAnalyzer>(cad_shape);
    ToolGeometryData geo_data = m_analyzer->analyze();
    
    // 2. --- "THE MUSCLE": INITIALIZE TOOL PHYSICS ---
    // We create the rotational solver for the tool mesh.
    std::cout << "[MillingStrategy] Loading RotationalFEA solver..." << std::endl;
    m_fea_solver = std::make_unique<RotationalFEA>(m_config, m_mesh);
    
    // Now we connect the "Brain" to the "Muscle".
    // We use the geometric data to tell the FEA solver which nodes are
    // the actual cutting edges.
    
    // TODO: This is a placeholder. A full implementation would map the
    // OCC `TopoDS_Edge` geometry to the visual `Mesh` nodes.
    // For now, we'll use the 'is_active_contact' flag from the FEA solver's
    // own internal (but less accurate) auto-detector.
    std::cout << "[MillingStrategy] (TODO: Mapping geometric edges to mesh...)" << std::endl;


    // 3. --- "THE WORKPIECE": INITIALIZE SPH MODEL ---
    std::cout << "[MillingStrategy] Loading SPHWorkpieceModel..." << std::endl;
    m_workpiece_model = std::make_unique<SPHWorkpieceModel>(m_config);

    // We need a new "workpiece_setup" block in the input.json to define
    // the block of material we are cutting.
    if (!m_config.contains("workpiece_setup")) {
        throw std::runtime_error("MillingStrategy: input.json is missing 'workpiece_setup' block.");
    }
    
    auto& wp_config = m_config["workpiece_setup"];
    Eigen::Vector3d wp_min(wp_config["min_corner"][0].get<double>(), 
                           wp_config["min_corner"][1].get<double>(), 
                           wp_config["min_corner"][2].get<double>());
    Eigen::Vector3d wp_max(wp_config["max_corner"][0].get<double>(),
                           wp_config["max_corner"][1].get<double>(),
                           wp_config["max_corner"][2].get<double>());
    
    m_workpiece_model->initialize_workpiece(wp_min, wp_max);
    
    std::cout << "[MillingStrategy] R&D Engine Initialized and Ready." << std::endl;
}

// This is the main loop for the R&D engine.
// It coordinates all the new modules in the correct order.
json MillingStrategy::run_time_step(double dt, int /*step_num*/) {
    if (!m_fea_solver || !m_workpiece_model) {
        throw std::runtime_error("MillingStrategy: Not initialized.");
    }

    // --- R&D PHYSICS LOOP ---

    // 1. "MUSCLE": Move the tool (rotation + translation)
    m_fea_solver->update_tool_motion(dt);

    // 2. "WORKPIECE": Calculates tool-particle collisions.
    //    This finds contacts, calculates forces, generates damage, 
    //    and transfers heat back to the tool nodes.
    m_workpiece_model->interact_with_tool(*m_mesh, dt);

    // 3. "MUSCLE": Update the tool's internal state.
    //    Based on the contacts and heat from step 2, it calculates its
    //    own stress, wear, and failure.
    json metrics = m_fea_solver->run_physics_step(dt);

    // 4. "WORKPIECE": Update all particle positions.
    //    Now that all forces for this step have been applied,
    //    move the SPH particles.
    m_workpiece_model->update_step(dt);

    // 5. Return the metrics (temp, stress, etc.) to the main simulation
    return metrics;
}

json MillingStrategy::get_final_results() {
    // Get final metrics from the tool solver
    const auto& last_metrics = m_parent_sim->time_series_output.back();
    return {
        {"total_accumulated_wear_m", last_metrics.value("total_accumulated_wear_m", 0.0)},
        {"total_time_s", last_metrics.value("time_s", 0.0)},
        {"final_fractured_nodes", last_metrics.value("cumulative_fractured_nodes", 0)}
    };
}

json MillingStrategy::get_visualization_data() {
    // Get particle data from the SPH model for rendering
    json particle_array = json::array();
    auto particles = m_workpiece_model->get_particles_for_visualization();

    for (const SPHParticle* p : particles) {
        particle_array.push_back({
            {"position", {p->position.x(), p->position.y(), p->position.z()}},
            {"velocity", {p->velocity.x(), p->velocity.y(), p->velocity.z()}},
            {"temperature", p->temperature},
            {"pressure", p->pressure},
            {"status", (p->status == ParticleStatus::CHIP_FLOWING ? "chip" : "workpiece")}
        });
    }
    
    // We create a JSON array of frames. For simplicity, we just add one frame here.
    // The main sim loop could be modified to call this every N steps.
    json frame;
    frame["step"] = m_parent_sim->time_series_output.size(); // Current step
    frame["particles"] = std::move(particle_array);
    
    return json::array({frame});
}