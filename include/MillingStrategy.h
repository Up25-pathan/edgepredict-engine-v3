#ifndef MILLING_STRATEGY_H
#define MILLING_STRATEGY_H

#include "IPhysicsStrategy.h"
#include "simulation.h"
#include "RotationalFEA.h"
#include "SPHWorkpieceModel.h"
#include "navier_stokes.h"  // CFD Support
#include "physics_models.h" // For ThermalModel
#include <memory>

class MillingStrategy : public IPhysicsStrategy {
public:
    MillingStrategy(Simulation* sim);
    ~MillingStrategy() override;

    void initialize(Mesh& mesh, const TopoDS_Shape& cad_shape, const json& config) override;
    json run_time_step(double dt, int step_num) override;
    json get_final_results() override;
    json get_visualization_data() override;

private:
    void detect_cutting_edges(Mesh& mesh);

    Simulation* m_parent_sim;
    Mesh* m_mesh;
    json m_config;

    // --- Physics Modules ---
    std::unique_ptr<RotationalFEA> m_fea_solver;
    std::unique_ptr<SPHWorkpieceModel> m_workpiece_model;
    
    // --- CFD Modules (Phase 4) ---
    std::unique_ptr<NavierStokesSolver> m_fluid_solver;
    std::unique_ptr<ThermalModel> m_thermal_helper; // Helper for heat transfer calcs
};

#endif // MILLING_STRATEGY_H