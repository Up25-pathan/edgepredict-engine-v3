#ifndef TURNING_STRATEGY_H
#define TURNING_STRATEGY_H

#include "IPhysicsStrategy.h"
#include "simulation.h"
#include "RotationalFEA.h"
#include "SPHWorkpieceModel.h"
#include "navier_stokes.h"
#include "physics_models.h"
#include <memory>

class TurningStrategy : public IPhysicsStrategy {
public:
    TurningStrategy(Simulation* sim);
    ~TurningStrategy() override;

    void initialize(Mesh& mesh, const TopoDS_Shape& cad_shape, const json& config) override;
    json run_time_step(double dt, int step_num) override;
    json get_final_results() override;
    json get_visualization_data() override;

    // --- NEW: Interface Implementation for G-Code & Adaptive Control ---
    void update_machine_state(const Eigen::Vector3d& pos, double feed, double rpm) override;
    void set_feed_rate(double f) override;
    void set_rotation_speed(double s) override;

private:
    void detect_cutting_edges(Mesh& mesh);

    Simulation* m_parent_sim;
    Mesh* m_mesh;
    json m_config;

    std::unique_ptr<RotationalFEA> m_fea_solver;
    std::unique_ptr<SPHWorkpieceModel> m_workpiece_model;
    std::unique_ptr<NavierStokesSolver> m_fluid_solver;
    std::unique_ptr<ThermalModel> m_thermal_helper;
};

#endif // TURNING_STRATEGY_H