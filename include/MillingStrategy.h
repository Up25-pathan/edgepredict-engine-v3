#ifndef MILLING_STRATEGY_H
#define MILLING_STRATEGY_H

#include "IPhysicsStrategy.h"
#include "RotationalFEA.h"     // The new "muscle"
#include "SPHWorkpieceModel.h" // The new "workpiece"
#include "GeometricAnalyzer.h" // The new "brain"

class MillingStrategy : public IPhysicsStrategy {
public:
    MillingStrategy(Simulation* sim);

    void initialize(Mesh& mesh, const json& config) override;
    json run_time_step(double dt, int step_num) override;
    json get_final_results() override;
    json get_visualization_data() override;

private:
    // This strategy holds all your NEW R&D components
    std::unique_ptr<GeometricAnalyzer> m_analyzer;
    std::unique_ptr<RotationalFEA> m_fea_solver;
    std::unique_ptr<SPHWorkpieceModel> m_workpiece_model;

    Simulation* m_parent_sim;
    Mesh* m_mesh;
    json m_config;
};

#endif // MILLING_STRATEGY_H