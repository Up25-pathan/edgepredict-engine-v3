#ifndef TURNING_STRATEGY_H
#define TURNING_STRATEGY_H

#include "IPhysicsStrategy.h"
#include "simulation.h"       // We will refactor this to be the *old* FEASolver
#include "navier_stokes.h"  //
#include "particle_system.h"//

// We will rename your original FEASolver to LinearFEASolver
// to avoid confusion.
using LinearFEASolver = FEASolver; 

class TurningStrategy : public IPhysicsStrategy {
public:
    TurningStrategy(Simulation* sim); // Pass the main sim object

    void initialize(Mesh& mesh, const json& config) override;
    json run_time_step(double dt, int step_num) override;
    json get_final_results() override;
    json get_visualization_data() override;

private:
    // This strategy holds all your OLD engine components
    std::unique_ptr<LinearFEASolver> m_fea_solver;
    std::unique_ptr<NavierStokesSolver> m_fluid_solver;
    std::unique_ptr<ParticleSystem> m_particle_system;
    
    Simulation* m_parent_sim; // To write progress
    Mesh* m_mesh;
    json m_config;
};

#endif // TURNING_STRATEGY_H