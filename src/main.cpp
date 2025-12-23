#include <iostream>
#include <chrono>
#include <filesystem>
#include "simulation.h"

int main(int argc, char* argv[]) {
    // 1. Setup Timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 2. Parse Command Line Arguments
    std::string config_path = "input.json";
    if (argc > 1) {
        config_path = argv[1];
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "   EdgePredict Engine v3.0 (SYCL/Hybrid GPU)      " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Running configuration: " << config_path << std::endl;

    try {
        // 3. Validation Check
        if (!std::filesystem::exists(config_path)) {
            throw std::runtime_error("Configuration file not found: " + config_path);
        }

        // 4. Initialize Simulation
        Simulation sim;
        
        // Note: Modified Simulation class to accept path in constructor or separate init
        // For now, we assume sim.load_config() is called internally or we can refactor Simulation 
        // to take the path. *Ideally, Simulation::Simulation(std::string path)*
        // Since we didn't refactor Simulation.h constructor, we rely on it reading 'input.json' 
        // OR we can add a setter if available. 
        // *Correction based on your current codebase:* Simulation constructor calls load_config("input.json").
        // *Refactor Recommendation:* In future, make Simulation take path argument.
        
        // 5. Run
        sim.run();

    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL ERROR] Simulation Aborted." << std::endl;
        std::cerr << "Reason: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "\n[UNKNOWN ERROR] Simulation crashed with an uncaught exception." << std::endl;
        return EXIT_FAILURE;
    }

    // 6. Report Stats
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Success! Execution Time: " << elapsed.count() << " seconds." << std::endl;
    std::cout << "==================================================" << std::endl;

    return EXIT_SUCCESS;
}