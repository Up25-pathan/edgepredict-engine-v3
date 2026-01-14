#include <iostream>
#include <chrono>
#include <filesystem>
#include "simulation.h"

int main(int argc, char* argv[]) {
    // 1. Setup Timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 2. Parse Command Line Arguments
    std::string config_path = "input.json"; // Default
    if (argc > 1) {
        config_path = argv[1];
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "   EdgePredict Engine v3.2 (Industry Ready)       " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Running configuration: " << config_path << std::endl;

    try {
        // 3. Validation Check
        if (!std::filesystem::exists(config_path)) {
            throw std::runtime_error("Configuration file not found: " + config_path);
        }

        // 4. Initialize Simulation with explicit path
        Simulation sim(config_path);
        
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