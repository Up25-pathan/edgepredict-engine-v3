/**
 * @file MachiningStrategyFactory.cpp
 * @brief Factory implementation for creating machining strategies
 */

#include "IMachiningStrategy.h"
#include "TurningStrategy.h"
#include "MillingStrategy.h"
#include "DrillingStrategy.h"
#include <iostream>

namespace edgepredict {

std::unique_ptr<IMachiningStrategy> MachiningStrategyFactory::create(MachiningType type) {
    switch (type) {
        case MachiningType::TURNING:
            std::cout << "[Factory] Creating TurningStrategy" << std::endl;
            return std::make_unique<TurningStrategy>();
            
        case MachiningType::MILLING:
            std::cout << "[Factory] Creating MillingStrategy" << std::endl;
            return std::make_unique<MillingStrategy>();
            
        case MachiningType::DRILLING:
            std::cout << "[Factory] Creating DrillingStrategy" << std::endl;
            return std::make_unique<DrillingStrategy>();
            
        case MachiningType::GRINDING:
            // For now, use milling as base (could create GrindingStrategy later)
            std::cout << "[Factory] Grinding not yet implemented, using MillingStrategy" << std::endl;
            return std::make_unique<MillingStrategy>();
            
        default:
            std::cout << "[Factory] Unknown type, defaulting to MillingStrategy" << std::endl;
            return std::make_unique<MillingStrategy>();
    }
}

std::unique_ptr<IMachiningStrategy> MachiningStrategyFactory::createFromConfig(const Config& config) {
    auto strategy = create(config.getMachiningType());
    if (strategy) {
        strategy->initialize(config);
    }
    return strategy;
}

} // namespace edgepredict
