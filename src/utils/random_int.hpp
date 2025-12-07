#pragma once
#include <random>

namespace Utils {

class Random {
public:
    // Returns a random integer in [min, max]
    static int int_in_range(int min, int max) {
        static thread_local std::mt19937 gen(seed());
        std::uniform_int_distribution<int> dist(min, max);
        return dist(gen);
    }

    // If you want deterministic behavior, replace rd() with a fixed constant
    static unsigned seed() {
        static std::random_device rd;
        return rd();
    }
};

} // namespace Utils
