#ifndef CEGO_UTILITIES_H
#define CEGO_UTILITIES_H

#include <set>
#include <iostream>
#include <fstream>

namespace CEGO{

/// Get N unique indices
template< class RNG >
std::set<std::size_t> get_N_unique(std::size_t Imin, std::size_t Imax, std::size_t N, RNG gen) {

    // Short circuit if you want as many indices as the length
    if (N == Imax-Imin) {
        std::set<std::size_t> out;
        for (auto i = Imin; i <= Imax; ++i) { out.insert(i); }
        return out;
    }
    // Otherwise, find N unique indices, each  >= 0 and <= Nindices
    std::uniform_int_distribution<> dis(Imin, Imax);
    std::set<std::size_t> indices;
    for (auto i = 0; i < N; ++i) {
        while (true) {
            // Trial index
            auto j = dis(gen);
            if (indices.find(j) != indices.end()) {
                // It's already in the list of indices to keep; keep trying
                continue;
            }
            else {
                // Not being used yet; keep it
                indices.insert(j);
                break;
            }
        }
    }
    return indices;
}

int get_env_int(const std::string& var, int def) {
    try {
        char* s = std::getenv(var.c_str());
        if (s == nullptr) {
            return def;
        }
        if (strlen(s) == 0) {
            return def;
        }
        return std::stoi(s, nullptr);
    }
    catch (...) {
        return def;
    }
}

// See https://docs.travis-ci.com/user/environment-variables/#default-environment-variables
bool is_CI() {
    try {
        char* s = std::getenv("CI");
        if (s == nullptr) {
            return false;
        }
        if (strlen(s) == 0) {
            return false;
        }
        return true;
    }
    catch (...) {
        return false;
    }
}

// Adapted from https://stackoverflow.com/a/34490647

static void seed_function() {} // we only use the address of this function
auto get_Mersenne_twister() {
    // Variables used in seeding
    static long long seed_counter = 0;
    int var;
    void* x = std::malloc(sizeof(int));
    free(x);

    std::seed_seq seed{
        // Time
        static_cast<long long>(std::chrono::high_resolution_clock::now()
                                   .time_since_epoch()
                                   .count()),
        // ASLR
        static_cast<long long>(reinterpret_cast<intptr_t>(&seed_counter)),
        static_cast<long long>(reinterpret_cast<intptr_t>(&var)),
        static_cast<long long>(reinterpret_cast<intptr_t>(x)),
        static_cast<long long>(reinterpret_cast<intptr_t>(&seed_function)),
        static_cast<long long>(reinterpret_cast<intptr_t>(&_Exit)),
        // Thread id
        static_cast<long long>(
            std::hash<std::thread::id>()(std::this_thread::get_id())),
        // counter
        ++seed_counter };
    std::mt19937 eng(seed);
    return eng;
}

// Adapted from https://stackoverflow.com/a/116220/1360263
inline std::string get_file_contents(const std::string& filename) {
    using std::ios;
    std::ifstream ifs(filename.c_str(), ios::in | ios::binary);
    if (!ifs) {
        throw std::invalid_argument("filename cannot be opened: " + filename);
    }
    return static_cast<std::stringstream const&>(std::stringstream() << ifs.rdbuf()).str();
}

} /* namespace CEGO */
#endif
