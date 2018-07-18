#ifndef CEGO_UTILITIES_H
#define CEGO_UTILITIES_H

#include <set>

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

} /* namespace CEGO */
#endif
