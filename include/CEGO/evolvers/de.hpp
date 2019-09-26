/*

This header implements the method of differential evolution of 
Storn & Price.  This stochastic optimization method merges individuals with
a number of different cross-over schemes based on differences between
individuals in the population

See for instance:
Storn, R. & Price, K., 1997, "Differential Evolution -- A Simple and Efficient Heuristic for 
Global Optimization over Continuous Spaces", J. Glob. Opt., v. 11, 341-359

*/

#ifndef CEGO_DE_H
#define CEGO_DE_H

#include "CEGO/datatypes.hpp"
#include "CEGO/CEGO.hpp"
#include <memory>

#include "nlohmann/json.hpp"

namespace CEGO{

    // *****************************************************************************************************
    // *****************************************************************************************************
    //                                             MUTANT GENERATORS
    // *****************************************************************************************************
    // *****************************************************************************************************

    /**
    @brief Generate one mutant individual given three individuals.  This is the one-difference-vector mutant generator of Storn and Price.
    */
    template<typename T>
    EArray<T> gen_1diff(const EArray<T>&c1, const EArray<T> &c2, const EArray<T>&c3, T F) {
        assert(c1.size() == c2.size());
        assert(c2.size() == c3.size()); 
        return c1 + F*(c2 - c3);
    }

    // *****************************************************************************************************
    // *****************************************************************************************************
    //                                             CROSSOVER
    // *****************************************************************************************************
    // *****************************************************************************************************



template<class RNG, typename TYPE>
pIndividual DE1bin(const NumericalIndividual<TYPE>&i0, 
                   const NumericalIndividual<TYPE>&i1,
                   const NumericalIndividual<TYPE>&i2,
                   const NumericalIndividual<TYPE>&i3,
                   RNG &gen, const IndividualFactory<TYPE> &factory, double F = 0.5, double CR = 0.9) 
{   
    // Copy of the coefficients for the base individual
    auto c0 = i0.get_coefficients();
    // The mutant obtained from original individual i1 and two others (i2 and i3) forming 
    // the difference
    const EArray<TYPE> cm = gen_1diff<TYPE>(i1.get_coefficients(),
                                      i2.get_coefficients(),
                                      i3.get_coefficients(), F);
    auto Ncoeff = c0.size();
    // R is the index that will definitely be used from the perturbed vector
    std::size_t R = std::uniform_int_distribution<std::size_t>(0, Ncoeff-1)(gen);
    std::uniform_real_distribution<> unireal(0, 1);
    for (auto i = 0; i < Ncoeff; ++i) {
        if (i == R || unireal(gen) < CR) {
            // Use the value from the mutant
            c0[i] = cm[i];
        }
        else { // Otherwise, use the old parameter from i0 
        }
    }
    return pIndividual(factory(std::move(c0)));
};

template<typename T, class RNG, typename TYPE>
pIndividual DE1exp(const pIndividual &i0, const pIndividual &i1, const pIndividual &i2, const pIndividual &i3,
                   RNG &gen, const IndividualFactory<TYPE> &factory, double F = 1.0, double CR = 0.9)
{
    // Copy of the coefficients
    auto c0 = static_cast<T*>(i0.get())->get_coefficients();
    // The mutant obtained from original individual i1 and two others (i2 and i3), forming 
    // the difference
    const auto cm = gen_1diff(static_cast<T*>(i1.get())->get_coefficients(),
                              static_cast<T*>(i2.get())->get_coefficients(),
                              static_cast<T*>(i3.get())->get_coefficients(), F);
    auto Ncoeff = c0.size();
    std::uniform_real_distribution<> unireal(0, 1);
    // n is the first index in the vector where a change is to be tried
    auto n = std::uniform_int_distribution<std::size_t>(0, Ncoeff - 1)(gen);
    for (auto i = 0; (unireal(gen) < CR) && (i < Ncoeff); ++i) {
        c0[n] = cm[n]; // Use the value from the mutant
        n = (n + 1) % (Ncoeff - 1); // Bump the index, and if you go above Ncoeff, go back to zero
    }
    return pIndividual(factory(std::move(c0)));
};

/// Flags for differential evolution.  If changed, make sure to modify the to/from_json functions below
struct DifferentialEvolutionFlags {
    std::size_t Nelite = 0; ///< The number of protected individuals in the elite
    double Fmin = 0.9, ///< The minimum value of F for DE 
           Fmax = 0.9; ///< The maximum value of F for DE
    double CR = 0.9;   ///< The crossover rate
};

inline void to_json(nlohmann::json& j, const DifferentialEvolutionFlags& f) {
    j = nlohmann::json{ { "Nelite", f.Nelite },{ "Fmin", f.Fmin },{ "Fmax", f.Fmax },{"CR",f.CR} };
}

inline void from_json(const nlohmann::json& j, DifferentialEvolutionFlags& f) {
    f.Nelite = j.at("Nelite").get<int>(); 
    f.Fmin = j.at("Fmin").get<double>();
    f.Fmax = j.at("Fmax").get<double>(); 
    f.CR = j.at("CR").get<double>();
}

/** Do differential evolution to generate a given population of individuals.  The individuals are not yet evaluated, 
  so that evaluation can be carried out in parallel
 */
template<typename T>
Population differential_evolution(const Population &this_layer, 
                                  const Population &older_layer, 
                                  const std::vector<Bound> &bounds, 
                                  const IndividualFactory<T> &factory, 
                                  std::mt19937 &rng,
                                  const DifferentialEvolutionFlags &flags)
{
    Population outputs;
    std::uniform_real_distribution<double> float_dis(0, 1);
    std::uniform_int_distribution<> this_int_selector = (this_layer.size() == 0) ? std::uniform_int_distribution<>(0, 1) : std::uniform_int_distribution<>(0, static_cast<int>(this_layer.size()) - 1);
    std::uniform_int_distribution<> older_int_selector = (older_layer.size() == 0) ? std::uniform_int_distribution<>(0, 1) : std::uniform_int_distribution<>(0, static_cast<int>(older_layer.size()) - 1);

    // Keep the first Nelite elements
    for (auto it = this_layer.cbegin(); it != this_layer.cbegin()+flags.Nelite; ++it){
        outputs.emplace_back((*it)->copy());
    }

    double Fmin = std::min(flags.Fmin, flags.Fmax), Fmax = std::max(flags.Fmin, flags.Fmax);
    double F = std::uniform_real_distribution<>(Fmin, Fmax)(rng);

    for (auto i = flags.Nelite; i < this_layer.size(); ++i)
    {
        Population candidates;
        std::set<int> uniques; uniques.insert(static_cast<int>(i));

        // Create a new individual
        std::size_t failure_count = 0;
        for (auto j = 0; j < 3; ++j)
        {
            if (failure_count > 10000) {
                throw std::range_error("Cannot populate individuals for differential evolution");
            }
            if (older_layer.size() == 0 || float_dis(rng) < 0.8) {
                // Pull from this generation
                auto k = static_cast<int>(this_int_selector(rng));
                if (uniques.find(k) == uniques.end()) {
                    uniques.insert(static_cast<int>(k));
                    candidates.emplace_back(this_layer[k]->copy());
                }
                else {
                    failure_count++; j--; // try again
                }
            }
            else {
                // Pull from the old generation
                auto k = static_cast<int>(older_int_selector(rng));
                if (uniques.find(-k) == uniques.end()) {
                    candidates.emplace_back(older_layer[k]->copy());
                    uniques.insert(static_cast<int>(-k));
                }
                else {
                    failure_count++; j--; // try again
                }
            }
        }
        
        auto other = DE1bin(static_cast<const NumericalIndividual<T>&>(*this_layer[i]),
                            static_cast<const NumericalIndividual<T>&>(*candidates[0]),
                            static_cast<const NumericalIndividual<T>&>(*candidates[1]),
                            static_cast<const NumericalIndividual<T>&>(*candidates[2]),
                            rng, factory, F, flags.CR);
        // Impose the bounds if bounds are provided
        if (!bounds.empty()) {
            const auto c = static_cast<const NumericalIndividual<T>&>(*other).get_coefficients();
            auto cnew = c; 
            for (auto i = 0; i < c.size(); ++i) {
                // If the value exceeds the bounds, then 
                // 1) try to reflect back into the bounds if possible, or
                // 2) randomly generate a number inside the bounds
                T cconstrained = static_cast<T>(bounds[i].reflect_then_random_out_of_bounds(rng,c[i]));
                cnew(i) = cconstrained;
            }
            other.reset(factory(std::move(cnew)));
        }
        outputs.emplace_back(std::move(other));
        // Set the age to the oldest age of any of the candidates
        std::vector<std::size_t> candidate_ages;
        for (auto &cand : candidates) {
            candidate_ages.push_back(cand->age());
        }
        outputs.back()->set_age(*std::max_element(candidate_ages.cbegin(), candidate_ages.cend()));
    }
    return outputs;
}

} /* namespace CEGO */
#endif
