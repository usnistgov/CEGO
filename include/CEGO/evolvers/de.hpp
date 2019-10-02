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

    /// Enumeration for how to select the first individual to go into the crossover
    enum class differential_evolution_selector { rand, best };
    /// Enumeration for how to crossover
    enum class differential_evolution_crossover { bin1, exp1, bin2, exp2 };

    /**
    @brief Generate one mutant individual given three individuals.  This is the one-difference-vector mutant generator of Storn and Price.
    */
    template<typename TYPE>
    EArray<TYPE> gen_1diff(const Population &inds, TYPE F) {
        auto get = [](const pIndividual& i) {
            return static_cast<const NumericalIndividual<TYPE>&>(*i).get_coefficients();
        };
        const auto& c1 = get(inds[0]), &c2 = get(inds[1]), &c3 = get(inds[2]);
        assert(c1.size() == c2.size());
        assert(c2.size() == c3.size()); 
        return c1 + F*(c2 - c3);
    }

    /**
    @brief Generate one mutant individual given three individuals.  This is the one-difference-vector mutant generator of Storn and Price.
    */
    template<typename TYPE>
    EArray<TYPE> gen_2diff(const Population& inds, TYPE F) {
        auto get = [](const pIndividual& i) {
            return static_cast<const NumericalIndividual<TYPE>&>(*i).get_coefficients();
        };
        const auto& c1 = get(inds[0]), & c2 = get(inds[1]), & c3 = get(inds[2]), & c4 = get(inds[3]), & c5 = get(inds[4]);
        assert(c1.size() == c2.size());
        assert(c2.size() == c3.size());
        return c1 + F * (c2 + c3 - c4 - c5);
    }

// *****************************************************************************************************
// *****************************************************************************************************
//                                             CROSSOVER
// *****************************************************************************************************
// *****************************************************************************************************

template<typename TYPE>
pIndividual DE1bin(const pIndividual& base_individual, 
                   const Population &others,
                   std::mt19937&gen, 
                   const IndividualFactory<TYPE> &factory, 
                   double F = 0.5, 
                   double CR = 0.9)
{   
    // Copy of the coefficients for the base individual
    auto c0 = static_cast<const NumericalIndividual<TYPE>&>(*base_individual).get_coefficients();
    // The mutant obtained from individual i1 and the two others (i2 and i3) forming 
    // the difference
    const EArray<TYPE> cm = (others.size() == 3) ? gen_1diff<TYPE>(others, F) : gen_2diff<TYPE>(others, F);
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

template<typename TYPE>
pIndividual DE1exp(const pIndividual &base_individual, 
                   const Population& others,
                   std::mt19937 &gen,
                   const IndividualFactory<TYPE> &factory, 
                   double F = 1.0, 
                   double CR = 0.9)
{
    // Copy of the coefficients
    auto c0 = static_cast<const NumericalIndividual<TYPE>&>(*base_individual).get_coefficients();
    // The mutant obtained from individual i1 and two others (i2 and i3), forming 
    // the difference
    const EArray<TYPE> cm = (others.size() == 3) ? gen_1diff<TYPE>(others, F) : gen_2diff<TYPE>(others, F);
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
           Fmax = 0.9, ///< The maximum value of F for DE
           CR = 0.9,   ///< The crossover rate
           prob_this_layer = 0.95; ///< The probability of selecting candidate from this layer
};

inline void to_json(nlohmann::json& j, const DifferentialEvolutionFlags& f) {
    j = nlohmann::json{ { "Nelite", f.Nelite },{ "Fmin", f.Fmin },{ "Fmax", f.Fmax },{"CR",f.CR},{"prob_this_layer",f.prob_this_layer} };
}

inline void from_json(const nlohmann::json& j, DifferentialEvolutionFlags& f) {
    f.Nelite = j.at("Nelite").get<int>(); 
    f.Fmin = j.at("Fmin").get<double>();
    f.Fmax = j.at("Fmax").get<double>(); 
    f.CR = j.at("CR").get<double>();
    j.at("prob_this_layer").get_to(f.prob_this_layer);
}

/** Do differential evolution to generate a given population of individuals.  The individuals are not yet evaluated, 
  so that evaluation can be carried out in parallel
 */
template<typename T>
Population differential_evolution(const Population &this_layer, 
                                  const Population &older_layer, 
                                  const std::vector<Bound> &bounds, 
                                  const differential_evolution_selector selector,
                                  const differential_evolution_crossover crossover,
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

        using sel = differential_evolution_selector;
        if (selector == sel::best) {
            // The best individual is always in index 0 because sorted by cost function
            candidates.emplace_back(this_layer[0]->copy());
        }
        using cr = differential_evolution_crossover;
        int total_ind_needed = (crossover == cr::bin1 || crossover == cr::exp1) ? 3 : 5;
        int num_ind_needed = total_ind_needed - static_cast<int>(candidates.size());

        // Create a new individual
        std::size_t failure_count = 0;
        for (auto j = 0; j < num_ind_needed; ++j)
        {
            if (failure_count > 10000) {
                throw std::range_error("Cannot populate individuals for differential evolution");
            }
            if (older_layer.size() == 0 || float_dis(rng) < flags.prob_this_layer) {
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

        pIndividual other;
        using cr = differential_evolution_crossover;
        switch (crossover) {
        case cr::bin1:
        case cr::bin2:
            other = DE1bin(this_layer[i], candidates, rng, factory, F, flags.CR); break;
        case cr::exp1:
        case cr::exp2:
            other = DE1exp(this_layer[i], candidates, rng, factory, F, flags.CR); break;
        default:
            throw std::invalid_argument("Not sure how this is possible, but crossver flag is invalid");
        }

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
        outputs.back()->set_age(0);
    }
    return outputs;
}

} /* namespace CEGO */
#endif
