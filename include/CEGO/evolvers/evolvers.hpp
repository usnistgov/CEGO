#ifndef CEGO_EVOLVERS_H
#define CEGO_EVOLVERS_H

#include "CEGO/datatypes.hpp"
#include "nlohmann/json.hpp"
#include "de.hpp"

namespace CEGO {

    enum class BuiltinEvolvers {differential_evolution};

    template<typename TYPE>
    class AbstractEvolver
    {
    public:
        virtual Population evolve_layer(const std::vector<Population> &layers, 
            const std::size_t ilayer, 
            const std::vector<Bound> &bounds, 
            std::mt19937&,
            const IndividualFactory<TYPE> &) const = 0;
        /// Destructor
        virtual ~AbstractEvolver() {};
        /// Set the flags in the derived class by passing a JSON dict
        virtual void set_flags(const nlohmann::json &JSON_flags) = 0;
        /// Get the flags in the derived class as a JSON dict
        virtual nlohmann::json get_flags() const = 0;
    };

    template<typename TYPE>
    class DE1BinEvolver : public AbstractEvolver<TYPE> {
    private:
        DifferentialEvolutionFlags m_flags;
    public:
        Population evolve_layer(
            const std::vector<Population> &pop_layers, 
            const std::size_t ilayer, 
            const std::vector<Bound> &bounds,
            std::mt19937 &rng,
            const IndividualFactory<TYPE> &factory) const {
            Population empty;
            return differential_evolution<TYPE>(
                pop_layers[ilayer],
                (ilayer > 0 ? pop_layers[ilayer - 1] : empty),  // older layer (if i > 0)
                bounds,
                factory,
                rng,
                m_flags);
        }
        /// Set the flags as a JSON dict
        void set_flags(const nlohmann::json &flags) { m_flags = flags; }
        /// Get the flags as a JSON dict
        nlohmann::json get_flags() const{ return m_flags; }
    };
}

#endif