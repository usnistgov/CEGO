#ifndef CEGO_H
#define CEGO_H

#include <memory>
#include <random>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cassert>
#include <iostream>
#include <iterator>
#include <functional>
#include <tuple>
#include <map>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cassert>
#include <future>       // std::packaged_task, std::future
#include <thread>       // std::thread, std::this_thread::sleep_for

#include <Eigen/Dense>
#include "CEGO/minimizers.hpp"
#include "CEGO/datatypes.hpp"
#include "CEGO/utilities.hpp"
#include "CEGO/lhs.hpp"
#include "CEGO/evolvers/evolvers.hpp"
#include "CEGO/concurrentqueue.h"
#include "ThreadPool.h"

#include "nlohmann/json.hpp"

namespace CEGO{

    struct Result {
        Eigen::ArrayXd c;
        double ssq;
        Result() {};
        Result(Eigen::ArrayXd &&c, double &&ssq) : c(c), ssq(ssq) {};
    };

    template<class T>
    inline std::string vec2string(const std::vector<T> &v, const std::string &sep = ", ") {
        using std::to_string;
        std::stringstream ss;
        for (auto &&el : v)
            ss << to_string(el) << sep;
        return ss.str();
    }

    inline std::string vec2string(const std::vector<std::vector<double> > &v, const std::string &sep = ", ") {
        std::stringstream ss;
        for (auto &el : v)
            ss << to_string(el) << sep;
        return ss.str();
    }

    /// Generate a random population of individuals
    template<typename T>
    Population random_population(const std::vector<CEGO::Bound> bounds, std::size_t count, const IndividualFactory<T> &factory, std::mt19937& rng)
    {
        auto length_ind = bounds.size();
        Population out; out.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            EArray<T> c(length_ind);
            for (std::size_t j = 0; j < length_ind; ++j) {
                auto &&bound = bounds[j];
                double d = 0; int integer = 0;
                bound.gen_uniform(rng, d, integer);

                switch (bound.m_lower.type) {
                case CEGO::numberish::types::DOUBLE:
                    c(j) = d; break;
                case CEGO::numberish::types::INT:
                    c(j) = integer; break;
                default:
                    throw std::invalid_argument("This can't happen...");
                }
            }
            assert(c.size() == length_ind);
            out.emplace_back(pIndividual(factory(std::move(c))));
        }
        return out;
    }

    /// Generate a population of individuals with the use of Latin-Hypercube sampling
    template<typename T>
    Population LHS_population(const std::vector<CEGO::Bound> bounds, std::size_t count, const IndividualFactory<T> &factory, std::mt19937 &rng)
    {
        // Generate the set of floating parameters in [0,1]
        Eigen::ArrayXXd population = LHS_samples(count, bounds.size(), rng);

        auto length_ind = bounds.size();
        Population out; out.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            EArray<T> c(length_ind);
            for (std::size_t j = 0; j < length_ind; ++j) {
                auto &&bound = bounds[j];

                switch (bound.m_lower.type) {
                case CEGO::numberish::types::DOUBLE:
                {
                    double w = population(i,j);
                    c(j) = bound.m_lower.as_double()*w + bound.m_upper.as_double()*(1-w); break;
                }
                case CEGO::numberish::types::INT:{
                    double w = population(i, j);
                    c(j) = int(round(bound.m_lower.as_int()*w + bound.m_upper.as_int()*(1 - w))); break;
                }
                default:
                    throw std::invalid_argument("This can't happen...");
                }
            }
            assert(c.size() == length_ind);
            out.emplace_back(pIndividual(factory(std::move(c))));
        }
        return out;
    }
    
    enum class LoggingScheme { none = 0, all, custom };
    enum class FilterOptions { accept, reject };
    enum class GenerationOptions { LHS, random };

    template<typename T>
    class Layers {
        
    private:
        std::vector<Population> m_layers;
        std::size_t m_generation = 0;
        moodycamel::ConcurrentQueue<Result> result_queue;
        std::mt19937 m_rng = get_Mersenne_twister();
        
        void initialize_layers() {

            // Generate all the individuals serially(!)
            MutantVector mutants;
            for (auto i = 0; i < Nlayers; ++i) {
                auto generator = (m_generation_flag == GenerationOptions::LHS) ? LHS_population<T> : random_population<T>;
                for (auto && ind : generator(m_bounds, Npop_size, get_individual_factory(), m_rng)) {
                    mutants.emplace_back(std::make_pair(i, std::move(ind)));
                }
            }

            // Evaluate all of the layers and individuals (in parallel, if desired)
            evaluate_mutants(mutants);

            // Recollect the individuals into layers, again in serial
            m_layers.resize(Nlayers);
            for (auto &&mut : mutants) {
                auto i = std::get<0>(mut);
                m_layers[i].emplace_back(std::move(std::get<1>(mut)));
            }

            // Sort all of the layers
            sort_all_layers();
        };
        std::unique_ptr<ThreadPool> m_pool;
        LoggingScheme m_log_scheme = LoggingScheme::none;
        std::function<FilterOptions(const Result &)> m_filter_function; 
        std::unique_ptr<AbstractEvolver<T> > m_evolver; ///< The functor that is to be used to evolve a layer
        std::vector<Bound> m_bounds;
        GenerationOptions m_generation_flag = GenerationOptions::LHS;
        CostFunction<T> m_cost_function; ///< Individual<T> -> cost
        DoubleCostFunction m_double_cost_function;
        DoubleGradientFunction m_double_gradient_function;
        std::size_t m_Nelite = 2;
    public:
        
        bool parallel = false;
        bool print_chunk_times = false;
        std::size_t parallel_threads = 6;
        std::size_t Nind_size, Npop_size, Nlayers, age_gap;

        Layers(const std::function<double(const EArray<T>&)> &function, std::size_t Nind_size, std::size_t Npop_size, std::size_t Nlayers, std::size_t age_gap = 5)
            : Nind_size(Nind_size), Npop_size(Npop_size), Nlayers(Nlayers), age_gap(age_gap){

            m_cost_function = [function](const CEGO::AbstractIndividual *pind) {
                const EArray<T> &c = static_cast<const CEGO::NumericalIndividual<T>*>(pind)->get_coefficients();
                return function(c);
            };
        };

        /// Constructor into which is passed a CostFunction and information about the layers
        Layers(CostFunction<T> &function, std::size_t Nind_size, std::size_t Npop_size, std::size_t Nlayers, std::size_t age_gap = 5) 
            : Nind_size(Nind_size), Npop_size(Npop_size), Nlayers(Nlayers), age_gap(age_gap), m_cost_function(function){ };

        /// Add gradient function to the class, taking in doubles, returning double
        template <typename Function, typename Function2>
        void add_gradient(const Function& f, const Function2&f2) {

            m_double_cost_function = [&f](const CEGO::EArray<double>& c) -> double {
                return f(c);
            };
            // The gradient function that will be used to do gradient optimization
            m_double_gradient_function = CEGO::ComplexStepGradient(
                [&f2](const CEGO::EArray<std::complex<double>>& c) {
                    return f2(c);
                }
            );
        }

        /// Specify the logging scheme that is to be employed
        void set_logging_scheme(LoggingScheme scheme){ m_log_scheme = scheme; }

        /// Get the logging scheme in use
        LoggingScheme get_logging_scheme() { return m_log_scheme; }

        /// Set the filtering function that should be used
        void set_filtering_function(const std::function<FilterOptions(const Result &)> &f){ m_filter_function = f; }
    
        /// Set the bounds on each element in the individual.  If a one-element vector, the same bounds are used for each parameter
        void set_bounds(const std::vector<Bound> &bounds){ m_bounds = bounds; };

        /// Get the bounds applied to each element in the individual
        const std::vector<Bound > & get_bounds (){ return m_bounds; };

        /// Get the flags for the evolver in JSON format
        const nlohmann::json get_evolver_flags() const {
            if (m_evolver == nullptr) {
                throw std::invalid_argument("Evolver has not been selected yet!");
            }
            return m_evolver->get_flags();
        }

        /// Set the flags for the evolver in JSON format
        const void set_evolver_flags(const nlohmann::json &flags) const {
            m_evolver->set_flags(flags);
        }

        /// Pick one of the builtin evolvers
        void set_builtin_evolver(BuiltinEvolvers e) {
            using sel = differential_evolution_selector;
            using cr = differential_evolution_crossover;
            switch (e) {
            case BuiltinEvolvers::differential_evolution:
            case BuiltinEvolvers::differential_evolution_rand1bin: 
                m_evolver.reset(new DEEvolver<T>(sel::rand, cr::bin1)); break;
            case BuiltinEvolvers::differential_evolution_rand1exp:
                m_evolver.reset(new DEEvolver<T>(sel::rand, cr::exp1)); break;
            case BuiltinEvolvers::differential_evolution_best1bin:
                m_evolver.reset(new DEEvolver<T>(sel::best, cr::bin1)); break; 
            case BuiltinEvolvers::differential_evolution_best1exp:
                m_evolver.reset(new DEEvolver<T>(sel::best, cr::exp1)); break;
            case BuiltinEvolvers::differential_evolution_rand2bin:
                m_evolver.reset(new DEEvolver<T>(sel::rand, cr::bin2)); break;
            case BuiltinEvolvers::differential_evolution_rand2exp:
                m_evolver.reset(new DEEvolver<T>(sel::rand, cr::exp2)); break;
            case BuiltinEvolvers::differential_evolution_best2bin:
                m_evolver.reset(new DEEvolver<T>(sel::best, cr::bin2)); break;
            case BuiltinEvolvers::differential_evolution_best2exp:
                m_evolver.reset(new DEEvolver<T>(sel::best, cr::exp2)); break;
            default:
                throw std::invalid_argument("Invalid builtin evolver");
            }
        }

        /// Get the cost function that is being used currently
        const CostFunction<T> get_cost_function() {
            return m_cost_function;
        }

        /// Set the flag to determine whether LHS or random (or other) is to be used to generate the population
        void set_generation_mode(GenerationOptions flag) {
            m_generation_flag = flag;
        }

        /// Get the flag to determine whether LHS or random is to be used to generate the population
        GenerationOptions get_generation_mode() {
            return m_generation_flag;
        }

        /// Get the function used to generate the population
        auto get_generator() {
            return (m_generation_flag == GenerationOptions::LHS) ? LHS_population<T> : random_population<T>;
        }

        /// Get a factory function to return an individual for a set of coefficients
        IndividualFactory<T> get_individual_factory() {
            
            auto &cost_function = m_cost_function;
            if (cost_function == nullptr) {
                throw std::bad_function_call();
            }

            // By default, individuals are "normal", and enhanced with a cost function that takes the individual as argument
            if (m_double_cost_function == nullptr && m_double_gradient_function == nullptr) {
                return [&cost_function](const EArray<T>&& c) {
                    return new NumericalIndividual<T>(std::move(c), cost_function);
                };
            }
            else {
                auto& double_cost_function = m_double_cost_function;
                auto& double_gradient_function = m_double_gradient_function;

                // Upgraded individual that also implements gradient function
                return [&cost_function, &double_cost_function, &double_gradient_function](const EArray<T>&& c) {
                    return new GradientIndividual<T>(std::move(c), cost_function, double_cost_function, double_gradient_function);
                };
            }
        }
    
        /** Iterate over the layers and find individuals that are too old for the given layer
         * 
         * First try to see if the elderly individual dominates any individual in a layer with a higher age limit
         * If it does, replace the individual in the higher age limit layer
         */
        void graduate_elderly_individuals() {

            // Iterate backwards through the N layers, starting at the N-1 layer, since the 
            // last layer has an infinite age limit, and you cannot age out of the 
            // highest age limit layer
            for (long ilayer = static_cast<long>(m_layers.size())-2; ilayer >= 0; --ilayer) {

                double age_threshold = static_cast<double>(age_gap*pow(2, ilayer));
                auto &this_layer = m_layers[ilayer];

                // Iterate backwards through the individuals in the layer
                for (int iind = static_cast<int>(this_layer.size())-1; iind >= static_cast<int>(m_Nelite); --iind) {
                    auto &ind = this_layer[iind];
                    if (ind->age() > age_threshold) {
                        // If higher age limit layer has an empty slot, use it directly
                        if (m_layers[ilayer+1].size() < Npop_size) {
                            m_layers[ilayer+1].push_back(std::move(ind));
                        }
                        // Or if it dominates the worst individual in the higher age limit layer, replace that one
                        // and sort the higher age limit layer
                        else if (ind->get_cost() < m_layers[ilayer + 1].back()->get_cost()) {
                            std::swap(m_layers[ilayer +1].back(), ind);
                            sort_layer(m_layers[ilayer + 1]);
                        }
                        // Say goodbye to this individual
                        this_layer.erase(this_layer.begin() + iind);
                    }
                }
            }
        }
        /// Repopulate the layer to replace removed old individuals
        void repopulate_layers() {

            // Generate the new individuals serially
            MutantVector mutants;
            for (auto i = 0; i < m_layers.size(); ++i) {
                auto &layer = m_layers[i];
                if (layer.size() < Npop_size) {
                    // How many individuals are missing?
                    auto missing_individuals_count = Npop_size - layer.size();
                    // Get the generator function
                    auto generator = get_generator();
                    // Then we pad out the population with new random individuals as needed (they start with an age of zero)
                    for (auto && ind : generator(m_bounds, missing_individuals_count, get_individual_factory(), m_rng)) {
                        mutants.emplace_back(std::make_pair(i, std::move(ind)));
                    }
                }
            }

            // Evaluate the individuals we just generated
            evaluate_mutants(mutants);

            // Recollect the individuals into layers, again in serial
            for (auto &&mut : mutants) {
                auto i = std::get<0>(mut);
                m_layers[i].emplace_back(std::move(std::get<1>(mut)));
            }

            // Sort all of the layers
            sort_all_layers();
        }

        /// Evaluate a single individual, and store the values if needed
        void evaluate_ind(pIndividual &ind) {
            if (ind->needs_evaluation()){
                ind->evaluate();
                switch (m_log_scheme) {
                    case LoggingScheme::none:
                        break;
                    case LoggingScheme::custom:{
                        if (!m_filter_function) {
                            throw std::invalid_argument("filtering function has not been provided, logging options are inconsistent!");
                        }
                        Result r(ind->get_coeffs_ArrayXd(), ind->get_cost());
                        // If the filter function returns the flag "accept", then store the result
                        switch (m_filter_function(r)) {
                        case FilterOptions::accept:
                            result_queue.enqueue(std::move(r)); break;
                        case FilterOptions::reject: break;
                        }
                        break;
                    }
                    case LoggingScheme::all:{
                        result_queue.enqueue(std::move(Result(ind->get_coeffs_ArrayXd(), ind->get_cost())));
                        break;
                    }
                    default: {
                        throw std::invalid_argument("logging flag is not set; this is an error");
                    }
                }
            }
        }
    
        /// Evaluate all of the layers
        void evaluate_layers() {
            for (auto &layer : m_layers) {
                for (auto &ind : layer) {
                    evaluate_ind(ind); // A no-op if the individual does not need to be evaluated
                }
            }
        };

        /// Sort all of the layers
        void sort_all_layers() {
            for (auto &layer : m_layers) {
                sort_layer(layer);
            }
        };

        /// Increase the age of all individuals
        void increase_all_ages() {
            for (auto &layer : m_layers) {
                for (auto &ind : layer) {
                    ind->increase_age();
                }
            }
        }

        /// Walk through all the individuals, apply some transformation
        /// Function is allowed to change the individuals!
        template <typename Function>
        void transform_individuals(Function F) {

            if (!parallel || parallel_threads == 0) {
                for (auto& layer : m_layers) {
                    for (auto& ind : layer) {
                        F(ind);
                    }
                }
            }
            else {
                for (auto& layer : m_layers) {
                    for (auto& ind : layer) {
                        m_pool->AddJob( [&F,&ind]() { F(ind); } );
                    }
                }
                // Wait until all the threads finish...
                m_pool->WaitAll();
            }
        }

        /// Calculate statistics of the costs of individuals in each layer
        std::vector<std::map<std::string, double> > cost_stats_each_layer() {
            std::vector<std::map<std::string, double> > out;
            for (auto &layer : m_layers) {
                Eigen::ArrayXd cost_layer(layer.size()), age_layer(layer.size());
                for (auto i = 0; i < layer.size(); ++i) {
                    cost_layer(i) = layer[i]->get_cost();
                    age_layer(i) = static_cast<double>(layer[i]->get_age());
                }
                std::map<std::string, double> this_layer_map;
                this_layer_map["max(cost)"] = cost_layer.maxCoeff();
                this_layer_map["min(cost)"] = cost_layer.minCoeff();
                this_layer_map["mean(cost)"] = cost_layer.mean();
                // See https://en.wikipedia.org/wiki/Standard_deviation#Discrete_random_variable
                this_layer_map["stddev(cost)"] = sqrt((cost_layer - cost_layer.mean()).square().sum()/cost_layer.size());
                this_layer_map["mean(age)"] = age_layer.mean();
                out.push_back(this_layer_map);
            }
            return out;
        }

        /// Sort a given layer
        void sort_layer(Population &pop) {
            auto sort_fcn = [](pIndividual &i1, pIndividual &i2) {
                return i1->get_cost() < i2->get_cost();
            };
            std::sort(pop.begin(), pop.end(), sort_fcn);
        }

        typedef std::vector<std::tuple<std::size_t, pIndividual> > MutantVector;

        void parallel_evaluator(MutantVector::iterator itstart, MutantVector::iterator itend, double &elap_sec) {
            
            std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
            start = std::chrono::high_resolution_clock::now();
            for (auto it = itstart; it != itend; ++it) {
                auto &ind = std::get<1>(*it);
                evaluate_ind(ind);
            }
            end = std::chrono::high_resolution_clock::now();
            elap_sec = std::chrono::duration<double>(end - start).count();
            return;
        };
        
        void init_thread_pool(short Nthreads){
            if (!m_pool || m_pool->GetThreads().size() != Nthreads){
                // Make a thread pool for the workers
                m_pool = std::unique_ptr<ThreadPool>(new ThreadPool(Nthreads));
            }
        }

        void evaluate_mutants(MutantVector &mutants) {
            if (!parallel || parallel_threads == 0) {
                MutantVector::iterator itstart = mutants.begin();
                MutantVector::iterator itend = mutants.end();
                double time = 0;
                parallel_evaluator(itstart, itend, time);
            }
            else {
                // Initialize the thread pool, no-op if already initialized and is the right size
                init_thread_pool(static_cast<short>(parallel_threads));
                std::vector< std::future<void> > futures;
                std::vector<std::thread> threads;
                std::size_t Lchunk = mutants.size() / parallel_threads; // Note this is an integer division, so a floor()!
                std::vector<double> times(parallel_threads);
                
                std::vector<std::size_t> chunksizes(parallel_threads, Lchunk);
                auto Nmax = mutants.size();
                std::size_t remainder = Nmax-Lchunk*parallel_threads;
                // Increase the first remainder chunk sizes
                for (auto i = 0; i < remainder; ++i){
                    chunksizes[i]++;
                }
                // Double-check we get the right sizes
                assert(std::accumulate(chunksizes.begin(), chunksizes.end(), static_cast<std::size_t>(0)) == mutants.size());
                std::size_t isum = 0;
                for (auto j = 0; j < parallel_threads; ++j)
                {
                    std::size_t cs = chunksizes[j];
                    auto itstart = mutants.begin() + isum;
                    auto itend = itstart + cs;
                    isum += cs;
                    double &time = times[j];
                    
                    std::function<void(void)> f = [this, itstart, itend, &time]() {
                        auto startTime = std::chrono::high_resolution_clock::now();
                        parallel_evaluator(itstart, itend, time);
                        auto endTime = std::chrono::high_resolution_clock::now();
                        time = std::chrono::duration<double>(endTime - startTime).count();
                    };
                    m_pool->AddJob(f);
                }
                // Wait until all the threads finish...
                m_pool->WaitAll();
                // Uncomment these lines to print out the times for each chunk; ideally they are all very close 
                // to each other.  Similar times means the work per thread is well divided
                if (print_chunk_times){
                    for (auto j = 0; j < parallel_threads; ++j) { 
                        std::cout << j << " " << times[j] << std::endl;
                    }
                }
            }
        }

        void evolve_parallel() {
            MutantVector mutants;
            // In serial, generate the mutants; generation of the mutants 
            // is very fast.  Store all mutants in a flat vector with the index of the layer
            for (auto i = 0; i < m_layers.size(); ++i) {
                for (auto &&el : m_evolver->evolve_layer(m_layers, i, get_bounds(), m_rng, get_individual_factory())) {
                    mutants.emplace_back(std::make_tuple(i, std::move(el)));
                }
            }

            // Evaluate all the mutants, either in parallel or serial
            evaluate_mutants(mutants);
            
            // Recollect the mutants into layers, again in serial
            std::vector<Population> new_layers(m_layers.size());
            for (auto &&mut : mutants) {
                auto i = std::get<0>(mut);
                new_layers[i].emplace_back(std::move(std::get<1>(mut)));
            }
            // Then determine which ones should be kept based on cost, greedily keeping
            // the one with the lower cost
            for (auto i = 0; i < m_layers.size(); ++i) {
                assert(new_layers[i].size() == m_layers[i].size());
                for (auto j = 0; j < m_layers[i].size(); ++j) {
                    if (new_layers[i][j]->get_cost() < m_layers[i][j]->get_cost()) {
                        std::swap(m_layers[i][j], new_layers[i][j]); // the other one will get thrown away
                    }
                    assert(new_layers[i][j]->get_cost() >= m_layers[i][j]->get_cost());
                }
            }
        }

        /// Evolve a given layer serially
        void evolve_layer(std::size_t i) {

            // Evolve the layer (elements are unevaluated)
            auto new_layer = m_evolver->evolve_layer(m_layers, i, get_bounds(), m_rng, get_individual_factory());

            for (auto j = 0; j < new_layer.size(); ++j) {
                evaluate_ind(new_layer[j]);
                double newcost = new_layer[j]->get_cost(), oldcost = m_layers[i][j]->get_cost();
                // Keep the one with the lower cost (new_layer is going to be the one that is 
                // going to be used)
                if (new_layer[j]->get_cost() > m_layers[i][j]->get_cost()) {
                    // Swap because the older one is better, we are collecting values in new_layer
                    std::swap(new_layer[j], m_layers[i][j]);
                }
                assert(new_layer[j]->get_cost() <= m_layers[i][j]->get_cost());
            }

            // How many individuals are missing?
            auto missing_individuals_count = Npop_size - new_layer.size();

            // Then we pad out the population with new random individuals as needed (they start with an age of zero)
            if (missing_individuals_count > 0) {
                auto generator = get_generator();
                Population random_inds = generator(m_bounds, missing_individuals_count, get_individual_factory(), m_rng);
                std::move(random_inds.begin(), random_inds.end(), std::back_inserter(new_layer));
            }

            // And move back to the layer
            std::swap(m_layers[i], new_layer);

            assert(m_layers[i].size() == Npop_size);
        }

        /// Carry out the steps for one generation
        void do_generation() {
            if (m_evolver == nullptr) {
                throw std::invalid_argument("Evolver has not been selected");
            }
            if (m_layers.size() == 0) {
                initialize_layers();
            }
            else if (m_generation % age_gap == 0 && m_layers.size() > 1) {

                // ========
                // Layer #1
                // ========

                // push into the next-from-bottom layer any individuals that dominate an individual in the higher layer
                // --
                // Join them all together into one population
                Population pop;
                // Move them into the population
                std::move(m_layers[1].begin(), m_layers[1].end(), std::back_inserter(pop));
                std::move(m_layers[0].begin(), m_layers[0].end(), std::back_inserter(pop));
                // Sort them in terms of increasing cost
                sort_layer(pop);
                // Remove the ones that are no longer needed;
                // Keep indices [0,Npop_size-1] inclusive
                for (auto i = pop.size()-1; i >= Npop_size; --i){
                    pop.erase(pop.begin()+i);
                }
                // Keep the best ones
                m_layers[1] = std::move(pop);

                // ========
                // Layer #0
                // ========

                // Generate new individuals in serial (fast)
                MutantVector mutants;
                auto generator = get_generator();
                for (auto && ind : generator(m_bounds, Npop_size, get_individual_factory(), m_rng)) {
                    mutants.emplace_back(std::make_pair(0,std::move(ind)));
                }
                // Evaluate the individuals in parallel
                evaluate_mutants(mutants);

                // Put them back into the 0-th layer
                m_layers[0].clear();
                for (auto &&mut : mutants) {
                    auto i = std::get<0>(mut);
                    m_layers[i].emplace_back(std::move(std::get<1>(mut)));
                }
            }
            graduate_elderly_individuals();
            repopulate_layers();
            evaluate_layers();
            if (parallel){
                evolve_parallel();
            }
            else{
                for (int i = static_cast<int>(m_layers.size()) - 1; i >= 0; i--) {
                    evolve_layer(i);
                }
            }
            evaluate_layers();
            increase_all_ages();
            sort_all_layers();
            m_generation++;
        }
        // Get the best individuals from each layer, starting with layer 0
        auto get_best_per_layer(){
            std::vector<std::tuple<double, const EArray<T> > > out;
            for (const auto &layer : m_layers) {
                const EArray<T> c = static_cast<const NumericalIndividual<T>*>(layer[0].get())->get_coefficients();
                out.emplace_back(std::make_tuple(layer[0]->get_cost(), c));
            }
            return out;
        };

        // Do gradient minimization for all individuals
        void gradient_minimizer() {

            // The bounds are universal, the same for all individuals
            auto bounds = get_bounds();
            EArray<double> ub(bounds.size()), lb(bounds.size()), xnew(bounds.size());
            auto i = 0;
            for (auto& b : bounds) {
                ub(i) = b.m_upper;
                lb(i) = b.m_lower;
                i++;
            }

            auto minimizer = [&ub, &lb](const CEGO::pIndividual& pind) {

                // Dynamically cast to a class with the appropriate methods
                CEGO::GradientIndividual<T>& ind = dynamic_cast<CEGO::GradientIndividual<T>&>(*pind);

                // The objective function to be minimized (Array<double> -> double)
                CEGO::DoubleObjectiveFunction obj = [&ind](const EArray<double>& c)-> double { return ind.cost(c); };
                // The gradient function used to do gradient optimization (Array<double> -> Array<double>)
                CEGO::DoubleGradientFunction g = [&ind](const EArray<double>& c)->EArray<double> { return ind.gradient(c); };
                
                // Store current values
                EArray<double> x0 = ind.get_coefficients().template cast<double>();
                auto F0 = ind.get_cost();
                
                // Configuration of the gradient minimizer
                CEGO::BoxGradientFlags flags;
                flags.Nmax = 30; // How many steps to take
                flags.VTR = F0*0.5; // If we get to this level, gradient step is accepted

                // Run the minimization
                double F;
                EArray<double> xnew;
                std::tie(xnew, F) = CEGO::box_gradient_minimization(obj, g, x0, lb, ub, flags);

                if (F < F0) {
                    // We reduced the cost function, yay!
                    ind.set_coefficients(xnew.cast<T>());
                    ind.calc_cost();
                    //std::cout << F / F0 << std::endl;
                }
                else {
                    //std::cout << "no reduction\n";
                }
            };
            transform_individuals(minimizer);
        };

        // Get the layer with the individual with the lowest (best) cost
        auto get_best() {
            auto B = get_best_per_layer();
            // The best individual might not be in the highest layer, let's find the best one
            return *std::min_element(B.begin(), B.end(), [](decltype(B.back()) &b1, decltype(B.back()) &b2) { return std::get<0>(b1) < std::get<0>(b2); });
        };
        /** Return a string with some diagnostic information for the best individual in the population
         * \sa get_best
         * \sa get_best_per_layer
         */
        std::string print_diagnostics() {
            auto [best_cost, c] = get_best();
            std::stringstream ss;
            ss << "i: " << static_cast<int>(m_generation - 1) << " best: " << best_cost << " c: " << c << " queue: " << result_queue.size_approx();
            return ss.str();
        }

        /// Get the results that have been logged during the course of this optimization
        std::vector<Result> get_results() {
            std::vector<Result> results(result_queue.size_approx());
            auto Nels = result_queue.try_dequeue_bulk(results.begin(), results.size());
            return results;
        }
    };
    template <typename TYPE>
    struct ALPSInputValues {
        std::vector<Bound> bounds; ///< The vector of bounds on the variables
        double VTR; ///< Value to reach (terminates on reaching this cost value)
        CostFunction<TYPE> f; ///< The cost function to be minimized
        bool parallel = false; ///< If true, evaluate each layer in a separate thread
        std::size_t max_gen = 1000; ///< Maximum number of generations that are allowed
        std::size_t NP = 40; ///< The number of individuals in a population (per layer)
        std::size_t Nlayer = 1; ///< The number of layers
        std::size_t age_gap = 5; ///< The number of generations between restarting the bottom layer
        bool disp = false; ///< If true, display diagnostics as you go to standard out
    };
    struct ALPSReturnValues {
        double fval; ///< The function value at termination
        double elapsed_sec; ///< The number of seconds to conduct the entire optimization
        std::string termination_reason; ///< Why the optimization stopped
    };

    template<typename T>
    ALPSReturnValues ALPS(ALPSInputValues<T> &in) {
        if (in.bounds.empty()){ throw std::invalid_argument("bounds variable must be provided"); }
        //if (in.VTR) { throw std::invalid_argument("VTR variable must be provided"); }
        if (!in.f) { throw std::invalid_argument("Cost function f must be provided"); }
        ALPSReturnValues out;
        auto D = in.bounds.size();
        auto layers = Layers<T>(in.f, D, in.NP, in.Nlayer, in.age_gap);
        layers.set_bounds(in.bounds);
        layers.parallel = in.parallel;
        layers.set_builtin_evolver(BuiltinEvolvers::differential_evolution);
        auto startTime = std::chrono::system_clock::now();
        for (auto i = 0; i < in.max_gen; ++i) {
            layers.do_generation();
            if (in.disp){ auto diag = layers.print_diagnostics(); std::cout << diag << std::endl; }
            auto best_layer = layers.get_best(); 
            if (std::get<0>(best_layer) < in.VTR) {
                out.termination_reason = "Reached VTR"; 
                break;
            }
            if (i == in.max_gen-1) { out.termination_reason = "Reached max # of generations"; }
        }
        
        auto endTime = std::chrono::system_clock::now();
        out.elapsed_sec = std::chrono::duration<double>(endTime - startTime).count();
        out.fval = std::get<0>(layers.get_best());
        return out;
    }

} /* namespace CEGO */
#endif
