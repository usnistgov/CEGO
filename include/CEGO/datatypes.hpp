#pragma once

#include <functional>
#include "CEGO/CEGO.hpp"
#include <iostream>

namespace CEGO {

    template <typename T> using EArray = Eigen::Array<T, Eigen::Dynamic, 1, 0, Eigen::Dynamic, 1>;
    template <typename T> using EVector = Eigen::Matrix<T, Eigen::Dynamic, 1, 0, Eigen::Dynamic, 1>;
    using DoubleObjectiveFunction = std::function<double(const EArray<double>&)>;
    using DoubleGradientFunction = std::function<EArray<double>(const EArray<double>&)>;

    template <typename T>
    T smallest_positive(const EArray<T>& c) {
        T val = std::numeric_limits<T>::max();
        for (auto i = 0; i < c.size(); ++i) {
            if ((c[i]) < val && c[i] > 0) {
                val = c[i];
            }
        }
        return val;
    };

    inline int double2int(double inval) {
        return std::lround(inval);
    }

    /// \brief A small class that allows for integer and double numbers
    ///  
    /// \note Integers up to 9,007,199,254,740,993 in double precision can be exactly represented: https://stackoverflow.com/a/3793950, so we use a double as the data store.
    struct numberish {
    public:
        double u_;
        
    public:
        enum types { EMPTY, INT, DOUBLE } type;
        
        numberish() { u_ = std::numeric_limits<double>::max(); type = EMPTY; }
        numberish(const int &value) { u_ = value; type = INT; }
        numberish(const long unsigned int& value) { u_ = static_cast<int>(value); type = INT; }
        numberish(const double &value) { u_ = value; type = DOUBLE; }

        const double u() const { return u_; }

        void operator=(const int& value) { u_ = value; type = INT; };
        void operator=(const double& value) { u_ = value; type = DOUBLE; };
        numberish operator-(const numberish& value) const {
            if (type != value.type) { throw std::logic_error("Cannot mix types in the - operator for numberish type"); }
            switch (type) {
            case INT:
                return numberish(static_cast<int>(u()) - static_cast<int>(value));
            case DOUBLE:
                return numberish(static_cast<double>(u()) - static_cast<double>(value));
            default:
                throw std::logic_error("Bad type");
            }
        };
        numberish operator+(const numberish& value) const {
            if (type != value.type) { throw std::logic_error("Cannot mix types in the + operator for numberish type"); }
            switch (type) {
            case INT:
                return numberish(static_cast<int>(u()) + static_cast<int>(value));
            case DOUBLE:
                return numberish(static_cast<double>(u()) + static_cast<double>(value));
            default:
                throw std::logic_error("Bad type");
            }
        };
        numberish operator*(const numberish& value) const {
            if (type != value.type) { throw std::logic_error("Cannot mix types in the * operator for numberish type; tried to do: " + to_string() + "*" + value.to_string()); }
            switch (type) {
            case INT:
                return numberish(static_cast<int>(u()) * static_cast<int>(value));
            case DOUBLE:
                return numberish(static_cast<double>(u()) * static_cast<double>(value));
            default:
                throw std::logic_error("Bad type");
            }
        };

        /// Get the value as an integer - stored double value will throw error
        operator int() const {
            if (type == DOUBLE) { 
                throw std::logic_error("Cannot downcast from double to int"); }
            return static_cast<int>(u());
        };
        /// Return the value as a double
        double as_double() const {
            return static_cast<double>(u());
        }
        /// Return the value as an integer
        int as_int() const {
            return static_cast<int>(*this);
        }
        /// Get the value as a double - stored integer will upcast to a double
        operator double() const {
            if (type == INT) { return static_cast<double>(u()); }
            else if (type == DOUBLE) { return static_cast<double>(u()); }
            else {
                throw std::logic_error("Bad type");
            }
        }
        /// Convert the value to a string
        std::string to_string() const {
            switch (type) {
            case INT:
                return std::to_string(static_cast<int>(u()));
            case DOUBLE:
                return std::to_string(static_cast<double>(u()));
            default:
                throw std::logic_error("Bad type to to_string");
            }
        };
    };

    inline numberish operator* (double k, const numberish& n)
    {
        if (n.type == numberish::types::DOUBLE) {
            double val = static_cast<double>(n)*k;
            return numberish(val);
        }
        else {
            int val = double2int(static_cast<int>(n)*k);
            return numberish(val);
        }
    }
    inline numberish operator* (const numberish& n, double d)
    {
        return d*n;
    }
    inline numberish operator+ (double k, const numberish& n)
    {
        if (n.type == numberish::types::DOUBLE) {
            double val = static_cast<double>(n) + k;
            return numberish(val);
        }
        else {
            int val = double2int(static_cast<int>(n) + k);
            return numberish(val);
        }
    }
    inline numberish operator+ (const numberish& n, double d)
    {
        return d+n;
    }
    inline numberish operator- (double k, const numberish& n)
    {
        if (n.type == numberish::types::DOUBLE) {
            double val = k-static_cast<double>(n);
            return numberish(val);
        }
        else {
            int val = double2int(k - static_cast<int>(n));
            return numberish(val);
        }
    }
    inline numberish operator- (const numberish& n, double d)
    {
        return -1.0*(d-n);
    }

    inline std::string to_string(const numberish &n) {
        return n.to_string();
    }

    inline std::string to_string(const std::vector<double> &v, const std::string &sep = ", ") {
        std::string o;
        for (auto &&el : v)
            o += std::to_string(el) + sep;
        return o;
    }

    std::ostream & operator << (std::ostream &out, const numberish &c)
    {
        out << to_string(c);
        return out;
    }

    struct Bound {
        numberish m_lower, m_upper;
        template<class T> Bound(const T& lower, const T& upper) : m_lower(lower), m_upper(upper) {};
        template<class T> Bound(const std::pair<T, T> &bounds) : m_lower(bounds.first), m_upper(bounds.second) {};
        template<typename URNG> void gen_uniform(URNG &gen, double &d, int &i) const {
            if (m_upper.type != m_lower.type) {
                throw std::logic_error("Upper and lower bounds are of different types; impossible");
            }
            if (m_upper.type == numberish::types::DOUBLE) {
                double lower = m_lower, upper = m_upper;
                d = std::uniform_real_distribution<>(lower, upper)(gen); i = -1;
            }
            else if (m_upper.type == numberish::types::INT) {
                i = std::uniform_int_distribution<>(m_lower, m_upper)(gen); d = -1;
            }
            else {
                throw std::logic_error("Invalid type");
            }
        }
        numberish enforce_bounds(const numberish &n) const {
            if (m_upper.type == numberish::types::DOUBLE) {
                return std::min(static_cast<double>(m_upper.u()), std::max(m_lower.u(), static_cast<double>(n)));
            }
            else if (m_upper.type == numberish::types::INT) {
                return std::min(static_cast<int>(m_upper.u()), std::max(static_cast<int>(m_lower.u()), static_cast<int>(n)));
            }
            else {
                throw std::logic_error("Invalid type");
            }
        }
        template<typename URNG> numberish random_out_of_bounds(URNG &gen, const numberish &n) const {
            if (m_upper.type == numberish::types::DOUBLE) {
                if (static_cast<double>(n) > m_lower.u().d && static_cast<double>(n) < m_upper.u().d){ return n; }
                else{
                    double dbl; int integer;
                    gen_uniform(gen, dbl, integer);
                    return dbl;
                }
            }
            else if (m_upper.type == numberish::types::INT) {
                if (static_cast<int>(n) > m_lower.u().i && static_cast<int>(n) < m_upper.u().i) { return n; }
                else {
                    double dbl; int integer;
                    gen_uniform(gen, dbl, integer);
                    return integer;
                }
            }
            else {
                throw std::logic_error("Invalid type");
            }
        }
        template<typename URNG> numberish reflect_then_random_out_of_bounds(URNG &gen, const numberish &val) const {
            // First try to reflect back into region, if that doesn't work, then generate a new value inside the range
            if (m_upper.type == numberish::types::DOUBLE) {
                // Upcasting is fine
                double n = static_cast<double>(val);
                
                // Input value is in range
                double upperval = static_cast<double>(m_upper.u()), lowerval = static_cast<double>(m_lower.u());
                if (n >= lowerval && n <= upperval) { return n; }
                else{
                    double exc_upper = n - upperval,  ///< Excursion above the upper bound
                           exc_lower = lowerval - n,  ///< Excursion below the lower bound
                           width = upperval- lowerval; ///< Width of the range
                    // Above the range, but reflection is possible
                    if (n > upperval && exc_upper <= width){
                        return upperval-exc_upper;
                    }
                    // Below the range, but reflection is possible
                    else if (n < lowerval && exc_lower <= width){
                        return lowerval + exc_lower;
                    }
                    // Out of range and reflection not possible
                    else {
                        double dbl; int integer;
                        gen_uniform(gen, dbl, integer);
                        return dbl;
                    }
                }
            }
            else if (m_upper.type == numberish::types::INT) {
                int i;
                if (val.type == numberish::types::INT){
                    i = static_cast<int>(val);
                }
                else {
                    i = double2int(val);
                }

                // Input value is in range
                int lowerval = static_cast<int>(m_lower.u()), upperval = static_cast<int>(m_upper.u());
                if (i >= lowerval && i <= upperval) { 
                    return i; 
                }
                else{
                    int exc_upper = i - upperval,  ///< Excursion above the upper bound
                        exc_lower = lowerval - i,  ///< Excursion below the lower bound
                        width = upperval - lowerval; ///< Width of the range
                    // Above the range, but reflection is possible
                    if (i > upperval && exc_upper <= width) {
                        return upperval - exc_upper;
                    }
                    // Below the range, but reflection is possible
                    else if (i < lowerval && exc_lower <= width) {
                        return lowerval + exc_lower;
                    }
                    // Out of range and reflection not possible
                    else {
                        double dbl; int integer;
                        gen_uniform(gen, dbl, integer);
                        return integer;
                    }
                }
            }
            else {
                throw std::logic_error("Invalid type");
            }
        }
    };

    struct RecombinationFlags {
        double p_same_w = 0.5; ///< The probability in [0,1] that the same w will be used to weight the 
                               ///< entire individual in the recombination, otherwise different w will be
                               ///< used for each coefficient in the individual
        double nonuniform_w_stddev = 1; ///< The standard deviation for the normal distribution for the weighting
                                        ///< factor between the individuals
        double uniform_w_stddev = 1; ///< The standard deviation for the normal distribution for the weighting
                                     ///< factor between the individuals
        bool enforce_bounds = true;
    };

    /// Forward declaration of AbstractIndividual
    class AbstractIndividual;

    /// A typedef for a  pointer to an individual
    typedef std::unique_ptr<AbstractIndividual> pIndividual;
    

    class AbstractIndividual
    {
    private:
        std::size_t m_age;
        bool m_needs_evaluation; /// True if the individual needs to have their objective function evaluated
    public:
        AbstractIndividual(std::size_t age) : m_age(age), m_needs_evaluation(true) {};
        virtual ~AbstractIndividual(){};
        /// Ask for an evaluation of this individual
        void request_evaluation() { m_needs_evaluation = true; };
        /// Set the evaluation state of this individual
        void set_needs_evaluation(bool needs_evaluation) { m_needs_evaluation = needs_evaluation; };
        /// Returns true if evaluation is needed
        bool needs_evaluation() const { return m_needs_evaluation; };
        /// Increase the age of the individual
        void increase_age() { m_age++; }
        /// Return the age of the individual
        std::size_t age() const { return m_age; }
        /// Return the age of the individual
        std::size_t get_age() const { return m_age; }
        /// Set the age of the individual
        void set_age(std::size_t age) { m_age = age; }
        /// Return the cost
        virtual double get_cost() = 0;
        /// Calculate the cost of this individual
        virtual void calc_cost() = 0;
        /// Get the coefficients as an array of doubles.  Throws if not possible
        virtual Eigen::ArrayXd get_coeffs_ArrayXd() const = 0;
        /// Return a copy of this individual
        virtual pIndividual copy() const = 0;
        /// Evaluate the given individual; no-op if no evaluation is needed
        void evaluate() {
            if (m_needs_evaluation) {
                calc_cost();
                m_needs_evaluation = false;
            }
        }
        /**
        *  @brief Merge this individual with another individual to obtain the offspring.
        *
        * This must be implemented by the derived class; deciding how to handle crossover will depend on
        * the data storage model (homogeneous or heterogeneous)
        */
        virtual pIndividual recombine_with(pIndividual &other,
            const std::vector<Bound> &bounds,
            std::mt19937 &rng,
            const RecombinationFlags &flags) = 0;
    };

    template <typename T> using IndividualFactory = std::function<AbstractIndividual* (const EArray<T>&&)>;
    /// A typedef for a population of individuals
    typedef std::vector<pIndividual > Population;
    /// A typedef for the cost function
    template <typename T> using CostFunction = std::function<T(const AbstractIndividual *)>;
    using GradientFunction = std::function<EArray<double>(const AbstractIndividual*)>;

    using DoubleCostFunction = std::function<double(const EArray<double>&)>;
    using DoubleGradientFunction = std::function<EArray<double>(const EArray<double>&)>;

    template<typename T>
    class NumericalIndividual : public AbstractIndividual {
    protected:
        double m_cost = 1e99;
        EArray<T> m_c;
        const CostFunction<T> m_f;
    public:
        NumericalIndividual(const EArray<T>&&c, const CostFunction<T> &f) : AbstractIndividual(0), m_c(c), m_f(f) {};
        NumericalIndividual(const EArray<T>&c, const CostFunction<T> &f) : AbstractIndividual(0), m_c(c), m_f(f) {};
        const EArray<T> & get_coefficients() const { return m_c; };
        void set_coefficients(const EArray<T> &c){ m_c = c; };
        void calc_cost() override {
            m_cost = m_f(this);
        }
        void set_cost(T cost){ m_cost = cost; };
        double get_cost() override {
            assert(needs_evaluation() == false);
            return m_cost;
        }
        Eigen::ArrayXd get_coeffs_ArrayXd() const override {
            Eigen::ArrayXd o(m_c.size());
            for (auto i = 0; i < m_c.size(); ++i) {
                o[i] = m_c[i];
            }
            return o;
        }
        template <typename TYPE>
        auto get_coeff_array() const {
            EArray<TYPE> o(m_c.size());
            for (auto i = 0; i < m_c.size(); ++i) {
                o[i] = m_c[i];
            }
            return o;
        }

        virtual pIndividual recombine_with(pIndividual& other,
            const std::vector<Bound>& bounds,
            std::mt19937 &rng,
            const RecombinationFlags &flags = {}) override
        {

            // Get the coefficients of the winner
            EArray<T> c0 = get_coefficients();

            // The size of the individual
            std::size_t N = c0.size();

            // Get the coefficients of the other one
            EArray<T> c1 = static_cast<NumericalIndividual<T>*>(other.get())->get_coefficients();

            EArray<T> cnew(N);

            // Get the new coefficients
            if (std::uniform_real_distribution<double>(0, 1)(rng) < flags.p_same_w) {
                // Different weighting for each coefficient taken from normal
                // distribution
                for (auto i = 0; i < N; ++i) {
                    double k = std::normal_distribution<double>(0, flags.uniform_w_stddev)(rng);
                    numberish c = c0[i] + k*(c1[i] - c0[i]);
                    if (!bounds.empty() && flags.enforce_bounds) {
                        cnew(i) = bounds[i].enforce_bounds(c);
                    }
                    else {
                        cnew(i) = c;
                    }
                }
            }
            else {
                // Uniform weighting for all coefficients
                double k = std::normal_distribution<double>(0, flags.nonuniform_w_stddev)(rng);
                for (auto i = 0; i < N; ++i) {
                    numberish c = c0[i] + k*(c1[i] - c0[i]);
                    if (!bounds.empty() && flags.enforce_bounds) {
                        cnew(i) = bounds[i].enforce_bounds(c);
                    }
                    else {
                        cnew(i) = c;
                    }
                }
            }
            pIndividual newone(new NumericalIndividual<T>(cnew, m_f));
            newone->set_age(std::max(this->age(), other->age()));
            assert(c0.size() == N);
            assert(c1.size() == N);
            assert(cnew.size() == N);
            return newone;
        }
        virtual pIndividual copy() const override {
            auto *newone = new NumericalIndividual<T>(m_c, m_f);
            newone->set_age(age());
            newone->set_needs_evaluation(needs_evaluation());
            newone->set_cost(m_cost);
            return pIndividual(newone);
        }
    };

    template <typename T> using pNumIndividual = std::unique_ptr<NumericalIndividual<T>>;

    template<typename TYPE>
    class GradientIndividual : public NumericalIndividual<TYPE> {
    protected:
        DoubleCostFunction m_double_cost_function;
        DoubleGradientFunction m_double_gradient_function;
    public:
        GradientIndividual(
            const EArray<TYPE>&& c, 
            const CostFunction<TYPE>& f, 
            const DoubleCostFunction &double_cost_function, 
            const DoubleGradientFunction &double_gradient_function)
            : NumericalIndividual<TYPE>(c,f), m_double_cost_function(double_cost_function), m_double_gradient_function(double_gradient_function)
        {}; 
        GradientIndividual(
            const EArray<TYPE>& c,
            const CostFunction<TYPE>& f,
            const DoubleCostFunction& double_cost_function,
            const DoubleGradientFunction& double_gradient_function
        ) : NumericalIndividual<TYPE>(c, f), m_double_cost_function(double_cost_function), m_double_gradient_function(double_gradient_function) {};
        double cost(const EArray<double>& c) const {
            return m_double_cost_function(c);
        }
        EArray<double> gradient(const EArray<double>&c) const { 
            return m_double_gradient_function(c);
        };
        pIndividual copy() const override {
            GradientIndividual<TYPE>* newone = new GradientIndividual<TYPE>(this->m_c, this->m_f, m_double_cost_function, m_double_gradient_function);
            newone->set_age(this->age());
            newone->set_needs_evaluation(this->needs_evaluation());
            newone->set_cost(this->m_cost);
            return pIndividual(newone);
        }
    };


} /* namespace CEGO*/

namespace Eigen {
template<> struct NumTraits<CEGO::numberish>: NumTraits<double> 
// Deriving from double permits to get the epsilon, dummy_precision, lowest, highest functions
{
  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 0,
    ReadCost = 1,
    AddCost = 3,
    MulCost = 3
  };
};

}; /* namespace Eigen */