#include "CEGO/CEGO.hpp"
#include <Eigen/Dense>
#include <atomic>

std::atomic_size_t Ncalls(0);

using CEGO::EArray;

/**
# Sadus, https://doi.org/10.1063/1.5041320, erratum: missing exponent of m
n is the repulsive exponent (the 12 of 12-6 LJ)
m is the attractive exponent (the 6 of 12-6 LJ)
*/
template <typename T>
auto B2_LennardJones(T Tstar, double n, double m){
    auto F = [&](auto y) {
        auto the_sum = 0.0;
        for (auto i = 0; i < 200; ++i) {
            auto my_factorial = [](auto k) { return tgamma(k + 1); };
            the_sum += tgamma((i * m - 3.0) / n) / my_factorial(i) * pow(y, i);
        }
        return pow(y, 3.0/(n - m)) * (tgamma((n - 3.0) / n) - 3.0/n*the_sum);
    };
    auto yn = pow(n/(n-m), n)*pow((n-m)/m,m)*pow(Tstar, -(n-m)); // y**n, Eq. 9
    auto y = pow(yn, 1.0/n);
    return 2*EIGEN_PI/3*F(y);
}

class FitClass {
public:
    EArray<double> m_T, B2, m_LHS, m_x;
    FitClass(const std::size_t Npts)
    {
        double Tmin = 0.1, Tmax = 10000;
        m_T = EArray<double>::LinSpaced(Npts, log10(Tmin), log10(Tmax)).exp();
        B2.resize(m_T.size());
        for (auto i = 0; i < m_T.size(); ++i) {
            B2(i) = B2_LennardJones(m_T(i), 12, 6);
        }
        // Set variables for fitting
        m_LHS = B2;
        m_x = 1/m_T;
    }
    template <typename TYPE> 
    auto eval_RHS(const EArray<double>& x, const EArray<TYPE> &c) 
    {
        std::decay_t<decltype(x)> val(x.size()); val.setZero();
        for (auto i = 0; i < c.size(); i += 2) {
            double a = c[i];
            double e = c[i + 1];
            val += a*x.pow(e);
        }
        return val.eval();
    }
    template <typename TYPE> TYPE objective(const EArray<TYPE>& c) {
        return ((eval_RHS(m_x, c) - m_LHS)).square().sum();
    }
    template <typename TYPE> EArray<TYPE> abs_rel_deviations(const EArray<TYPE>& c) {
        return (eval_RHS(m_x, c) - m_LHS).eval();
    }
    double objective(const CEGO::AbstractIndividual *pind) {
        const auto &c = dynamic_cast<const CEGO::NumericalIndividual<CEGO::numberish>*>(pind)->get_coeff_array<CEGO::numberish>();
        return objective(c);
    }
};

int do_one()
{
    //std::srand((unsigned int)time(0));
    std::size_t Nterms = 3;

    //EArray<CEGO::numberish> c1(3); c1 << 1.0, 2, 3;
    //EArray<CEGO::numberish> c2(3); c2 << 2.0, 3, 4;
    //EArray<CEGO::numberish> c3(3); c3 << 3.0, 4, 5;
    //auto oo = (c1 - c2).eval();
    //auto o2 = (c3 - c2).eval();
    //auto o3 = (o2*0.7).eval();

    // Construct the bounds
    std::vector<CEGO::Bound> bounds;
    for (auto i = 0; i < Nterms; ++i) {
        bounds.push_back(CEGO::Bound(std::make_pair(-1000.0, 1000.0)));
        bounds.push_back(CEGO::Bound(std::make_pair(0.0, 10.0))); 
    }    
    std::size_t Npts = 100;
    FitClass rp(Npts);
   
    auto Ncalls = 0;
    CEGO::CostFunction<CEGO::numberish> cost_wrapper = [&rp](const CEGO::AbstractIndividual*pind) {return rp.objective(pind); };
    auto Ntotal_individuals = 1000;
    auto Nlayers = 7;
    auto layers = CEGO::Layers<CEGO::numberish>(cost_wrapper, bounds.size(), Ntotal_individuals/Nlayers, Nlayers, 3);
    layers.parallel = false;
    layers.parallel_threads = 6;
    layers.set_bounds(bounds);
    layers.set_generation_mode(CEGO::GenerationOptions::LHS);
    layers.set_builtin_evolver(CEGO::BuiltinEvolvers::differential_evolution_best1bin);
    auto f = [&rp](const CEGO::EArray<double>& c) {return rp.objective<double>(c); };
    //auto f2 = [&rp](const CEGO::EArray<std::complex< double >>& c) {return rp.objective<std::complex<double>>(c); };
    //layers.add_gradient(f, f2);

    auto flags = layers.get_evolver_flags();
    flags["Nelite"] = 1;
    flags["Fmin"] = 0.1;
    flags["Fmax"] = 1.1;
    flags["CR"] = 0.9;
    layers.set_evolver_flags(flags);

    std::vector<double> best_costs; 
    const double VTR = 2e-4;
    auto startTime = std::chrono::system_clock::now();
    bool success = false;
    for (auto counter = 0; counter < 15000; ++counter) {
        layers.do_generation();

        /*if (counter % 1000 == 0) {
            layers.gradient_minimizer();
        }*/
        
        auto [best_cost, best_coeffs] = layers.get_best();
        if (counter % 50 == 0) {
            std::cout << counter << ": best: " << best_cost << std::endl;
            //std::cout << counter << ": best coeffs: " << c << "||" << std::endl;
            //std::cout << counter << ": obj again: " << rp.objective(c) << "||" << std::endl;
        }
        if (best_cost < VTR) { success = true;  break; }
    }
    auto best_layer = layers.get_best();
    auto best_coeffs = std::get<1>(best_layer);
    //std::cout << rp.abs_rel_deviations(best_coeffs) << std::endl;
    auto endTime = std::chrono::system_clock::now();
    double elap = std::chrono::duration<double>(endTime - startTime).count();
    std::cout << "run:" << elap << " s\n";
    std::cout << "NFE:" << Ncalls << std::endl;
    return success;
}

int main() {
    int N = (CEGO::is_CI() ? 3 : 100);
    int good = 0;
    for (auto i = 0; i < N; ++i) {
        good += do_one();
    }
    std::cout << "success:" << good << "/" << N << std::endl;
}