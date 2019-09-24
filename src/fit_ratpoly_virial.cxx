#include "CEGO/CEGO.hpp"
#include <Eigen/Dense>
#include <atomic>

std::atomic_size_t Ncalls(0);

using CEGO::EArray;

auto linfit_coeffs(const EArray<double>& x, const EArray<double>& y, const std::size_t Nnum, const std::size_t Nden) {
    Eigen::MatrixXd A(x.size(), Nnum + Nden);
    Eigen::MatrixXd b = y;
    for (auto i = 0; i < Nnum; ++i) {
        A.col(i) = x.pow(i);
    }
    for (auto i = 0; i < Nden; ++i) {
        A.col(i + Nnum) = -y*x.pow(i+1);
    }
    auto c = A.colPivHouseholderQr().solve(b).eval();
    auto ycheck = (A * c).eval();
    return c;
}

class RatPoly {
public:
    EArray<double> m_T, B2, m_LHS, m_x;
    std::size_t m_Nnum, m_Nden;
    
    RatPoly(const std::size_t Nnum, const std::size_t Nden) : m_Nnum(Nnum), m_Nden(Nden)
    {
        // Data from Garberoglio, DOI: 10.1039/C8FD00092A
        m_T.resize(21);
        B2.resize(21);
        m_T << 200, 225, 250, 273.15, 300, 325, 350, 375, 400, 450, 500, 550, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000;
        B2 << -2.3825E-02, -8.6200E-03, -3.9850E-03, -2.2590E-03, -1.3301E-03, -8.9350E-04, -6.3650E-04, -4.7840E-04, -3.7240E-04, -2.4480E-04, -1.7240E-04, -1.2830E-04, -9.8700E-05, -6.2460E-05, -4.1280E-05, -2.7580E-05, -1.8550E-05, -5.0400E-06, 2.0800E-06, 6.2000E-06, 8.8800E-06;
        m_LHS = B2;
        m_x = 200/m_T;
    }
    void solve_linear() {
        std::cout << "----------- LINEAR --------- " << std::endl;
        Eigen::ArrayXi indices = Eigen::ArrayXi::LinSpaced(m_Nnum + m_Nden, 0L, static_cast<int>(m_x.size()));
        EArray<double> xx = m_x(indices), yy = m_LHS(indices);
        EArray<double> c = linfit_coeffs(xx, yy, m_Nnum, m_Nden);
        std::cout << c << std::endl;
        double o = objective(c);
        std::cout << "objective: " << o << std::endl;
        EArray<double> devs = abs_rel_deviations(c).eval() * 100;
        std::cout << "relative(%) devs: " << devs << std::endl;
    }
    template <typename TYPE> EArray<TYPE> eval_RHS(const EArray<double>& x, const EArray<TYPE> &c) {
        EArray<TYPE> num = EArray<TYPE>::Zero(x.size()), den = EArray<TYPE>::Zero(x.size());
        assert(m_Nnum + m_Nden == c.size());
        for (auto i = 0; i < m_Nnum; ++i) {
            num += c[i]*x.pow(i);
        }
        for (auto i = 0; i < m_Nden; ++i) {
            den += c[m_Nnum+i]*x.pow(i+1);
        }
        return num/(1+den);
    }
    template <typename TYPE> TYPE objective(const EArray<TYPE>& c) {
        return ((eval_RHS(m_x, c) - m_LHS)/m_LHS).square().sum();
    }
    template <typename TYPE> EArray<TYPE> abs_rel_deviations(const EArray<TYPE>& c) {
        return ((eval_RHS(m_x, c) - m_LHS)/m_LHS).eval();
    }
    double objective(const CEGO::AbstractIndividual *pind) {
        const auto &c = static_cast<const CEGO::NumericalIndividual<double>*>(pind)->get_coeff_array<double>().eval();
        return objective(c);
    }  
};

int do_one()
{
    //std::srand((unsigned int)time(0));
    std::size_t Nnum = 3, Nden = 5;

    // Construct the bounds
    std::vector<CEGO::Bound> bounds;
    for (auto i = 0; i < Nnum; ++i) { 
        bounds.push_back(CEGO::Bound(std::pair<double, double>(-1000, 1000)));
    }
    for (auto i = 0; i < Nden; ++i){
        bounds.push_back(CEGO::Bound(std::pair<double, double>(-1000, 1000))); 
    }    
    RatPoly rp(Nnum, Nden);
   
    auto Ncalls = 0;
    CEGO::CostFunction<double> cost_wrapper = [&rp](const CEGO::AbstractIndividual*pind) {return rp.objective(pind); };
    auto Nlayers = 1;
    auto layers = CEGO::Layers<double>(cost_wrapper, bounds.size(), 1000, Nlayers, 3);
    layers.parallel = true;
    layers.parallel_threads = 6;
    layers.set_bounds(bounds);
    layers.set_generation_mode(CEGO::GenerationOptions::LHS);
    layers.set_builtin_evolver(CEGO::BuiltinEvolvers::differential_evolution);
    auto f = [&rp](const CEGO::EArray<double>& c) {return rp.objective<double>(c); };
    auto f2 = [&rp](const CEGO::EArray<std::complex< double >>& c) {return rp.objective<std::complex<double>>(c); };
    layers.add_gradient(f, f2);

    auto flags = layers.get_evolver_flags();
    flags["Nelite"] = 1;
    flags["Fmin"] = 0.1;
    flags["Fmax"] = 1.1;
    flags["CR"] = 1;
    layers.set_evolver_flags(flags);

    std::vector<double> best_costs; 
    const double VTR = 2e-4;
    auto startTime = std::chrono::system_clock::now();
    bool success = false;
    for (auto counter = 0; counter < 15000; ++counter) {
        layers.do_generation();

        if (counter % 1000 == 0) {
            layers.gradient_minimizer();
        }

        auto best_layer = layers.get_best();
        double best_cost;
        std::vector<double> best_coeffs;
        best_cost = std::get<0>(best_layer); best_costs.push_back(best_cost);
        best_coeffs = std::get<1>(best_layer);
        EArray<double> c = Eigen::Map<const Eigen::ArrayXd>(&(best_coeffs[0]), best_coeffs.size());
        if (counter % 50 == 0) {
            std::cout << counter << ": best: " << best_cost << std::endl;
        }
        if (counter % 200 == -1) {
            std::cout << counter << ": best coeffs: " << c << std::endl;
        }
        if (best_cost < VTR) { success = true;  break; }
    }
    auto best_layer = layers.get_best();
    auto best_coeffs = std::get<1>(best_layer);
    EArray<double> c = Eigen::Map<const Eigen::ArrayXd>(&(best_coeffs[0]), best_coeffs.size());
    std::cout << rp.abs_rel_deviations(c)*100 << std::endl;
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