#include "CEGO/CEGO.hpp"

// See http://stackoverflow.com/a/4609795
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

double Rosenbrock(const std::vector<double> &x) {
    return 100 * pow(pow(x[0], 2) - x[1], 2) + pow(1 - x[0], 2);
}
double Corana(const std::vector<double> &x) {
    std::vector<double> d = { 1., 1000., 10., 100. };
    double r = 0;
    for (auto j = 0; j < 4; ++j) {
        double zj = floor(std::abs(x[j] / 0.2) + 0.49999) * sgn(x[j]) * 0.2;
        if (std::abs(x[j] - zj) < 0.05) {
            r += 0.15 * pow(zj - 0.05*sgn(zj), 2) * d[j];
        }
        else {
            r += d[j] * x[j] * x[j];
        }
    }
    return r;
}
template <typename T>
T Griewangk(const std::vector<T> &x) {
    double sum1 = 0, prod1 = 1;
    for (auto i = 0; i < x.size(); ++i) {
        sum1 += static_cast<double>(x[i] * x[i]);
        prod1 *= cos(static_cast<double>(x[i]) / sqrt(i + 1));
    };
    return sum1/4000.0 - prod1 + 1;
}

double HyperEllipsoid(const std::vector<double> &x) {
    double sum1 = 0;
    for (auto i = 0; i < x.size(); ++i) {
        sum1 += (i+1)*(i+1)*x[i]*x[i];
    };
    return sum1;
}

/// A sample problem that is very easy plus a strong penalty to force the values to be integers
/// The solution is x[i] = i
double IntTest(const std::vector<double> &x) {
    double sum1 = 0, sum2 = 0;
    for (auto i = 0; i < x.size(); ++i) {
        sum1 += (x[i]-i)*(x[i]-i);
        auto q = x[i]-floor(x[i]);
        sum2 += q*(1-q);
    };
    return sum1+100000*sum2;
}

class StornTest{
protected:
    std::function<double(const std::vector<double>&)> m_f;
    int m_NP;
    double m_F, m_CR, m_VTR;
    std::vector<CEGO::Bound> m_bounds;
    
public:
    int Ncalls = 0;
    StornTest(std::function<double(const std::vector<double>&)> &f, int NP, double F, double CR, double VTR, std::vector<CEGO::Bound> bounds)
        : m_f(f), m_NP(NP), m_F(F), m_CR(CR), m_VTR(VTR), m_bounds(bounds)
    {}
    double objective(const CEGO::AbstractIndividual *pind) {
        Ncalls++;
        const std::vector<double> &c = static_cast<const CEGO::NumericalIndividual<double>*>(pind)->get_coefficients();
        return m_f(c);
    }
    void run(){
        Ncalls = 0;
        CEGO::CostFunction<double> cost_wrapper = std::bind((double (StornTest::*)(const CEGO::AbstractIndividual *)) &StornTest::objective, this, std::placeholders::_1);
        auto Npop_size = m_NP * m_bounds.size();
        short Nlayers = 1;
        auto layers = CEGO::Layers<double>(cost_wrapper, m_bounds.size(), Npop_size, Nlayers, 5000000000000);
        layers.parallel = false;
        layers.parallel_threads = 1;
        layers.set_bounds(m_bounds);
        layers.set_builtin_evolver(CEGO::BuiltinEvolvers::differential_evolution);

        auto flags = layers.get_evolver_flags();
        flags["Nelite"] = 0;
        flags["Fmin"] = m_F;
        flags["Fmax"] = m_F;
        flags["CR"] = m_CR;
        layers.set_evolver_flags(flags);

        for (auto counter = 0; counter < 50000; ++counter) {
            layers.do_generation();
            auto best_cost = std::get<0>(layers.get_best());
            auto best_coeffs = std::get<1>(layers.get_best());
            if (counter % 5 == 0) {
                std::cout << counter << ": best: " << best_cost << "\n ";
                //std::cout << Eigen::Map<const Eigen::ArrayXd>(&(best_coeffs[0]), best_coeffs.size()) << std::endl;
            }
            if (best_cost < m_VTR) { break; }
        }

        auto best_cost = std::get<0>(layers.get_best());
        auto best_coeffs = std::get<1>(layers.get_best());
        std::cout << "best: " << best_cost << "\n ";
        std::cout << Eigen::Map<const Eigen::ArrayXd>(&(best_coeffs[0]), best_coeffs.size()) << std::endl;
    }
};

int main() {
    {
        // Integer problem #1
        std::function<double(const std::vector<double> &)> f = IntTest;
        StornTest IntTest(f, 10, 0.5, 0, 1e-16, std::vector<CEGO::Bound>(10, CEGO::Bound(std::pair<double, double>(-1000, 1000))));
        IntTest.run();
        std::cout << IntTest.Ncalls << std::endl;
    }
    { 
        // Test problem 6
        std::function<double(const std::vector<double> &)> f = Corana;
        StornTest CoranaTest(f, 10, 0.5, 0, 1e-6, std::vector<CEGO::Bound>(4, CEGO::Bound(std::pair<double, double>(-1000, 1000))));
        CoranaTest.run();
        std::cout << CoranaTest.Ncalls << std::endl;
    } 
    {
        // Test problem 7
        std::function<double(const std::vector<double> &)> f = Griewangk<double>;
        StornTest GriewangkTest(f, 25, 0.5, 0.2, 1e-6, std::vector<CEGO::Bound>(10, CEGO::Bound(std::pair<double, double>(-400, 400)) ));
        GriewangkTest.run();
        std::cout << GriewangkTest.Ncalls << std::endl;
    }
    {
        // Test problem 11, two variants
        std::function<double(const std::vector<double> &)> f = HyperEllipsoid;
        for (int D : {30, 100}){
            StornTest HyperEllipsoidTest(f, 20, 0.5, 0.1, 1e-10, std::vector<CEGO::Bound>(D, CEGO::Bound(std::pair<double, double>(-1, 1))));
            HyperEllipsoidTest.run();
            std::cout << HyperEllipsoidTest.Ncalls << std::endl;
        }
    }
}