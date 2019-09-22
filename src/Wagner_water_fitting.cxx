#include "CEGO/CEGO.hpp"
#include "CEGO/minimizers.hpp"
#include <Eigen/Dense>

// autodiff include
#include "Eigen/Dense"
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

std::atomic_size_t Ncalls(0);

template <typename T> using EArray = Eigen::Array<T, Eigen::Dynamic, 1>;

constexpr double Tt = 273.16;
constexpr double Tc = 647.096;

const std::vector<double> b = { 1.99274064, 1.09965342, -0.510839303, -1.75493479, -45.5170352, -6.746944503e5 };
const std::vector<double> c = { 1 / 3, 2 / 3, 5 / 3, 16 / 3, 43 / 3, 110 / 3 };

template <class T, class... Ts> struct is_any : std::disjunction<std::is_same<T, Ts>...> {};

template <typename T, typename B, typename C>
auto evaluate_RHS(const T &theta, const B &b, const C &c) { // theta = 1-T/Tc
    T o = 1.0 + 0*theta;
    for (auto i = 0; i < b.size(); ++i) {
        o += b[i] * pow(theta, c[i]);
    }
    return o;
}

auto evaluate_RHS(const EArray<double> &theta, const EArray<std::complex<double>> &b, const EArray<std::complex<double>> &c) { // theta = 1-T/Tc
    EArray<std::complex<double>> o = 1.0 + 0*theta;
    for (auto i = 0; i < b.size(); ++i) {
        o += b[i] * theta.pow(c[i]);
    }
    return o;
}

template <typename Coeffs, typename Theta>
auto eval_fit(const Coeffs &coeffs, const Theta &theta) {
    if constexpr (is_any<Coeffs, std::vector<double>, std::vector<std::complex<double>>, std::vector<CEGO::numberish>>::value) {
        Coeffs b(coeffs.begin(), coeffs.begin() + coeffs.size() / 2);
        Coeffs c(coeffs.begin() + coeffs.size() / 2, coeffs.end());
        return evaluate_RHS(theta, b, c);
    }
    else {
        Coeffs b = coeffs.head(coeffs.size() / 2),
            c = coeffs.tail(coeffs.size() / 2);
        return evaluate_RHS(theta, b, c);
    }
}

template <typename Coeffs, typename Theta, typename Yval>
auto objective(const Coeffs& coeffs, const Theta &theta, const Yval &yval) {
    return (eval_fit(coeffs, theta) - yval).square().sum();
};

template <typename Theta, typename Yval>
auto objective(const CEGO::AbstractIndividual* pind, const Theta& theta, const Yval& yval) {
    const std::vector<CEGO::numberish>& coeffs = static_cast<const CEGO::NumericalIndividual<CEGO::numberish>*>(pind)->get_coefficients();
    return objective(coeffs, theta, yval);
};

struct Inputs{
    std::string root = "";
    std::size_t parallel_threads = 1;
    std::vector<std::size_t> Nlayersvec = { 1 };
    std::size_t i = 0;
    std::size_t gradmin_mod = 5;
    std::size_t Nmax_gradient = 5;
};

inline void to_json(nlohmann::json& j, const Inputs& f) {
    j = nlohmann::json{ { "root", f.root },{ "parallel_threads", f.parallel_threads },{"i",f.i},{"gradmin_mod",f.gradmin_mod},{"Nmax_gradient",f.Nmax_gradient} };
}

inline void from_json(const nlohmann::json& j, Inputs& f) {
    j.at("root").get_to(f.root);
    j.at("parallel_threads").get_to(f.parallel_threads);
    j.at("i").get_to(f.i);
    j.at("gradmin_mod").get_to(f.gradmin_mod);
    j.at("Nmax_gradient").get_to(f.Nmax_gradient);
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

void do_one(Inputs& inputs)
{
    std::srand((unsigned int)time(0));

    EArray<double> theta = 1 - EArray<double>::LinSpaced(1000, Tt, Tc) / Tc;
    EArray<double> yval = evaluate_RHS(theta, b, c);

    // Construct the bounds
    std::vector<CEGO::Bound> bounds;
    for (auto i = 0; i < b.size(); ++i) {
        double v0 = 0.1 * b[i], v1 = 10 * b[i];
        bounds.push_back(CEGO::Bound(std::make_pair(std::min(v0, v1), std::max(v0, v1))));
    }
    for (auto i = 0; i < c.size(); ++i) {
        double v0 = 0.1 * c[i], v1 = 4 * c[i];
        bounds.push_back(CEGO::Bound(std::make_pair(std::min(v0, v1), std::max(v0, v1))));
    }

    CEGO::CostFunction<CEGO::numberish> cost_wrapper = [&theta, &yval](const CEGO::AbstractIndividual* pind) {return objective(pind, theta, yval); };
    auto Npop_size = 15*bounds.size();
    auto Nlayers = 1;
    auto layers = CEGO::Layers<CEGO::numberish>(cost_wrapper, bounds.size(), Npop_size, Nlayers, 5);
    layers.parallel = (inputs.parallel_threads > 1);
    layers.parallel_threads = inputs.parallel_threads;
    layers.set_builtin_evolver(CEGO::BuiltinEvolvers::differential_evolution);
    layers.set_bounds(bounds);

    auto flags = layers.get_evolver_flags();
    flags["Nelite"] = 3;
    flags["Fmin"] = 0.1;
    flags["Fmax"] = 1.0;
    flags["CR"] = 1;
    layers.set_evolver_flags(flags);

    std::vector<double> best_costs; 
    std::vector<std::vector<double> > objs;
    double VTR = 1e-16, best_cost = 999999.0;
    auto startTime = std::chrono::system_clock::now();
    for (auto counter = 0; counter < 5000; ++counter) {
        layers.do_generation();

        // Do a single step of gradient minimization for all individuals
        auto minimizer = [&layers, &theta, &yval, &inputs](const CEGO::pIndividual &pind) {
            auto bounds = layers.get_bounds();
            Eigen::ArrayXd ub(bounds.size()), lb(bounds.size()), xnew(bounds.size());
            auto i = 0;
            for (auto& b : layers.get_bounds()) {
                ub(i) = b.m_upper;
                lb(i) = b.m_lower;
                i++;
            }
            // Store current values
            auto current_cost = pind->get_cost();

            auto ind = static_cast<CEGO::NumericalIndividual<CEGO::numberish>*>(pind.get());
            const std::vector<CEGO::numberish>& c = ind->get_coefficients();
            Eigen::ArrayXd x0(c.size());
            for (auto i = 0; i < c.size(); ++i) {
                x0[i] = c[i];
            }

            // Try to minimize with gradient
            //
            // The objective function to be minimized
            CEGO::DoubleObjectiveFunction obj = [&layers, &theta, &yval](const CEGO::EArray<double>& c) -> double {
                return objective(c, theta, yval); 
            };
            auto F0 = obj(x0);
            // The gradient function used to do gradient optimization
            CEGO::DoubleGradientFunction g = CEGO::ComplexStepGradient(
                [&layers, &theta, &yval](const CEGO::EArray<std::complex<double>>& c) -> std::complex<double>{
                    return objective(c, theta, yval);
                }
            );
            auto g0 = g(x0);
            double F;
            CEGO::BoxGradientFlags flags;
            flags.Nmax = inputs.Nmax_gradient;
            flags.VTR = 1e-16;
            std::tie(xnew, F) = CEGO::box_gradient_minimization(obj, g, x0, lb, ub, flags);
            if (F < F0) {
                std::vector<CEGO::numberish> cnew;
                for (auto i = 0; i < xnew.size(); ++i){
                    cnew.push_back(xnew(i));
                }
                ind->set_coefficients(cnew);
                ind->calc_cost();
            }
            else {
                //std::cout << "no reduction\n";
            }
        };
        if (counter % inputs.gradmin_mod == 0 && counter > 0) {
            layers.transform_individuals(minimizer);
        }

        // Store the best objective function in each layer
        std::vector<double> oo;
        for (auto &&cost_coefficients : layers.get_best_per_layer()) {
            oo.push_back(std::get<0>(cost_coefficients));
        }
        objs.push_back(oo);
        auto stats = layers.cost_stats_each_layer();

        // For the overall best result, print it, and write JSON to file
        auto best_layer = layers.get_best();
        auto best_coeffs = std::get<1>(best_layer);
        best_cost = std::get<0>(best_layer); best_costs.push_back(best_cost);
        if (counter % 50 == 0) {
            std::cout << counter << ": best: " << best_cost << "\n ";
            //std::cout << bumps.to_realworld(best_coeffs//)-bumps.c0 << "\n ";// << CEGO::vec2string(bumps.c0) << "\n";
        }
        if (best_cost < VTR){ break; }
    }
    auto endTime = std::chrono::system_clock::now();
    double elap = std::chrono::duration<double>(endTime - startTime).count();
    std::cout << "run:" << elap << " s\n";
    std::cout << "NFE:" << Ncalls << std::endl;
}

int main() {
    Inputs in;
    in.root = "shaped-";
    in.Nlayersvec = {1};
    auto Nrepeats = get_env_int("NREPEATS", 1);
    in.gradmin_mod = get_env_int("GRADMOD", 100);
    in.parallel_threads = get_env_int("NTHREADS", 4);
    in.Nmax_gradient = get_env_int("NMAX_gradient", 5);
    nlohmann::json j = in;
    std::cout << j << std::endl;
    for (in.i = 0; in.i < Nrepeats; ++in.i) {
        do_one(in);
    }
}