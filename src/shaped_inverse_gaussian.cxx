#include "CEGO/CEGO.hpp"
#include "CEGO/minimizers.hpp"
#include <Eigen/Dense>

// autodiff include
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

#if defined(PYBIND11)
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
namespace py = pybind11;
#endif

std::atomic_size_t Ncalls(0);

template <typename T> using EArray = Eigen::Array<T, Eigen::Dynamic, 1>;

class Bumps {
public:
    std::size_t Nbumps;
    Eigen::ArrayXd c0, xp, yp, zp;
    double gamma = 10;
    const std::vector<CEGO::Bound> m_bounds;
    
    Bumps(std::size_t Nbumps, std::size_t Npoints, const std::vector<CEGO::Bound> &bounds) : Nbumps(Nbumps), m_bounds(bounds)
    {
        // Initialize the random number generator
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

        // Calculate the initial set of coefficients for the bump characteristics
        c0.resize(Nbumps * 6); 
        for (auto i = 0; i < bounds.size(); ++i) {
            double dbl; int integer;
            bounds[i].gen_uniform(gen, dbl, integer);
            if (bounds[i].m_lower.type == bounds[i].m_lower.DOUBLE){
                c0[i] = dbl;
            }
            else {
                c0[i] = static_cast<double>(integer);
            }
        }

        // Generate some random points in the domain [0.1,1.0] for both variables
        xp = (1-0.1)*(Eigen::ArrayXd::Random(Npoints)+1)/2 + 0.1;
        yp = (1-0.1)*(Eigen::ArrayXd::Random(Npoints)+1)/2 + 0.1;
        zp = f_givenxy(c0, xp, yp);

        double checkval = objective(to_scaled(c0));
        std::cout << "c0: " << c0 << std::endl; 
        
        assert(std::abs(checkval) < 1e-16);
        if (std::abs(checkval) > 1e-16) {
            throw std::invalid_argument("Did not start out with zero objective function!");
        }
    }
    /**
     * @brief Calculate the functional value for a set of vectors of points
     * @brief xb The x coordinate of the center of the bump
     * @brief x The x coordinate of the points to be evaluated
     * @brief y The y coordinate of the points to be evaluated
     */
    template <typename T>
    EArray<T> f_givenxy(const EArray<T> &c, const EArray<T> &x, const EArray<T> &y) 
    {
        Ncalls++;
        EArray<T> s = EArray<T>::Zero(x.size());
        auto chunksize = 6; 
        assert(c.size()%chunksize==0);
        for (long i = 0; i < c.size(); i += chunksize) {
            s += x.pow(c[i+0])*y.pow(c[i+1])*(c[i+2]*(x-c[i+3]).square() +c[i+4]*(y-c[i+5]).square()).exp();
        }
        return s.eval();
    }
    double objective(const CEGO::AbstractIndividual *pind) {
        const std::vector<CEGO::numberish> &c = static_cast<const CEGO::NumericalIndividual<CEGO::numberish>*>(pind)->get_coefficients();
        Eigen::ArrayXd cc(c.size());
        for (auto i = 0; i < cc.size(); ++i) {
            cc[i] = c[i];
        }
        return objective(cc);
    }
    double objective(const EArray<double>& cscaled) {
        return (f_givenxy<double>(to_realworld<double>(cscaled), xp, yp) - zp).square().sum();
    }
    std::complex<double> objective(const EArray<std::complex<double>>& cscaled) {
        return (f_givenxy<std::complex<double>>(to_realworld<std::complex<double>>(cscaled), xp, yp) - zp).square().sum();
    }

    autodiff::dual objective(const EArray<autodiff::dual>& cscaled) {
        EArray<autodiff::dual> creal = to_realworld(cscaled);
        EArray<autodiff::dual> zmodel = f_givenxy<autodiff::dual>(creal, xp, yp);
        EArray<autodiff::dual> err = zmodel - zp.cast<autodiff::dual>();
        return err.square().sum();
    }

    // Inspired by scipy, keep all variables scaled in 0,1
    template <typename T>
    EArray<T> to_realworld(const EArray<T>&x){
        EArray<T> o(x.size());
        for (auto i = 0; i < o.size(); ++i){
            T lower = static_cast<T>(m_bounds[i].m_lower);
            T upper = static_cast<T>(m_bounds[i].m_upper);
            o[i] = lower*(static_cast<T>(1.0) - x[i]) + upper*x[i];
        }
        return o.eval();
    }
    Eigen::ArrayXd to_scaled(const Eigen::ArrayXd &x) {
        Eigen::ArrayXd o(x.size());
        for (auto i = 0; i < o.size(); ++i) {
            double lower = static_cast<double>(m_bounds[i].m_lower);
            double upper = static_cast<double>(m_bounds[i].m_upper);
            o[i] = (x[i]-lower)/(upper-lower);
        }
        return o;
    }
       
    void plot_surface() {
        #if defined(PYBIND11)
        using namespace pybind11::literals;
        py::module plt = py::module::import("matplotlib.pyplot"); // Import matplotlib
        std::size_t Nx = 100, Ny = 100;
        Eigen::MatrixXd X = Eigen::RowVectorXd::LinSpaced(Nx, 0.1, 1).replicate(Ny, 1);
        Eigen::MatrixXd Y = Eigen::VectorXd::LinSpaced(Ny, 0.1, 1).replicate(Nx, 1);
        X.resize(Nx*Ny,1); Y.resize(Nx*Ny,1);
        Eigen::MatrixXd Z = f_givenxy(c0, X.array(), Y.array()).matrix();
        X.resize(Nx, Ny); Y.resize(Nx, Ny); Z.resize(Nx, Ny);
        
        Eigen::ArrayXd levels = Eigen::ArrayXd::LinSpaced(300, Z.minCoeff(), Z.maxCoeff());
        try{
            plt.attr("contourf")(X, Y, Z, levels);
        }
        catch (std::exception &e) {
            std::cout << e.what()  << std::endl;
        }
        plt.attr("colorbar")(); 
        plt.attr("scatter")(xp, yp);
        plt.attr("show")();
        #else
        std::cout << "No support for pybind11, so no plots\n";
        #endif
    }
    void plot_trace(const std::vector<double> &best_costs) {
        #if defined(PYBIND11)
        py::module plt = py::module::import("matplotlib.pyplot"); // Import matplotlib
        plt.attr("plot")(best_costs);
        plt.attr("show")();
        #else
        std::cout << "No support for pybind11, so no plots\n";
        #endif
    }
};

struct BumpsInputs{
    std::string root;
    std::size_t parallel_threads;
    std::size_t Nbumps;
    std::vector<std::size_t> Nlayersvec;
    std::size_t i;
};

void do_one(BumpsInputs &inputs)
{
    std::srand((unsigned int)time(0));

    // Construct the bounds
    std::size_t Npoints = inputs.Nbumps*6*10;
    std::vector<CEGO::Bound> bounds;
    for (auto i = 0; i < inputs.Nbumps; ++i) {
        bounds.push_back(CEGO::Bound(std::pair<double, double>(0.1, 1))); // ex
        bounds.push_back(CEGO::Bound(std::pair<double, double>(1, 3))); // ey
        bounds.push_back(CEGO::Bound(std::pair<double, double>(-50, -10))); // gx
        bounds.push_back(CEGO::Bound(std::pair<double, double>(0.1, 1))); // xb
        bounds.push_back(CEGO::Bound(std::pair<double, double>(-50, -10))); // gy
        bounds.push_back(CEGO::Bound(std::pair<double, double>(0.1, 1))); // yb
    } 

    // Normalized bounds in [0,1]
    std::vector<CEGO::Bound> nbounds;
    for (auto i = 0; i < inputs.Nbumps; ++i) {
        nbounds.push_back(CEGO::Bound(std::pair<double, double>(0, 1))); // ex
        nbounds.push_back(CEGO::Bound(std::pair<double, double>(0, 1))); // ey
        nbounds.push_back(CEGO::Bound(std::pair<double, double>(0, 1))); // gx
        nbounds.push_back(CEGO::Bound(std::pair<double, double>(0, 1))); // xb
        nbounds.push_back(CEGO::Bound(std::pair<double, double>(0, 1))); // gy
        nbounds.push_back(CEGO::Bound(std::pair<double, double>(0, 1))); // yb
    }
    
    Bumps bumps(inputs.Nbumps, Npoints, bounds);
    //bumps.plot_surface();

    for (auto Nlayers : inputs.Nlayersvec){
        Ncalls = 0;
        CEGO::CostFunction<CEGO::numberish> cost_wrapper = std::bind((double (Bumps::*)(const CEGO::AbstractIndividual *)) &Bumps::objective, bumps, std::placeholders::_1);
        auto Npop_size = 15*bounds.size();
        auto layers = CEGO::Layers<CEGO::numberish>(cost_wrapper, bounds.size(), Npop_size, Nlayers, 5);
        layers.parallel = (inputs.parallel_threads > 1);
        layers.parallel_threads = inputs.parallel_threads;
        layers.set_builtin_evolver(CEGO::BuiltinEvolvers::differential_evolution);
        layers.set_bounds(nbounds);

        auto flags = layers.get_evolver_flags();
        flags["Nelite"] = 1;
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
            auto minimizer = [&layers, &bumps](const CEGO::pIndividual &pind) {
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
                CEGO::DoubleObjectiveFunction obj = [&layers, &bumps](const CEGO::EArray<double>& c) -> double {
                    return bumps.objective(c);
                };
                auto F0 = obj(x0);
                // The gradient function that will be used to do gradient optimization
                CEGO::DoubleGradientFunction g = CEGO::ComplexStepGradient(
                    [&layers, &bumps](const CEGO::EArray<std::complex<double>>& c) {
                    return bumps.objective(c);
                    }
                );
                auto g0 = g(x0);
                double F;
                std::tie(xnew, F) = CEGO::gradient_linesearch(obj, g, x0, lb, ub);
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
            if (counter % 100 == 0) {
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

        //bumps.plot_trace(best_costs);
        std::string fname = inputs.root + "Nbumps"+std::to_string(inputs.Nbumps)+"-Nlayers"+std::to_string(Nlayers) + "-run" + std::to_string(inputs.i) + ".txt";
        FILE* fp = fopen(fname.c_str(), "w");
        for (auto j = 0; j < best_costs.size(); ++j){
            fprintf(fp, "%12.8e", best_costs[j]);
            if (j < best_costs.size() - 1) {
                fprintf(fp, ", ");
            }
        }
        fclose(fp);
        /*std::cout << bumps.xb0 << std::endl;
        std::cout << bumps.yb0 << std::endl;*/
        std::cout << "NFE:" << Ncalls << std::endl;
    }
}

int main() {
    #if defined(PYBIND11)
    py::scoped_interpreter interp{};
    #endif
    BumpsInputs in;
    in.root = "shaped-";
    in.Nlayersvec = {1};
    for (in.parallel_threads = 4; in.parallel_threads <= 4; in.parallel_threads *= 2){
        for (in.Nbumps = 5; in.Nbumps < 6; ++in.Nbumps){
            std::cout << "Nbumps: " << in.Nbumps << std::endl;
            for (in.i = 0; in.i < 1; ++in.i) {
                do_one(in);
            }
        }
    }
}
