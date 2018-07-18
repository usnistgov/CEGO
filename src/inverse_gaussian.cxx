#include "CEGO/CEGO.hpp"
#include <Eigen/Dense>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
namespace py = pybind11;
#include <atomic>

std::atomic_size_t Ncalls(0);

class Bumps {
public:
    std::size_t Nbumps;
    Eigen::ArrayXd xb0, yb0, xp, yp, zp;
    double gamma = 5;
    
    Bumps(std::size_t Nbumps, std::size_t Npoints) : Nbumps(Nbumps) {
        xb0 = (Eigen::ArrayXd::Random(Nbumps)*10).round();
        yb0 = (Eigen::ArrayXd::Random(Nbumps)*10).round();
        
        xp = Eigen::ArrayXd::Random(Npoints)*10;
        yp = Eigen::ArrayXd::Random(Npoints)*10;
        zp = f_givenxy(xb0, yb0, xp, yp);

        Eigen::ArrayXd c = xb0; c.conservativeResize(xb0.size() * 2); c.tail(Nbumps) = yb0;
        double checkval = objective(c);
        assert(std::abs(checkval) < 1e-16);
    }
    /**
     * @brief Calculate the functional value for a set of vectors of points
     * @brief xb The x coordinate of the center of the bump
     * @brief yb The y coordinate of the center of the bump
     * @brief x The x coordinate of the points to be evaluated
     * @brief y The y coordinate of the points to be evaluated
     */
    Eigen::ArrayXd f_givenxy(const Eigen::ArrayXd &xb, const Eigen::ArrayXd &yb, const Eigen::ArrayXd&x, const Eigen::ArrayXd &y) {
        Eigen::ArrayXd s = Eigen::ArrayXd::Zero(x.size());
        for (auto i = 0; i < xb.size(); ++i) {
            s += (-gamma*(x-xb[i]).square() - gamma*(y-yb[i]).square()).exp();
        }
        return s;
    }
    double objective(const CEGO::AbstractIndividual *pind) {
        const std::vector<double> &c = static_cast<const CEGO::NumericalIndividual<double>*>(pind)->get_coefficients();
        return objective(Eigen::Map<const Eigen::ArrayXd>(&(c[0]), c.size()) );
    }
    double objective_vec(const std::vector<double> &c) {
        return objective(Eigen::Map<const Eigen::ArrayXd>(&(c[0]), c.size()));
    }
    double penalty_vec(const std::vector<double> &c) {
        return penalty(Eigen::Map<const Eigen::ArrayXd>(&(c[0]), c.size()));
    }
    double penalty(const Eigen::ArrayXd &c) {
        Eigen::ArrayXd q = c - c.floor();
        constexpr double MY_PI = 3.14159265358979323846;
        Eigen::ArrayXd penalty = 0.25*(0.5 - 0.5*cos(2*MY_PI*q));
        return 100*penalty.sum();
    }
    double objective(const Eigen::ArrayXd &c) {
        Ncalls ++;
        return (f_givenxy(c.head(Nbumps), c.tail(Nbumps), xp, yp) - zp).square().sum() + penalty(c);
    }
    void plot_surface() {
        using namespace pybind11::literals;
        py::module plt = py::module::import("matplotlib.pyplot"); // Import matplotlib
        std::size_t Nx = 100, Ny = 100;
        Eigen::MatrixXd X = Eigen::RowVectorXd::LinSpaced(Nx, -1, 1).replicate(Ny, 1);
        Eigen::MatrixXd Y = Eigen::VectorXd::LinSpaced(Ny, -1, 1).replicate(Nx, 1);
        X.resize(Nx*Ny,1); Y.resize(Nx*Ny,1);
        Eigen::MatrixXd Z = f_givenxy(xb0, yb0, X.array(), Y.array()).matrix();
        X.resize(Nx, Ny); Y.resize(Nx, Ny); Z.resize(Nx, Ny);
        plt.attr("contourf")(X, Y, Z, "N"_a=3000);
        plt.attr("scatter")(xp, yp);
        plt.attr("show")();
    }
    void plot_trace(const std::vector<double> &best_costs) {
        py::module plt = py::module::import("matplotlib.pyplot"); // Import matplotlib
        plt.attr("plot")(best_costs);
        plt.attr("show")();
    }
};

void do_one(const std::size_t Nbumps, const std::string &root, std::size_t i) {

    std::srand((unsigned int)time(0));

    std::size_t Npoints = Nbumps*10;
    Bumps bumps(Nbumps, Npoints);
    //bumps.plot_surface();

    auto D = 2*Nbumps; // twice because they are pairs
    CEGO::CostFunction cost_wrapper = std::bind((double (Bumps::*)(const CEGO::AbstractIndividual *)) &Bumps::objective, bumps, std::placeholders::_1);
    auto layers = CEGO::Layers<double>(cost_wrapper, D, 30*D, 1, 10);
    // Apply the bounds (all in [-1,1])
    layers.set_bounds(std::vector<CEGO::Bound>(D, CEGO::Bound(std::pair<double, double>(-15.0, 15.0))));
    layers.parallel = true;
    layers.parallel_threads = 4;
    layers.set_builtin_evolver(CEGO::BuiltinEvolvers::differential_evolution); 
    auto fl = layers.get_evolver_flags();
    fl["CR"] = 0.9;
    fl["Fmin"] = 0.5; 
    fl["Fmax"] = 0.5;
    fl["Nelite"] = 1;
    layers.set_evolver_flags(fl);

    auto f = [](const CEGO::Result &r) { 
        if (r.ssq < 1) { 
            return CEGO::FilterOptions::accept; 
        }
        else {
            return CEGO::FilterOptions::reject;
        }
    };
    layers.set_filtering_function(f);
    layers.set_logging_scheme(CEGO::LoggingScheme::custom);

    std::vector<double> best_costs; 
    std::vector<std::vector<double> > objs;
    double VTR = 1e-14, best_cost = 999999.0;
    auto startTime = std::chrono::system_clock::now();
    for (auto counter = 0; counter < 500000; ++counter) {
        layers.do_generation();

        // Store the best objective function in each layer
        std::vector<double> oo;
        for (auto &&cost_coefficients : layers.get_best_per_layer()) {
            oo.push_back(std::get<0>(cost_coefficients));
        }
        objs.push_back(oo);

        // For the overall best result, print it, and write JSON to file
        auto best_layer = layers.get_best();
        best_cost = std::get<0>(best_layer); best_costs.push_back(best_cost);
        auto best_coeffs = std::get<1>(best_layer);
        if (counter % 10 == 0){
            std::cout << counter << ": best-penalty: " << best_cost-bumps.penalty_vec(best_coeffs) << "; best: " << best_cost << "\n ";// << CEGO::vec2string(best_coeffs) << "\n";
        }
        if (best_cost < VTR){ break; }
    }
    auto endTime = std::chrono::system_clock::now();
    double elap = std::chrono::duration<double>(endTime - startTime).count();
    std::cout << "run:" << elap << " s\n";

    //bumps.plot_trace(best_costs);
    FILE* fp = fopen((root + "run-" + std::to_string(i) + ".txt").c_str(), "w");
    for (auto j = 0; j < best_costs.size(); ++j){
        fprintf(fp, "%12.8e", best_costs[j]);
        if (j < best_costs.size() - 1) {
            fprintf(fp, ", ");
        }
    }
    fclose(fp);
    std::cout << bumps.xb0 << std::endl;
    std::cout << bumps.yb0 << std::endl;
    std::cout << "NFE:" << Ncalls << std::endl;
}

int main() {
    for (auto i = 0; i < 10; ++i) {
        do_one(5, "N5-", i);
    }
    for (auto i = 0; i < 10; ++i) {
        do_one(10, "N10-", i);
    }
}
