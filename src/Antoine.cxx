#include "CEGO/concurrentqueue.h"
#include "CEGO/CEGO.hpp"
#include <Eigen/Dense>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
namespace py = pybind11;
#include <atomic>
#include <fstream>

std::atomic_size_t Ncalls(0);

inline bool ValidNumber(double x)
{
    // Idea from http://www.johndcook.com/IEEE_exceptions_in_cpp.html
    return (x <= DBL_MAX && x >= -DBL_MAX);
};

class Antoine {
public:
    double m_Tt, m_Tc, m_pc, m_Dc;
    Eigen::ArrayXd m_LHS, m_T;
    std::string m_name;
    Antoine(const std::string &name) :  m_name(name)
    {
        py::module ct = py::module::import("ctREFPROP.ctREFPROP"); // Import cytpes wrapper of REFPROP dll
        std::string RPPREFIX(getenv("RPPREFIX")); 
        auto R = ct.attr("REFPROPFunctionLibrary")(RPPREFIX + "/REFPRP64.DLL");
        auto version = R.attr("RPVersion")().cast<std::string>();
        auto setup_info = R.attr("SETUPdll")(1, name+".FLD", "HMX.BNC","DEF");
        if (setup_info.attr("ierr").cast<int>() != 0) {
            throw std::invalid_argument( setup_info.attr("herr").cast<std::string>() );
        }
        auto info = R.attr("INFOdll")(1);
        m_pc = info.attr("Pc").cast<double>();
        m_Tc = info.attr("Tc").cast<double>();
        m_Tt = info.attr("Ttrp").cast<double>();
        m_Dc = info.attr("Dc").cast<double>();

        std::size_t N = 300;
        m_LHS.resize(N);
        m_T.resize(N);
        std::vector<double> z(1,1);
        Eigen::ArrayXd Tvec = Eigen::ArrayXd::LinSpaced(N, m_Tc*0.999, 0.99*m_Tt);
        Eigen::Index j = 0;

        for (auto i = 0; i < Tvec.size(); ++i){
            auto o = R.attr("TQFLSHdll")(Tvec[i], 1, &z, 0);
            int ierr = o.attr("ierr").cast<int>();
            
            if (ierr <= 100){
                double p = o.attr("P").cast<double>();
                if (!ValidNumber(p)){ continue; }
                m_LHS(j) = p;
            }
            m_T(j) = Tvec(i);
            j++;
        }
        m_LHS.conservativeResize(j);
        m_T.conservativeResize(j);
    }
    void plot_curve(){
        py::module plt = py::module::import("matplotlib.pyplot"); // Import matplotlib
        plt.attr("plot")(1/m_T, m_LHS);
        plt.attr("show")();
    }
    Eigen::ArrayXd eval_RHS(const Eigen::ArrayXd& T, const Eigen::ArrayXd &c) {
        return c[3]*pow(10,c[0]-c[1]/(c[2]+T));
    }
    double objective(const CEGO::AbstractIndividual *pind) {
        const std::vector<double> &c = static_cast<const CEGO::NumericalIndividual<double>*>(pind)->get_coefficients();
        return objective(Eigen::Map<const Eigen::ArrayXd>(&(c[0]), c.size()) );
    }
    double objective(const Eigen::ArrayXd &c) {
        Ncalls++;
        double ssq = ((eval_RHS(m_T, c) - m_LHS)/m_LHS).square().sum();
        return ssq;
    }
    void plot_trace(const std::vector<double> &best_costs) {
        py::module plt = py::module::import("matplotlib.pyplot"); // Import matplotlib
        plt.attr("plot")(best_costs);
        plt.attr("show")();
    }
    const Eigen::ArrayXd &get_T(){ return m_T; }
};

int main()
{
    py::scoped_interpreter interp{};
    std::srand((unsigned int)time(0));

    // Construct the bounds
    std::vector<CEGO::Bound> bounds;
    for (auto i = 0; i < 3; ++i) { 
        bounds.push_back(CEGO::Bound(std::pair<double, double>(-10000, 10000)));
    }
    for (auto i = 0; i < 1; ++i) {
        bounds.push_back(CEGO::Bound(std::pair<double, double>(0.1,100)));
    }
    Antoine rp("PROPANE");
    //rp.plot_curve();
   
    CEGO::CostFunction cost_wrapper = std::bind((double (Antoine::*)(const CEGO::AbstractIndividual *)) &Antoine::objective, &rp, std::placeholders::_1);
    auto Nlayers = 5;
    auto layers = CEGO::Layers<double>(cost_wrapper, bounds.size(), 2000, Nlayers, 3);
    layers.parallel = false;
    layers.set_bounds(bounds);

    auto flags = layers.get_evolver_flags();
    flags["Nelite"] = 2;
    flags["Fmin"] = 0.5;
    flags["Fmax"] = 1.0;
    flags["CR"] = 0.7;
    layers.set_evolver_flags(flags);

    std::vector<double> best_costs; 
    std::vector<std::vector<double> > objs;
    double VTR = 1e-6, best_cost = 999999.0;
    auto startTime = std::chrono::system_clock::now();
    for (auto counter = 0; counter < 3000; ++counter) {
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
        if (counter % 50 == 0) {
            std::cout << counter << ": best: " << best_cost << "\n ";// << CEGO::vec2string(best_coeffs) << "\n";
        }
        if (best_cost < VTR){ break; }
    }
    auto best_layer = layers.get_best();
    auto c = std::get<1>(best_layer);
    auto endTime = std::chrono::system_clock::now();
    double elap = std::chrono::duration<double>(endTime - startTime).count();

    // Get the results stored in the thread-safe queue
    std::vector<CEGO::Result> results; Eigen::MatrixXd mat;
    results = layers.get_results();

    std::cout << "cbest" << Eigen::Map<Eigen::ArrayXd>(&(c[0]), c.size()) << std::endl;
        
    std::cout << "run:" << elap << " s\n";
    std::cout << "NFE:" << Ncalls << std::endl;
}