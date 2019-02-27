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

inline bool ValidNumber(double x)
{
    // Idea from http://www.johndcook.com/IEEE_exceptions_in_cpp.html
    return (x <= DBL_MAX && x >= -DBL_MAX);
};

enum fitting_options{ FIT_P, FIT_RHOV, FIT_RHOL};

class RatPolyAncillary {
public:
    double m_Tt, m_Tc, m_pc, m_Dc;
    std::size_t m_Nnum, m_Nden;
    Eigen::ArrayXd m_LHS, m_THETA, m_T;
    std::string m_name;
    fitting_options to_fit = FIT_RHOV;
    RatPolyAncillary(const std::string &name, const std::size_t Nnum, const std::size_t Nden) : m_Nnum(Nnum), m_Nden(Nden), m_name(name)
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
        m_THETA.resize(N);
        m_T.resize(N);
        std::vector<double> z(1,1);
        Eigen::ArrayXd Tvec = Eigen::ArrayXd::LinSpaced(N, m_Tc*0.999, 0.5*m_Tt);
        Eigen::Index j = 0;

        for (auto i = 0; i < Tvec.size(); ++i){
            auto o = R.attr("TQFLSHdll")(Tvec[i], 1, &z, 0);
            int ierr = o.attr("ierr").cast<int>();
            if (ierr <= 100){
                //std::cout << Tvec(i) << " " << o << std::endl;
                if (to_fit == FIT_P){
                    double p = o.attr("P").cast<double>();
                    if (!ValidNumber(p)){ continue; }
                    m_LHS(j) = log(p/m_pc)*Tvec(i)/m_Tc;
                }
                else if (to_fit == FIT_RHOL) {
                    double DL = o.attr("Dl").cast<double>();
                    if (!ValidNumber(DL)) { continue; }
                    m_LHS(j) = DL/m_Dc;
                }
                else if (to_fit == FIT_RHOV) {
                    double DV = o.attr("Dv").cast<double>();
                    if (!ValidNumber(DV)) { continue; }
                    m_LHS(j) = log(DV / m_Dc);//*Tvec(i)/m_Tc;
                }
                else {
                    throw std::invalid_argument("Argument for what to fit is invalid");
                }
                m_THETA(j) = 1-(Tvec(i)/m_Tc);
                m_T(j) = Tvec(i);
                j++;
            }
        }
        m_LHS.conservativeResize(j-1);
        m_THETA.conservativeResize(j-1);
        m_T.conservativeResize(j - 1);
    }
    void plot_curve(){
        py::module plt = py::module::import("matplotlib.pyplot"); // Import matplotlib
        plt.attr("plot")(m_THETA, m_LHS);
        plt.attr("show")();
    }
    void plot_deviation(const Eigen::ArrayXd &c) {

        py::module plt = py::module::import("matplotlib.pyplot"); // Import matplotlib
        Eigen::ArrayXd T = m_Tc*(1+m_THETA), deviation;
        std::string ylabel;
        if (to_fit == FIT_P){
            Eigen::ArrayXd pfit = eval_RHS(m_THETA, c).exp();
            Eigen::ArrayXd peos = m_LHS.exp();
            deviation = (pfit/peos-1)*100;
            ylabel = "Pressure deviation (%)";
        }
        else if (to_fit == FIT_RHOV) {
            Eigen::ArrayXd rhofit = eval_RHS(m_THETA, c).exp()*m_Dc;
            Eigen::ArrayXd rhoeos = m_LHS.exp()*m_Dc;
            deviation = (rhofit / rhoeos - 1) * 100;
            ylabel = "Density deviation (%)";
        }
        else if (to_fit == FIT_RHOL) {
            Eigen::ArrayXd rhofit = eval_RHS(m_THETA, c)*m_Dc;
            Eigen::ArrayXd rhoeos = m_LHS;
            deviation = (rhofit / rhoeos - 1) * 100;
            ylabel = "Density deviation (%)";
        }
        using namespace pybind11::literals;
        plt.attr("plot")(1 / (m_THETA + 1), deviation);
        plt.attr("ylabel")(ylabel);
        plt.attr("axvline")(m_Tt / m_Tc, "dashes"_a = std::vector<double>(2, 2));
        plt.attr("xlabel")("$T/T_c$");
        plt.attr("savefig")(m_name + ".pdf"); 
        plt.attr("close")();
    }
    Eigen::ArrayXd eval_RHS(const Eigen::ArrayXd& x, const Eigen::ArrayXd &c) {
        Eigen::ArrayXd num = Eigen::ArrayXd::Zero(x.size()), den = Eigen::ArrayXd::Zero(x.size());
        assert(m_Nnum + m_Nden == c.size());
        for (auto i = 0; i < m_Nnum; ++i) {
            num += c[i]*x.pow(i);
        }
        for (auto i = 0; i < m_Nden; ++i) {
            den += c[m_Nnum+i]*x.pow(i+1);
        }
        return num/(1+den);
    }
    double objective(const CEGO::AbstractIndividual *pind) {
        const std::vector<double> &c = static_cast<const CEGO::NumericalIndividual<double>*>(pind)->get_coefficients();
        return objective(Eigen::Map<const Eigen::ArrayXd>(&(c[0]), c.size()) );
    }
    double objective(const Eigen::ArrayXd &c) {
        //Ncalls++;
        return (eval_RHS(m_THETA, c) - m_LHS).square().sum();
    }
    void plot_trace(const std::vector<double> &best_costs) {
        py::module plt = py::module::import("matplotlib.pyplot"); // Import matplotlib
        plt.attr("plot")(best_costs);
        plt.attr("show")();
    }
};

int main()
{
    py::scoped_interpreter interp{};
    std::srand((unsigned int)time(0));

    std::size_t Nnum = 4, Nden = 4;

    // Construct the bounds
    std::vector<CEGO::Bound> bounds;
    for (auto i = 0; i < Nnum; ++i) { 
        bounds.push_back(CEGO::Bound(std::pair<double, double>(-10000, 10000)));
    }
    for (auto i = 0; i < Nden; ++i){
        bounds.push_back(CEGO::Bound(std::pair<double, double>(-10000, 10000))); 
    }    
    RatPolyAncillary rp("PROPANE", Nnum, Nden);
    rp.plot_curve();
   
    auto Ncalls = 0;
    CEGO::CostFunction cost_wrapper = std::bind((double (RatPolyAncillary::*)(const CEGO::AbstractIndividual *)) &RatPolyAncillary::objective, &rp, std::placeholders::_1);
    auto Nlayers = 1;
    auto layers = CEGO::Layers<double>(cost_wrapper, bounds.size(), 40, Nlayers, 5);
    layers.parallel = true;
    layers.set_bounds(bounds);

    auto flags = layers.get_evolver_flags();
    flags["Nelite"] = 2;
    flags["Fmin"] = 0.5;
    flags["Fmax"] = 0.5;
    flags["CR"] = 0.9;
    layers.set_evolver_flags(flags);

    std::vector<double> best_costs; 
    std::vector<std::vector<double> > objs;
    double VTR = 1e-6, best_cost = 999999.0;
    auto startTime = std::chrono::system_clock::now();
    for (auto counter = 0; counter < 50000; ++counter) {
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
            //auto c = std::get<1>(best_layer);
            //rp.plot_deviation(Eigen::Map<const Eigen::ArrayXd>(&(c[0]), c.size()));
        }
        if (best_cost < VTR){ break; }
    }
    auto best_layer = layers.get_best();
    auto c = std::get<1>(best_layer);
    rp.plot_deviation(Eigen::Map<const Eigen::ArrayXd>(&(c[0]), c.size()));
    auto endTime = std::chrono::system_clock::now();
    double elap = std::chrono::duration<double>(endTime - startTime).count();
    std::cout << "run:" << elap << " s\n";
    std::cout << "NFE:" << Ncalls << std::endl;

}
