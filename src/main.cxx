#include <string>
#include <atomic>

// See http://dr.library.brocku.ca/bitstream/handle/10464/10416/Brock_Opoku-Amankwaah_Audrey_2016.pdf?sequence=1

#include "CEGO/CEGO.hpp"
#include "CEGO/lhs.hpp"
#include "CEGO/evolvers/evolvers.hpp"
#include "CEGO/minimizers.hpp"

// autodiff include
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

// See http://stackoverflow.com/a/4609795
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

static std::atomic_size_t Ncalls(0);

template <typename T>
auto pow2(T x0) {
    return x0*x0;
}

template <typename T>
auto Rosenbrock(T x0, T x1) {
    return 100.0*pow2(pow2(x0) - x1) + pow2(1.0-x0);
}

template <typename T>
CEGO::EArray<T> Rosenbrock_exact_gradient(const T x0, const T x1) {
    CEGO::EArray<T> o(2);
    o(0) = 200* (pow2(x0) - x1)*(2*x0) -2*(1 - x0);
    o(1) = 200*(pow2(x0) - x1) * -1;
    return o;
}

template <typename T>
T Rosenbrockvec(const Eigen::Matrix<T, Eigen::Dynamic, 1,0, Eigen::Dynamic, 1> & x) {
    return Rosenbrock(x[0], x[1]);
}

autodiff::dual Rosenbrockvek(const CEGO::EVector<autodiff::dual> & x) {
    return Rosenbrock(x[0], x[1]); //100.0 * (x(0)*x(0)-x(1)) * (x(0)*x(0)-x(1)) + (1.0-x(0)) * (1.0-x(0));
}
autodiff::dual Rosenbrockvek1(const CEGO::EVector<autodiff::dual>& x) {
    return 100.0 * pow2(x(0) * x(0) - x(1)) + pow2(1.0 - x(0));
}
autodiff::dual Rosenbrockvek2(const CEGO::EVector<autodiff::dual>& x) {
    return 100.0 * pow2(pow2(x(0)) - x(1)) + pow2(1.0 - x(0));
}
//
//template <typename T>
//T RosenbrockI(const CEGO::AbstractIndividual* pind) {
//    return Rosenbrockvec(pind->get_coeffs_ArrayXd());
//}

template <typename T>
T Griewangk(const std::vector<T> &x) {
    Ncalls ++;
    double sum1 = 0, prod1 = 1;
    for (auto i = 0; i < x.size(); ++i){ 
        sum1 += static_cast<double>(x[i]*x[i]); 
        prod1 *= cos(static_cast<double>(x[i])/sqrt(i+1));
    }; 
    return sum1/4000.0 - prod1 + 1;
}
double Corana(const std::vector<double> &x) {
    Ncalls++;
    std::vector<double> d = {1., 1000., 10., 100.};
    double r = 0;
    for (auto j = 0; j < 4; ++j){
        double zj = floor(std::abs(x[j] / 0.2) + 0.49999) * sgn(x[j]) * 0.2;
        if (std::abs(x[j] - zj) < 0.05){
            r += 0.15 * pow(zj - 0.05*sgn(zj), 2) * d[j];
        }
        else{
            r += d[j]*x[j]*x[j];
        }
    }
    return r;
}

#ifndef PYBIND11

void test_bounds() {
    using namespace CEGO;
    std::vector<Bound> bounds;
    bounds.push_back(Bound(std::make_pair(-1000, 1000)));
    bounds.push_back(Bound(-1000, 1000));
    bounds.push_back(Bound(-1000.0, 1000.0));
    int rr =0;
}

template<typename T, typename F, typename G>
void do_minimization(F f, G g) {
    using namespace CEGO;
    Ncalls = 0;
    ALPSInputValues<T> in;
    in.f = f;
    auto D = 10;
    for (auto i = 0; i < D; ++i) { in.bounds.push_back(std::pair<T,T>( -600, 600 )); }
    in.age_gap = 5;
    in.Nlayer = 1;
    in.NP = 20;
    in.VTR = 1e-3;
    ALPSReturnValues out = CEGO::ALPS<T>(in);
    std::cout << "run: " << out.elapsed_sec << " s\n";
    std::cout << "Ncalls: " << Ncalls << std::endl;
}

/// Minimize the Rosenbrock function with complex step derivatives to build the quasi-exact gradient
void minimize_Rosenbrock_CSD() {
    
    // Start and box bounds
    Eigen::VectorXd x0(2); x0 << -0.3, 0.5;
    Eigen::VectorXd lbvec(2); lbvec << -1, -1;
    Eigen::VectorXd ubvec(2); ubvec << 1, 1; 

    // Define functors for function and its gradient
    CEGO::DoubleObjectiveFunction func = Rosenbrockvec<double>;
    const std::function<std::complex<double>(CEGO::EArray<std::complex<double>>&) > csdf = Rosenbrockvec<std::complex<double>>;
    CEGO::DoubleGradientFunction gradcsd = CEGO::ComplexStepGradient(csdf);

    // Check that the right gradient is returned
    Eigen::ArrayXd gcsd = gradcsd(x0);
    Eigen::ArrayXd gexact = Rosenbrock_exact_gradient(x0(0), x0(1));
    double graderror = (gexact - gcsd).cwiseAbs().sum();
    if (graderror > 1e-13) {
        throw std::invalid_argument("Did not get the right gradient; error is " + std::to_string(graderror));
    }
    
    auto tic = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd xsoln; 
    double F;
    CEGO::BoxGradientFlags flags;
    flags.Nmax = 10000;
    flags.VTR = 1e-14;
    std::tie(xsoln, F) = CEGO::box_gradient_minimization(func, gradcsd, x0, lbvec, ubvec, flags);
    auto toc = std::chrono::high_resolution_clock::now();
    double elap = std::chrono::duration<double>(toc - tic).count();
    std::cout << elap << std::endl;

    Eigen::VectorXd xsolnexact(2); xsolnexact << 1, 1;
    double solnerror = (xsoln - xsolnexact).cwiseAbs().sum();
    if (solnerror > 1e-6) {
        throw std::invalid_argument("Did not get the right solution; solution is (" + std::to_string(xsoln(0))+","+ std::to_string(xsoln(1)) + "); should be (1,1)");
    }
}

void minimize_Rosenbrock_autodiff() {
    Eigen::VectorXd x0(2); x0 << -0.3, 0.5;
    Eigen::ArrayXd gexact = Rosenbrock_exact_gradient(x0(0), x0(1));
    std::cout << gexact << std::endl;

    Eigen::VectorXdual x0dual = x0.cast<autodiff::dual>();
    Eigen::VectorXd g30 = autodiff::forward::gradient(Rosenbrockvec<autodiff::dual>, autodiff::wrt(x0dual), autodiff::forward::at(x0dual));
    Eigen::VectorXd g3 = autodiff::forward::gradient(Rosenbrockvek, autodiff::wrt(x0dual), autodiff::forward::at(x0dual));
    std::cout << g30.array() - gexact.array() << " should be zero\n";
    std::cout << g3.array() - gexact.array() << " should be zero\n";

    Eigen::VectorXd lbvec(2); lbvec << -1, -1;
    Eigen::VectorXd ubvec(2); ubvec << 1, 1;
    CEGO::DoubleObjectiveFunction func = Rosenbrockvec<double>;
    const std::function<autodiff::dual(CEGO::EVector<autodiff::dual>&)> adf = Rosenbrockvec<autodiff::dual>;
    CEGO::DoubleGradientFunction grad = CEGO::AutoDiffGradient(adf);

    Eigen::ArrayXd g31 = autodiff::forward::gradient(Rosenbrockvek1, autodiff::wrt(x0dual), autodiff::forward::at(x0dual));
    Eigen::ArrayXd g32 = autodiff::forward::gradient(Rosenbrockvek2, autodiff::wrt(x0dual), autodiff::forward::at(x0dual));
}

int main(){

    //test_bounds();
    //do_minimization<double>(RosenbrockI<double>, nullptr);
    //do_minimization<CEGO::numberish>(RosenbrockI<CEGO::numberish>, nullptr);
    minimize_Rosenbrock_CSD();
}
#else

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/* Trampoline class; passes calls up into the derived class in python
 * One overload is needed for each virtual function you want to be able to overload
 */
template<typename TYPE = double>
class PyFuncWrapper : public CEGO::FuncWrapper<TYPE> {
public:
    /* Inherit the constructors */
    using CEGO::FuncWrapper<TYPE>::FuncWrapper;

    double call(const CEGO::CRefEArray<TYPE> & x) const override {
        PYBIND11_OVERRIDE(
            double,                   /* Return type */
            CEGO::FuncWrapper<TYPE>,  /* Parent class */
            call,                     /* Name of function in C++ (must match Python name) */
            x                         /* Argument(s) */
        );
    }
};

template<class T >
void upgrade_Layers(py::class_<CEGO::Layers<T>> &layers){
    using namespace CEGO;
    typedef Layers<T> MyLayers;

    layers.def(py::init<const FuncWrapper<T>&, int, int, int, int>());
    layers.def(py::init<std::function<double(const CRefEArray<T> &)>, int, int, int, int>());
    layers.def("do_generation", &MyLayers::do_generation);
    layers.def("print_diagnostics", &MyLayers::print_diagnostics);
    layers.def("get_best", &MyLayers::get_best);
    layers.def("set_bounds", &MyLayers::set_bounds);
    layers.def("get_results", &MyLayers::get_results);
    layers.def("get_logging_scheme", &MyLayers::get_logging_scheme);
    layers.def("set_logging_scheme", &MyLayers::set_logging_scheme);
    layers.def("get_generation_mode", &MyLayers::get_generation_mode);
    layers.def("set_generation_mode", &MyLayers::set_generation_mode);
    layers.def("set_filtering_function", &MyLayers::set_filtering_function);
    layers.def("cost_stats_each_layer", &MyLayers::cost_stats_each_layer);
    layers.def("set_builtin_evolver", &MyLayers::set_builtin_evolver);
    layers.def("get_evolver_flags", [](MyLayers &layers) { return layers.get_evolver_flags().dump(); });
    layers.def("set_evolver_flags", [](MyLayers &layers, const std::string &s) { return layers.set_evolver_flags(nlohmann::json::parse(s)); });
}

void init_PyCEGO(py::module &m) {
    using namespace CEGO;

    py::class_<Bound >(m, "Bound")
        .def(py::init<const double &, const double &>())
        .def(py::init<const int &, const int &>())
        ;

    py::class_<numberish >(m, "Numberish")
        .def(py::init<const double &>())
        .def(py::init<const int &>())
        .def("as_double",&numberish::as_double)
        .def("as_int", &numberish::as_int)
        .def("__float__", [](const numberish &n){ return n.as_double(); })
        
        .def(double() * py::self)
        .def(py::self * double())
        .def(py::self * py::self)

        .def(double() + py::self)
        .def(py::self + double())
        .def(py::self + py::self)

        .def(double() - py::self)
        .def(py::self - double())
        .def(py::self - py::self)
        
        ;

    py::class_<Result>(m, "Result")
        .def_readonly("c", &Result::c)
        .def_readonly("ssq", &Result::ssq)
        ;

    typedef LoggingScheme LS;
    py::enum_<LoggingScheme >(m, "LoggingScheme")
        .value("none", LS::none)
        .value("all", LS::all)
        .value("custom", LS::custom)
        ;

    typedef GenerationOptions go;
    py::enum_<GenerationOptions >(m, "GenerationOptions")
        .value("LHS", go::LHS)
        .value("random", go::random)
        ;

    typedef FilterOptions fo;
    py::enum_< fo >(m, "FilterOptions")
        .value("accept", fo::accept)
        .value("reject", fo::reject)
        ;

    typedef CEGO::BuiltinEvolvers be;
    py::enum_< be >(m, "BuiltinEvolvers")
        .value("differential_evolution", be::differential_evolution)
        .value("differential_evolution_rand1bin", be::differential_evolution_rand1bin)
        .value("differential_evolution_rand1exp", be::differential_evolution_rand1exp)
        .value("differential_evolution_best1bin", be::differential_evolution_best1bin)
        .value("differential_evolution_best1exp", be::differential_evolution_best1exp)
        .value("differential_evolution_rand2bin", be::differential_evolution_rand2bin)
        .value("differential_evolution_rand2exp", be::differential_evolution_rand2exp)
        .value("differential_evolution_best2bin", be::differential_evolution_best2bin)
        .value("differential_evolution_best2exp", be::differential_evolution_best2exp)
        ; 

    py::class_<FuncWrapper<double>, PyFuncWrapper<double>> (m, "DoubleFuncWrapper")
        .def(py::init<>())
        ;

    py::class_<FuncWrapper<CEGO::numberish>, PyFuncWrapper<CEGO::numberish>> (m, "NumberishFuncWrapper")
        .def(py::init<>())
        ;

    typedef Layers<double> MyLayers;
    py::class_<MyLayers> layers(m, "DoubleLayers");
    upgrade_Layers(layers);

    typedef Layers<CEGO::numberish> NumberishLayers;
    py::class_<NumberishLayers> numberish_layers(m, "NumberishLayers");
    upgrade_Layers(numberish_layers);
    PYBIND11_NUMPY_DTYPE(CEGO::numberish, u_);

    m.def("indexer", [](std::function<CEGO::numberish(const RefEArray<CEGO::numberish> &)> &f){ 
        EArray<CEGO::numberish> a(10); a = 3.9; f(a); return;
    });

    m.def("first_entry", [](const std::vector<CEGO::numberish> x) {
        return x[0];
    });

    m.def("mutator", [](
        std::function<void(const py::array_t<CEGO::numberish>&)>& f,
        const py::array_t<CEGO::numberish>& x) {
        f(x);
    });

    m.def("LHS_samples", [](int Npop, int Nparam){std::mt19937 m_rng = get_Mersenne_twister(); return LHS_samples(Npop, Nparam, m_rng);} );

}

PYBIND11_MODULE(PyCEGO, m){
    m.doc() = "PyCEGO: Python wrapper of CEGO implementation of Ian Bell in C++17";
    m.attr("__version__") = CEGOVERSION;
    init_PyCEGO(m);
}

#endif
