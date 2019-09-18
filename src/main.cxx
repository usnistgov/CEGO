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

template <typename T>
T RosenbrockI(const CEGO::AbstractIndividual* pind) {
    return Rosenbrockvec(pind->get_coeffs_ArrayXd());
}

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

int main(){

    //test_bounds();
    //do_minimization<double>(RosenbrockI<double>, nullptr);
    //do_minimization<CEGO::numberish>(RosenbrockI<CEGO::numberish>, nullptr);

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

    const std::function<std::complex<double>(CEGO::EVector<std::complex<double>>&) > csdf = Rosenbrockvec<std::complex<double>>;
    CEGO::DoubleGradientFunction gradcsd = CEGO::ComplexStepGradient(csdf);
    Eigen::ArrayXd g1 = grad(x0);
    Eigen::ArrayXd g1csd = gradcsd(x0);
    CEGO::DoubleGradientFunction gradexact = [](const CEGO::EArray<double>& c) -> CEGO::EArray<double> {
        return Rosenbrock_exact_gradient(c(0), c(1));
    };
   
    Eigen::ArrayXd g31 = autodiff::forward::gradient(Rosenbrockvek1, autodiff::wrt(x0dual), autodiff::forward::at(x0dual));
    Eigen::ArrayXd g32 = autodiff::forward::gradient(Rosenbrockvek2, autodiff::wrt(x0dual), autodiff::forward::at(x0dual));
    
    std::cout << g1csd-gexact << " must be zero (CSD)\n";
    auto tic = std::chrono::high_resolution_clock::now();
    CEGO::box_gradient_minimization(func, gradexact, x0, lbvec, ubvec);
    auto toc = std::chrono::high_resolution_clock::now();
    double elap = std::chrono::duration<double>(toc-tic).count();
    std::cout << elap << std::endl;

    int rr =  0;
}
#else

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

template<class T >
void upgrade_Layers(py::class_<CEGO::Layers<T>> &layers){
    using namespace CEGO;
    typedef Layers<T> MyLayers;

    layers.def(py::init<std::function<double(const std::vector<T> &)>&, int, int, int, int>());
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
        .value("differential_evolution", be::differential_evolution);

    typedef Layers<double> MyLayers;
    py::class_<MyLayers> layers(m, "DoubleLayers");
    upgrade_Layers(layers);

    typedef Layers<CEGO::numberish> NumberishLayers;
    py::class_<NumberishLayers> numberish_layers(m, "NumberishLayers");
    upgrade_Layers(numberish_layers);

    m.def("LHS_samples", &LHS_samples);

}

PYBIND11_PLUGIN(PyCEGO) {
    py::module m("PyCEGO", "Python wrapper of CEGO implementation of Ian Bell in C++11");
    init_PyCEGO(m);
    return m.ptr();
}

#endif
