#pragma once

// autodiff includes
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

namespace CEGO {

    template <typename T> using EArray = Eigen::Array<T, Eigen::Dynamic, 1, 0, Eigen::Dynamic, 1>;
    template <typename T> using EVector = Eigen::Matrix<T, Eigen::Dynamic, 1, 0, Eigen::Dynamic, 1>;
    using DoubleObjectiveFunction = std::function<double(const EArray<double>&)>;
    using DoubleGradientFunction = std::function<EArray<double>(const EArray<double>&)>;

    /**
    A function of a vector of duals is passed into this wrapper; the function must have the signature:

    autodiff::dual func(Eigen::Array<autodiff::dual>)

    The autodiff library is used to calculate the gradient of the function

    */
    auto AutoDiffGradient(const std::function<autodiff::dual(EVector<autodiff::dual>&)>& func) {
        return [&func](const EArray<double>& x0) -> EArray<double> {
            // Upcast the doubles to duals (copy)
            Eigen::VectorXdual x = x0.cast<autodiff::dual>();
            // Call the gradient function; the argument must support VectorXdual arguments
            // The outputs are double again
            return autodiff::forward::gradient(func, autodiff::wrt(x), autodiff::forward::at(x));
        };
    }

    auto ComplexStepGradient(const std::function<std::complex<double>(EArray<std::complex<double>>&)> func) {
        return [func](const EArray<double>& x0) -> EArray<double> {
            EArray<double> o(x0.size());
            // Upcast the doubles to complex<double>
            Eigen::ArrayXcd x = x0.cast <std::complex<double> >();
            double h = 1e-100;
            // Call the objective function with the j-th element shifted by +h*j
            for (auto k = 0; k < x.size(); ++k) {
                Eigen::ArrayXcd xp = x;
                xp(k) += std::complex<double>(0, h);
                o(k) = func(xp).imag() / h;
            }
            return o;
        };
    }

    /**
    In this generic function, the gradient function is provided explicitly, convenient if
    you know the analytic gradient (for simple objective functions), or if you can 
    calculate it with automatic differentiation
    */
    auto gradient_linesearch(DoubleObjectiveFunction& objfunc, DoubleGradientFunction& gradfunc, const Eigen::ArrayXd& x, const Eigen::ArrayXd& lbvec, const Eigen::ArrayXd& ubvec, std::size_t Nmax_linesearch) {
        double c = 0.5, tau = 0.5;
        // Evaluate the objective and gradient functions for double arguments
        auto F = objfunc(x);
        auto g = gradfunc(x);
        // Check upper and lower bounds to determine the largest allowed value for alpha
        Eigen::ArrayXd alphaub = ubvec / g.array(), alphalb = lbvec / g.array();
        double alpha = std::max(alphaub.maxCoeff(), alphalb.maxCoeff());
        // The termination condition for the reduction in objective function
        double t = c * (g.square()).sum();
        for (auto j = 0; j < Nmax_linesearch; ++j) {
            alpha *= tau;
            auto fnew = objfunc((x - alpha * g.array()));
            double diff = F - fnew;
            if (diff > alpha * t) {
                return std::make_tuple((x - (alpha * g).array()).eval(), fnew); 
            }
        }
        return std::make_tuple(x, F);
    }

    struct BoxGradientFlags {
        double VTR = 1e-16;
        std::size_t Nmax = 100;
        std::size_t Nmax_linesearch = 50;
    };

    auto box_gradient_minimization(
        DoubleObjectiveFunction& funcdouble, 
        DoubleGradientFunction& gradfunc, 
        const Eigen::ArrayXd& x, 
        const Eigen::ArrayXd& lbvec, 
        const Eigen::ArrayXd& ubvec,
        const BoxGradientFlags &flags = BoxGradientFlags())
    {
        Eigen::ArrayXd xnew = x;
        double F;
        for (auto counter = 0; counter <= flags.Nmax; ++counter) {
            std::tie(xnew, F) = gradient_linesearch(funcdouble, gradfunc,  xnew, lbvec, ubvec, flags.Nmax_linesearch);
            if (F < flags.VTR) {
                return std::make_tuple(xnew, F);
            }
        }
        return std::make_tuple(xnew, F);
    }

}; // namespace CEGO