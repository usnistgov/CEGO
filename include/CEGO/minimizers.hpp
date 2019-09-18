#pragma once

// autodiff includes
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

namespace CEGO {

    template <typename T> using EArray = Eigen::Array<T, Eigen::Dynamic, 1>;
    using DoubleObjectiveFunction = std::function<double(const EArray<double>&)>;
    using DoubleGradientFunction = std::function<EArray<double>(const EArray<double>&)>;

    /**
    In this generic function, the gradient function is provided explicitly, convenient if
    you know the analytic gradient (for simple objective functions)
    */
    auto gradient_linesearch(DoubleObjectiveFunction& objfunc, DoubleGradientFunction& gradfunc, const Eigen::ArrayXd& x, const Eigen::ArrayXd& lbvec, const Eigen::ArrayXd& ubvec) {
        double c = 0.5, tau = 0.5;
        // Evaluate the objective and gradient functions for double arguments
        auto F = objfunc(x);
        auto g = gradfunc(x);
        // Check upper and lower bounds to determine the largest allowed value for alpha
        Eigen::ArrayXd alphaub = ubvec / g.array(), alphalb = lbvec / g.array();
        double alpha = std::max(alphaub.maxCoeff(), alphalb.maxCoeff());
        // The termination condition for the reduction in objective function
        double t = c * (g.square()).sum();
        for (auto j = 0; j < 30; ++j) {
            alpha *= tau;
            auto fnew = objfunc((x - alpha * g.array()));
            double diff = F - fnew;
            if (diff > alpha * t) {
                break;
            }
        }
        return std::make_tuple((x - (alpha * g).array()).eval(), F);
    }

    /**
    A function of a vector of duals is passed into this wrapper; the function must have the signature:

    autodiff::dual func(Eigen::Array<autodiff::dual>)

    The autodiff library is used to calculate the gradient of the function

    */
    template <typename Function>
    auto AutoDiffGradient(const Function &func) {
        return [&func](const EArray<double>& x0) -> EArray<double> {
            // Upcast the doubles to duals (copy)
            Eigen::VectorXdual x = x0.cast<autodiff::dual>();
            // Call the gradient function; the argument must support VectorXdual arguments
            // The outputs are double again
            return autodiff::forward::gradient(func, autodiff::wrt(x), autodiff::forward::at(x));
        };
    }

    void box_gradient_minimization(DoubleObjectiveFunction& funcdouble, DoubleGradientFunction& gradfunc, const Eigen::ArrayXd& x, const Eigen::ArrayXd& lbvec, const Eigen::ArrayXd& ubvec) {
        Eigen::ArrayXd xnew = x;
        double F;
        for (auto counter = 0; counter <= 10000; ++counter) {
            std::tie(xnew, F) = gradient_linesearch(funcdouble, gradfunc,  xnew, lbvec, ubvec);
            if (counter % 100 == 0) {
                std::cout << counter << " " << F << " " << std::endl;
            }
            if (std::abs(F) < 1e-10) {
                break;
            }
        }
    }

}; // namespace CEGO