#pragma once

// autodiff includes
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

template <typename Function, typename RealFunction>
auto gradient_linesearch(Function& func, RealFunction& funcreal, Eigen::VectorXdual& x, const Eigen::ArrayXd& lbvec, const Eigen::ArrayXd& ubvec) {
    using namespace autodiff;
    dual F;
    double c = 0.5, tau = 0.5;

    // Use autodiff to determine the gradient; steepest descent direction is 
        // the negative of the gradient
    Eigen::VectorXd g = gradient(func, wrt(x), at(x), F);
    // Get real values as an array
    const Eigen::ArrayXd xx = x.cast<double>();
    // Check upper and lower bounds to determine the largest allowed value for alpha
    Eigen::ArrayXd alphaub = ubvec / g.array(), alphalb = lbvec / g.array();
    double alpha = std::max(alphaub.maxCoeff(), alphalb.maxCoeff());
    // The termination condition for the reduction in objective function
    double t = c * (g.array().square()).sum();
    for (auto j = 0; j < 30; ++j) {
        alpha *= tau;
        auto fnew = funcreal((xx - alpha * g.array()).matrix());
        double diff = val(F) - val(fnew);
        if (diff > alpha * t) {
            break;
        }
    }
    return std::make_tuple(x - (alpha * g).cast<autodiff::dual>(), F);
}

template <typename Function, typename RealFunction>
void box_gradient_minimization(Function& func, RealFunction& funcreal, const Eigen::VectorXd& x0, const Eigen::ArrayXd &lbvec, const Eigen::ArrayXd& ubvec) {
    using namespace autodiff;
    Eigen::VectorXdual x = x0.cast<autodiff::dual>();  // the input vector x, casted to VectorXd
    dual F;
    for (auto counter = 0; counter <= 10000; ++counter) {
        std::tie(x, F) = gradient_linesearch(func, funcreal, x, lbvec, ubvec);
        if (counter % 100 == 0) {
            std::cout << counter << " " << F << " " << std::endl;
        }
        if (std::abs(val(F)) < 1e-10) {
            break;
        }
    }
}