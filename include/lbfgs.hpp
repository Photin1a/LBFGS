/**
* @file lbfgs.hpp
* @brief An unconstrained optimization solver
* @version 0.1.0
* @date 2023-11-23
* @author YuanShuang Yang <2574195342@qq.com>
*
* @copyright Copyright (c) 2023, HCC Laboratory.
* SPDX-License-Identifier: Apache-2.0
*/

#ifndef _LBFGS_HPP_
#define _LBFGS_HPP_

#include <algorithm>
#include <chrono>
#include <deque>
#include <iostream>
#include <vector>

#include <Eigen/Core>

class LBFGS {
private:
  double grad_tolerance_ = 1e-4;
  double cost_tolerance_ = 1e-4;
  double armijo_stp_ = 0.5;

  double wolfe_stp_ = 0.1;
  double wolfe_sigama_ = 0.9;
  double cautious_epsilon_ = 1e-6;
  int m_max_ = 30;
  Eigen::VectorXd alpha_;
  int m_ = 0;

  Eigen::VectorXd x_;
  std::vector<double> fx_;
  size_t iterations_;

public:
  struct Solution {
    size_t iterations;
    Eigen::VectorXd x;
    double fx;
    double use_time;
  } solution_;

private:
  // Armijo Line-Search step
  inline double ArmijoLineSearch(const Eigen::VectorXd &x,
                                 const Eigen::VectorXd &grad,
                                 const Eigen::VectorXd &descend_direction) {
    double t = 1;
    auto cdg = armijo_stp_ * descend_direction.dot(grad);
    while (F(x + t * descend_direction) > F(x) + t * cdg)
      t /= 2.0;
    return t;
  }

  // Wolfe Line-Search step
  inline double WolfeLineSearch(Eigen::VectorXd &x, const Eigen::VectorXd &grad,
                                const Eigen::VectorXd &descend_direction,
                                Eigen::VectorXd &next_x,
                                Eigen::VectorXd &next_grad) {

    double a = 0;
    double b = 9999999;
    double j = 0;
    double t = 1;
    auto cdg = wolfe_stp_ * descend_direction.dot(grad);
    while (true) {
      if (F(next_x = (x + t * descend_direction)) > F(x) + t * cdg) {
        j++;
        b = t;
        t = (t + a) / 2;
        continue;
      }
      if (Grad(next_x, next_grad).dot(descend_direction) <
          wolfe_sigama_ * grad.dot(descend_direction)) {
        a = t;
        t = std::min(2 * t, (t + b) / 2);
        continue;
      }
      break;
    }
    return t;
  }

  // L-BFGS update
  inline Eigen::VectorXd &TwoLoop(const Eigen::VectorXd &grad,
                                  Eigen::VectorXd &descend_direction,
                                  std::deque<Eigen::VectorXd> &s,
                                  std::deque<Eigen::VectorXd> &y,
                                  std::deque<double> &ro) {
    descend_direction = -grad;
    for (int i = m_ - 1; i >= 0; i--) {
      alpha_(i) = ro[i] * s[i].dot(descend_direction);
      descend_direction -= alpha_(i) * y[i];
    }
    descend_direction /= ro[m_ - 1] * y[m_ - 1].squaredNorm();
    for (int i = 0; i < m_; i++) {
      descend_direction +=
          s[i] * (alpha_(i) - ro[i] * y[i].dot(descend_direction));
    }
    return descend_direction;
  }
  virtual inline double F(const Eigen::VectorXd &x) { return .0; };
  virtual inline Eigen::VectorXd &Grad(const Eigen::VectorXd &x,
                                       Eigen::VectorXd &grad) {
    return grad;
  }

public:
  // Constructor
  LBFGS() { alpha_.resize(m_max_); }

  // L_BFGS Solver
  Solution Solve(const Eigen::VectorXd &x0) {
    auto start_time = std::chrono::high_resolution_clock::now();
    x_ = x0;
    iterations_ = 0;
    Eigen::VectorXd grad, next_grad, descend_direction, next_x;
    std::deque<Eigen::VectorXd> s, y;
    std::deque<double> ro;
    Grad(x0, grad);
    descend_direction = -grad;
    auto grad_tolerance_square = grad_tolerance_ * grad_tolerance_;
    while (grad.squaredNorm() > grad_tolerance_square) {
      // Wolfe condition:calulate descend_direction, next_x, next_grad
      WolfeLineSearch(x_, grad, descend_direction, next_x, next_grad);

      // si,yi
      auto si = next_x - x_;
      auto yi = next_grad - grad;

      // cautious update
      if (yi.dot(si) > cautious_epsilon_ * grad.norm() * si.squaredNorm()) {
        // store si,yi
        if (m_ >= m_max_) {
          s.pop_front();
          y.pop_front();
          ro.pop_front();
          m_--;
        }
        s.push_back(si);
        y.push_back(yi);
        ro.push_back(1 / si.dot(yi));
        m_++;
        TwoLoop(grad, descend_direction, s, y, ro); // update B(k+1)<==>d
      }                                             // else d = last_d

      grad = next_grad;
      x_ = next_x;
      iterations_++;
      std::cout << "iter:" << iterations_ << std::endl;
      std::cout << "s_size:" << s.size() << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    solution_.x = x_;
    solution_.fx = F(x_);
    solution_.iterations = iterations_;
    solution_.use_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             end_time - start_time)
                             .count();
    return solution_;
  } // wolfe
};
#endif
