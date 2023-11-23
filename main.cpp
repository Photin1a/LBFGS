/**
* @file main.cpp
* @brief main
* @version 
* @date 2023-11-23
* @author YuanShuang Yang <2574195342@qq.com>
*
* @copyright Copyright (c) 2023, HCC Laboratory.
* SPDX-License-Identifier: Apache-2.0
*/

#include "lbfgs.hpp"
#include <Eigen/Core>
#include <iostream>
#include <memory>

using Eigen::VectorXd;

class RosenBrock : public LBFGS {
public:
  int N_;
  RosenBrock(int N) : LBFGS() { N_ = N; }

  inline double F(const Eigen::VectorXd &x) override {
    double fx = 0;
    for (int i = 1; i <= x.rows() / 2; i++)
      fx = fx + 100 * pow(pow(x(2 * i - 2), 2) - x(2 * i - 1), 2) +
           pow(x(2 * i - 2) - 1, 2);
    return fx;
  }
  inline Eigen::VectorXd &Grad(const Eigen::VectorXd &x,
                               Eigen::VectorXd &grad) override {
    grad = Eigen::VectorXd::Zero(x.rows());
    for (int i = 1; i <= x.rows() / 2; i++) {
      grad(2 * i - 2) =
          grad(2 * i - 2) +
          400 * (pow(x(2 * i - 2), 2) - x(2 * i - 1)) * x(2 * i - 2) +
          2 * (x(2 * i - 2) - 1);
      grad(2 * i - 1) =
          grad(2 * i - 1) - 200 * (pow(x(2 * i - 2), 2) - x(2 * i - 1));
    }
    return grad;
  }
};

int main() {
  int N = 100;
  VectorXd x0 = Eigen::VectorXd::Zero(N);
  auto rosebrock = RosenBrock(N);
  rosebrock.Solve(x0);
  std::cout << "x:" << rosebrock.solution_.x << std::endl;
  std::cout << "iters:" << rosebrock.solution_.iterations << std::endl;
  std::cout << "fx:" << rosebrock.solution_.fx << std::endl;
  std::cout << "use_time/us:" << rosebrock.solution_.use_time << std::endl;
}