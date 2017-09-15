#include <iostream>
#include <iomanip>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

  // Initializing vector with the RMSE for each variable
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // Verify if estimations is not empty and has same lenght of ground_truth
  if(estimations.size() == 0 || estimations.size() != ground_truth.size()) {
  	cout << "Invalid estimation or ground truth data" << endl;
  	return rmse;
  }

  // Accumulate squared residuals
  for(unsigned int i = 0; i < estimations.size(); ++i) {
  	VectorXd residual = estimations[i] - ground_truth[i];

  	residual = residual.array()*residual.array();
  	rmse += residual;
  }

  // Calculate the mean
  rmse = rmse / estimations.size();

  // Calculate the square root
  rmse = rmse.array().sqrt();

  // print RMSE
  //cout << fixed << setprecision(4);
  //cout << "-------------------------------------------------" << endl;
  //cout << "RMSE\t\t>> " << rmse(0) << "  " << rmse(1) << "  "  << rmse(2) << "  " << rmse(3) << endl;
  //cout << "RMSE (max)\t>> 0.1100  0.1100  0.5200  0.5200" << endl;

  // Return the Root Mean Squared Error
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  
  // Initializing Jacobian matrix
  MatrixXd Hj(3, 4);

  // State parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float c = px*px + py*py;

  // Check division by zero
  if(fabs(c) < 0.0001) {
  	cout << "CalculateJacobian(): Error - Division by zero" << endl;
  	return Hj;
  }

  // Jacobian matrix
  Hj << px/sqrt(c), py/sqrt(c), 0, 0,
  		  -py/c, px/c, 0, 0,
  		  (py*(vx*py-vy*px))/pow(c, 1.5), (px*(vy*px-vx*py))/pow(c, 1.5), px/sqrt(c), py/sqrt(c);

  return Hj;
}
