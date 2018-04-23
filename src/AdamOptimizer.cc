#include <optimize.h>

AdamOptimizer::AdamOptimizer(float lr, float beta1, float beta2, float epsilon) {
  this->lr = lr;
  this->beta1 = beta1;
  this->beta2 = beta2;
  this->epsilon = epsilon;
}

void AdamOptimizer::optimize(Optimizable*) {
}
