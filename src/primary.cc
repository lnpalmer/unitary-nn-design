#include <Primary.h>

PrimaryUnit::PrimaryUnit(int i) {
  this->i = i;
}

void PrimaryUnit::connectTo(PrimaryUnit* to, float value) {
  b.push_back(pair<PrimaryUnit*, float>(to, value));
  to->f.push_back(pair<PrimaryUnit*, float>(this, value));
}

void PrimaryUnit::connectFrom(PrimaryUnit* from, float value) {
  f.push_back(pair<PrimaryUnit*, float>(from, value));
  from->b.push_back(pair<PrimaryUnit*, float>(this, value));
}

void PrimaryUnit::disconnectTo(PrimaryUnit* to) {
  for (auto it = b.begin(); it != b.end(); it++) {
    if ((*it).first == to) b.erase(it);
  }
  for (auto it = to->f.begin(); it != to->f.end(); it++) {
    if ((*it).first == this) b.erase(it);
  }
}

void PrimaryUnit::disconnectFrom(PrimaryUnit* from) {
  for (auto it = f.begin(); it != f.end(); it++) {
    if ((*it).first == from) f.erase(it);
  }
  for (auto it = from->b.begin(); it != from->b.end(); it++) {
    if ((*it).first == this) b.erase(it);
  }
}

void PrimaryUnit::setWeightTo(PrimaryUnit* to, float value) {
  for (auto it = b.begin(); it != b.end(); it++) {
    if ((*it).first == to) (*it).second = value;
  }
  for (auto it = to->f.begin(); it != to->f.end(); it++) {
    if ((*it).first == this) (*it).second = value;
  }
}

void PrimaryUnit::setWeightFrom(PrimaryUnit* from, float value) {
  for (auto it = f.begin(); it != f.end(); it++) {
    if ((*it).first == from) (*it).second = value;
  }
  for (auto it = from->b.begin(); it != from->b.end(); it++) {
    if ((*it).first == this) (*it).second = value;
  }
}

void PrimaryUnit::forward() {

  preactivation = bias;

  for (int i = 0; i < f.size(); i++) {
    preactivation += f[i].first->activation * f[i].second;
  }

  activation = nonlinearity(preactivation);

}

void PrimaryUnit::backward() {

  for (int i = 0; i < b.size(); i++) {
    gradient += f[i].first->activation *
                f[i].first->nonlinearityDerivative(f[i].first->preactivation) *
                f[i].second;
  }

}

float PrimaryUnit::nonlinearity(float x) {
  return tanhf(x);
}

float PrimaryUnit::nonlinearityDerivative(float x) {
  return 1 - tanhf(x) * tanhf(x);
}

PrimaryNetwork::PrimaryNetwork(int N, int I, int O) {
  this->N = N;
  this->I = I;
  this->O = O;

  for (int i = 0; i < I; i++) populate(i);
  for (int i = N - O; i < N; i++) populate(i);
}

void PrimaryNetwork::populate(int i) {
  units[i] = new PrimaryUnit(i);
}

void PrimaryNetwork::remove(int i) {
  delete units[i];
  units.erase(units.find(i));
}

void PrimaryNetwork::connect(int i, int j, float value) {
  units[i]->connectTo(units[j], value);
}

void PrimaryNetwork::disconnect(int i, int j) {
  units[i]->disconnectTo(units[j]);
}

vector<float> PrimaryNetwork::forward(vector<float> x) {
  /* ??? */
}

vector<float> PrimaryNetwork::backward(vector<float> dy) {
  /* ??? */
}
