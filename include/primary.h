#include <map>
#include <vector>
#include <math.h>
using namespace std;

class PrimaryUnit {
public:
  PrimaryUnit(int i = 0);
  void connectTo(PrimaryUnit*, float);
  void connectFrom(PrimaryUnit*, float);
  void disconnectTo(PrimaryUnit*);
  void disconnectFrom(PrimaryUnit*);
  void setWeightTo(PrimaryUnit*, float);
  void setWeightFrom(PrimaryUnit*, float);
  void forward();
  void backward();
  float nonlinearity(float);
  float nonlinearityDerivative(float);
private:
  int i;
  vector<pair<PrimaryUnit*, float>> f;
  vector<pair<PrimaryUnit*, float>> b;
  float bias;
  float preactivation;
  float activation;
  float gradient;
};

class PrimaryNetwork {
public:
  PrimaryNetwork(int, int, int);
  void populate(int);
  void remove(int);
  void connect(int, int, float);
  void disconnect(int, int);
  vector<float> forward(vector<float>);
  vector<float> backward(vector<float>);
private:
  int N, I, O;
  map<int, PrimaryUnit*> units;
};
