#include <map>
using namespace std;

class Optimizable {
public:
  virtual map<string, pair<float, float>> getParameters() = 0;
  virtual void setParameters(map<long long, float>) = 0;
};

class Optimizer {
public:
  virtual void optimize(Optimizable*) = 0;
};

class AdamOptimizer : public Optimizer {
public:
  AdamOptimizer(float, float, float, float);
  void optimize(Optimizable*);
private:
  float lr;
  float beta1;
  float beta2;
  float epsilon;
  map<Optimizable*, map<long long, pair<float, float>>> moments;
};
