#include <map>
#include <vector>
using namespace std;

class PrimaryUnit {
public:
  PrimaryUnit(int, int, int);
private:
  vector<PrimaryUnit*> F;
  vector<PrimaryUnit*> B;
};

class PrimaryNetwork {
public:
  PrimaryNetwork(int, int, int);
private:
  int N, I, O;
  map<int, PrimaryUnit> units;
};
