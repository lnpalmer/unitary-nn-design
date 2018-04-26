#include <torch/torch.h>

#include <vector>
#include <map>
#include <queue>
#include <iostream>

using namespace std;
using namespace at;
using namespace torch::autograd;

// TODO: optimize, clean

struct FEntry {
public:
  FEntry(int i = -1) {
    this->i = i;
  }

  bool operator<(const FEntry& other) const {
    return i > other.i;
  }

  int i;
  vector< pair<int, float> > c;
};

struct BEntry {
public:
  BEntry(int j = -1) {
    this->j = j;
  }

  bool operator<(const BEntry& other) const {
    return j < other.j;
  }

  int j;
  vector< pair<int, float> > c;
};

Tensor dtanh_from_tanh(Tensor y) {
  return 1.0f - (y * y);
}

Tensor dagnn_forward(
  Tensor x,
  Tensor W,
  Tensor b,
  Tensor i,
  Tensor o
) {

  int M = x.size(0);
  int I = *(i.toIntData());
  int O = *(o.toIntData());
  int N = W.size(0);

  W = W.coalesce();
  auto W_i = W._indices();
  auto W_v = W._values();
  auto W_i_acs = W_i.accessor<long, 2>();
  auto W_v_acs = W_v.accessor<float, 1>();

  // parse W for forward connections
  auto F_ = map<int, FEntry>();
  for (int k = 0; k < W_v.size(0); k++) {

    int i = W_i_acs[0][k];
    int j = W_i_acs[1][k];
    float w_ij = W_v_acs[k];

    if (F_.find(i) == F_.end()) {
      F_[i] = FEntry(i);
    }

    F_[i].c.push_back(pair<int, float>(j, w_ij));

  }

  // move to a priority queue
  priority_queue<FEntry> F;
  for (auto it = F_.begin(); it != F_.end(); it++) {
    F.push((*it).second);
  }

  // load input
  auto a = cat({
    x.transpose(0, 1),
    make_variable(CPU(kFloat).zeros({N - I, M}))
  });

  int d0 = 0;
  int d1 = 0;

  // actual forward pass
  while (!F.empty()) {

    d0 ++;

    FEntry F_i = F.top();
    int i = F_i.i;

    auto z = make_variable(CPU(kFloat).zeros({M}).fill_(b[i]));

    for (auto jw_ij : F_i.c) {

      int j = jw_ij.first;
      float w_ij = jw_ij.second;

      z += a[j] * w_ij;

      d1 ++;

    }

    if (i >= N - O) a[i] = z;
    else a[i] = z.tanh();

    F.pop();

  }

  return a.transpose(0, 1);

}

vector<at::Tensor> dagnn_backward(
  at::Tensor W,
  at::Tensor b,
  at::Tensor i,
  at::Tensor o,
  at::Tensor a,
  at::Tensor da
) {

  int M = a.size(0);
  int I = *(i.toIntData());
  int O = *(o.toIntData());
  int N = W.size(0);

  W = W.coalesce();
  auto W_i = W._indices();
  auto W_v = W._values();
  auto W_i_acs = W_i.accessor<long, 2>();
  auto W_v_acs = W_v.accessor<float, 1>();

  // parse W for backward connections
  auto B_ = map<int, BEntry>();
  for (int k = 0; k < W_v.size(0); k++) {

    int i = W_i_acs[0][k];
    int j = W_i_acs[1][k];
    float w_ij = W_v_acs[k];

    if (B_.find(j) == B_.end()) {
      B_[j] = BEntry(j);
    }

    B_[j].c.push_back(pair<int, float>(i, w_ij));

  }

  // move to a priority queue
  priority_queue<BEntry> B;
  for (auto it = B_.begin(); it != B_.end(); it++) {
    B.push((*it).second);
  }

  int posW = 0;

  // actual backward pass
  auto dz = da.clone();

  dz = dz.transpose(0, 1);
  da = da.transpose(0, 1);
  a = a.transpose(0, 1);

  auto dW = W.clone();
  auto dW_i = dW._indices();
  auto dW_v = dW._values();
  auto dW_i_acs = dW_i.accessor<long, 2>();

  while (!B.empty()) {

    BEntry B_j = B.top();
    int j = B_j.j;

    auto da_j = da[j].clone();
    for (auto iw_ij : B_j.c) {

      int i = iw_ij.first;
      float w_ij = iw_ij.second;

      da_j += dz[i] * w_ij;

      dW_i_acs[0][posW] = i;
      dW_i_acs[1][posW] = j;
      dW_v[posW] = (dz[i] * a[j]).sum();
      posW ++;

    }

    if (j < I) dz[j] = da_j;
    else dz[j] = da_j * dtanh_from_tanh(a[j]);

    B.pop();

  }

  dz = dz.transpose(0, 1);
  auto db = dz.sum(0);

  return {
    dz.slice(1, 0, I),
    dW,
    db
  };

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dagnn_forward, "DagNN forward");
  m.def("backward", &dagnn_backward, "DagNN backward");
}
