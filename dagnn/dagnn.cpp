#include <torch/torch.h>

#include <vector>
#include <map>
#include <queue>
#include <iostream>

using namespace std;
using namespace at;
using namespace torch::autograd;

// TODO: optimize, clean

struct Array1D {
  Array1D(int rows, float fill = 0.0f) {
    this->rows = rows;
    this->data = vector<float>(rows, fill);
  }

  float get(int i) {
    return data[i];
  }

  void set(int i, float value) {
    data[i] = value;
  }

  int rows;
  vector<float> data;
};

struct Array2D {
  Array2D(int rows, int cols, float fill = 0.0f) {
    this->rows = rows;
    this->cols = cols;
    this->data = vector<float>(rows * cols, fill);
  }

  float get(int i, int j) {
    return data[i * cols + j];
  }

  void set(int i, int j, float value) {
    data[i * cols + j] = value;
  }

  int rows;
  int cols;
  vector<float> data;
};

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

float dtanh_from_tanh(float y) {
  return 1.0f - (y * y);
}

Array1D var_to_arr1d(Tensor var) {
  auto arr = Array1D(var.size(0));
  auto acs = var.accessor<float, 1>();

  for (int i = 0; i < arr.rows; i++) {
    arr.set(i, acs[i]);
  }

  return arr;
}

Array2D var_to_arr2d(Tensor var) {
  auto arr = Array2D(var.size(0), var.size(1));
  auto acs = var.accessor<float, 2>();

  for (int i = 0; i < arr.rows; i++) {
    for (int j = 0; j < arr.cols; j++) {
      arr.set(i, j, acs[i][j]);
    }
  }

  return arr;
}

Tensor arr1d_to_var(Array1D arr) {
  auto var = CPU(kFloat).zeros({arr.rows});
  auto acs = var.accessor<float, 1>();

  for (int i = 0; i < arr.rows; i++) {
    acs[i] = arr.get(i);
  }

  return make_variable(var);
}

Tensor arr2d_to_var(Array2D arr) {
  auto var = CPU(kFloat).zeros({arr.rows, arr.cols});
  auto acs = var.accessor<float, 2>();

  for (int i = 0; i < arr.rows; i++) {
    for (int j = 0; j < arr.cols; j++) {
      acs[i][j] = arr.get(i, j);
    }
  }

  return make_variable(var);
}

vector<Tensor> dagnn_forward(
  Tensor x_var,
  Tensor W,
  Tensor b_var,
  Tensor i,
  Tensor o
) {

  int M = x_var.size(0);
  int I = *(i.toIntData());
  int O = *(o.toIntData());
  int N = W.size(0);

  W = W.coalesce();
  auto W_i = W._indices();
  auto W_v = W._values();
  auto W_i_acs = W_i.accessor<long, 2>();
  auto W_v_acs = W_v.accessor<float, 1>();

  auto b = var_to_arr1d(b_var);
  auto x_T = var_to_arr2d(x_var.transpose(0, 1));

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

  auto z = Array2D(N, M);
  auto a = Array2D(N, M);

  // load input
  for (int i = 0; i < I; i++) {
    for (int m = 0; m < M; m++) {
      z.set(i, m, x_T.get(i, m));
      a.set(i, m, x_T.get(i, m));
    }
  }

  // actual forward pass
  while (!F.empty()) {

    FEntry F_i = F.top();
    int i = F_i.i;

    for (auto jw_ij : F_i.c) {

      int j = jw_ij.first;
      float w_ij = jw_ij.second;

      for (int m = 0; m < M; m++) {
        z.set(i, m, z.get(i, m) + a.get(j, m) * w_ij);
      }

    }

    if (i >= N - O) for (int m = 0; m < M; m++) a.set(i, m, z.get(i, m));
    else for (int m = 0; m < M; m++) a.set(i, m, tanhf(z.get(i, m)));

    F.pop();

  }

  return {
    arr2d_to_var(a).transpose(0, 1).contiguous(),
    arr2d_to_var(z).transpose(0, 1).contiguous()
  };

}

vector<at::Tensor> dagnn_backward(
  at::Tensor W,
  at::Tensor b,
  at::Tensor i,
  at::Tensor o,
  at::Tensor a_var,
  at::Tensor dy
) {

  int M = a_var.size(0);
  int I = *(i.toIntData());
  int O = *(o.toIntData());
  int N = W.size(0);

  auto da_var = cat({
    make_variable(CPU(kFloat).zeros({M, N - O})),
    dy
  }, 1);

  auto a_T = var_to_arr2d(a_var.transpose(0, 1));
  auto da_T = var_to_arr2d(da_var.transpose(0, 1));
  auto dz_T = var_to_arr2d(da_var.transpose(0, 1));

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
  auto dW = W.clone();
  auto dW_i = dW._indices();
  auto dW_v = dW._values();
  auto dW_i_acs = dW_i.accessor<long, 2>();

  while (!B.empty()) {

    BEntry B_j = B.top();
    int j = B_j.j;

    for (auto iw_ij : B_j.c) {

      int i = iw_ij.first;
      float w_ij = iw_ij.second;

      for (int m = 0; m < M; m++) {
        da_T.set(j, m, da_T.get(j, m) + dz_T.get(i, m) * w_ij);
      }

      dW_i_acs[0][posW] = i;
      dW_i_acs[1][posW] = j;
      float dW_v_posW = 0.0f;
      for (int m = 0; m < M; m++) {
        dW_v_posW += dz_T.get(i, m) * a_T.get(j, m);
      }
      dW_v[posW] = dW_v_posW;
      posW ++;

    }

    if (j < I) for (int m = 0; m < M; m++) dz_T.set(j, m, da_T.get(j, m));
    else for (int m = 0; m < M; m++) {
      dz_T.set(j, m, da_T.get(j, m) * dtanh_from_tanh(a_T.get(j, m)));
    }

    B.pop();

  }

  auto db = Array1D(N);
  for (int i = 0; i < N; i++) {
    for (int m = 0; m < M; m++) {
      db.set(i, db.get(i) + dz_T.get(i, m));
    }
  }

  return {
    arr2d_to_var(dz_T).transpose(0, 1).contiguous(),
    dW,
    arr1d_to_var(db)
  };

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dagnn_forward, "DagNN forward");
  m.def("backward", &dagnn_backward, "DagNN backward");
}
