#include <Python.h>

#include <gsl/gsl_randist.h>

static const double ALPHA = 0.0001;
static const int ITERATIONS = 10000;
static const int BURN = 1000;
static const int THIN = 100;

static gsl_rng *rng;

static PyObject *clda_infer_topics_gibbs(PyObject *self, PyObject *args) {
  PyObject *doc0;
  PyObject *topics0;
  if (!PyArg_ParseTuple(args, "OO", &doc0, &topics0))
    return NULL;
  int N = (int)PySequence_Length(doc0);
  int K = (int)PySequence_Length(topics0);
  int V = (int)PySequence_Length(PySequence_GetItem(topics0, 0));
  int *doc = malloc(N * sizeof(int));
  double *topics = malloc(K * V * sizeof(double));
  int i, j, k;

  for (i = 0; i < N; i++)
    doc[i] = (int)PyLong_AsLong(PySequence_GetItem(doc0, i));

  PyObject *topic0;
  for (k = 0; k < K; k++) {
    topic0 = PySequence_GetItem(topics0, k);
    for (i = 0; i < V; i++)
      topics[k*V + i] = PyFloat_AsDouble(PySequence_GetItem(topic0, i));
  }


  double *alpha = malloc(K * sizeof(double));
  double *theta = malloc(K * sizeof(double));
  int *zs = malloc(N * sizeof(int));
  int *topic_counts = malloc(K * sizeof(int));
  double *ps = malloc(K * sizeof(double));
  double *sum_thetas = malloc(K * sizeof(double));
  for (k = 0; k < K; k++) sum_thetas[k] = 0;
  int n_thetas = 0;

  for (k = 0; k < K; k++) alpha[k] = ALPHA;
  gsl_ran_dirichlet(rng, K, alpha, theta);

  for (j = 0; j < ITERATIONS; j++) {
    for (i = 0; i < N; i++) {
      double sum_ps = 0;
      for (k = 0; k < K; k++) {
        ps[k] = theta[k] * topics[k*V + doc[i]];
        sum_ps += ps[k];
      }
      // if p = 0, k-1 = first nonzero k. if p = 1, k-1 = K-1:
      double p = gsl_rng_uniform(rng);
      for (k = 0; p >= 0 && k < K; k++)
        p -= ps[k] / sum_ps;
      zs[i] = k - 1;
    }

    for (k = 0; k < K; k++) alpha[k] = ALPHA;
    for (i = 0; i < N; i++) alpha[zs[i]]++;
    gsl_ran_dirichlet(rng, K, alpha, theta);

    if (j > BURN && j % THIN == 0) {
      for (k = 0; k < K; k++)
        sum_thetas[k] += theta[k];
      n_thetas++;
    }
  }

  PyObject *pytheta = PyList_New(K);
  for (int k = 0; k < K; k++)
    PyList_SET_ITEM(pytheta, k, PyFloat_FromDouble(sum_thetas[k] / n_thetas));

  free(doc);
  free(topics);
  //free(alpha);
  free(theta);
  free(zs);
  free(topic_counts);
  free(ps);
  free(sum_thetas);

  return pytheta;
}


static PyMethodDef CldaMethods[] = {
  {"infer_topics_gibbs", clda_infer_topics_gibbs, METH_VARARGS,
   "Infer topics using Gibbs sampling."},
  {NULL, NULL, 0, NULL}
};

/* static struct PyModuleDef cldamodule = { */
/*   PyModuleDef_HEAD_INIT, */
/*   "clda", */
/*   NULL, */
/*   -1, */
/*   CldaMethods */
/* }; */

/* PyMODINIT_FUNC */
/* PyInit_clda(void) { */
/*   return PyModule_Create(&cldamodule); */
/* } */

PyMODINIT_FUNC
initclda(void) {
  gsl_rng_env_setup();
  rng = gsl_rng_alloc(gsl_rng_default);

  Py_InitModule("clda", CldaMethods);
}
