#include <Python.h>

// NLP C module
#include "nlp.h"

PyObject *counterType;

static PyObject* maxent_log_probs(PyObject *self, PyObject *args) {
  PyObject *features, *weights, *labels;
  PyObject *label;
  Py_ssize_t i;
  int ok;

  if (!PyArg_ParseTuple(args, "OOO", &features, &weights, &labels))
    return NULL;

  // This is crap for testing - should see if weights is a countermap, not just a dict
  // (at min, check if values of weights are dicts)
  if (!PyDict_Check(features) || ! PyDict_Check(weights) || ! PyAnySet_Check(labels))
	return NULL;

// log_probs = Counter()
  PyObject *log_probs = NlpCounter_New();

// for label in self.labels:
  i = 0;
  while (_PySet_Next(labels, &i, &label)) {
//   log_probs[label] = sum((weights[label] * datum_features).itervalues())
	PyObject *newValue;
	double sum = 0.0;
	Py_ssize_t j;
	PyObject *featureKey, *featureCount;
	PyObject *labelWeights = PyDict_GetItem(weights, label);;
	
	j = 0;
	while (PyDict_Next((PyObject*)features, &j, &featureKey, &featureCount)) {
	  PyObject *weight = PyDict_GetItem(labelWeights, featureKey);

	  sum += PyFloat_AsDouble(featureCount) * PyFloat_AsDouble(weight);
	}

	newValue = PyFloat_FromDouble(sum);
	ok = PyDict_SetItem(log_probs, label, newValue);

	Py_DECREF(newValue);
	if (ok < 0) {
	  Py_DECREF(log_probs);
	  return NULL;
	}
  }

  // log_probs.log_normalize()
  NlpCounter_LogNormalize(log_probs);

  return log_probs;
}

PyDoc_STRVAR(module_doc, "Maximum Entropy function implementation");

static PyMethodDef maxent_methods[] = {
  {"get_log_probabilities", maxent_log_probs, METH_VARARGS, "Calculate the log probs for a datum"},
  {NULL, NULL, 0, NULL} /* Sentinel */
};

PyMODINIT_FUNC
initmaxent(void) 
{
    PyObject* m;
    m = Py_InitModule3("maxent", maxent_methods, module_doc);
	if (m == NULL)
	  return;

	if (import_nlp() < 0)
	  return;
}
