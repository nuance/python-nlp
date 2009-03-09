#include "Python.h"

#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "sloppy-math.h"

#define NLP_MODULE

#if PY_VERSION_HEX < 0x02050000
  typedef int Py_ssize_t;
  #define PY_SSIZE_T_MAX INT_MAX
  #define PY_SSIZE_T_MIN INT_MIN
#endif

#include "nlp.h"

typedef enum {
  false=0,
  true
} bool;

typedef struct {
  PyDictObject dict;
  double default_value;

  bool frozen;

  double frozen_arg_max;
} cnterobject;

/* counter type *********************************************************/

static int cnter_init(PyObject *self, PyObject *args, PyObject *kwds); /* Forward */
static PyObject * cnter_repr(cnterobject *dd);

PyDoc_STRVAR(cnter_missing_doc,
"__missing__(key) # Called by __getitem__ for missing key; pseudo-code:\n\
  if self.default_factory is None: raise KeyError((key,))\n\
  self[key] = value = self.default_factory()\n\
  return value\n\
");

static PyObject *
cnter_missing(cnterobject *dd, PyObject *key)
{
	PyObject *value = PyFloat_FromDouble(dd->default_value);
	if (PyObject_SetItem((PyObject *)dd, key, value) < 0) {
		Py_DECREF(value);
		return NULL;
	}
	return value;
}

PyDoc_STRVAR(cnter_copy_doc, "D.copy() -> a shallow copy of D.");

static PyObject *
cnter_copy(cnterobject *dd)
{
	/* This calls the object's class.  That only works for subclasses
	   whose class constructor has the same signature.  Subclasses that
	   define a different constructor signature must override copy().
	*/
  return PyObject_CallFunctionObjArgs((PyObject *)((PyObject*)dd)->ob_type, dd,
									  PyFloat_FromDouble(dd->default_value), NULL);
}

static PyObject *
cnter_reduce(cnterobject *dd)
{
	/* __reduce__ must return a 5-tuple as follows:

	   - factory function
	   - tuple of args for the factory function
	   - additional state (here None)
	   - sequence iterator (here None)
	   - dictionary iterator (yielding successive (key, value) pairs

	   This API is used by pickle.py and copy.py.

	   For this to be useful with pickle.py, the default_factory
	   must be picklable; e.g., None, a built-in, or a global
	   function in a module or package.

	   Both shallow and deep copying are supported, but for deep
	   copying, the default_factory must be deep-copyable; e.g. None,
	   or a built-in (functions are not copyable at this time).

	   This only works for subclasses as long as their constructor
	   signature is compatible; the first argument must be the
	   optional default_factory, defaulting to None.
	*/
	PyObject *items, *args, *result, *default_value;

	default_value = PyFloat_FromDouble(dd->default_value);
	args = PyTuple_Pack(1, default_value);
	Py_DECREF(default_value);

	if (args == NULL)
	  return NULL;

	items = PyObject_CallMethod((PyObject *)dd, "iteritems", "()");
	if (items == NULL) {
		Py_DECREF(args);
		return NULL;
	}

	result = PyTuple_Pack(5, ((PyObject*)dd)->ob_type, args, Py_None, Py_None, items);
	Py_DECREF(args);
	Py_DECREF(items);

	return result;
}

PyDoc_STRVAR(reduce_doc, "Return state information for pickling.");

static PyObject *
cnter_normalize(cnterobject *dd)
{
	Py_ssize_t i;
	PyObject *key, *value;
	double sum = 0.0;

	i = 0;
	while (PyDict_Next((PyObject*)dd, &i, &key, &value)) {
		sum += PyFloat_AsDouble(value);
	}
	
	if (sum == 0.0) {
	  Py_ssize_t len = PyDict_Size((PyObject*)dd);
	  PyObject *uniform = PyFloat_FromDouble(1.0 / (double)len);

	  i = 0;
	  while (PyDict_Next((PyObject*)dd, &i, &key, &value)) {
		int ok;

		ok = PyDict_SetItem((PyObject*)dd, key, uniform);
		if (ok < 0) {
		  Py_DECREF(uniform);
		  return NULL;
		}
	  }
	  Py_DECREF(uniform);

	  Py_INCREF(Py_None);
	  return Py_None;
	}

	i = 0;
	while (PyDict_Next((PyObject*)dd, &i, &key, &value)) {
	  int ok;

	  PyObject *newValue = PyFloat_FromDouble(PyFloat_AsDouble(value) / sum);
	  ok = PyDict_SetItem((PyObject*)dd, key, newValue);
	  Py_DECREF(newValue);

	  if (ok < 0) return NULL;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}

PyDoc_STRVAR(cnter_normalize_doc, "D.normalize() -> normalizes the values in D, returns None");

static PyObject *
cnter_log_normalize(cnterobject *dd)
{
	Py_ssize_t i;
	PyObject *key, *value;
	double log_sum = 0.0;

	i = 0;
	while (PyDict_Next((PyObject*)dd, &i, &key, &value)) {
	  log_sum += sloppy_exp(PyFloat_AsDouble(value));
	}

	log_sum = log(log_sum);

	i = 0;
	while (PyDict_Next((PyObject*)dd, &i, &key, &value)) {
	  int ok;

	  PyObject *newValue = PyFloat_FromDouble(PyFloat_AsDouble(value) - log_sum);
	  ok = PyDict_SetItem((PyObject*)dd, key, newValue);
	  Py_DECREF(newValue);

	  if (ok < 0) return NULL;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}

PyDoc_STRVAR(cnter_log_normalize_doc, "D.log_normalize() -> normalizes the log counts in D, returns None");

static PyObject *
cnter_log(cnterobject *dd)
{
	Py_ssize_t i;
	PyObject *key, *value;

	i = 0;
	while (PyDict_Next((PyObject*)dd, &i, &key, &value)) {
	  int ok;

	  PyObject *newValue = PyFloat_FromDouble(log(PyFloat_AsDouble(value)));
	  ok = PyDict_SetItem((PyObject*)dd, key, newValue);
	  Py_DECREF(newValue);

	  if (ok < 0) return NULL;
	}
	
	dd->default_value = log(dd->default_value);
	
	Py_INCREF(Py_None);
	return Py_None;
}

PyDoc_STRVAR(cnter_log_doc, "D.log() -> in place logs the counts in D and the default value, returns None");

static PyObject *
cnter_pow(PyObject *dd, PyObject *other, PyObject *modulo)
{
	Py_ssize_t i;
	PyObject *key, *value;

	if ((PyInt_Check(dd) || PyFloat_Check(dd) || PyLong_Check(dd)) && NlpCounter_Check(other))
	  return cnter_pow(other, dd, modulo);
	
	if (!(PyInt_Check(other) || PyFloat_Check(other) || PyLong_Check(other)) && NlpCounter_Check(dd)) {
	  PyErr_SetString(PyExc_ValueError, "Counter __pow__ requires a counter and a number"); 
	  return NULL;
	}

	double scalar;

	if (PyInt_Check(other)) scalar = (double)PyInt_AsLong(other);
	else if (PyLong_Check(other)) scalar = (double)PyLong_AsLong(other);
	else scalar = PyFloat_AsDouble(other);

	PyObject *ret_cnter = NlpCounter_Type.tp_new(&NlpCounter_Type, NULL, NULL);

	/* Set the default value*/
	((cnterobject *)ret_cnter)->default_value = pow(((cnterobject *)dd)->default_value, scalar);

	i = 0;
	while (PyDict_Next(dd, &i, &key, &value)) {
	  int ok;

	  PyObject *newValue = PyFloat_FromDouble(pow(PyFloat_AsDouble(value), scalar));
	  ok = PyDict_SetItem(ret_cnter, key, newValue);
	  Py_DECREF(newValue);

	  if (ok < 0) return NULL;
	}
	
	return (PyObject*) ret_cnter;
}

PyDoc_STRVAR(cnter_pow_doc, "D ** i -> raises the counts in D and the default value to the power i and returns the results. NOTE: this ignore the optional modulo argument");

static PyObject *
cnter_exp(cnterobject *dd)
{
	Py_ssize_t i;
	PyObject *key, *value;

	i = 0;
	while (PyDict_Next((PyObject*)dd, &i, &key, &value)) {
	  int ok;

	  PyObject *newValue = PyFloat_FromDouble(sloppy_exp(PyFloat_AsDouble(value)));
	  ok = PyDict_SetItem((PyObject*)dd, key, newValue);
	  Py_DECREF(newValue);

	  if (ok < 0) return NULL;
	}
	
	dd->default_value = sloppy_exp(dd->default_value);
	
	Py_INCREF(Py_None);
	return Py_None;
}

PyDoc_STRVAR(cnter_exp_doc, "D.exp() -> in place exps the counts in D and the default value, returns None");

static PyObject *
cnter_total_count(cnterobject *dd)
{
	Py_ssize_t i;
	PyObject *key, *value;
	double sum = 0.0;
	int seen = 0;
	
	i = 0;
	while (PyDict_Next((PyObject*)dd, &i, &key, &value)) {
		sum += PyFloat_AsDouble(value);
		seen = 1;
	}

	if (seen)
		return PyFloat_FromDouble(sum);
	else
		return PyFloat_FromDouble(dd->default_value);
}

PyDoc_STRVAR(cnter_total_count_doc, "D.total_count() -> sum of the values in D");

static PyObject *
cnter_arg_max(cnterobject *dd)
{
	Py_ssize_t i;
	PyObject *key, *value;
	PyObject *arg_max = NULL;
	double max = 0.0;
	double running;

	i = 0;
	while (PyDict_Next((PyObject*)dd, &i, &key, &value)) {
		running = PyFloat_AsDouble(value);
		if (arg_max == NULL || max < running) {
			Py_XDECREF(arg_max);
			arg_max = key;
			Py_INCREF(arg_max);
			max = running;
		}
	}

	if (!arg_max) {
		Py_INCREF(Py_None);
		return Py_None;
	}
	
	return arg_max;
}

PyDoc_STRVAR(cnter_arg_max_doc, "D.arg_max() -> arg max of the items in D");

static PyObject *
cnter_max(cnterobject *dd)
{
	Py_ssize_t i;
	PyObject *key, *value;
	double max = 0.0;
	double running;
	int found = 0;

	i = 0;
	while (PyDict_Next((PyObject*)dd, &i, &key, &value)) {
		running = PyFloat_AsDouble(value);
		if (max < running || found == 0) {
			max = running;
			found = 1;
		}
	}
	
	if (found == 0)
		return PyFloat_FromDouble(dd->default_value);

	return PyFloat_FromDouble(max);
}

PyDoc_STRVAR(cnter_max_doc, "D.max() -> max of the items in D");

static PyObject *
cnter_inner_product(PyObject *dd, PyObject *other)
{
	Py_ssize_t i;
	PyObject *key, *value;

	if (!NlpCounter_Check(dd) || !NlpCounter_Check(other)) {
	  PyErr_SetString(PyExc_ValueError, "Counter inner_product requires two counters"); 
	  return NULL;
	}

	double ret = 0.0;

	/* Walk through all the keys in other and add value * dd->default if they're not in dd */
	i = 0;
	while (PyDict_Next(other, &i, &key, &value)) {
		int contains = PyDict_Contains(dd, key);

		if (contains == 0) {
		  ret += ((cnterobject*)dd)->default_value * PyFloat_AsDouble(value);
		} 
		else if (contains < 0) {
		  return NULL;
		}
	}

	i = 0;
	while (PyDict_Next(dd, &i, &key, &value)) {
	  PyObject *otherValue = PyDict_GetItem(other, key);

	  if (otherValue != NULL) {
		ret += PyFloat_AsDouble(PyDict_GetItem(other, key)) * PyFloat_AsDouble(value);
	  }
	  else {
		ret += ((cnterobject*)other)->default_value * PyFloat_AsDouble(value);
	  } 
	}

	return PyFloat_FromDouble(ret);;
}

PyDoc_STRVAR(cnter_inner_product_doc, "D.inner_product(O) -> inner product of D and O");

#define SCALAR_OP(fn_name, OP) \
static PyObject *\
fn_name(cnterobject *cnter, PyObject *other)\
{\
  Py_ssize_t i;\
  PyObject *key, *value;\
  double scalar;\
\
  if (PyInt_Check(other)) scalar = (double)PyInt_AsLong(other);\
  else if (PyLong_Check(other)) scalar = (double)PyLong_AsLong(other);\
  else scalar = PyFloat_AsDouble(other);\
\
  cnterobject *ret_cnter = NULL;\
\
  ret_cnter = (cnterobject *)NlpCounter_Type.tp_new(&NlpCounter_Type, NULL, NULL);\
\
  ret_cnter->default_value = cnter->default_value OP scalar;\
\
  /* Copy cnter into ret_cnter */ \
  if (PyDict_Update((PyObject*)ret_cnter, (PyObject*)cnter) < 0)\
   	return NULL;\
\
  i = 0;\
  while (PyDict_Next((PyObject*)&(ret_cnter->dict), &i, &key, &value)) {\
	int ok;\
\
	PyObject *newValue = PyFloat_FromDouble(PyFloat_AsDouble(value) OP scalar);\
	ok = PyDict_SetItem((PyObject*)&(ret_cnter->dict), key, newValue);\
	Py_DECREF(newValue);\
\
	if (ok < 0) {\
	  Py_DECREF((PyObject*)ret_cnter);\
	  return NULL;\
	}\
  }\
\
  return (PyObject*)ret_cnter;\
}

#define CNTER_OP(FN_NAME, OP) \
SCALAR_OP(FN_NAME ## _scalar, OP)\
\
static PyObject *\
FN_NAME(PyObject *dd, PyObject *other)\
{\
  Py_ssize_t i;\
  PyObject *key, *value;\
\
  if ((PyInt_Check(other) || PyFloat_Check(other) || PyLong_Check(other)) && NlpCounter_Check(dd)) \
		return FN_NAME ## _scalar((cnterobject*)dd, other);\
\
  if ((PyInt_Check(dd) || PyFloat_Check(dd) || PyLong_Check(dd)) && NlpCounter_Check(other)) \
		return FN_NAME ## _scalar((cnterobject*)other, dd);\
\
  if (!NlpCounter_Check(dd) || !NlpCounter_Check(other)) {\
    PyErr_SetString(PyExc_ValueError, "Counter " #OP " requires two counters or a counter and a scalar");\
    return NULL;\
  }\
\
  cnterobject *dd_cnter = (cnterobject*)dd;\
  cnterobject *other_cnter = (cnterobject*)other;\
  cnterobject *ret_cnter = (cnterobject *)NlpCounter_Type.tp_new(&NlpCounter_Type, NULL, NULL);\
\
  /* Set the default value*/\
  ret_cnter->default_value = dd_cnter->default_value OP other_cnter->default_value;\
\
  int source;\
  for (source = 0; source < 2; source++) {\
		PyObject *src, *oth;\
\
		if (source == 0) {\
		  src = other;\
		  oth = (PyObject*)dd;\
		}\
		else {\
		  oth = other;\
		  src = (PyObject*)dd;\
		}\
\
		i = 0;\
		while (PyDict_Next(src, &i, &key, &value)) {\
		  int contains = PyDict_Contains(oth, key);\
\
		  /* If we have a key other doesn't have, set the new value to our value OP other->default*/\
		  if (contains == 0) {\
			PyObject *newValue;\
			if (src == dd) newValue = PyFloat_FromDouble(PyFloat_AsDouble(value) OP ((cnterobject*)oth)->default_value);\
			else newValue = PyFloat_FromDouble(((cnterobject*)oth)->default_value OP PyFloat_AsDouble(value));\
				int ok = PyDict_SetItem((PyObject*)ret_cnter, key, newValue);\
				Py_DECREF(newValue);\
\
				if (ok < 0) {\
			  	Py_DECREF((PyObject*)ret_cnter);\
			  	return NULL;\
				}\
		  }\
		  /* If we both have the key, OP the values (only on the first time around)*/\
		  else if (contains > 0 && src == other) {\
				/* both counters have the key, so just increment our count if we're the first source (not dd)*/\
				PyObject *currentValue = PyDict_GetItem((PyObject*)dd, key);\
				PyObject *newValue = PyFloat_FromDouble(PyFloat_AsDouble(currentValue) OP PyFloat_AsDouble(value));\
				int ok = PyDict_SetItem((PyObject*)ret_cnter, key, newValue);\
				Py_DECREF(newValue);\
\
				if (ok < 0) {\
			  	Py_DECREF((PyObject*)ret_cnter);\
			  	return NULL;\
				}\
	  	}\
	  	/* Else, there was an error*/\
	  	else if (contains < 0) {\
				Py_DECREF((PyObject*)ret_cnter);\
				return NULL;\
	  	}\
		}\
  }\
\
  return (PyObject*)ret_cnter;\
}

#define SCALAR_IOP(FN_NAME, OP) \
static PyObject *\
FN_NAME(cnterobject *dd, PyObject *other)\
{\
  Py_ssize_t i;\
  PyObject *key, *value;\
  double scale;\
\
  if (PyInt_Check(other)) scale = (double)PyInt_AsLong(other);\
  else if (PyLong_Check(other)) scale = (double)PyLong_AsLong(other);\
  else scale = PyFloat_AsDouble(other);\
\
  dd->default_value = dd->default_value OP scale; \
\
  i = 0;\
  while (PyDict_Next((PyObject*)dd, &i, &key, &value)) {\
	int ok;\
\
	PyObject *newValue = PyFloat_FromDouble(PyFloat_AsDouble(value) OP scale);\
	ok = PyDict_SetItem((PyObject*)dd, key, newValue);\
	Py_DECREF(newValue);\
\
	if (ok < 0) {\
	  return NULL;\
	}\
  }\
\
  Py_INCREF((PyObject*)dd);\
  return (PyObject*)dd;\
}

#define CNTER_IOP(FN_NAME, OP) \
SCALAR_IOP(FN_NAME ## _scalar, OP)\
\
static PyObject *\
FN_NAME(PyObject *dd, PyObject *other)\
{\
	Py_ssize_t i;\
	PyObject *key, *value;\
\
	if ((PyInt_Check(other) || PyFloat_Check(other) || PyLong_Check(other)) && NlpCounter_Check(dd)) \
	  return FN_NAME ## _scalar((cnterobject*)dd, other);\
\
	if ((PyInt_Check(dd) || PyFloat_Check(dd) || PyLong_Check(dd)) && NlpCounter_Check(other)) \
	  return FN_NAME ## _scalar((cnterobject*)other, dd);\
\
	if (!NlpCounter_Check(dd) || !NlpCounter_Check(other)) {\
	  PyErr_SetString(PyExc_ValueError, "Counter in-place " #OP " requires two counters or a counter and a scalar"); \
	  return NULL;\
	}\
\
	/* Walk through all the keys in other and fetch them from dd, thus creating 0.0 items for any missing keys*/\
	i = 0;\
	PyObject *defaultValue = PyFloat_FromDouble(((cnterobject*)dd)->default_value);\
	while (PyDict_Next(other, &i, &key, &value)) {\
		int contains = PyDict_Contains(dd, key);\
		/* If the key is not in the dictionary, try to set it to the default value (and fail on exception as appropriate)*/\
		if (contains == 0 && PyDict_SetItem(dd, key, defaultValue) < 0) {\
		  Py_DECREF(defaultValue);\
		  return NULL;\
		}\
		else if (contains < 0) {\
		  Py_DECREF(defaultValue);\
		  return NULL;\
		}\
	}\
\
	/* Update the default values */\
	((cnterobject*)dd)->default_value = ((cnterobject*)dd)->default_value OP ((cnterobject*)other)->default_value;\
\
	Py_DECREF(defaultValue);\
	defaultValue = PyFloat_FromDouble(((cnterobject*)other)->default_value);\
	i = 0;\
	while (PyDict_Next(dd, &i, &key, &value)) {\
	  int ok;\
	  PyObject *otherValue = PyDict_GetItem(other, key);\
	  if (otherValue == NULL) otherValue = defaultValue;\
		\
	  PyObject *newValue = PyFloat_FromDouble(PyFloat_AsDouble(value) OP PyFloat_AsDouble(otherValue));\
	  ok = PyDict_SetItem(dd, key, newValue);\
	  Py_DECREF(newValue);\
\
	  if (ok < 0) {\
		Py_DECREF(defaultValue);\
		return NULL;\
	  }\
	}\
\
	Py_DECREF(defaultValue);\
	Py_INCREF(dd);\
	return dd;\
}

CNTER_OP(cnter_mul, *)

CNTER_OP(cnter_div, /)

CNTER_OP(cnter_add, +)

CNTER_OP(cnter_sub, -)

CNTER_IOP(cnter_imul, *)

CNTER_IOP(cnter_idiv, /)

CNTER_IOP(cnter_iadd, +)

CNTER_IOP(cnter_isub, -)

static PyObject *
cnter_freeze(cnterobject *dd)
{
  if (dd->frozen) {
	Py_RETURN_NONE;
  }

  dd->frozen_arg_max = PyFloat_AsDouble(cnter_arg_max(dd));
  dd->frozen = true;

  Py_RETURN_NONE;
}

PyDoc_STRVAR(cnter_freeze_doc, "D.freeze() -> computes some static statistics, sets D.frozen to True");

static PyObject *
cnter_sample(cnterobject *dd)
{
  double point = ((double)rand() / (((double)RAND_MAX) + 1));
  double running = 0.0;
  Py_ssize_t i = 0;
  PyObject *key, *value;

  while (PyDict_Next((PyObject*)dd, &i, &key, &value)) {
	running += PyFloat_AsDouble(value);
	if (running >= point) {
	  Py_INCREF(key);
	  return key;
	}
  }

  PyErr_SetString(PyExc_ValueError, "sampling didn't find a point!");
  return NULL;
}

PyDoc_STRVAR(cnter_sample_doc, "D.sample() -> Randomly samples from the counter, assumes (for now) that it's a 0-1 distribution");

static PyMethodDef cnter_methods[] = {
	{"__missing__", (PyCFunction)cnter_missing, METH_O,
	 cnter_missing_doc},
	{"copy", (PyCFunction)cnter_copy, METH_NOARGS,
	 cnter_copy_doc},
	{"__copy__", (PyCFunction)cnter_copy, METH_NOARGS,
	 cnter_copy_doc},
	{"__reduce__", (PyCFunction)cnter_reduce, METH_NOARGS,
	 reduce_doc},
	{"normalize", (PyCFunction)cnter_normalize, METH_NOARGS,
	 cnter_normalize_doc},
	{"log_normalize", (PyCFunction)cnter_log_normalize, METH_NOARGS,
	 cnter_log_normalize_doc},
	{"log", (PyCFunction)cnter_log, METH_NOARGS, cnter_log_doc},
	{"exp", (PyCFunction)cnter_exp, METH_NOARGS, cnter_exp_doc},
	{"total_count", (PyCFunction)cnter_total_count, METH_NOARGS,
	 cnter_total_count_doc},
	{"arg_max", (PyCFunction)cnter_arg_max, METH_NOARGS,
	 cnter_arg_max_doc},
	{"max", (PyCFunction)cnter_max, METH_NOARGS, cnter_max_doc},
	{"inner_product", (PyCFunction)cnter_inner_product, METH_O, cnter_inner_product_doc},
	{"freeze", (PyCFunction)cnter_freeze, METH_NOARGS, cnter_freeze_doc},
	{"sample", (PyCFunction)cnter_sample, METH_NOARGS, cnter_sample_doc},
	{NULL}
};

static PyObject *
cnter_getdefault(cnterobject *self, void *unused) {
  return PyFloat_FromDouble(self->default_value);
}

static int
cnter_setdefault(cnterobject *self, PyObject *number, void *unused)
{
  if (!(PyInt_Check(number) || PyFloat_Check(number) || PyLong_Check(number)))
	return 1;

  if (PyInt_Check(number)) self->default_value = (double)PyInt_AsLong(number);
  else if (PyLong_Check(number)) self->default_value = (double)PyLong_AsLong(number);
  else self->default_value = PyFloat_AsDouble(number);

  return 0;
}

static PyGetSetDef cnter_getset[] = {
	{"default", (getter)cnter_getdefault, (setter)cnter_setdefault},
	{NULL}
};

// 
// static PyMemberDef cnter_members[] = {
// 	{NULL}
// };

static void
cnter_dealloc(cnterobject *dd)
{
	PyDict_Type.tp_dealloc((PyObject *)dd);
}

static int
cnter_print(cnterobject *dd, FILE *fp, int flags)
{
	int sts;
	fprintf(fp, "counter(");
	sts = PyDict_Type.tp_print((PyObject *)dd, fp, 0);
	fprintf(fp, ", default: %f)", dd->default_value);
	return sts;
}

static PyObject *
cnter_repr(cnterobject *dd)
{
	PyObject *baserepr;
	PyObject *result;
	baserepr = PyDict_Type.tp_repr((PyObject *)dd);
	if (baserepr == NULL)
		return NULL;
	result = PyString_FromFormat("counter(%s, default=%d)", PyString_AS_STRING(baserepr), (int)dd->default_value);
	Py_DECREF(baserepr);
	return result;
}

static int
cnter_traverse(PyObject *self, visitproc visit, void *arg)
{
  return PyDict_Type.tp_traverse(self, visit, arg);
}

static int
cnter_init(PyObject *self, PyObject *args, PyObject *kwds)
{
  PyObject *newargs;
  PyObject *newdefault = NULL;

  if (args == NULL || !PyTuple_Check(args))
	newargs = PyTuple_New(0);
  else {
	Py_ssize_t n = PyTuple_GET_SIZE(args);

	if (n == 1) {
	  newdefault = PyTuple_GET_ITEM(args, 0);
	  if (!(PyInt_Check(newdefault) || PyFloat_Check(newdefault) || PyLong_Check(newdefault))) {
		newdefault = NULL;
		newargs = args;
		Py_INCREF(newargs);
	  }
	  else {
		newargs = PyTuple_New(0);
	  }
	} else if (n == 2) {
	  newdefault = PyTuple_GET_ITEM(args, 1);
	  if (!(PyInt_Check(newdefault) || PyFloat_Check(newdefault) || PyLong_Check(newdefault))) {
		PyErr_SetString(PyExc_TypeError,
						"second argument must be float");                           
		return -1;
	  }
	  else {
		newargs = PySequence_GetSlice(args, 0, 1);
	  }
	} else if (n == 0) {
	  newargs = args;
	  Py_INCREF(newargs);
	} else {
		PyErr_SetString(PyExc_TypeError,
						"counter takes at most 2 arguments");                           
		return -1;
	}
  }

  if (newargs == NULL)
	return -1;

  if (kwds == NULL || !PyDict_Check(kwds))
	kwds = PyDict_New();
  else
	Py_INCREF(kwds);

  int result = PyDict_Type.tp_init(self, newargs, kwds);
  Py_DECREF(newargs);
  Py_DECREF(kwds);

  if (newdefault)
	((cnterobject*)self)->default_value = PyFloat_AsDouble(newdefault);
  else
	((cnterobject*)self)->default_value = 0.0;

  ((cnterobject*)self)->frozen = false;

  return result;
}

/* C interfaces */

PyObject *
NlpCounter_New(void)
{
  cnterobject *mp;

  mp = (cnterobject *)NlpCounter_Type.tp_new(&NlpCounter_Type, NULL, NULL);

  return (PyObject *)mp;
}

int
NlpCounter_Normalize(PyObject *op)
{
  cnterobject *mp;

  if (!NlpCounter_Check(op)) {
	PyErr_BadArgument();
	return -1;
  }
  
  mp = (cnterobject*)op;
  cnter_normalize(mp);
  return 1;
}

int
NlpCounter_LogNormalize(PyObject *op)
{
  cnterobject *mp;

  if (!NlpCounter_Check(op)) {
	PyErr_BadArgument();
	return -1;
  }
  
  mp = (cnterobject*)op;
  cnter_log_normalize(mp);
  return 1;
}

// Convenience method for either getting the value for the key
// or returning the default value
// NOTE: This doesn't do any type checking (for speed purposes)
// NOTE: same ref-counting semantics as PyDict_GetItem
PyObject*
NlpCounter_XGetItem(PyObject *cnter, PyObject *key)
{
  PyObject *value = PyDict_GetItem(cnter, key);

  if (!value)
	return PyFloat_FromDouble(((cnterobject*)cnter)->default_value);

  return value;
}

double
NlpCounter_XGetDouble(PyObject *cnter, PyObject *key)
{
  PyObject *value = PyDict_GetItem(cnter, key);

  if (!value)
	return ((cnterobject*)cnter)->default_value;

  return PyFloat_AsDouble(value);
}

int
NlpCounter_SetDefault(PyObject *cnter, double new_default)
{
  ((cnterobject*)cnter)->default_value = new_default;
  return 1;
}

double
NlpCounter_GetDefault(PyObject *cnter)
{
  return ((cnterobject*)cnter)->default_value;
}


/****************/

PyDoc_STRVAR(cnter_doc,
"counter() --> dict with default of 0.0\n\
\n\
The default factory is called without arguments to produce\n\
a new value when a key is not present, in __getitem__ only.\n\
A counter compares equal to a dict with the same items.\n\
");

/* See comment in xxsubtype.c */
#define DEFERRED_ADDRESS(ADDR) 0

static PyNumberMethods cnter_as_number = {
    (binaryfunc) cnter_add,				/*nb_add*/
	(binaryfunc) cnter_sub,				/*nb_subtract*/
    (binaryfunc) cnter_mul,				/*nb_multiply*/
	(binaryfunc) cnter_div,				/*nb_divide*/
	0,				/*nb_remainder*/
	0,				/*nb_divmod*/
	(ternaryfunc) cnter_pow,				/*nb_power*/
	0,				/*nb_negative*/
	0,				/*nb_positive*/
	0,				/*nb_absolute*/
	0,				/*nb_nonzero*/
	0,				/*nb_invert*/
	0,				/*nb_lshift*/
	0,				/*nb_rshift*/
	0,				/*nb_and*/
	0,				/*nb_xor*/
	0,				/*nb_or*/
	0,				/*nb_coerce*/
	0,				/*nb_int*/
	0,				/*nb_long*/
	0,				/*nb_float*/
	0,				/*nb_oct*/
	0, 				/*nb_hex*/
	(binaryfunc) cnter_iadd,		/*nb_inplace_add*/
    (binaryfunc) cnter_isub,		/*nb_inplace_subtract*/
	(binaryfunc) cnter_imul,		/*nb_inplace_multiply*/
	(binaryfunc) cnter_idiv,		/*nb_inplace_divide*/
	0,				/*nb_inplace_remainder*/
	0,				/*nb_inplace_power*/
	0,				/*nb_inplace_lshift*/
	0,				/*nb_inplace_rshift*/
	0,				/*nb_inplace_and*/
	0,				/*nb_inplace_xor*/
	0,				/*nb_inplace_or*/
};

PyTypeObject NlpCounter_Type = {
	PyObject_HEAD_INIT(DEFERRED_ADDRESS(&PyType_Type))
	0,				/* ob_size */
	"nlp.counter",	/* tp_name */
	sizeof(cnterobject),		/* tp_basicsize */
	0,				/* tp_itemsize */
	/* methods */
	(destructor)cnter_dealloc,	/* tp_dealloc */
	(printfunc)cnter_print,	/* tp_print */
	0,				/* tp_getattr */
	0,				/* tp_setattr */
	0,				/* tp_compare */
	(reprfunc)cnter_repr,		/* tp_repr */
	&cnter_as_number,				/* tp_as_number */
	0,				/* tp_as_sequence */
	0,				/* tp_as_mapping */
	0,	       			/* tp_hash */
	0,				/* tp_call */
	0,				/* tp_str */
	PyObject_GenericGetAttr,	/* tp_getattro */
	0,				/* tp_setattro */
	0,				/* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC |
		Py_TPFLAGS_HAVE_WEAKREFS | Py_TPFLAGS_CHECKTYPES,	/* tp_flags */
	cnter_doc,			/* tp_doc */
	cnter_traverse,		/* tp_traverse */
	0,				/* tp_clear */
	0,				/* tp_richcompare */
	0,				/* tp_weaklistoffset*/
	0,				/* tp_iter */
	0,				/* tp_iternext */
	cnter_methods,		/* tp_methods */
	0, //cnter_members,		/* tp_members */
	cnter_getset,				/* tp_getset */
	DEFERRED_ADDRESS(&PyDict_Type),	/* tp_base */
	0,				/* tp_dict */
	0,				/* tp_descr_get */
	0,				/* tp_descr_set */
	0,				/* tp_dictoffset */
	(initproc)cnter_init,			/* tp_init */
	PyType_GenericAlloc,		/* tp_alloc */
	0,				/* tp_new */
	PyObject_GC_Del,		/* tp_free */
};

/* module level code ********************************************************/

PyDoc_STRVAR(module_doc,
"High performance nlp data structures, based on collections code.\n\
- counter:  dict subclass, defaults to 0.0 value & implements some extra functionality\n\
");

PyMODINIT_FUNC
initnlp(void)
{
	PyObject *m;
	static void *NlpCounter_API[NlpCounter_API_pointers];
	PyObject *c_api_object;

	m = Py_InitModule3("nlp", NULL, module_doc);
	if (m == NULL)
		return;

	NlpCounter_Type.tp_base = &PyDict_Type;
	if (PyType_Ready(&NlpCounter_Type) < 0)
		return;
	Py_INCREF(&NlpCounter_Type);
	PyModule_AddObject(m, "counter", (PyObject *)&NlpCounter_Type);

	NlpCounter_API[0] = (void *)NlpCounter_New;
	NlpCounter_API[1] = (void *)NlpCounter_Normalize;
	NlpCounter_API[2] = (void *)NlpCounter_LogNormalize;
	NlpCounter_API[3] = (void *)NlpCounter_XGetItem;
	NlpCounter_API[4] = (void *)NlpCounter_XGetDouble;
	NlpCounter_API[5] = (void *)NlpCounter_SetDefault;
	NlpCounter_API[6] = (void *)NlpCounter_GetDefault;

	c_api_object = PyCObject_FromVoidPtr((void *)NlpCounter_API, NULL);

	srandomdev();

	if (c_api_object != NULL)
	  PyModule_AddObject(m, "_C_API", c_api_object);

	return;
}
