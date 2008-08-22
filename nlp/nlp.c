#include "Python.h"
#include <math.h>

/* counter type *********************************************************/

typedef struct {
  PyDictObject dict;
} cnterobject;

static PyTypeObject cnter_type; /* Forward */

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
	PyObject *value = PyFloat_FromDouble(0.0);
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
	return PyObject_CallFunctionObjArgs((PyObject *)dd->dict.ob_type,
					    dd, NULL);
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
	PyObject *items;
	PyObject *result;
	items = PyObject_CallMethod((PyObject *)dd, "iteritems", "()");
	if (items == NULL) {
		Py_DECREF(items);
		return NULL;
	}
	result = PyTuple_Pack(4, dd->dict.ob_type,
			      Py_None, Py_None, items);
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
	while (PyDict_Next((PyObject*)&(dd->dict), &i, &key, &value)) {
		sum += PyFloat_AsDouble(value);
	}
	
	i = 0;
	while (PyDict_Next((PyObject*)&(dd->dict), &i, &key, &value)) {
	  int ok;

	  PyObject *newValue = PyFloat_FromDouble(PyFloat_AsDouble(value) / sum);
	  ok = PyDict_SetItem((PyObject*)&(dd->dict), key, newValue);
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
	while (PyDict_Next((PyObject*)&(dd->dict), &i, &key, &value)) {
	  log_sum += exp(PyFloat_AsDouble(value));
	}
	
	log_sum = log(log_sum);

	i = 0;
	while (PyDict_Next((PyObject*)&(dd->dict), &i, &key, &value)) {
	  int ok;

	  PyObject *newValue = PyFloat_FromDouble(PyFloat_AsDouble(value) - log_sum);
	  ok = PyDict_SetItem((PyObject*)&(dd->dict), key, newValue);
	  Py_DECREF(newValue);

	  if (ok < 0) return NULL;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}

PyDoc_STRVAR(cnter_log_normalize_doc, "D.log_normalize() -> normalizes the log counts in D, returns None");

static PyObject *
cnter_total_count(cnterobject *dd)
{
	Py_ssize_t i;
	PyObject *key, *value;
	double sum = 0.0;
	
	i = 0;
	while (PyDict_Next((PyObject*)&(dd->dict), &i, &key, &value)) {
		sum += PyFloat_AsDouble(value);
	}

	return PyFloat_FromDouble(sum);
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
	while (PyDict_Next((PyObject*)&(dd->dict), &i, &key, &value)) {
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
cnter_imul(cnterobject *dd, PyObject *other)
{
	Py_ssize_t i;
	PyObject *key, *value;
	// TODO: check that other is a counter
	
	cnterobject *other_cnter = (cnterobject*)other;

	// Walk through all the keys in other and fetch them from dd, thus creating 0.0 items for any missing keys
	i = 0;
	PyObject *defaultValue = PyFloat_FromDouble(0.0);
	while (PyDict_Next((PyObject*)&(other_cnter->dict), &i, &key, &value)) {
		int contains = PyDict_Contains((PyObject*)&(dd->dict), key);
		// If the key is not in the dictionary, try to set it to the default value (and fail on exception as appropriate)
		if (contains == 0 && PyDict_SetItem((PyObject*)&(dd->dict), key, defaultValue) < 0) {
		  Py_DECREF(defaultValue);
		  return NULL;
		}
		else if (contains < 0) {
		  Py_DECREF(defaultValue);
		  return NULL;
		}
	}
	
	i = 0;
	while (PyDict_Next((PyObject*)&(dd->dict), &i, &key, &value)) {
	  int ok;
	  PyObject *otherValue = PyDict_GetItem((PyObject*)&(other_cnter->dict), key);

	  if (otherValue == NULL) otherValue = defaultValue;
		
	  PyObject *newValue = PyFloat_FromDouble(PyFloat_AsDouble(value) * PyFloat_AsDouble(otherValue));
	  ok = PyDict_SetItem((PyObject*)&(dd->dict), key, newValue);
	  Py_DECREF(newValue);

	  if (ok < 0) {
		Py_DECREF(defaultValue);
		return NULL;
	  }
	}
	
	Py_DECREF(defaultValue);
	Py_INCREF((PyObject*)dd);
	return (PyObject*)dd;
}

static PyObject *
cnter_scale(cnterobject *dd, PyObject *other)
{
  Py_ssize_t i;
  PyObject *key, *value;
  double scale;

  if (PyInt_Check(other)) scale = (double)PyInt_AsLong(other);
  else if (PyLong_Check(other)) scale = (double)PyLong_AsLong(other);
  else scale = PyFloat_AsDouble(other);

  cnterobject *ret_cnter = NULL;

  ret_cnter = (cnterobject *)cnter_type.tp_new(&cnter_type, NULL, NULL);

  // Copy dd into ret_cnter
  if (PyDict_Update((PyObject*)&(ret_cnter->dict), (PyObject*)&(dd->dict)) < 0)
   	return NULL;

  i = 0;
  while (PyDict_Next((PyObject*)&(ret_cnter->dict), &i, &key, &value)) {
	int ok;

	PyObject *newValue = PyFloat_FromDouble(PyFloat_AsDouble(value) * scale);
	ok = PyDict_SetItem((PyObject*)&(ret_cnter->dict), key, newValue);
	Py_DECREF(newValue);

	if (ok < 0) {
	  Py_DECREF((PyObject*)ret_cnter);
	  return NULL;
	}
  }
	
  return (PyObject*)ret_cnter;
}

static PyObject *
cnter_mul(cnterobject *dd, PyObject *other)
{
  Py_ssize_t i;
  PyObject *key, *value;
  // TODO: check that other is a counter or a number

  if (PyInt_Check(other) || PyFloat_Check(other) || PyLong_Check(other))
	return cnter_scale(dd, other);

  cnterobject *other_cnter = (cnterobject*)other;
  cnterobject *ret_cnter = (cnterobject *)cnter_type.tp_new(&cnter_type, NULL, NULL);

  // Copy dd into ret_cnter
  if (PyDict_Update((PyObject*)ret_cnter, (PyObject*)dd) < 0) {
	Py_DECREF((PyObject*)ret_cnter);
	return NULL;
  }

  // Walk through all the keys in other and fetch them from ret_cnter, thus creating 0.0 items for any missing keys
  i = 0;
  PyObject *defaultValue = PyFloat_FromDouble(0.0);

  while (PyDict_Next((PyObject*)&(other_cnter->dict), &i, &key, &value)) {
	int contains = PyDict_Contains((PyObject*)&(ret_cnter->dict), key);
	// If the key is not in the dictionary, try to set it to the default value (and fail on exception as appropriate)
	if (contains == 0 && PyDict_SetItem((PyObject*)&(ret_cnter->dict), key, defaultValue) < 0) {
	  Py_DECREF((PyObject*)ret_cnter);
	  Py_DECREF(defaultValue);
	  return NULL;
	}
	else if (contains < 0) {
	  Py_DECREF((PyObject*)ret_cnter);
	  Py_DECREF(defaultValue);
	  return NULL;
	}
  }

  i = 0;
  while (PyDict_Next((PyObject*)&(dd->dict), &i, &key, &value)) {
	int ok;

	PyObject *otherValue = PyDict_GetItem((PyObject*)&(other_cnter->dict), key);
	if (otherValue == NULL) otherValue = defaultValue;
		
	PyObject *newValue = PyFloat_FromDouble(PyFloat_AsDouble(value) * PyFloat_AsDouble(otherValue));
	ok = PyDict_SetItem((PyObject*)&(ret_cnter->dict), key, newValue);
	Py_DECREF(newValue);

	if (ok < 0) {
	  Py_DECREF((PyObject*)ret_cnter);
	  Py_DECREF(defaultValue);
	  return NULL;
	}
  }

  Py_DECREF(defaultValue);
  return (PyObject*)ret_cnter;
}

static PyObject *
cnter_iadd(cnterobject *dd, PyObject *other)
{
	Py_ssize_t i;
	PyObject *key, *value;
	// TODO: check that other is a counter
	cnterobject *other_cnter = (cnterobject*)other;

	// Walk through all the keys in other and fetch them from dd, thus creating 0.0 items for any missing keys
	i = 0;
	PyObject *defaultValue = PyFloat_FromDouble(0.0);
	while (PyDict_Next((PyObject*)&(other_cnter->dict), &i, &key, &value)) {
		int contains = PyDict_Contains((PyObject*)&(dd->dict), key);
		// If the key is not in the dictionary, try to set it to the default value (and fail on exception as appropriate)
		if (contains == 0 && PyDict_SetItem((PyObject*)&(dd->dict), key, defaultValue) < 0) {
		  Py_DECREF(defaultValue);
		  return NULL;
		}
		else if (contains < 0) {
		  Py_DECREF(defaultValue);
		  return NULL;
		}
	}

	i = 0;
	while (PyDict_Next((PyObject*)&(dd->dict), &i, &key, &value)) {
	  int ok;
	  PyObject *otherValue = PyDict_GetItem((PyObject*)&(other_cnter->dict), key);
	  if (otherValue == NULL) otherValue = defaultValue;
		
	  PyObject *newValue = PyFloat_FromDouble(PyFloat_AsDouble(value) + PyFloat_AsDouble(otherValue));
	  ok = PyDict_SetItem((PyObject*)&(dd->dict), key, newValue);
	  Py_DECREF(newValue);

	  if (ok < 0) {
		Py_DECREF(defaultValue);
		return NULL;
	  }
	}

	Py_DECREF(defaultValue);
	Py_INCREF((PyObject*)dd);
	return (PyObject*)dd;
}

static PyObject *
cnter_isub(cnterobject *dd, PyObject *other)
{
	Py_ssize_t i;
	PyObject *key, *value;
	// TODO: check that other is a counter
	cnterobject *other_cnter = (cnterobject*)other;

	// Walk through all the keys in other and fetch them from dd, thus creating 0.0 items for any missing keys
	i = 0;
	PyObject *defaultValue = PyFloat_FromDouble(0.0);
	while (PyDict_Next((PyObject*)&(other_cnter->dict), &i, &key, &value)) {
		int contains = PyDict_Contains((PyObject*)&(dd->dict), key);
		// If the key is not in the dictionary, try to set it to the default value (and fail on exception as appropriate)
		if ((contains == 0 && PyDict_SetItem((PyObject*)&(dd->dict), key, defaultValue) < 0) || contains < 0) {
		  // if the set throws an error or contains threw an error, return NULL
		  Py_DECREF(defaultValue);
		  return NULL;
		}
	}

	i = 0;
	while (PyDict_Next((PyObject*)&(dd->dict), &i, &key, &value)) {
	  int ok;
	  PyObject *otherValue = PyDict_GetItem((PyObject*)&(other_cnter->dict), key);
	  if (otherValue == NULL) otherValue = defaultValue;
		
	  PyObject *newValue = PyFloat_FromDouble(PyFloat_AsDouble(value) - PyFloat_AsDouble(otherValue));
	  ok = PyDict_SetItem((PyObject*)&(dd->dict), key, newValue);
	  Py_DECREF(newValue);

	  if (ok < 0) {
		Py_DECREF(defaultValue);
		return NULL;
	  }
	}

	Py_DECREF(defaultValue);
	Py_INCREF((PyObject*)dd);
	return (PyObject*)dd;
}

static PyObject *
cnter_sub(cnterobject *dd, PyObject *other)
{
  Py_ssize_t i;
  PyObject *key, *value;
  // TODO: check that other is a counter or a number

  cnterobject *other_cnter = (cnterobject*)other;
  cnterobject *ret_cnter = (cnterobject *)cnter_type.tp_new(&cnter_type, NULL, NULL);

  // Copy dd into ret_cnter
  if (PyDict_Update((PyObject*)ret_cnter, (PyObject*)dd) == -1) {
	Py_DECREF((PyObject*)ret_cnter);
	return NULL;
  }

  // Walk through all the keys in other and fetch them from ret_cnter, thus creating items for any missing keys
  i = 0;
  while (PyDict_Next((PyObject*)&(other_cnter->dict), &i, &key, &value)) {
	int contains = PyDict_Contains((PyObject*)&(ret_cnter->dict), key);
	// If the key is not in the dictionary, try to set it to the default value (and fail on exception as appropriate)
	if (contains == 0 && PyDict_SetItem((PyObject*)&(ret_cnter->dict), key, PyFloat_FromDouble(- PyFloat_AsDouble(value))) < 0) {
	  Py_DECREF((PyObject*)ret_cnter);
	  return NULL;
	}
	else if (contains == 1) {
	  int ok;

	  PyObject *myValue = PyDict_GetItem((PyObject*)&(ret_cnter->dict), key);
	  PyObject *newValue = PyFloat_FromDouble(PyFloat_AsDouble(myValue) - PyFloat_AsDouble(value));

	  ok = PyDict_SetItem((PyObject*)&(ret_cnter->dict), key, newValue);
	  Py_DECREF(newValue);

	  if (ok < 0) {
		Py_DECREF((PyObject*)ret_cnter);
		return NULL;
	  }
	}
	else if (contains < 0) {
	  Py_DECREF((PyObject*)ret_cnter);
	  return NULL;
	}
  }
	
  return (PyObject*)ret_cnter;
}

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
	{"total_count", (PyCFunction)cnter_total_count, METH_NOARGS,
	 cnter_total_count_doc},
	{"arg_max", (PyCFunction)cnter_arg_max, METH_NOARGS,
	 cnter_arg_max_doc},
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
	fprintf(fp, ")");
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
	result = PyString_FromFormat("counter(%s)",
				     PyString_AS_STRING(baserepr));
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

  if (args == NULL || !PyTuple_Check(args))
	newargs = PyTuple_New(0);
  else {
	Py_INCREF(args);
	newargs = args;
  }

  if (kwds == NULL || !PyDict_Check(kwds))
	kwds = PyDict_New();
  else
	Py_INCREF(kwds);

  int result = PyDict_Type.tp_init(self, newargs, kwds);
  Py_DECREF(newargs);
  Py_DECREF(kwds);
  return result;
}

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
	0,				/*nb_add*/
	(binaryfunc) cnter_sub,				/*nb_subtract*/
    (binaryfunc) cnter_mul,				/*nb_multiply*/
	0,				/*nb_divide*/
	0,				/*nb_remainder*/
	0,				/*nb_divmod*/
	0,				/*nb_power*/
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
    (binaryfunc) cnter_isub,				/*nb_inplace_subtract*/
	(binaryfunc) cnter_imul,		/*nb_inplace_multiply*/
	0,				/*nb_inplace_divide*/
	0,				/*nb_inplace_remainder*/
	0,				/*nb_inplace_power*/
	0,				/*nb_inplace_lshift*/
	0,				/*nb_inplace_rshift*/
	0,				/*nb_inplace_and*/
	0,				/*nb_inplace_xor*/
	0,				/*nb_inplace_or*/
};

static PyTypeObject cnter_type = {
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
	0,				/* tp_getset */
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

	m = Py_InitModule3("nlp", NULL, module_doc);
	if (m == NULL)
		return;

	cnter_type.tp_base = &PyDict_Type;
	if (PyType_Ready(&cnter_type) < 0)
		return;
	Py_INCREF(&cnter_type);
	PyModule_AddObject(m, "counter", (PyObject *)&cnter_type);

	return;
}

