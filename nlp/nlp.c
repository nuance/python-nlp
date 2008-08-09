#include "Python.h"

/* counter type *********************************************************/

typedef struct {
	PyDictObject dict;
} cnterobject;

static PyTypeObject cnter_type; /* Forward */

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
	Py_ssize_t len = PyDict_Size((PyObject*)&(dd->dict));
	double values[len];
	
	i = 0;
	while (PyDict_Next((PyObject*)&(dd->dict), &i, &key, &value)) {
		values[i] = PyFloat_AsDouble(value);
		sum += values[i];		
	}
	
	i = 0;
	while (PyDict_Next((PyObject*)&(dd->dict), &i, &key, &value)) {
		PyObject *newValue = PyFloat_FromDouble(values[i] / sum);
		PyDict_SetItem((PyObject*)&(dd->dict), key, newValue);
		Py_DECREF(value);
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}

PyDoc_STRVAR(cnter_normalize_doc, "D.normalize() -> normalizes the values in D, returns None");

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
			arg_max = key;
			max = running;
		}
	}
	
	Py_INCREF(arg_max);
	return arg_max;
}

PyDoc_STRVAR(cnter_arg_max_doc, "D.arg_max() -> arg max of the items in D");

static PyObject *
cnter_imul(cnterobject *dd, PyObject *other_cnter)
{
	Py_INCREF(Py_None);
	return Py_None;
}

PyDoc_STRVAR(cnter_imul_doc, "D *= a -> key-wise multiplies D by values in other counter");

static PyObject *
cnter_iadd(cnterobject *dd, PyObject *other_cnter)
{
	Py_INCREF(Py_None);
	return Py_None;
}

PyDoc_STRVAR(cnter_iadd_doc, "D *= a -> key-wise adds D by values in other counter");

static PyMethodDef cnter_methods[] = {
	{"__missing__", (PyCFunction)cnter_missing, 1,
	 cnter_missing_doc},
	{"copy", (PyCFunction)cnter_copy, METH_NOARGS,
	 cnter_copy_doc},
	{"__copy__", (PyCFunction)cnter_copy, METH_NOARGS,
	 cnter_copy_doc},
	{"__reduce__", (PyCFunction)cnter_reduce, METH_NOARGS,
	 reduce_doc},
	{"normalize", (PyCFunction)cnter_normalize, METH_NOARGS,
	 cnter_normalize_doc},
	{"total_count", (PyCFunction)cnter_total_count, METH_NOARGS,
	 cnter_total_count_doc},
	{"arg_max", (PyCFunction)cnter_arg_max, METH_NOARGS,
	 cnter_arg_max_doc},
	{"__imul__", (PyCFunction)cnter_imul, 1,
	 cnter_imul_doc},
	{"__iadd__", (PyCFunction)cnter_iadd, 1,
	 cnter_iadd_doc},
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
cnter_tp_clear(cnterobject *dd)
{
	return PyDict_Type.tp_clear((PyObject *)dd);
}

static int
cnter_init(PyObject *self, PyObject *args, PyObject *kwds)
{
	PyObject *newargs;
	int result;
	if (args == NULL || !PyTuple_Check(args))
		newargs = PyTuple_New(0);
	else {
		Py_ssize_t n = PyTuple_GET_SIZE(args);
		newargs = PySequence_GetSlice(args, 0, n);
	}
	if (newargs == NULL)
		return -1;
	result = PyDict_Type.tp_init(self, newargs, kwds);
	Py_DECREF(newargs);
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
	0,				/* tp_as_number */
	0,				/* tp_as_sequence */
	0,				/* tp_as_mapping */
	0,	       			/* tp_hash */
	0,				/* tp_call */
	0,				/* tp_str */
	PyObject_GenericGetAttr,	/* tp_getattro */
	0,				/* tp_setattro */
	0,				/* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC |
		Py_TPFLAGS_HAVE_WEAKREFS,	/* tp_flags */
	cnter_doc,			/* tp_doc */
	cnter_traverse,		/* tp_traverse */
	(inquiry)cnter_tp_clear,	/* tp_clear */
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
	cnter_init,			/* tp_init */
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

