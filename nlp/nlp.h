#ifndef Py_NLPMODULE_H
#define Py_NLPMODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#define PyNlp_API_pointers 0

PyAPI_DATA(PyTypeObject) NlpCounter_Type;

#define NlpCounter_API_pointers 7

#ifdef NLP_MODULE

  static PyObject* NlpCounter_New(void);
  static int NlpCounter_Normalize(PyObject *mp);
  static int NlpCounter_LogNormalize(PyObject *mp);
  static PyObject* NlpCounter_XGetItem(PyObject *cnter, PyObject *key);
  static double NlpCounter_XGetDouble(PyObject *cnter, PyObject *key);
  static int NlpCounter_SetDefault(PyObject *cnter, double new_default);
  static double NlpCounter_GetDefault(PyObject *cnter);

#define NlpCounter_Check(op) PyObject_TypeCheck(op, &NlpCounter_Type)
#define NlpCounter_CheckExact(op) ((op)->ob_type == &NlpCounter_Type)

#else

  static void **Nlp_API;

#define NlpCounter_New (*(PyObject* (*)(void)) Nlp_API[0])
#define NlpCounter_Normalize (*(int (*)(PyObject *mp)) Nlp_API[1])
#define NlpCounter_LogNormalize (*(int (*)(PyObject *mp)) Nlp_API[2])
#define NlpCounter_XGetItem (*(PyObject* (*)(PyObject *cnter, PyObject *key)) Nlp_API[3])
#define NlpCounter_XGetDouble (*(double (*)(PyObject *cnter, PyObject *key)) Nlp_API[4])
#define NlpCounter_SetDefault (*(int (*)(PyObject *cnter, double new_default)) Nlp_API[5])
#define NlpCounter_GetDefault (*(double (*)(PyObject *cnter)) Nlp_API[6])

  static int
  import_nlp(void)
  {
	PyObject *module = PyImport_ImportModule("nlp");

	if (module != NULL) { 
	  PyObject *c_api_object = PyObject_GetAttrString(module, "_C_API"); 

	  if (c_api_object == NULL) 
		return -1; 
	  if (PyCObject_Check(c_api_object)) 
		Nlp_API = (void **)PyCObject_AsVoidPtr(c_api_object); 
	  Py_DECREF(c_api_object);
	  return 0;
	}

	PyErr_SetString(PyExc_ImportError, "Couldn't import nlp module");
	return -1;
  }

#endif

#ifdef __cplusplus
}
#endif

#endif /* _NLP_H_ */
