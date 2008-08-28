#ifndef Py_NLPMODULE_H
#define Py_NLPMODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#define PyNlp_API_pointers 0

PyAPI_DATA(PyTypeObject) NlpCounter_Type;

#define NlpCounter_Check(op) PyObject_TypeCheck(op, &NlpCounter_Type)
#define NlpCounter_CheckExact(op) ((op)->ob_type == &NlpCounter_Type)

#define NlpCounter_API_pointers 3

#ifdef NLP_MODULE

static PyObject* NlpCounter_New(void);
static void NlpCounter_Normalize(PyObject *mp);
static void NlpCounter_LogNormalize(PyObject *mp);

#else

static void **Nlp_API;

#define NlpCounter_New (*(PyObject* (*)(void)) Nlp_API[0])
#define NlpCounter_Normalize (*(void (*)(PyObject *mp)) Nlp_API[1])
#define NlpCounter_LogNormalize (*(void (*)(PyObject *mp)) Nlp_API[2])

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
   } 
  return 0;
}

#endif

#ifdef __cplusplus
}
#endif

#endif /* _NLP_H_ */
