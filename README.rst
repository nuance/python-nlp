Overview
========
An acceptably pythonic bayesian natural language processing / machine learning library.

Goals
=====
Key goals are simplicity, speed, elegance, and conciseness of user code.

Status
======
Both a naive bayes and a maximum entropy classifier is implemented. Both rely on a mixture of C and python. Expect more interesting models in the near future.

Core code is mostly restricted to python, although inner loops are optimized in C (see maxent.c for an example). Fallbacks to python are possible although they currently do not happen gracefully.

License
=======
The code is licensed under the GNU GPL v2. If you're interested in using the code under a different license please send a message my github account and we'll work something out.
