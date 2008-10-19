Overview
========
An acceptably pythonic bayesian natural language processing / machine learning library.

Goals
=====
Key goals are simplicity, speed, elegance, and conciseness of user code.

Status
======
A naive bayes, a maximum entropy classifier, and a first-order exact hidden markov model are implemented. All rely on a mixture of C and python. Expect more interesting models in the near future.

Core code is mostly restricted to python, although inner loops are optimized in C (see maxent.c for an example). Fallbacks to python are possible although they currently do not happen gracefully (this would be as simple as using a try:except ImportError: around the optimization code, but I haven't felt it to be worthwhile).

My current focus is on fleshing out chain models and generalizing some of the shared code (higher-order class code could be designed to be generic, as would the viterbi decoder and any future approximate inference schemes).

License
=======
The code is licensed under the GNU GPL v2. If you're interested in using the code under a different license please send a message my github account and we'll work something out.
