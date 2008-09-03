#!/usr/bin/env python
 
from distutils.core import setup, Extension
 
setup(
  name             = 'nlp',
  version          = '0.01',
  description      = 'Fast nlp utilities for Python.',
  long_description = 'Fast nlp utilities implemented in C.',
  author           = 'Matt Jones',
  author_email     = 'matt@mhjones.org',
  license          = 'BSD License',
  ext_modules      = [Extension(name='nlp', sources=['nlp.c', 'sloppy-math.c']),
					  Extension(name='maxent', sources=['maxent.c']),]
)
