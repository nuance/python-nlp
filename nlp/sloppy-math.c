#include "sloppy-math.h"
#include <math.h>

/* Adapted from java code by Dan Klein and Teg Grenager
 */

double sloppy_exp(double x)
{
  if ((x > 0 && x < 0.001) || (x < 0 && x > -0.001))
	return 1 + x;
  return exp(x);
}

double sloppy_log_add(double logs[], int length)
{
  int maxIdx = -1;
  double max = log(0);
  int i;

  if (length == 1) return max;

  for (i = 0; i < length; i++)
  {
	if (logs[i] > max)
	{
	  max = logs[i];
	  maxIdx = i;
	}
  }

  if (maxIdx == -1) return max;

  double threshold = max - 20;
  double sumNegativeDifferences = 0.0;

  for (i = 0; i < length; i++)
	if (i != maxIdx && logs[i] > threshold)
	  sumNegativeDifferences += exp(logs[i] - max);

  if (sumNegativeDifferences > 0.0)
	return max + log(1.0 + sumNegativeDifferences);
  return max;
}
