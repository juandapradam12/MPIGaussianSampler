#ifndef BOX_MULLER_H
#define BOX_MULLER_H

/*
 * Marsaglia polar form of the Box–Muller transform.
 * Draws a sample from N(mu, sigma^2) using pairs of uniform(-1, 1) draws.
 */
double box_muller_normal(double mu, double sigma);

#endif /* BOX_MULLER_H */
