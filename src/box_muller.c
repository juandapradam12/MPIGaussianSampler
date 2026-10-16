#include "box_muller.h"

#include <math.h>
#include <stdlib.h>

double box_muller_normal(double mu, double sigma)
{
    static double spare;
    static int has_spare = 0;

    if (has_spare) {
        has_spare = 0;
        return mu + sigma * spare;
    }

    double u;
    double v;
    double s;

    do {
        u = -1.0 + 2.0 * ((double)rand() / (double)RAND_MAX);
        v = -1.0 + 2.0 * ((double)rand() / (double)RAND_MAX);
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    const double scale = sqrt(-2.0 * log(s) / s);
    spare = v * scale;
    has_spare = 1;

    return mu + sigma * (u * scale);
}
