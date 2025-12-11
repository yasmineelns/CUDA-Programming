#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <vector>

using namespace std;
#define M_PI 3.14159265358979323846

double F_ref = 0.0685896;

void randomPointOnSquare(double L, double &x, double &y, mt19937 &gen)
{
    uniform_real_distribution<double> dist(-L / 2.0, L / 2.0);
    x = dist(gen);
    y = dist(gen);
}

void randomCosineDirection(double &dx, double &dy, double &dz, mt19937 &gen)
{
    uniform_real_distribution<double> dist(0.0, 1.0);
    double r1 = dist(gen);
    double r2 = dist(gen);

    double phi = 2.0 * M_PI * r1;
    double z = sqrt(1.0 - r2);
    double r = sqrt(r2);

    dx = r * cos(phi);
    dy = r * sin(phi);
    dz = z;
}

bool intersectsSquareB(double Ax, double Ay, double Az,
                       double dx, double dy, double dz,
                       double L, double distZ)
{
    if (dz <= 0)
        return false;
    double t = (distZ - Az) / dz;
    if (t <= 0)
        return false;

    double X = Ax + t * dx;
    double Y = Ay + t * dy;

    return (abs(X) <= L / 2.0 && abs(Y) <= L / 2.0);
}

double estimateViewFactor(long long N, double L, double d, mt19937 &gen)
{

    int hits = 0;

    for (long long i = 0; i < N; i++)
    {
        double Ax, Ay, dx, dy, dz;
        randomPointOnSquare(L, Ax, Ay, gen);
        randomCosineDirection(dx, dy, dz, gen);
        if (intersectsSquareB(Ax, Ay, 0.0, dx, dy, dz, L, d))
            hits++;
    }

    return (double)hits / (double)N;
}

int main()
{
    vector<long long> Ns = {1'000, 10'000, 100'000, 1'000'000, 10'000'000};
    ofstream file("errors.csv");
    file << "N,F_est,AbsError\n";
    mt19937 gen(42);
    double L = 1.0;
    double d = 2.0;

    for (auto N : Ns)
    {
        double F_est = estimateViewFactor(N, L, d, gen);
        double abs_err = fabs(F_est - F_ref);
        cout << "N = " << N << "  F = " << F_est << "  Err = " << abs_err << endl;
        file << N << "," << F_est << "," << abs_err << "\n";
    }

    file.close();
    return 0;
}
