#include <iostream>
#include <random>
#include <cmath>
using namespace std;
#define M_PI 3.14159265358979323846

void randomPointOnSquare(double L, double& x, double& y, mt19937& gen) {
    uniform_real_distribution<double> dist(-L/2.0, L/2.0);
    x = dist(gen);
    y = dist(gen);
}
// Modèle de la réflexion diffuse (Loi de Lambert).
void randomCosineDirection(double& dx, double& dy, double& dz, mt19937& gen) {
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

    if (dz <= 0) return false;  

    double t = (distZ - Az) / dz;
    if (t <= 0) return false;

    double X = Ax + t * dx;
    double Y = Ay + t * dy;

    return (abs(X) <= L/2.0 && abs(Y) <= L/2.0);
}

int main() {
    const double L = 1.0;       // longueur des carrés
    const double d = 2.0;       // distance entre A et B
    const int N = 1'000'000;    // nombre de rayons

    mt19937 gen(42);

    int hits = 0;

    for (int i = 0; i < N; i++) {

        double Ax, Ay;
        randomPointOnSquare(L, Ax, Ay, gen);
        double Az = 0.0;
        
        double dx, dy, dz;
        randomCosineDirection(dx, dy, dz, gen);

        if (intersectsSquareB(Ax, Ay, Az, dx, dy, dz, L, d))
            hits++;
    }

    double F = (double)hits / (double)N;

    cout << "Rays        = " << N << "\n";
    cout << "Hits        = " << hits << "\n";
    cout << "View Factor = " << F << endl;

    return 0;
}
