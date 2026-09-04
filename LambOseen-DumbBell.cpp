#include <fstream>
#include <cmath>
#include <iostream>
#include <vector>
#include <complex>
#include <random>

int x = 64;
int y = 64;

double r0 = 0.05;
double L = 60*r0;
double dx = L/x;
double dy = L/y;

double xc;
double yc;
int sign;

double rho = 1.2;
double mu = 1.8e-5;
double nu = mu/rho;

// double Re = 560;
double Gamma = 2.0 * M_PI;
// double tau_f = 2.0 * M_PI * r0 * r0 / Gamma;
// double u_theta_max = (Gamma/(2.0*M_PI*r0)) * (1 - std::exp(-1));
double w0 = (Gamma/(M_PI*r0*r0));
// const double alpha_oseen = 1.2564;

double T = 50;
double dt = 1e-3;

// dumbell
double M = 1.0;
double S = 1.0;
double l = 1;
double St = 5.0e-2;
int nDumbBell = 500;

using Matrix = std::vector<std::vector<double>>;
using CVector = std::vector<std::complex<double>>;
using CMatrix = std::vector<CVector>;

struct DumbBell {
    // bead 1
    double bead1_x = 0.0;
    double bead1_y = 0.0; 
    double bead1_u = 0.0; 
    double bead1_v = 0.0;

    // bead 2
    double bead2_x = 0.0; 
    double bead2_y = 0.0; 
    double bead2_u = 0.0; 
    double bead2_v = 0.0;

    // center of mass of dumbbell
    double x_cm = 0.0; 
    double y_cm = 0.0; 
    double u_cm = 0.0; 
    double v_cm = 0.0;

    // rotational orientation of dumbbell
    double alpha_c = 0.0; 
    double omega = 0.0;

    // fluid's velocity at bead's location
    double bead1_uf = 0.0;
    double bead1_vf = 0.0;
    double bead2_uf = 0.0;
    double bead2_vf = 0.0;
};

void setInitialCondition(CMatrix& w) {
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            // 1st quadrant
            if (i*dx > 0.5*L && j*dy > 0.5*L) {
                xc = 0.75*L;
                yc = 0.75*L;
                sign = +1;
            } else if (i*dx > 0.5*L && j*dy <= 0.5*L) { // 4th quadrant
                xc = 0.75*L;
                yc = 0.25*L;
                sign = -1;
            } else if (i*dx <= 0.5*L && j*dy > 0.5*L) { // 2nd quadrant
                xc = 0.25*L;
                yc = 0.75*L;
                sign = -1;
            } else { // 3rd quadrant
                xc = 0.25*L;
                yc = 0.25*L;
                sign = +1;
            }

            double r2 = (i*dx - xc)*(i*dx - xc) + (j*dy - yc)*(j*dy - yc);
            w[i][j] = sign * w0 * std::exp(-r2/(r0*r0));
        }
    }
}

// =================================================================
// FUNCTIONS RELATED TO FOURIER TRANSFORM
// =================================================================
std::complex<double> omegaPower(int p, int n) {
    return std::polar(1.0, -p * 2.0 * M_PI / n);
}

std::complex<double> omegaPowerInverse(int p, int n) {
    return std::polar(1.0, p * 2.0 * M_PI / n);
}

void FFT(std::vector<std::complex<double>>& Y) {
    int n = Y.size();

    if (n == 1) return;

    std::vector<std::complex<double>> Ye, Yo;
    for (int i = 0; i < n/2; ++i) {
        Ye.push_back(Y[2*i]);
        Yo.push_back(Y[2*i + 1]);
    }

    FFT(Ye);
    FFT(Yo);
    
    for (int j = 0; j < n/2; ++j) {
        Y[j] = Ye[j] + omegaPower(j, n)*Yo[j];
        Y[j + n/2] = Ye[j] - omegaPower(j, n)*Yo[j];
    }
}

void IFFT(std::vector<std::complex<double>>& y) {
    int n = y.size();

    if (n == 1) return;

    std::vector<std::complex<double>> ye, yo;
    for (int i = 0; i < n/2; ++i) {
        ye.push_back(y[2*i]);
        yo.push_back(y[2*i + 1]);
    }

    IFFT(ye);
    IFFT(yo);

    for (int j = 0; j < n/2; ++j) {
        y[j] = ye[j] + omegaPowerInverse(j, n)*yo[j];
        y[j + n/2] = ye[j] - omegaPowerInverse(j, n)*yo[j];
    }
}

void normalizeIFFT(std::vector<std::complex<double>>& y, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] /= n;
    }
}

void FFT2(CMatrix& Y) {
    for (int i = 0; i < x; i++) {
        FFT(Y[i]);
    }

    std::vector<std::complex<double>> temp(x, 0.0);
    for (int j = 0; j < y; ++j) {
        for (int i = 0; i < x; ++i) {
            temp[i] = Y[i][j];
        }
        FFT(temp);

        for (int i = 0; i < x; ++i) {
            Y[i][j] = temp[i];
        }
    }
}

void IFFT2(CMatrix& Y) {
    for (int i = 0; i < x; ++i) {
        IFFT(Y[i]);
        normalizeIFFT(Y[i], x);
    }

    std::vector<std::complex<double>> temp(x, 0.0);
    for (int j = 0; j < y; ++j) {
        for (int i = 0; i < x; ++i) {
            temp[i] = Y[i][j];
        }
        IFFT(temp);
        normalizeIFFT(temp, y);

        for (int i = 0; i < x; ++i) {
            Y[i][j] = temp[i];
        }
    }
}

// =================================================================
// =================================================================

double crossProduct(const std::vector<double>& a, const std::vector<double>& b) {
    return b[1]*a[0] - a[1]*b[0];
}

// ========================== INTERPOLATION =======================================
void biCubicSpline(const CMatrix& data1, const CMatrix& data2, std::vector<DumbBell>& dumb) {
    // y rows and x-1 splines
    std::vector<std::vector<double>> coeff_a(y, std::vector<double>(x-1, 0.0));
    std::vector<std::vector<double>> coeff_b(y, std::vector<double>(x-1, 0.0));
    std::vector<std::vector<double>> coeff_c(y, std::vector<double>(x-1, 0.0));
    std::vector<std::vector<double>> coeff_d(y, std::vector<double>(x-1, 0.0));

    auto solve_1d = [](const std::vector<double>& a_vals, double h, std::vector<double>& a_out, std::vector<double>& b_out, std::vector<double>& c_out, std::vector<double>& d_out) {
        int n = a_vals.size() - 1;
        std::vector<double> alpha(n+1, 0.0);
        
        for (int i = 1; i <= n-1; ++i) {
            alpha[i] = (3.0 / h) * (a_vals[i+1] - a_vals[i]) - (3.0 / h) * (a_vals[i] - a_vals[i-1]);
        }
        
        std::vector<double> l_vec(n+1, 1.0);
        std::vector<double> mu(n+1, 0.0);
        std::vector<double> z(n+1, 0.0);
        
        for (int i = 1; i <= n-1; ++i) {
            l_vec[i] = 4.0 * h - h * mu[i-1];
            mu[i] = h / l_vec[i];
            z[i] = (alpha[i] - h * z[i-1]) / l_vec[i];
        }
        
        std::vector<double> c_temp(n+1, 0.0);
        for (int j = n-1; j >= 0; --j) {
            c_temp[j] = z[j] - mu[j] * c_temp[j+1];
            b_out[j] = (a_vals[j+1] - a_vals[j]) / h - h * (c_temp[j+1] + 2.0 * c_temp[j]) / 3.0;
            d_out[j] = (c_temp[j+1] - c_temp[j]) / (3.0 * h);
            a_out[j] = a_vals[j];
            c_out[j] = c_temp[j];
        }
    };

    // ===================    U    =====================
    for (int j = 0; j < y; ++j) {
        std::vector<double> horizontal_data(x);
        for (int i = 0; i < x; ++i) {
            horizontal_data[i] = data1[i][j].real();
        }
        solve_1d(horizontal_data, dx, coeff_a[j], coeff_b[j], coeff_c[j], coeff_d[j]);
    }

    std::vector<double> temp_y_vals1(y, 0.0);
    std::vector<double> v_a1(y-1, 0.0);
    std::vector<double> v_b1(y-1, 0.0);
    std::vector<double> v_c1(y-1, 0.0);
    std::vector<double> v_d1(y-1, 0.0);

    std::vector<double> temp_y_vals2(y, 0.0);
    std::vector<double> v_a2(y-1, 0.0);
    std::vector<double> v_b2(y-1, 0.0);
    std::vector<double> v_c2(y-1, 0.0);
    std::vector<double> v_d2(y-1, 0.0);

    for (auto& _dumb : dumb) {
        double px1 = std::fmod(_dumb.bead1_x, L);
        double py1 = std::fmod(_dumb.bead1_y, L);
        double px2 = std::fmod(_dumb.bead2_x, L);
        double py2 = std::fmod(_dumb.bead2_y, L);
        
        int ix1 = static_cast<int>(std::floor(px1 / dx));
        int ix2 = static_cast<int>(std::floor(px2 / dx));
        ix1 = std::max(0, std::min(ix1, x-2));
        ix2 = std::max(0, std::min(ix2, x-2));
        double delta_x1 = px1 - (ix1 * dx);
        double delta_x2 = px2 - (ix2 * dx);
        
        for (int j = 0; j < y; ++j) {
            temp_y_vals1[j] = coeff_a[j][ix1] + coeff_b[j][ix1]*delta_x1 + coeff_c[j][ix1]*delta_x1*delta_x1 + coeff_d[j][ix1]*delta_x1*delta_x1*delta_x1;
            temp_y_vals2[j] = coeff_a[j][ix2] + coeff_b[j][ix2]*delta_x2 + coeff_c[j][ix2]*delta_x2*delta_x2 + coeff_d[j][ix2]*delta_x2*delta_x2*delta_x2;
        }
        
        solve_1d(temp_y_vals1, dy, v_a1, v_b1, v_c1, v_d1);
        solve_1d(temp_y_vals2, dy, v_a2, v_b2, v_c2, v_d2);
        
        int iy1 = static_cast<int>(std::floor(py1 / dy));
        int iy2 = static_cast<int>(std::floor(py2 / dy));
        iy1 = std::max(0, std::min(iy1, y-2));
        iy2 = std::max(0, std::min(iy2, y-2));
        double delta_y1 = py1 - (iy1*dy);
        double delta_y2 = py2 - (iy2*dy);
        
        _dumb.bead1_uf = v_a1[iy1] + v_b1[iy1]*delta_y1 + v_c1[iy1]*delta_y1*delta_y1 + v_d1[iy1]*delta_y1*delta_y1*delta_y1;
        _dumb.bead2_uf = v_a2[iy2] + v_b2[iy2]*delta_y2 + v_c2[iy2]*delta_y2*delta_y2 + v_d2[iy2]*delta_y2*delta_y2*delta_y2;
                           
    }

    // ===================    V    =====================
    for (int j = 0; j < y; ++j) {
        std::vector<double> horizontal_data(x);
        for (int i = 0; i < x; ++i) {
            horizontal_data[i] = data2[i][j].real();
        }
        solve_1d(horizontal_data, dx, coeff_a[j], coeff_b[j], coeff_c[j], coeff_d[j]);
    }

    for (auto& _dumb : dumb) {
        double px1 = std::fmod(_dumb.bead1_x, L);
        double py1 = std::fmod(_dumb.bead1_y, L);
        double px2 = std::fmod(_dumb.bead2_x, L);
        double py2 = std::fmod(_dumb.bead2_y, L);
        
        int ix1 = static_cast<int>(std::floor(px1 / dx));
        int ix2 = static_cast<int>(std::floor(px2 / dx));
        ix1 = std::max(0, std::min(ix1, x-2));
        ix2 = std::max(0, std::min(ix2, x-2));
        double delta_x1 = px1 - (ix1*dx);
        double delta_x2 = px2 - (ix2*dx);
        
        for (int j = 0; j < y; ++j) {
            temp_y_vals1[j] = coeff_a[j][ix1] + coeff_b[j][ix1]*delta_x1 + coeff_c[j][ix1]*delta_x1*delta_x1 + coeff_d[j][ix1]*delta_x1*delta_x1*delta_x1;
            temp_y_vals2[j] = coeff_a[j][ix2] + coeff_b[j][ix2]*delta_x2 + coeff_c[j][ix2]*delta_x2*delta_x2 + coeff_d[j][ix2]*delta_x2*delta_x2*delta_x2;
        }
        
        solve_1d(temp_y_vals1, dy, v_a1, v_b1, v_c1, v_d1);
        solve_1d(temp_y_vals2, dy, v_a2, v_b2, v_c2, v_d2);
        
        int iy1 = static_cast<int>(std::floor(py1 / dy));
        int iy2 = static_cast<int>(std::floor(py2 / dy));
        iy1 = std::max(0, std::min(iy1, y-2));
        iy2 = std::max(0, std::min(iy2, y-2));
        double delta_y1 = py1 - (iy1*dy);
        double delta_y2 = py2 - (iy2*dy);
        
        _dumb.bead1_vf = v_a1[iy1] + v_b1[iy1]*delta_y1 + v_c1[iy1]*delta_y1*delta_y1 + v_d1[iy1]*delta_y1*delta_y1*delta_y1;
        _dumb.bead2_vf = v_a2[iy2] + v_b2[iy2]*delta_y2 + v_c2[iy2]*delta_y2*delta_y2 + v_d2[iy2]*delta_y2*delta_y2*delta_y2;
                           
    }
}

// compute wavenumber.
// done at the start of the simulation
void computeWavenumber(Matrix& kx, Matrix& ky) {
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            if (i <= x/2) {
                kx[i][j] = 2.0 * M_PI * i / L;
            } else {
                kx[i][j] = 2.0 * M_PI * (i-x) / L;
            }
            if (j <= y/2) {
                ky[i][j] = 2.0 * M_PI * j / L;
            } else {
                ky[i][j] = 2.0 * M_PI * (j-y) / L;
            }
        }
    }
}

// to compute velocity form streamfunction.
// required to solve vorticity transport equation.
// called at each intermediate steps
void computeVelocity(CMatrix& psi, CMatrix& u, CMatrix& v, const Matrix& kx, const Matrix& ky) {
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            std::complex<double> ikx(0.0, kx[i][j]);
            std::complex<double> iky(0.0, ky[i][j]);

            u[i][j] = psi[i][j] * iky;
            v[i][j] = -psi[i][j] * ikx;
        }
    }

    IFFT2(u);
    IFFT2(v);
}

/*
to compute derivatives of vorticity
input vorticity is already fourier transformed (it is done before computing stream function (fourier transformed))
*/
void derivatives(CMatrix& w, CMatrix& dwx, CMatrix& ddwx, CMatrix& dwy, CMatrix& ddwy, const Matrix& kx, const Matrix& ky) {
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {    
            std::complex<double> ikx(0.0, kx[i][j]);
            std::complex<double> iky(0.0, ky[i][j]);
            std::complex<double> iikx(-kx[i][j]*kx[i][j], 0.0);
            std::complex<double> iiky(-ky[i][j]*ky[i][j], 0.0);

            dwx[i][j] = w[i][j] * ikx;
            dwy[i][j] = w[i][j] * iky;
            ddwx[i][j] = w[i][j] * iikx;
            ddwy[i][j] = w[i][j] * iiky;
        }
    }
    IFFT2(dwx);
    IFFT2(dwy);
    IFFT2(ddwx);
    IFFT2(ddwy);
}

/*
to initialize dumbbells.
done only once
*/
void dumbbellInit(std::vector<DumbBell>& d) {
    std::mt19937 gen(23);
    std::uniform_real_distribution<double> r_dis(0, L);
    std::uniform_real_distribution<double> rand_alpha(0, M_PI);

    for (int p = 0; p < nDumbBell; ++p) {
        d[p].x_cm = r_dis(gen);
        d[p].y_cm = r_dis(gen);
        d[p].alpha_c = rand_alpha(gen);
    }
}

void computeBeadPosition(std::vector<DumbBell>& d) {
    for (int p = 0; p < nDumbBell; ++p) {
        // Bead 1
        d[p].bead1_x = d[p].x_cm - 0.5*l*cos(d[p].alpha_c);
        d[p].bead1_y = d[p].y_cm - 0.5*l*sin(d[p].alpha_c);
        
        // Bead 2
        d[p].bead2_x = d[p].x_cm + 0.5*l*cos(d[p].alpha_c);
        d[p].bead2_y = d[p].y_cm + 0.5*l*sin(d[p].alpha_c);
    }
}

void computeBeadVelocity(std::vector<DumbBell>& d) {
    for (int p = 0; p < nDumbBell; ++p) {
        double rx1 = d[p].bead1_x - d[p].x_cm;
        double ry1 = d[p].bead1_y - d[p].y_cm;

        d[p].bead1_u = d[p].u_cm - ry1 * d[p].omega;
        d[p].bead1_v = d[p].v_cm + rx1 * d[p].omega;

        double rx2 = d[p].bead2_x - d[p].x_cm;
        double ry2 = d[p].bead2_y - d[p].y_cm;

        d[p].bead2_u = d[p].u_cm - ry2 * d[p].omega; 
        d[p].bead2_v = d[p].v_cm + rx2 * d[p].omega;
    }
}

void RungeKutta4(CMatrix& w, CMatrix& psi, CMatrix& u, CMatrix& v, const Matrix& kx, const Matrix& ky, std::vector<DumbBell>& d) {
    CMatrix w_hat = w;
    FFT2(w_hat);

    computeVelocity(psi, u, v, kx, ky);
    CMatrix dwx(x, CVector(y, 0.0));
    CMatrix ddwx(x, CVector(y, 0.0));
    CMatrix dwy(x, CVector(y, 0.0));
    CMatrix ddwy(x, CVector(y, 0.0));

    CMatrix w_temp = w;
    std::vector<DumbBell> d_temp = d;

    CMatrix k1(x, CVector(y, 0.0));
    CMatrix k2 = k1;
    CMatrix k3 = k1;
    CMatrix k4 = k1;

    std::vector<double> k_x1(nDumbBell);
    std::vector<double> k_y1 = k_x1; 
    std::vector<double> k_u1 = k_x1;
    std::vector<double> k_v1 = k_x1;
    std::vector<double> k_omega1 = k_x1;
    std::vector<double> k_alpha1 = k_x1;

    std::vector<double> k_x2 = k_x1;
    std::vector<double> k_y2 = k_x1; 
    std::vector<double> k_u2 = k_x1;
    std::vector<double> k_v2 = k_x1;
    std::vector<double> k_omega2 = k_x1;
    std::vector<double> k_alpha2 = k_x1;

    std::vector<double> k_x3 = k_x1;
    std::vector<double> k_y3 = k_x1; 
    std::vector<double> k_u3 = k_x1;
    std::vector<double> k_v3 = k_x1;
    std::vector<double> k_omega3 = k_x1;
    std::vector<double> k_alpha3 = k_x1;

    std::vector<double> k_x4 = k_x1;
    std::vector<double> k_y4 = k_x1; 
    std::vector<double> k_u4 = k_x1;
    std::vector<double> k_v4 = k_x1;
    std::vector<double> k_omega4 = k_x1;
    std::vector<double> k_alpha4 = k_x1;

    // ================================
    //  ------------- 1 ---------------
    // ================================
    derivatives(w_hat, dwx, ddwx, dwy, ddwy, kx, ky);
    computeBeadPosition(d_temp);
    computeBeadVelocity(d_temp);
    biCubicSpline(u, v, d_temp);

    for (int p = 0; p < nDumbBell; ++p) {
        k_u1[p] = dt * ((1.0/((1.0+M)*St)) * (S*(d_temp[p].bead1_uf - d_temp[p].bead1_u) + (d_temp[p].bead2_uf - d_temp[p].bead2_u)));
        k_v1[p] = dt * ((1.0/((1.0+M)*St)) * (S*(d_temp[p].bead1_vf - d_temp[p].bead1_v) + (d_temp[p].bead2_vf - d_temp[p].bead2_v)));
        k_x1[p] = dt * d_temp[p].u_cm;
        k_y1[p] = dt * d_temp[p].v_cm;
        std::vector<double> l_vec = {d_temp[p].bead2_x - d_temp[p].bead1_x, d_temp[p].bead2_y - d_temp[p].bead1_y};
        std::vector<double> uf1 = {d_temp[p].bead1_uf, d_temp[p].bead1_vf};
        std::vector<double> uf2 = {d_temp[p].bead2_uf, d_temp[p].bead2_vf};
        k_omega1[p] = dt * (1.0/(M*St)) * ((M*crossProduct(l_vec, uf2) - S*crossProduct(l_vec, uf1)) - ((S*l*l*d_temp[p].omega)/(1.0+M) + (M*M*l*l*d_temp[p].omega)/(1.0+M)));
        k_alpha1[p] = dt * d_temp[p].omega;

        d_temp[p].x_cm = std::fmod(d[p].x_cm + 0.5*k_x1[p] + L, L);
        d_temp[p].y_cm = std::fmod(d[p].y_cm + 0.5*k_y1[p] + L, L);
        d_temp[p].u_cm = d[p].u_cm + 0.5*k_u1[p];
        d_temp[p].v_cm = d[p].v_cm + 0.5*k_v1[p];
        d_temp[p].omega = d[p].omega + 0.5*k_omega1[p];
        d_temp[p].alpha_c = d[p].alpha_c + 0.5*k_alpha1[p];
    }

    // vorticity
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            k1[i][j] = dt * (-u[i][j] * dwx[i][j] - v[i][j] * dwy[i][j] + nu * ddwx[i][j] + nu * ddwy[i][j]);
            w_temp[i][j] = w[i][j] + 0.5*k1[i][j];
        }
    }

    // ===========================
    // ----------- 2 -------------
    // ===========================
    FFT2(w_temp);
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            if (i == 0 && j == 0) {
                psi[i][j] = 0.0;
            } else {
                psi[i][j] = -w_temp[i][j] / (kx[i][j]*kx[i][j] + ky[i][j]*ky[i][j]);
            }
        }
    }

    computeVelocity(psi, u, v, kx, ky);
    derivatives(w_temp, dwx, ddwx, dwy, ddwy, kx, ky);
    computeBeadPosition(d_temp);
    computeBeadVelocity(d_temp);
    biCubicSpline(u, v, d_temp);

    for (int p = 0; p < nDumbBell; ++p) {
        k_u2[p] = dt * ((1.0/((1.0+M)*St)) * (S*(d_temp[p].bead1_uf - d_temp[p].bead1_u) + (d_temp[p].bead2_uf - d_temp[p].bead2_u)));
        k_v2[p] = dt * ((1.0/((1.0+M)*St)) * (S*(d_temp[p].bead1_vf - d_temp[p].bead1_v) + (d_temp[p].bead2_vf - d_temp[p].bead2_v)));
        k_x2[p] = dt * d_temp[p].u_cm;
        k_y2[p] = dt * d_temp[p].v_cm;
        std::vector<double> l_vec = {d_temp[p].bead2_x - d_temp[p].bead1_x, d_temp[p].bead2_y - d_temp[p].bead1_y};
        std::vector<double> uf1 = {d_temp[p].bead1_uf, d_temp[p].bead1_vf};
        std::vector<double> uf2 = {d_temp[p].bead2_uf, d_temp[p].bead2_vf};
        k_omega2[p] = dt * (1.0/(M*St)) * ((M*crossProduct(l_vec, uf2) - S*crossProduct(l_vec, uf1)) - ((S*l*l*d_temp[p].omega)/(1.0+M) + (M*M*l*l*d_temp[p].omega)/(1.0+M)));
        k_alpha2[p] = dt * d_temp[p].omega;

        d_temp[p].x_cm = std::fmod(d[p].x_cm + 0.5*k_x2[p] + L, L);
        d_temp[p].y_cm = std::fmod(d[p].y_cm + 0.5*k_y2[p] + L, L);
        d_temp[p].u_cm = d[p].u_cm + 0.5*k_u2[p];
        d_temp[p].v_cm = d[p].v_cm + 0.5*k_v2[p];
        d_temp[p].omega = d[p].omega + 0.5*k_omega2[p];
        d_temp[p].alpha_c = d[p].alpha_c + 0.5*k_alpha2[p];
    }

    // vorticity
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            k2[i][j] = dt * (-u[i][j] * dwx[i][j] - v[i][j] * dwy[i][j] + nu * ddwx[i][j] + nu * ddwy[i][j]);
            w_temp[i][j] = w[i][j] + 0.5*k2[i][j];
        }
    }

    // ----------------- 3 ------------
    FFT2(w_temp);
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            if (i == 0 && j == 0) {
                psi[i][j] = 0.0;
            } else {
                psi[i][j] = -w_temp[i][j] / (kx[i][j]*kx[i][j] + ky[i][j]*ky[i][j]);
            }
        }
    }

    computeVelocity(psi, u, v, kx, ky);
    derivatives(w_temp, dwx, ddwx, dwy, ddwy, kx, ky);
    computeBeadPosition(d_temp);
    computeBeadVelocity(d_temp);
    biCubicSpline(u, v, d_temp);

    for (int p = 0; p < nDumbBell; ++p) {
        k_u3[p] = dt * ((1.0/((1.0+M)*St)) * (S*(d_temp[p].bead1_uf - d_temp[p].bead1_u) + (d_temp[p].bead2_uf - d_temp[p].bead2_u)));
        k_v3[p] = dt * ((1.0/((1.0+M)*St)) * (S*(d_temp[p].bead1_vf - d_temp[p].bead1_v) + (d_temp[p].bead2_vf - d_temp[p].bead2_v)));
        k_x3[p] = dt * d_temp[p].u_cm;
        k_y3[p] = dt * d_temp[p].v_cm;
        std::vector<double> l_vec = {d_temp[p].bead2_x - d_temp[p].bead1_x, d_temp[p].bead2_y - d_temp[p].bead1_y};
        std::vector<double> uf1 = {d_temp[p].bead1_uf, d_temp[p].bead1_vf};
        std::vector<double> uf2 = {d_temp[p].bead2_uf, d_temp[p].bead2_vf};
        k_omega3[p] = dt * (1.0/(M*St)) * ((M*crossProduct(l_vec, uf2) - S*crossProduct(l_vec, uf1)) - ((S*l*l*d_temp[p].omega)/(1.0+M) + (M*M*l*l*d_temp[p].omega)/(1.0+M)));
        k_alpha3[p] = dt * d_temp[p].omega;

        d_temp[p].x_cm = std::fmod(d[p].x_cm + k_x3[p] + L, L);
        d_temp[p].y_cm = std::fmod(d[p].y_cm + k_y3[p] + L, L);
        d_temp[p].u_cm = d[p].u_cm + k_u3[p];
        d_temp[p].v_cm = d[p].v_cm + k_v3[p];
        d_temp[p].omega = d[p].omega + k_omega3[p];
        d_temp[p].alpha_c = d[p].alpha_c + k_alpha3[p];
    }

    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            k3[i][j] = dt * (-u[i][j] * dwx[i][j] - v[i][j] * dwy[i][j] + nu * ddwx[i][j] + nu * ddwy[i][j]);
            w_temp[i][j] = w[i][j] + k3[i][j];
        }
    }

    // -------------- 4 ---------------
    FFT2(w_temp);
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            if (i == 0 && j == 0) {
                psi[i][j] = 0.0;
            } else {
                psi[i][j] = -w_temp[i][j] / (kx[i][j]*kx[i][j] + ky[i][j]*ky[i][j]);
            }
        }
    }

    computeVelocity(psi, u, v, kx, ky);
    derivatives(w_temp, dwx, ddwx, dwy, ddwy, kx, ky);
    computeBeadPosition(d_temp);
    computeBeadVelocity(d_temp);
    biCubicSpline(u, v, d_temp);

    for (int p = 0; p < nDumbBell; ++p) {
        k_u4[p] = dt * ((1.0/((1.0+M)*St)) * (S*(d_temp[p].bead1_uf - d_temp[p].bead1_u) + (d_temp[p].bead2_uf - d_temp[p].bead2_u)));
        k_v4[p] = dt * ((1.0/((1.0+M)*St)) * (S*(d_temp[p].bead1_vf - d_temp[p].bead1_v) + (d_temp[p].bead2_vf - d_temp[p].bead2_v)));
        k_x4[p] = dt * d_temp[p].u_cm;
        k_y4[p] = dt * d_temp[p].v_cm;
        std::vector<double> l_vec = {d_temp[p].bead2_x - d_temp[p].bead1_x, d_temp[p].bead2_y - d_temp[p].bead1_y};
        std::vector<double> uf1 = {d_temp[p].bead1_uf, d_temp[p].bead1_vf};
        std::vector<double> uf2 = {d_temp[p].bead2_uf, d_temp[p].bead2_vf};
        k_omega4[p] = dt * (1.0/(M*St)) * ((M*crossProduct(l_vec, uf2) - S*crossProduct(l_vec, uf1)) - ((S*l*l*d_temp[p].omega)/(1.0+M) + (M*M*l*l*d_temp[p].omega)/(1.0+M)));
        k_alpha4[p] = dt * d_temp[p].omega;
    }

    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            k4[i][j] = dt * (-u[i][j] * dwx[i][j] - v[i][j] * dwy[i][j] + nu * ddwx[i][j] + nu * ddwy[i][j]);
        }
    }

    // final
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            w[i][j] += (1.0/6.0) * (k1[i][j] + 2.0*k2[i][j] + 2.0*k3[i][j] + k4[i][j]);
        }
    }

    FFT2(w);
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            if (i == 0 && j == 0) {
                psi[i][j] = 0.0;
            } else {
                psi[i][j] = -w[i][j] / (kx[i][j]*kx[i][j] + ky[i][j]*ky[i][j]);
            }
        }
    }
    IFFT2(w);

    computeVelocity(psi, u, v, kx, ky);

    for (int p = 0; p < nDumbBell; ++p) {
        d[p].x_cm = std::fmod(d[p].x_cm + (1.0/6.0) * (k_x1[p] + 2.0*k_x2[p] + 2.0*k_x3[p] + k_x4[p]) + L, L);
        d[p].y_cm = std::fmod(d[p].y_cm + (1.0/6.0) * (k_y1[p] + 2.0*k_y2[p] + 2.0*k_y3[p] + k_y4[p]) + L, L);
        d[p].u_cm += (1.0/6.0) * (k_u1[p] + 2.0*k_u2[p] + 2.0*k_u3[p] + k_u4[p]);
        d[p].v_cm += (1.0/6.0) * (k_v1[p] + 2.0*k_v2[p] + 2.0*k_v3[p] + k_v4[p]);
        d[p].alpha_c += (1.0/6.0) * (k_alpha1[p] + 2.0*k_alpha2[p] + 2.0*k_alpha3[p] + k_alpha4[p]);
        d[p].omega += (1.0/6.0) * (k_omega1[p] + 2.0*k_omega2[p] + 2.0*k_omega3[p] + k_omega4[p]);
    }

    computeBeadPosition(d);
    computeBeadVelocity(d);
}

void textComplex(const CMatrix& Y) {
    std::ofstream file("data.txt");

    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            file << i*dx << " " << j*dy << " " << Y[i][j].real() << " " << Y[i][j].imag() << std::endl;
        }
    }
}

void exportDumbBells(const std::vector<DumbBell>& d, int iter) {
    std::string name = "dumbbell" + std::to_string(iter) + ".txt";
    std::ofstream file(name);

    for (int i = 0; i < nDumbBell; ++i) {
        file << d[i].x_cm << " " << d[i].y_cm << " " << d[i].bead1_x << " " << d[i].bead1_y << " " << d[i].bead2_x << " " << d[i].bead2_y << std::endl;
    }
}

void solve(CMatrix& w, CMatrix& psi, CMatrix& u, CMatrix& v, const Matrix& kx, const Matrix& ky, std::vector<DumbBell>& d) {
    int iter = 0;

    FFT2(w);
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            if (i == 0 && j == 0) {
                psi[i][j] = 0.0;
            } else {
                psi[i][j] = -w[i][j] / (kx[i][j]*kx[i][j] + ky[i][j]*ky[i][j]);
            }
        }
    }
    IFFT2(w);
    computeVelocity(psi, u, v, kx, ky);
    dumbbellInit(d);
    // exportDumbBells(d, 0);

    for (double t = dt; t < T; t+=dt) {
        RungeKutta4(w, psi, u, v, kx, ky, d);
        if (iter%10 == 0) {
            std::cout << iter << std::endl;
        }
        if (iter%50 == 0) {
            exportDumbBells(d, iter);
        }
        iter++;
        
    } 
    // exportParticles(p_p, p_v);
    // corefiles.close();
    
}

int main() {
    // field
    CMatrix w(x, CVector(y, 0.0));
    CMatrix psi(x, CVector(y, 0.0));
    CMatrix u(x, CVector(y, 0.0));
    CMatrix v(x, CVector(y, 0.0));

    // particles
    std::vector<DumbBell> d(nDumbBell);

    // wavenumbers
    Matrix kx(x, std::vector<double>(y, 0.0));
    Matrix ky = kx;

    setInitialCondition(w);
    computeWavenumber(kx, ky);
    
    solve(w, psi, u, v, kx, ky, d);

    // CMatrix U(x, CVector(y, 0.0));

    // for (int i = 0; i < x; ++i) {
    //     for (int j = 0; j < y; ++j) {
    //         U[i][j] = std::sqrt(u[i][j]*u[i][j] + v[i][j]*v[i][j]);
    //     }
    // }
    // textComplex(U);
    //exportDumbBells(d, 1);
    return 0;
}
