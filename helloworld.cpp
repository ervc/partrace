#include <iostream>
#include <array>

template <int Ndim>
class Domain
{
public:
	Domain() {
		int default_size = 64;
		for (int i=0;i<Ndim;i++) {
			nx[i] = default_size;
			xlo[i] = 0.0;
			xhi[i] = 1.0;
		}

		for (int d = 0; d < Ndim; ++d) dx[d] = (xhi[d] - xlo[d]) / nx[d];
        for (int d = 0; d < Ndim; ++d) invdx[d] = 1. / dx[d];

        vol = 1;
        for (int d = 0; d < Ndim; ++d) vol *= dx[d];
        invvol = 1. / vol;
	}
    Domain(std::array<int, Ndim> const& nx,
        std::array<double, Ndim> const& xlo,
        std::array<double, Ndim> const& xhi) :
            nx(nx), xlo(xlo), xhi(xhi)
    {
        for (int d = 0; d < Ndim; ++d) dx[d] = (xhi[d] - xlo[d]) / nx[d];
        for (int d = 0; d < Ndim; ++d) invdx[d] = 1. / dx[d];

        vol = 1;
        for (int d = 0; d < Ndim; ++d) vol *= dx[d];
        invvol = 1. / vol;
    }

    std::array<double, Ndim> getCellCoords(std::array<int, Ndim> const& idx) const
    {
        std::array<double, Ndim> x;
        for (int d = 0; d < Ndim; ++d)
        {
            x[d] = xlo[d] + (idx[d] + 0.5) * dx[d];
        }

        return x;
    }

    std::array<double, Ndim> getPointCoords(std::array<int, Ndim> const& idx) const
    {
        std::array<double, Ndim> x;
        for (int d = 0; d < Ndim; ++d)
        {
            x[d] = xlo[d] + idx[d] * dx[d];
        }

        return x;
    }

    std::array<int, Ndim> nx;
    std::array<double, Ndim> xlo;
    std::array<double, Ndim> xhi;

    std::array<double, Ndim> dx;
    std::array<double, Ndim> invdx;

    double vol;
    double invvol;
};

int main(int argc, char const *argv[])
{
	const int NDIM = 3;
	std::array<int,NDIM> nx = {64,64,1};
	std::array<double,NDIM> xlo = {0,0,0};
	std::array<double,NDIM> xhi = {1,1,1};

	Domain<NDIM> dom;
	std::cout << dom.vol << "\n";
	return 0;
}