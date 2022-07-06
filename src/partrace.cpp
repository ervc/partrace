#include <iostream>
#include <stdlib.h>
#include <array>

#ifndef NDIM
#define NDIM 3
#endif

class Grid
{
public:
	int nx,ny,nz;
	int total_cells;
	Grid(int nx,int ny,int nz) : nx(nx), ny(ny), nz(nz) {
		total_cells = nx*ny*nz;
	}
	int getIndex(int i, int j, int k) {
		return nx*ny*k + nx*j + i;
	}
};

class Mesh
{
public:
	int nx[3] = { 0 };
	double xlo[3] = { 0 };
	double xhi[3] = { 0 };

	// Mesh() {
	// 	nx = {64,64,1};
	//  	xlo = {-0.5,-0.5,-0.5};
	//  	xhi = {0.5,0.5,0.5};
	// }

	Mesh(
		int const& nx[3],
		double xlo[3],
		double xhi[3]) : 
			nx(nx), xlo(xlo), xhi(xhi) {}

	~Mesh() {};

	void print_nx() {
		std::cout << "{";
		for (int i=0;i<NDIM;i++) {
			std::cout << nx[i];
			if (i < NDIM-1) {
				std::cout << ", ";
			}
		}
		std::cout << "}" <<"\n";
	}
	
};

int main(int argc, char const *argv[]) {
	int nx[3] = {24,24,1};
	double xlo[3] = {-0.5,-0.5,-0.5};
	double xhi[3] = {0.5,0.5,0.5};
	Mesh mesh(nx,xlo,xhi);
	mesh.print_nx();

	return 0;
}


/*
int main(int argc, char const *argv[])
{
	const int NX = 3;
	const int NY = 5;
	const int NZ = 2;
	// const int NCELL = NX*NY*NZ;
	Grid grid(NX,NY,NZ);
	const int NCELL = grid.total_cells;
	fprintf(stdout, "%d\n", NCELL);

	// std::array<double, NCELL> rho = {{1}};
	double rho[NCELL] = {};
	double vel[3][NCELL] = {{}};
	double cell[3][NCELL] = {{}};
	// std::array<double, NCELL> velx = {};
	// std::array<double, NCELL> vely;
	// std::array<double, NCELL> velz;

	for (int i = 0; i < NX; i++) {
		for (int j = 0; j < NY; j++) {
			for (int k = 0; k < NZ; k++) {
				int ind = grid.getIndex(i,j,k);
				rho[ind] = 1.;
				fprintf(stdout, "%d\n", ind);
				fprintf(stdout, "%d  %d  %d\n", i,j,k);
			}
		}
	}

	return 0;
}
*/