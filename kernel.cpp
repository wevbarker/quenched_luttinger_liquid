#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_fft_complex.h>

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

using namespace std;
using std::cout;
using std::setw;

//***begin parameters***

const int d = 100;		//Hilbert space dimension
const int Number = 50;	//Number of fermions
const double xi = 1;	//quasi-one-dimensionality  parameter
const double eta = Number/(2*xi);	//confinement parameter
const int n = pow(2,11);	//size of fft arrays; must be a power of 2

//***end parameters***

gsl_matrix_complex * Rxy = gsl_matrix_complex_alloc(n,n);		//real space kernel
gsl_matrix_complex * Rkxy = gsl_matrix_complex_alloc(n,n);		//fft taken in x
gsl_matrix_complex * Rkxky = gsl_matrix_complex_alloc(n,n);		//fft taken in y
gsl_vector_complex * v = gsl_vector_complex_alloc(n);	//an intermediate vector we'll need

void printProgress (double percentage)	//this function is very useful for monitoring the progress of a calculation, preventing wasted time
{
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf ("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush (stdout);
}

double kernel(double x, double y)	//the real space kernel in the Hartree & exchange integrals defined for the principal unit cell in the lattice of parameter 2l_0
{
	if((x<1.0) && (y<1.0))
	{
		return (sqrt(M_PI)*eta)*exp(pow(eta*(x-y),2))*erfc(abs(eta*(x-y)));	//the effective pair potential: here that for the harmonic wire but can be completely general
	}
	else
	{
		return 0.0;
	}
}

int main (void)
{
	cout<<"\n"<<"Initializing real-space kernel...";
	for (int i=0;i<n;i++)
	{
		for (int j=0;j<n;j++)
		{
			gsl_matrix_complex_set(Rxy,i,j,gsl_complex_rect(kernel(2*(double)i/(double)n,2*(double)j/(double)n),0));	//initilizes the kernel in Rxy
		}
	}
	
	cout<<"\n"<<"Fourier-transforming x:"<<endl;
	for (int i=0;i<n;i++)	//the first fft in x fills the matrix Rkxy
	{	
		gsl_matrix_complex_get_col(v,Rxy,i);
		gsl_complex_packed_array data = v->data;
		size_t stride = v->stride;
		size_t m = v->size;
		gsl_fft_complex_radix2_forward (data, stride, m);
		for (int j=0;j<n/2;j++)
		{
			gsl_matrix_complex_set(Rkxy,n/2+j,i,gsl_complex_rect(data[2*j],data[2*j+1]));
			gsl_matrix_complex_set(Rkxy,j,i,gsl_complex_rect(data[n+2*j],data[n+2*j+1]));
		}
		printProgress ((double)(i+1)/((double)n));
	}

	cout<<"\n"<<"Fourier-transforming y:"<<endl;
	for (int i=0;i<n;i++)	//the second fft in y fills the matrix Rkxky
	{	
		gsl_matrix_complex_get_row(v,Rkxy,i);
		gsl_complex_packed_array data = v->data;
		size_t stride = v->stride;
		size_t m = v->size;
		gsl_fft_complex_radix2_forward (data, stride, m);
		for (int j=0;j<n/2;j++)
		{
			gsl_matrix_complex_set(Rkxky,i,n/2+j,gsl_complex_rect(data[2*j],data[2*j+1]));
			gsl_matrix_complex_set(Rkxky,i,j,gsl_complex_rect(data[n+2*j],data[n+2*j+1]));
		}
		printProgress ((double)(i+1)/((double)n));
	}

	cout<<"\n"<<"Printing to kernel.dat...";
	FILE * kernel = fopen ("kernel.dat", "wb");	//outputs the final double fft for creation of Hartree & exchange integrals
	gsl_matrix_complex_fwrite (kernel, Rkxky);
	cout<<"\n"<<"Write complete";

	return 0;
}
