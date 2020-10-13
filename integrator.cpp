#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <complex>
#include <fstream>
#include <iomanip>
#include <stdio.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

using namespace std;
using std::cout;
using std::setw;

//***begin parameters***

int n = pow(2,11);	//size of fft arrays; must be a power of 2
const int d = 100;	//Hilbert space dimension

//***end parameters***

int Ha_indices[4] = {0};	//indices of the operator form of the kernel that correspond to a `Hartree matric'
int Ex_indices[4] = {0};	//indices of the operator form of the kernel that correspond to an `exchange matric'
void * Ha_point = &Ha_indices;
void * Ex_point = &Ex_indices;
long int Ha_lookup[d][d] = {0};	//an array of addresses to be used in conjunction with hartree.dat
long int Ex_lookup[d][d] = {0};	//an array of addresses to be used in conjunction with exchange.dat
gsl_vector_complex * Q = gsl_vector_complex_alloc(4*d);	//an intermediate matrix we'll need
gsl_vector_complex * P = gsl_vector_complex_alloc(4*d);	//an intermediate matrix we'll need
gsl_matrix_complex * M = gsl_matrix_complex_alloc(n,n);	//the matrix into which we load the fft kernel

void printProgress (double percentage)	//this function is very useful for monitoring the progress of a calculation, preventing wasted time
{
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf ("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush (stdout);
}

gsl_complex disc(int a, int b, int c, int d)	//finds elements of the atomic-basis fourth-rank Cartesian tensor form of the pair potential from the fft kernel
{
	int p = 0, q = 0;
	int i = a+1, j = b+1, k = c+1, l = d+1;	//steps necessary because C++ counts from 1
	if((i<k&&i<l)||(j<k&&j<l)||(k>i&&k>l)||(l>i&&l>j))	//this enforces the assumed symmetries on the elements artificially (we know the symmetries exist), and also saves time
	{
		q = i;
		p = j;
		i = k;
		j = l;
		k = q;
		l = p;
	}
	if(i<j)
	{
		p = i;
		i = j;
		j = p;
	}
	if(k<l)
	{
		p = k;
		k = l;
		l = p;
	}

	gsl_complex zo = GSL_COMPLEX_ZERO;

	for(int q=0;q<2;q++)	//sample the fft kernel sixteen times per element - probably reducable by symmetry if you think about it...
	{
		zo.dat[q]=	(double)(gsl_matrix_complex_get(M,i+j+n/2,k+l+n/2).dat[q])-
					(double)(gsl_matrix_complex_get(M,i+j+n/2,k-l+n/2).dat[q])-
					(double)(gsl_matrix_complex_get(M,i+j+n/2,l-k+n/2).dat[q])+
					(double)(gsl_matrix_complex_get(M,i+j+n/2,-k-l+n/2).dat[q])-

					(double)(gsl_matrix_complex_get(M,i-j+n/2,k+l+n/2).dat[q])+
					(double)(gsl_matrix_complex_get(M,i-j+n/2,k-l+n/2).dat[q])+
					(double)(gsl_matrix_complex_get(M,i-j+n/2,l-k+n/2).dat[q])-
					(double)(gsl_matrix_complex_get(M,i-j+n/2,-k-l+n/2).dat[q])-

					(double)(gsl_matrix_complex_get(M,j-i+n/2,k+l+n/2).dat[q])+
					(double)(gsl_matrix_complex_get(M,j-i+n/2,k-l+n/2).dat[q])+
					(double)(gsl_matrix_complex_get(M,j-i+n/2,l-k+n/2).dat[q])-
					(double)(gsl_matrix_complex_get(M,j-i+n/2,-k-l+n/2).dat[q])+

					(double)(gsl_matrix_complex_get(M,-i-j+n/2,k+l+n/2).dat[q])-
					(double)(gsl_matrix_complex_get(M,-i-j+n/2,k-l+n/2).dat[q])-
					(double)(gsl_matrix_complex_get(M,-i-j+n/2,l-k+n/2).dat[q])+
					(double)(gsl_matrix_complex_get(M,-i-j+n/2,-k-l+n/2).dat[q]);

	}

	return zo;
}

int main (void)
{
	cout<<"opening hartree.dat..."<<endl;
	FILE * Ha = fopen ("hartree.dat", "wb");	//this .dat stores the library of Hartree matrices directly as gsl complex matrices
	cout<<"opening exchange.dat..."<<endl;
	FILE * Ex = fopen ("exchange.dat", "wb");	//this .dat stores the library of exchange matrices directly as gsl complex matrices
	cout<<"opening kernel.dat..."<<endl;
	FILE * g = fopen ("kernel.dat", "rb");
	gsl_matrix_complex_fread (g, M);	//fill the fft kernel matrix from file
	cout<<"initializing and writing integral matrices:"<<endl;
	gsl_vector_complex_set_zero(P);	//an intermediate matrix we'll need
	gsl_vector_complex_set_zero(Q);	//an intermediate matrix we'll need

	for (int i=0;i<d;i++)
	{
		for (int j=0;j<d;j++)
		{
			Ha_lookup[i][j] = ftell(Ha);	//fills the address matrix from the current position in file
			Ex_lookup[i][j] = ftell(Ex);
			gsl_matrix_complex * A = gsl_matrix_complex_alloc(d,d); //an intermediate matrix we'll need
			gsl_matrix_complex * B = gsl_matrix_complex_alloc(d,d); //an intermediate matrix we'll need
			gsl_matrix_complex_set_zero (A);
			gsl_matrix_complex_set_zero (B);
			for (int k=0;k<d;k++)
			{
				for (int l=0;l<d;l++)
				{
					gsl_matrix_complex_set(A,k,l,disc(i,j,k,l));	//fills A as the relevant Hartree matrix from disc function
					gsl_matrix_complex_set(B,k,l,disc(i,l,k,j));	//fills B as the relevant exchange matrix from disc function
				}
			}
			gsl_matrix_complex_fwrite (Ha, A); //writes Hartree matrix to file
			gsl_matrix_complex_fwrite (Ex, B);
			gsl_matrix_complex_free(A);
			gsl_matrix_complex_free(B);
		}
		printProgress ((double)(i+1)/((double)d));
	}

	fclose (Ha);
	fclose (Ex);

	ofstream Ha_lookup_out("Ha_lookup.dat", ios::out | ios::binary);	//prints the Hartree matrix addresses to Ha_lookup.dat, as a conventional array
	Ha_lookup_out.write((char *) &Ha_lookup, sizeof Ha_lookup);
	Ha_lookup_out.close();

	ofstream Ex_lookup_out("Ex_lookup.dat", ios::out | ios::binary);
	Ex_lookup_out.write((char *) &Ex_lookup, sizeof Ex_lookup);
	Ex_lookup_out.close();
	cout<<endl;
	cout<<"write completed"<<endl;

	return 0;
}
