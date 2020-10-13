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
#include <sstream>
#include <string>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_integration.h>

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
#define GSL_COMPLEX_I gsl_complex_rect(0,1)

using namespace std;
using std::cout;
using std::setw;

//***begin switchboard*** various options for different kinds of simulations - this was much larger in the full code

const bool quench = true;	//instructs the program whether to use a quench projection on the initial states
const bool interactions = false;	//instructs the program whether to form a Fock matrix or a trivial Hamiltonian
const bool phases = true;	//instructs the program whether to predict the N+-q ground states in the shunted frame
const bool prepare_noninteracting_gas = true; //instructs the program whether to use an initial ground state for a free...
const bool prepare_interacting_gas = false;	//...or interacting gas

//***end switchboard***
//***begin parameters***

double c = 2e-6;	//weights the off-diagonal terms, acting like a square coupling constant to tune the interaction strength
const double q = 10.0000001;	//the `q' parameter of the quench
const int d = 200;	//dimension of Hilbert space
const int Number = 60;	//number of electrons
const int IT = 500;	//TDHFT evolution iterations
const int AT = 5;	//HF ground state convergence iterations
const double m = M_PI;	//mass of the electrons, pi is very convenient
const double eps = 0.5/(Number*IT);	//time step for time evolution, this is set up to end the simulation when the compression front has covered a quarter of the crystal length

///***end parameters***

long int Ha_lookup[d][d] = {0};				//Hartree addresses
long int Ex_lookup[d][d] = {0};

gsl_matrix_complex * Fa = gsl_matrix_complex_alloc (d, d);	//atomic basis fock matrix
gsl_matrix_complex * U = gsl_matrix_complex_alloc (d, d);	//time-evolution operator
gsl_matrix_complex * C = gsl_matrix_complex_alloc (d, Number);	//molecular basis, this is the `S'-matrix in the project report
gsl_matrix_complex * Qaf = gsl_matrix_complex_alloc (d, d);	//quench projection operator
gsl_matrix_complex * C_l = gsl_matrix_complex_alloc (d, abs(Number-(int)q));	//equivalent state of subsonic rarefied phase
gsl_matrix_complex * C_h = gsl_matrix_complex_alloc (d, Number+(int)q);	//equivalent state of subsonic compressed phase
gsl_matrix_complex * C_0 = gsl_matrix_complex_alloc (d, Number);	//initial state
gsl_matrix_complex * R = gsl_matrix_complex_alloc (d, d);	//diagonalization rotation matrix
gsl_matrix_complex * D = gsl_matrix_complex_alloc (d, d);	//diagonal time-evolution matrix
gsl_matrix_complex * P = gsl_matrix_complex_alloc (d, d);	//the `P'-matrix as it appears in the project report
gsl_matrix_complex * h = gsl_matrix_complex_alloc (d, d);	//kinetic energy matrix
gsl_matrix_complex * Q = gsl_matrix_complex_alloc (d, d);	//temporary matrix we'll meed
gsl_matrix_complex * M = gsl_matrix_complex_alloc(d,d);		//temporary matrix we'll need
gsl_vector * E = gsl_vector_alloc (d); //a vector to store energy eigenvalues

FILE * g = fopen ("hartree.dat", "rb");
FILE * f = fopen ("exchange.dat", "rb");

gsl_vector_complex * aleph = gsl_vector_complex_alloc(d);	//temporary
gsl_vector_complex * bet = gsl_vector_complex_alloc(d);		//temporary

struct field_operators	//this object is a legacy from when the code was equipped to calculate the stress-energy tensor
{
    double number_density;
    double Hamiltonian_density;
    gsl_complex momentum_density;
    gsl_complex momentum_flux_density;
};

void printProgress (double percentage) //this function is very useful for monitoring the progress of a calculation, preventing wasted time
{
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf ("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush (stdout);
}

double basis(double x, int n)	//the basis wavefunctions of the shunted state
{
	return sqrt(2)*sin((n+1)*x*M_PI);
}

void field_operator_filler(double x, gsl_matrix_complex * M, field_operators * A) //again, this used to calculate the stress-energy tensor but now ony calculates densities
{
	int N = (int)M->size2;
	for(int i = 0; i<N; i++)
	{
		gsl_complex phi = GSL_COMPLEX_ZERO;
		for(int k = 0; k<d; k++)
		{
			phi = gsl_complex_add(phi,gsl_complex_mul_real(gsl_matrix_complex_get(M,k,i),basis(x,k)));
		}
		A->number_density += gsl_complex_abs(gsl_complex_mul(gsl_complex_conjugate(phi),phi));
	}
}

gsl_complex y(double u) //a function used in the creation of the shunt projection matrix, see the project report for the form of the matrix
{
	if(q==u)
	{
		return gsl_complex_rect(q,0);
	}
	else
	{
		gsl_complex z = GSL_COMPLEX_ONE;
		z = gsl_complex_sub(z,gsl_complex_mul_real(gsl_complex_exp(gsl_complex_rect(0,-M_PI*q)),cos(M_PI*u)));
		z = gsl_complex_sub(z,gsl_complex_mul_real(gsl_complex_exp(gsl_complex_rect(0,-M_PI*(q+1))),(u/q)*sin(M_PI*u)));
		return gsl_complex_mul_real(z,q/(M_PI*(pow(q,2)-pow((u),2))));
	}
}

void projection_initializer() //generates the shunt projection matrix
{
	for (int j=0;j<d;j++)
	{
		for (int k=0;k<d;k++)
		{
			if(q==0)
			{
				if(j==k)
				{
					gsl_matrix_complex_set(Q,j,k,GSL_COMPLEX_ONE);
				}
				else
					gsl_matrix_complex_set(Q,j,k,GSL_COMPLEX_ZERO);
			}
			else
			{
				gsl_matrix_complex_set(Q,j,k,gsl_complex_sub(y(k-j),y(k+j+2)));
			}
		}
	}
}

gsl_complex trace(long int L, int k) //takes the trace of P with the Hartree or exchange matrix (determined by k) at the relevant address
{
	if(k == 0)
	{
		fseek(g,L,SEEK_SET);
		gsl_matrix_complex_fread (g, M);
	}
	else
	{
		fseek(f,L,SEEK_SET);
		gsl_matrix_complex_fread (f, M);
	}
	gsl_complex value = GSL_COMPLEX_ZERO;
	gsl_complex value2 = GSL_COMPLEX_ZERO;
	for (int i=0;i<d;i++)
	{
		gsl_matrix_complex_get_row(aleph,P,i);
		gsl_matrix_complex_get_col(bet,M,i);
		gsl_blas_zdotu (aleph,bet,&value2);
		value = gsl_complex_add(value,value2);
	}
	return 	value;
}

void print_groundstates(int a,gsl_matrix_complex * M)	//produces a two-column text file with x and rho(x) for each of the compressed and rarefied ground states
{
	ofstream groundstates;
	if(a==0)
	{
	    groundstates.open("groundstates_l.txt");
	}
	if(a==1)
	{
	    groundstates.open("groundstates_h.txt");
	}
	for (int l=0;l<1000;l++)
	{
		double x = l/1000.0;
		field_operators A = {};
		field_operator_filler(x,M,&A);
		groundstates<<x<<std::setw(20)<<A.number_density<<endl;
	}
	groundstates.close();
}

void simulator_printer(int it) //produces a two-column text file with x and rho(x) for the current iteration of the density
{
	cout<<"printing frame to file..."<<endl;
	ofstream data;
	data.open("data.txt");
	for (int l=0;l<1000;l++)
	{
		double x = l/1000.0;
		field_operators A = {};
		field_operator_filler(x,C,&A);
		data<<x
		<<std::setw(20)<<A.number_density
		<<endl;
	}
	data.close();
}

void piper(int it) //this is a Gnuplot pipe, it takes the local density text files and deposits the frame of the animation in the folder "frames"
{
	FILE* pipe = _popen("C:/gnuplot/bin/gnuplot.exe", "w");	//the path to "gnuplot.exe" goes here - on non-Windows machines you will have a different root and slash
	if (pipe != NULL)
	{
		fprintf(pipe, "set term win\n");
		fprintf(pipe, "set terminal pngcairo dashed size 800,500 enhanced font 'cmb10, 15'\n");
		fprintf(pipe, "set style line 1 lt 1 lc rgb 'black' lw 1\n");
		fprintf(pipe, "set style line 2 lt 1 lc rgb 'red' lw 1\n");
		fprintf(pipe, "set style line 3 lt 1 lc rgb 'blue' lw 1\n");
		fprintf(pipe, "set encoding utf8\n");
		fprintf(pipe, "unset key\n");
		fprintf(pipe, "set xlabel 'crystal coordinate'\n");
		fprintf(pipe, "set title 'TDHF %d fermions, %d-dimensional basis'\n",Number,d);
		fprintf(pipe, "set ylabel '{/Symbol Y}{/Symbol Y}^*'\n");
		fprintf(pipe, "set xrange [0:1]\n");
		fprintf(pipe, "set yrange [%d:%d]\n",0,2*Number);
		fprintf(pipe, "set label 'Time %e' at 0.55,7.0 \n",it*eps);
//		fprintf(pipe, "set arrow from 0,500+3 to '%e',3  \n",((Number+q)*eps*it));
//		fprintf(pipe, "set arrow from 0,2.5 to '%e',2.5  \n",((Number-q)*eps*it));
//		fprintf(pipe, "set arrow from '%e','%e' to '%e','%e'  \n",(Number*eps*it+0.05),(Number+0.5*q+10),(Number*eps*it),(Number+0.5*q));
		fprintf(pipe, "set label 'Compression front' at '%e','%e' \n",(Number*eps*it),(Number-15.0));
		fprintf(pipe, "set label 'Compressed phase' at 0.5,80 \n");
		fprintf(pipe, "set label 'Rarefied phase' at 0.5,40 \n");
		fprintf(pipe, "set arrow from 0.5,40 to 0.5,50  \n");
		fprintf(pipe, "set arrow from 0.5,80 to 0.5,70  \n");
		fprintf(pipe, "set arrow from '%e','%e' to '%e',60  \n",((Number-q)*eps*it),(Number-10.0),((Number-q)*eps*it));
		fprintf(pipe, "set arrow from '%e','%e' to '%e',60  \n",((Number+q)*eps*it),(Number-10.0),((Number+q)*eps*it));
		fprintf(pipe, "set arrow from '%e','%e' to '%e','%e' nohead  \n",((Number-q)*eps*it),(Number-10.0),((Number+q)*eps*it),(Number-10.0));
		fprintf(pipe, "set output 'frames/frame_%d.png'\n",it);
		fprintf(pipe, "plot "
				"'data.txt' using 1:2 w l ls 1 dashtype 1,"
//				"'data.txt' using 1:3 w l ls 2 dashtype 1,"
//				"'data.txt' using 1:4 w l ls 3 dashtype 1,"
//				"'data.txt' using 1:5 w l ls 4 dashtype 1,"
//				"'data.txt' using 1:5 w l ls 5 dashtype 1,";
		);
		if (phases)
		{
			fprintf(pipe, "'groundstates_l.txt' using 1:2 w l ls 1 lc rgb '#99555555' dashtype 1,"
							" 'groundstates_h.txt' using 1:2 w l ls 1 lc rgb '#99555555' dashtype 1\n");		
		}
		else
		{
			fprintf(pipe, "\n");
		}
		fprintf(pipe, "unset output\n");
		fflush(pipe);
		cout<<"FRAME "<<it<<" COMPLETE"<<endl;
	}
	else puts("cannot call Gnuplot: check the path\n");
	_pclose(pipe);
	remove("data.txt");	//this stops text files building up in the home directory
}

void lookup_initializer()	//fills the addresses for the Hartree & exchange operators from file
{
	if (interactions)
	{
		ifstream Ha_lookup_in("Ha_lookup.dat", ios::in | ios::binary);
		Ha_lookup_in.read((char *) &Ha_lookup, sizeof Ha_lookup);
		Ha_lookup_in.close();

		ifstream Ex_lookup_in("Ex_lookup.dat", ios::in | ios::binary);
		Ex_lookup_in.read((char *) &Ex_lookup, sizeof Ex_lookup);
		Ex_lookup_in.close();
	}
}

void kinetic_initializer()	//sets up the diagonal free Hamiltonian
{
	for (int i=0;i<d;i++)
	{
		gsl_matrix_complex_set (h, i, i, gsl_complex_rect((1.0/(2.0*m))*pow((M_PI*i+1),2),0.0));
	}
}

void fock_initializer(gsl_matrix_complex * M)	//calculates the off-diagonal terms of the Fock matrix and adds the diagonal Hamiltonian to the final product
{
	cout<<"initializing Fock matrix:"<<endl;
	if (interactions)
	{
		gsl_matrix_complex * w6 = gsl_matrix_complex_alloc (d, d);
		gsl_blas_zgemm (CblasNoTrans,CblasConjTrans,GSL_COMPLEX_ONE,M,M,GSL_COMPLEX_ZERO,w6);
		gsl_matrix_complex_memcpy (P, w6);
		gsl_matrix_complex_free (w6);
		for (int i=0;i<d;i++)
		{
			for (int k=0;k<i+1;k++)
			{
				gsl_complex inter = gsl_complex_mul_real(gsl_complex_sub(trace(Ha_lookup[i][k],0),trace(Ex_lookup[i][k],1)),c);
				gsl_matrix_complex_set (Fa,i,k,inter);
				gsl_matrix_complex_set (Fa,k,i,gsl_complex_conjugate(inter));
			}
			printProgress ((double)(i+1)/((double)d));
		}
	}
	gsl_matrix_complex_add (Fa,h);
	cout<<endl;
}

void fock_diagonalizer()	//diagonalizes the Fock matrix
{
	cout<<"diagonalizing Fock matrix..."<<endl;
	gsl_eigen_hermv_workspace * w = gsl_eigen_hermv_alloc (d);
	gsl_eigen_hermv (Fa,E,R,w);
	gsl_eigen_hermv_free (w);
	gsl_eigen_hermv_sort (E, R, GSL_EIGEN_SORT_VAL_ASC);
}

void fock_convergence(gsl_matrix_complex * M)	//see `converger'
{
	int N = (int)M->size2;
	gsl_matrix_complex_view w9 = gsl_matrix_complex_submatrix(R,0,0,d,N);
	gsl_matrix_complex_memcpy (M, &(w9.matrix));
}

void zero() //this resets several matrices
{
	gsl_matrix_complex_set_zero (Fa);
	gsl_matrix_complex_set_zero (D);
	gsl_matrix_complex_set_zero (R);
	gsl_matrix_complex_set_zero (U);
	gsl_vector_set_zero(E);
}

void converger(gsl_matrix_complex * M) //implements regular old Hartree Fock groundstate convergence routine
{
	gsl_matrix_complex_set_identity(M);
	if(interactions)
	{
		zero();
		for (int it=0;it<AT;it++)
		{
			fock_initializer(M);
			fock_diagonalizer();
			fock_convergence(M);
		}
	}
}

void state_initializer() //sets up the initial `S' matric (i.e. C) depending on the options at the switchboard
{
	if (prepare_noninteracting_gas)
	{
		gsl_matrix_complex_set_identity (C);
		gsl_matrix_complex_memcpy (C_0, C);
	}
	if (prepare_interacting_gas)
	{
		gsl_matrix_complex_set_identity (C);
		converger(C);
		gsl_matrix_complex_memcpy (C_0, C);
	}
	if (quench)
	{
		gsl_matrix_complex * w8 = gsl_matrix_complex_alloc (d, Number);
		gsl_blas_zgemm (CblasNoTrans,CblasNoTrans,GSL_COMPLEX_ONE,Q,C,GSL_COMPLEX_ZERO,w8);
		gsl_matrix_complex_memcpy (C, w8);
		gsl_matrix_complex_free (w8);
		gsl_matrix_complex_memcpy (Qaf, Q);
	}
}

void groundstates() //calculates the compressed/rarefied phases and sends them for printing
{
	if (phases)
	{
		cout<<"finding SCF rarefied groundstate..."<<endl;
		converger(C_l);
		print_groundstates(0,C_l);
		
		cout<<"finding SCF compressed groundstate..."<<endl;
		converger(C_h);
		print_groundstates(1,C_h);
	}
}

void fock_exponentiator() //part of time-evolution, exponentiates the diagonalized matrix
{
	cout<<"exponentiating Fock matrix..."<<endl;
	for(int i = 0; i<d; i++)
	{
		gsl_matrix_complex_set(D,i,i,gsl_complex_exp(gsl_complex_rect(0.0,-1.0*eps*gsl_vector_get(E,i))));
	}
	gsl_matrix_complex * w3 = gsl_matrix_complex_alloc (d, d);
	gsl_blas_zgemm (CblasNoTrans,CblasNoTrans,GSL_COMPLEX_ONE,R,D,GSL_COMPLEX_ZERO,w3);
	gsl_blas_zgemm (CblasNoTrans,CblasConjTrans,GSL_COMPLEX_ONE,w3,R,GSL_COMPLEX_ZERO,U);
	gsl_matrix_complex_free (w3);
}

void fock_propagator() //implements time evolution step
{
	cout<<"propagating quantum state..."<<endl;
	int N = (int)C->size2;
	gsl_matrix_complex * w1 = gsl_matrix_complex_alloc (d, N);
	gsl_blas_zgemm (CblasNoTrans,CblasNoTrans,GSL_COMPLEX_ONE,U,C,GSL_COMPLEX_ZERO,w1);
	gsl_matrix_complex_memcpy (C, w1);
	gsl_matrix_complex_free (w1);
}

void simulator() //the main body of the simulation, dictates what is involved at each iteration
{
	cout<<"commencing TDHF simulation..."<<endl;
	state_initializer();
	for (int it=0;it<IT;it++)
	{
		zero();
		if (it==0)
		{

		}
		else
		{
			fock_initializer(C);
			fock_diagonalizer();
			fock_exponentiator();
			fock_propagator();
		}
		simulator_printer(it);
		piper(it);
	}
}

void banner_begin()
{
	cout<<"*************************************************************"<<endl;
	cout<<"*************************************************************"<<endl;
	cout<<"| | | SIMULATION COMMENCING -- TIME DEPENDENT HARTREE FOCK THEORY | | |"<<endl;
}

void banner_end()
{
	cout<<"*************************************************************"<<endl;
	cout<<"*************************************************************"<<endl;
	cout<<"| | | SIMULATION CONCULDED -- TIME DEPENDENT HARTREE FOCK THEORY | | |"<<endl;
}

int main (void)
{
	banner_begin();

	lookup_initializer(); //set up the addresses of the Hartree/exchange matrixes from file
	projection_initializer(); //set up the shunt projection matrix
	kinetic_initializer(); //set up the trivial diagonal Hamiltonian

	groundstates(); //if needed, find the ground states for the compressed/rarefied phases by convergence
	simulator(); //do the simulation

	banner_end();

	return 0;
}
