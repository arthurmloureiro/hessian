#include "Hessian.h"

int main(){

	std::cout << "trying to see if I get same results as Alan's code for the Hessian" << std::endl;

	// variables:
	int L		=	10;
	int	nbins	=	4 ;
	int n 		= 	L*nbins;

	double DeltaG 	=	0.00001;
	double DeltaA	=	0.00001;

	bool verbose = true;

	//Generating data from a normal distribution N(0, 1)
	random_device rd{};
    mt19937 gen{rd()};
    normal_distribution<double> gaussian(0.0, 1.0);

	matrixType G0ini(nbins, nbins); 
	for (int i = 0; i < nbins; i++){
		for (int j = 0; j < nbins; j++){
			G0ini(i,j) = gaussian(gen);
		}
	}

	// trying to make G0 symmetrical:
	//matrixType G0 = G0ini + G0ini.transpose().eval();
	//cout << "G0 is: \n" << G0 << endl;

	// for debugging:
	matrixType G0(nbins,nbins);
	G0 << 0.187512,  0.896667,   3.11734, -0.213435, 
          0.896667, -0.662912, -0.994966,  -1.78307,
          3.11734, -0.994966, 3.86366,  -1.23029,
          -0.213435,  -1.78307,  -1.23029,  -2.01497;
	
	
	matrixType sigma(nbins, nbins);
	matrixType sigmatilde0(nbins, nbins);

	matrixType mean_am 	= 	MatrixXd::Zero(L, nbins);
	matrixType am 		= 	MatrixXd::Zero(L, nbins);
	
	vectorType W0(G0.rows());
	matrixType U0(G0.rows(), G0.cols());
	matrixType C0(G0.rows(), G0.cols());
	matrixType invC0(G0.rows(), G0.cols());
	GtoC(G0, W0, U0, C0, invC0); 

	// multivariate gaussian with zero mean:
	normal_random_variable sample { C0 };
	
	// Generate the m values separately (all field values are real).  
	// am is a subpart of the whole a (corresponding to a single m, for all fields).
	// FIXME: commented out for debugging!
	// for (int m = 0; m < L; m++){
	// 	am.row(m) += sample();
	// 	//cout << " TEST MULTIVARIATE TEST TEST TEST: " << am.row(m) <<  endl;
	// }

	// matrixType a0;
	// a0 = am.transpose().eval();
	// a0.resize(1, n);

	matrixType a0(1, n);
	a0 << 9.88701,  -0.844261,    17.2041,   -2.42976,   -9.89706,    3.26295,
                    -19.9852,    1.64855,   -10.2665,    2.36963,   -21.3809,   2.65065,
                    -0.247704,  -0.503445,    1.48576, -0.0450003,    8.14745,  -0.235597,
                    13.3251,   -1.88637,   -6.79681,  -0.397248,   -12.5753,     2.8178,
                    1.2276,   -1.89707,    3.07532,    0.67641,   -13.6552,    -1.0081,
                    -21.7761,    4.60199,   -3.98565,   -1.17426,   -6.13525,    2.01446,
                    3.50602,    1.52371,    4.31075,   -1.19933;
	
	
	// Test sigma_from_a routine at fiducial point:
	matrixType sigma0(nbins, nbins);
	sigma0 = sigma_from_a(a0, L, nbins);
	//Combine sigma with two factors of U:
	sigmatilde0 = tilde(sigma0, U0);

	matrixType phi0(nbins,nbins);
	NArrayType psi0(boost::extents[nbins][nbins][nbins]);

	// In the following, I have replaced W0 by -W0, since we want the inverse of C:
	Ortiz_matrices(phi0, psi0, -W0, nbins);

	// Covariance matrix (well, G) is random, so this is probably the right prior...
	int q0 = 0; 
	matrixType invN0	= 	MatrixXd::Identity(n,n); // Make the noise white
	matrixType Z0 		= 	invN0; //  Make the transform trivial, for simplicity
	matrixType x0 		= 	MatrixXd::Zero(1,n); // Set data vector to zero. Why not?
	
	double func0 = neglnpost(a0, G0, x0, sigma0, invN0, Z0, L, q0);
	double prior0 = neglnprior(G0, q0);
	//cout << "TARGET: " << target(sigma0, G0) << endl;

	cout << "\t Calculating first derivatives w.r.t. G" << endl;

	matrixType numerical_G(nbins,nbins);
	matrixType analytic_G(nbins, nbins);

	for (int i=0; i < nbins; i++){
		for (int j=0; j < nbins; j++){
			matrixType G1 = perturb(G0, i, j, DeltaG);

			double func1 = neglnpost(a0, G1, x0, sigma0, invN0, Z0, L, q0);

			numerical_G(i,j) = (func1 - func0)/DeltaG;
			analytic_G(i,j) = grad_G(sigmatilde0, U0, phi0, i, j, q0, L, nbins);
			
		}
	}


	cout << "\t Calculating first derivatives w.r.t. alms" << endl;
	
	matrixType numerical_a(1,n);
	matrixType sigma1(nbins, nbins);
	
	for (int i=0; i < n; i++){
		matrixType a1 = a0;
		a1(i) += DeltaA;
		sigma1 = sigma_from_a(a1, L, nbins); 
		
		double func2 = neglnpost(a1, G0, x0, sigma1, invN0, Z0, L, q0);
		//cout << "post0 = " << func0 << " // post1 = " << func2 << "// post1 - post0 = " << (func2 - func0) << endl;
		numerical_a(i) = (func2 - func0)/DeltaA;

	}


	cout << "\t Calculating the second derivatives w.r.t. G" << endl;

	matrixType secondDerivG(1, intPow(nbins, 4));
	matrixType numerical_G0(nbins,nbins);
	
	int index = 0;

	for (int i = 0; i < nbins; i++){
		for (int j = 0; j < nbins; j++){

			matrixType G1 = perturb(G0, i, j, DeltaG);

			double func1 = neglnpost(a0, G1, x0, sigma0, invN0, Z0, L, q0);
			numerical_G0(i,j) = (func1 - func0)/DeltaG;


			// Now second derivatives.  Perturb also element G[kk,ll]:
			for (int kk = 0; kk < nbins; kk++){
				for (int ll = 0; ll < nbins; ll++){

					matrixType G2	= 	perturb(G0, kk, ll, DeltaG);
					double func2 	= 	neglnpost(a0, G2, x0, sigma0, invN0, Z0, L, q0);
					
					matrixType G3	=	perturb(G2, i, j, DeltaG);
					double func3	=	neglnpost(a0, G3, x0, sigma0, invN0, Z0, L, q0);

					secondDerivG(index) = (func3 - func2)/DeltaG;

					index += 1;

				}
			} 

		}
	}

	cout << "\t Calculating the second derivatives w.r.t. alms" << endl;

	matrixType secondDerivA(1, intPow(nbins, 4));
	matrixType numerical_a0(1, n);
	matrixType numerical_a2(1, n);
	matrixType Hessian_aa_Num(n,n);
	matrixType sigma2(nbins, nbins);

	index = 0;

	// compute the gradient at a fiducial point:
	for (int i=0; i < n; i++){
		matrixType a1 = a0;
		a1(i) += DeltaA;
		sigma1 = sigma_from_a(a1, L, nbins); 
		
		double func1 = neglnpost(a1, G0, x0, sigma1, invN0, Z0, L, q0);
		//cout << "post0 = " << func0 << " // post1 = " << func2 << "// post1 - post0 = " << (func2 - func0) << endl;
		numerical_a0(i) = (func1 - func0)/DeltaA;

	}

	// Perturb a elements in sequence and find gradients at perturbed points:
	// why is this being repeated again????
	for (int j = 0; j < n; j++){
		matrixType a1 = a0;
		a1(j) += DeltaA;
		sigma1 = sigma_from_a(a1, L, nbins);

		double func1 = neglnpost(a1, G0, x0, sigma1, invN0, Z0, L, q0);

		for (int i = 0; i < n; i++){
			matrixType a2 = a1;
			a2(i) += DeltaA;
			sigma2 = sigma_from_a(a2, L, nbins);

			double func2 = neglnpost(a2, G0, x0, sigma2, invN0, Z0, L, q0);

			numerical_a2(i) = (func2  - func1)/DeltaA;

		}

		Hessian_aa_Num.row(j) = (numerical_a2 - numerical_a0)/DeltaA;

	}


	if (verbose == true){
		cout << "G0: \n"			<< G0 				<< endl;
		cout << "a0: \n" 			<< a0 				<< endl;
		cout << "U0: \n" 			<< U0 				<< endl;
		cout << "W0: \n" 			<< W0 				<< endl;
		cout << "sig0: \n" 			<< sigma0 			<< endl;
		cout << "C0: \n" 			<< C0 				<< endl;
		cout << "InvC0: \n" 		<< invC0			<< endl;
		cout << "SigTilde0 \n" 		<< sigmatilde0 		<< endl;
		cout << "Post0: " 			<< func0			<< endl;
		cout << "[N] dPost/dG = \n"	<< numerical_G 		<< endl;
		cout << "[A] dPost/dG = \n"	<< analytic_G 		<< endl;
		//cout << "dPost/da = \n" 	<< numerical_a 		<< endl;
		//cout << "d2Post/dG2 = \n"	<< secondDerivG 	<< endl;
		//cout << "Hessian_AA = \n"	<< Hessian_aa_Num	<< endl;
		cout << "Phi0 = \n"			<< phi0 			<< endl;
		cout << "Psi0[1][1][1] ="	<< psi0[1][1][1]	<< endl;
	}
}


int intPow(int x, int p) {
  if (p == 0) return 1;
  if (p == 1) return x;
  return x * intPow(x, p-1);
}

matrixType perturb(matrixType G, int ii, int jj, double DeltaG){
	// Change the [i,j] and [j,i] entries of array G by adding DeltaG

	matrixType tempG = G;

	if (ii == jj){
		tempG(ii,jj) = G(ii,jj) + DeltaG;
	} else {
		tempG(ii,jj) = G(ii,jj) + DeltaG; 
		tempG(jj,ii) = G(jj,ii) + DeltaG; 
	}

	return tempG;
}

double neglnpost(matrixType a, matrixType G, matrixType x, matrixType sigma, matrixType invN, matrixType Z, int L, int q){
	// returns the neglog posterior

	double firstPart = (2*L+1)*(0.5)*target(sigma, G);
	double likepart = neglnlike(a, x, invN, Z);
	double priorpart = (2*L+1)*neglnprior(G, q);
	
	return firstPart + likepart + priorpart;
}


double neglnlike(matrixType a, matrixType x, matrixType invN, matrixType Z){
	/* 	the neg-ln likelihood
		in python:
			 dx = x-np.dot(Z,a.T)
    		 neglnlike = 0.5*np.dot(dx,np.dot(invN,dx.T)) 
    FIXME: There is a potential problem here
    */

	double neglnlk;

	matrixType dx(1, x.cols());
	matrixType Za(1, x.cols());
	
	Za = (Z*a.transpose());
	//cout << "[ZA]  rows = " << Za.rows() << " cols = " << Za.cols() << endl;
	
	dx = x - Za.transpose();
	//cout << "[dx]  rows = " << dx.rows() << " cols = " << dx.cols() << endl;
	
	matrixType invNdx = invN * dx.transpose();
	matrixType dxInvNdx = dx*invNdx;
	//cout << "[dxInvNdx] rows = " << dxInvNdx.rows() << " cols = " << dxInvNdx.cols() << endl;
	//cout << "dx*invNdx = " << dx*invNdx << endl;
	
	// dxInvNdx is a matrix even though it's a scalar... this (0,0) "converts" to a double:
	neglnlk = 0.5*(dxInvNdx(0,0)); 
	
	return neglnlk;
}

double neglnprior(matrixType G, double q){
	// Returns -ln(prior)+0.5*lndetC for a Jeffreys prior |C|^q 

	double lndetC = G.trace();

	return 0.5*(1. - 2.*q)*lndetC;
}

double target(matrixType sigma, matrixType G){
	// computes the Trace(e^(G) x sigma)

	// necessary local variables:
	vectorType W0f(G.rows());
	matrixType U0f(G.rows(), G.cols());
	matrixType C0f(G.rows(), G.cols());
	matrixType invC0f(G.rows(), G.cols());

	GtoC(G, W0f, U0f, C0f, invC0f); 

	double result = 0.0;
	matrixType invCSig(invC0f.rows(), sigma.cols());
	invCSig = invC0f*sigma;

	result = invCSig.trace();

	// for (int i = 0; i < invCSig.cols(); i++){
	// 	result += invCSig(i,i);
	// }

	return result;
}

void GtoC(matrixType G, vectorType& W, matrixType& U, matrixType& C, matrixType& invC){
	// returns eigen-values and eigen-vectors of the Matrix G
	// also C and its inverse;
	
	// calculating eigen-values and eig-vectors:
	//cout << "G:     " << G << endl;
	SelfAdjointEigenSolver<matrixType>	eigensolver(G);

	if (eigensolver.info() != Success) abort();

	// Rotate:  W has eigenvalues, U eigenvectors. 
	W = eigensolver.eigenvalues();
	U = eigensolver.eigenvectors();
	//U.normalize();

	//cout << "U0: " << U << endl;
	//matrixType U2 = U;

	// Create the matrix e^G, by exponentiating the diagonals of W:
	vectorType expW(W.size());
	vectorType expNW(W.size());
	for (int i = 0; i < W.size(); i++){
		expW(i) = exp(W(i));
		expNW(i) = exp(-W(i));
	}

	// creates a diagonal matrix with exp(W)
	matrixType A(W.size(), W.size());
	matrixType A2(W.size(), W.size());
	for (int i = 0; i < W.size(); i++){
		for (int j = 0; j < W.size(); j++){
			if (i == j){
				A(i,j) 	= expW(i);
				A2(i,j) = expNW(i);
			} else {
				A(i,j) 	= 0.0;
				A2(i,j) = 0.0;
			}
		}
	}

	matrixType AU(A.rows(), U.cols()); 
	AU = A*U.transpose().eval();
	matrixType expG(U.rows(), AU.cols());
	C = U*AU;
	//cout << "C0 inside the function is: \n" << C << endl;
	//C = expG;

	matrixType AU2(A2.rows(), U.cols());
	AU2 = A2*U.transpose().eval();
	matrixType expNG(U.rows(), AU2.cols());
	invC = U*AU2;
	//invC = expNG;

	//cout << "exp(W) is: " << expW << endl;
	
}

matrixType sigma_from_a(matrixType a, int L, int nbins){

	//Compute the sigma matrix from the a vector (concatenated individual harmonic coefficients):
	
	matrixType sigma = MatrixXd::Zero(nbins, nbins);

	for (int i = 0; i < nbins; i++){
		for (int j = 0; j < nbins; j++){
			for (int m = 0; m < L; m++){
				//cout << "AAA " << a(k*nbins + i) << endl;
				//cout << "BBB " << a(k*nbins + j) << endl;
				sigma(i,j) += (a(m*nbins + i)*a(m*nbins + j));
			}
		}
	}

	return sigma/(2*L+1);
}

matrixType tilde(matrixType sigma, matrixType U){

	matrixType sigU(sigma.rows(), U.cols());
	sigU = sigma*U;

	matrixType sigmaTilde(U.transpose().rows(), sigU.cols());
	sigmaTilde = (U.transpose().eval())*sigU;

	return sigmaTilde;
}


void Ortiz_matrices(matrixType& phi, NArrayType &psi, matrixType W, int m){
	// Construct Ortiz et al's phi (2D) and psi (3D) objects

	for (int i = 0; i < m; i++){
		for (int j = 0; j < m; j++){
			if (i==j){
				phi(i,j) = exp(W(i));
			} else {
				phi(i,j) = (exp(W(i)) - exp(W(j)))/(W(i) - W(j));
			}
		}
	}

	// Create 3-index matrix F(,,) of Ortiz et al. eq. (25):

	for (int a = 0; a < m; a++){
		double Ea = exp(W(a));
		for (int b = 0; b < m; b++){
			double Eb = exp(W(b));
			for (int c = 0; c < m; c++){
				double Ec = exp(W(c));
				if (a==b && a==c){
					psi[a][b][c] = Ea/2.;
				} else {
					if (a==b){
						psi[a][b][c] = ((-1 + W(a) - W(c))*Ea + Ec)/pow((W(a) - W(c)), 2.);
					} else if (a==c) {
						psi[a][b][c] = ((-1 + W(a) - W(b))*Ea + Eb)/pow((W(a) - W(b)), 2.);
					} else if (b==c) {
						psi[a][b][c] = ((-1 + W(b) - W(a))*Eb + Ea)/pow((W(a) - W(b)), 2.);
					} else {
						psi[a][b][c] = ( (W(b) - W(c))*Ea + (W(c) -W(a))*Eb + (W(a) - W(b))*Ec )/( (W(a) - W(b)) * (W(a) - W(c)) * (W(b) - W(c)) );
					}
				}
			}
		}
	}

}

double grad_G(matrixType sigmaTilde, matrixType U, matrixType phi, int ii, int jj, int q, int L, int nbins){

	double GradG;

	GradG = -0.5 * (2*L+1) * OrtizFirstDerivatives(sigmaTilde, U, phi, ii, jj, nbins);

	if(ii==jj){
		GradG += 0.5*(1 - 2*q)*(2*L + 1);
	}

	return GradG;
}

double OrtizFirstDerivatives(matrixType sigmaTilde, matrixType U, matrixType phi, int ii, int jj, int nbins){
	// Compute first derivative using Ortiz relations

	double OritzDeriv = 0.0;

	for (int eta = 0; eta < nbins; eta++){
		for (int nu = 0; nu < nbins; nu++){
			OritzDeriv += 2.0 * sigmaTilde(eta, nu) * ( U(ii, nu)*U(jj, eta) ) * phi(nu, eta);
		}
	}

	if(ii==jj){
		OritzDeriv *= 0.5;
	}

	return OritzDeriv;
}