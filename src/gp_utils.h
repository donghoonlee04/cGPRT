#ifndef GP_UTILS_H
#define GP_UTILS_H

#include <armadillo>

void gp_marginal_likelihood(
	const arma::fmat & f,
	const arma::fmat & Q, 
	const float& sn,
	float & f0,
	float & dsn,
	float & sk
	) {

	int n = Q.n_rows;
	int q = Q.n_cols;
	int m = f.n_cols;
		
	float sn2 = std::exp(sn*2);		

	arma::fmat qe(q, q, arma::fill::eye);
	arma::fmat Qt = trans(Q);
	arma::fmat L = arma::chol(Qt*Q / sn2 + qe);
	arma::fmat QtQe = Qt*(Q / sn2);
	for (int i=0; i<q; ++i)
		QtQe(i, i) += 1;
	arma::fmat alpha = -(1.0f / sn2 / sn2) * (Q * arma::solve(QtQe, (Qt*f))) + (1.0f / sn2) * f;
	float tryalpha_m = (1.0f / m) * arma::trace(arma::trans(f) * alpha);
	float sk2 = (1.0f / n) * tryalpha_m;

	f0 = (1.0f/ 2.0f / sk2) * tryalpha_m + arma::sum(arma::log(L.diag())) + (n / 2.0f) *std::log(2*3.141592*sn2*sk2);		
	dsn = -(1.0f / m / sk2) * arma::trace(arma::trans(alpha) * alpha) + (1.0f/sn2 * n - (1.0f/sn2/sn2) * arma::trace(arma::solve(QtQe, Qt*Q)));
	dsn = dsn * sn2;
	sk = (1.0f / 2.0f) * std::log(sk2);
}

void gp_optimize (
	const arma::fmat & f,
	const arma::fmat & Q,
	const int sn,
	const int length,
	float& sk_opt,
	float& sn_opt		
	) {

	int i = 0;
	bool is_failed = 0;

	float X, Z, s, X0, dF0, df3, df0, ff0, ff3, FF0, FF;
	float INT, EXT, MAX, RATIO, SIG, RHO, red, f0, fX, d0, x3, F0, x2, f2, d2, f3, d3, x1, f1, d1, A, B, x4, f4, d4, ff2, ff1, ff4;

	INT = 0.1; EXT = 3.0; MAX = 8; RATIO = 10; SIG = 0.1; RHO = SIG / 2; red = 1;

	X = sn;

	gp_marginal_likelihood(f, Q, X, f0, df0, ff0);

	Z = X;
		
	fX = f0;
	++i;
	s = -df0;
	d0 = -s*s;
	x3 = red/(1-d0);

	while(i<std::abs(length))
	{
		i = i + (length > 0);
		X0 = X; F0 = f0; dF0 = df0; FF0 = ff0;
		float M = std::min(MAX, (float)(-length-i));

		while (1)
		{
			try
			{
				x2 = 0; f2 = f0; d2 = d0; f3 = f0; df3 = df0; ff2 = ff0;
				if (M>0)
				{
					M = M - 1; i = i + (length<0);                        
					Z = X + x3*s;
					gp_marginal_likelihood(f, Q, Z, f3, df3, ff3);
				}
				if (f3 < F0)
				{
					X0 = X+x3*s;
					F0 = f3; 
					dF0 = df3;
					FF0 = ff3;
				}
				d3 = df3*s;
				if (d3 > SIG*d0 || f3 > f0+x3*RHO*d0 || M == 0)					
					break;

				x1 = x2; f1 = f2; d1 = d2; ff1 = ff2; 
				x2 = x3; f2 = f3; d2 = d3; ff2 = ff3;
				A = 6*(f1-f2)+3*(d2+d1)*(x2-x1);
				B = 3*(f2-f1)-(2*d1+d2)*(x2-x1);
				x3 = x1-d1*(x2-x1)*(x2-x1)/(B+sqrt(B*B-A*d1*(x2-x1)));

				if (x3 < 0 || _isnan(x3))
					x3 = x2*EXT;
				else
				{
					
					if (x3 > x2*EXT )
						x3 = x2*EXT;   
					else if (x3 < x2+INT*(x2-x1))
						x3 = x2+INT*(x2-x1);
				}
			}
			catch (std::exception& e)
			{
				x3 = x2*EXT;
			}
		}

		while ((abs(d3) > -SIG*d0 || f3 > f0+x3*RHO*d0) && M > 0 )
		{
			try
			{
				
				if (d3 > 0 || f3 > f0+x3*RHO*d0) { x4 = x3; f4 = f3; d4 = d3; ff4 = ff3; }
				else { x2 = x3; f2 = f3; d2 = d3; ff2 = ff3; }
				if (f4 > f0) { x3 = x2-(0.5*d2*(x4-x2)*(x4-x2))/(f4-f2-d2*(x4-x2));}
				else 
				{
					A = 6*(f2-f4)/(x4-x2)+3*(d4+d2); B = 3*(f4-f2)-(2*d2+d4)*(x4-x2); 		
					x3 = x2+(std::sqrt((float)(B*B-A*d2*(x4-x2)*(x4-x2)-B)))/A; 
					if (_isnan(x3))
						x3 = (x2+x4)/2;
				}
				
				x3 = std::max(std::min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2));
				

				Z = X + x3*s;
				gp_marginal_likelihood(f, Q, Z, f3, df3, ff3);
				if (f3 < F0)
				{
					X0 = X+x3*s; 
					F0 = f3; 
					dF0 = df3;
					FF0 = ff3;
				}
				M = M - 1; i = i + (length<0);
				d3 = df3*s;
			}
			catch (std::exception& e)
			{
				x3 = (x2+x4)/2;
			}
		}

		if (std::abs(d3) < -SIG*d0 && f3 < f0+x3*RHO*d0 ) 
		{ 
			X = X+x3*s; f0 = f3; ff0 = ff3; FF = ff0;
			s = (df3*df3-df0*df3)/(df0*df0)*s - df3;
			df0 = df3;
			d3 = d0; d0 = df0*s;
			if (d0 > 0)
				s = -df0; d0 = -s*s;
			x3 = x3 * std::min(RATIO, d3/(d0-std::numeric_limits<float>::min()));
			is_failed = 0;
		}
		else
		{
			X = X0; f0 = F0; df0 = dF0; FF = FF0;
			if (is_failed || i > std::abs(length))
				break;                        
			s = -df0; d0 = -s*s;
			x3 = 1/(1-d0);
			is_failed = 1;
		}
	}
	sn_opt = X;
	sk_opt = FF;
}
#endif
