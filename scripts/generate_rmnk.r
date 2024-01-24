# From: https://sourceforge.net/p/mocobench/code/HEAD/tree/trunk/rmnk/generator/rmnkGenerator.R
#!/usr/bin/env Rscript

#     This library is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; version 3 
#     of the License.
#
#     This library is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with this library; if not, write to the Free Software
#     Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Contact: http://mocobench.sourceforge.net
#
# Authors:
#     Arnaud Liefooghe <arnaud.liefooghe@lifl.fr>
#     Sebastien Verel  <sebastien.verel@inria.fr>
#
# 1.1 2014-03-21  Manuel López-Ibáñez <manuel.lopez-ibanez@ulb.ac.be>
#
#   * Bug fix in the generation of links.
#
# 1.2 2014-07-10  Arnaud Liefooghe <arnaud.liefooghe@lifl.fr>
#
#   * Bug fix in the generation of links when K=0.
#

# Syntax of the command line:
# ./rmnkGenerator.R rho M N K seed instance.dat
# Example:
# ./rmnkGenerator.R 0.8 2 18 2 0 instance.dat
#
# Or for some systems:
# R --slave --no-restore --file=rmnkGenerator.R --args 0.8 2 18 2 0 instance.dat
library(matrixcalc)

args <- commandArgs(TRUE)

# parameter: correlation between table contributions
rho <- as.numeric(args[1])

# parameter: number of objectve functions
M <- as.numeric(args[2])

# parameter: size of the bit string
N <- as.numeric(args[3])

# parameter: number of epistatic interactions
K <- as.numeric(args[4])

# parameter: seed number used by the generator
s <- as.numeric(args[5])

# parameter: output file name 
fileName <- args[6]

library(MASS)

## seed the random generator number
set.seed(s)

## generation of links

links <- matrix(0, K, N)
v <- seq(0, N - 1)
for (i in seq(1, N)) links[, i] <- sample(v[-i], K)

## generation of the tables of contributions

# generation of matrix correlation
R <- matrix(rep(rho, M), M, M)
diag(R) <- rep(1, M)

# Added for MOVE:
# for large number of objectives with high negative correlation, the matrix can be non positive definite
# Ensure R is positive definite
# Multiply R by its transpose
RR <- R %*% t(R)

# Check if RR is positive definite
if (!is.positive.definite(RR)) {
    # Use Cholesky Decomposition to make it positive definite
    RR <- chol(RR)
    RR <- RR %*% t(RR)
}

# pdf of multivariate normal law
data <- pnorm(mvrnorm(n = N * 2^(K + 1), mu = rep(0, M), Sigma = RR))

## print the function

f <- file(fileName, open = "w")

# heading of the file

cat("c file generated by rmnkGenerator.R with seed ", s, "the", format(Sys.time(), "%m/%d/%Y %H:%M:%S"), "\n", file = f)
cat("c version 1.1\n", file = f)
cat("c the links are random and identical for every objective functions\n", file = f)
cat("p rMNK", rho, M, N, K, "\n", file = f)

# links

cat("p links\n", file = f)

for (j in seq(1, N)) {
	for (m in seq(1, M)) cat(j - 1, " ", file = f)
	cat("\n", file = f)

	if (K > 0) {
		for (i in seq(1, K)) {
			for (m in seq(1, M)) cat(links[i, j], " ", file = f)
			cat("\n", file = f)
		}
	}
}

# tables of contributions

cat("p tables\n", file = f)

for (i in seq(1, N * 2^(K + 1))) {
	for (m in seq(1, M)) cat(data[i, m], " ", file = f)
	cat("\n", file = f)
}

close(f)
