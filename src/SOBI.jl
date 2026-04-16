"""
takes a collection of signals and calculates independent sources based on
second-order blind identification (SOBI). Algorithm taken from EEGLab but uses
`Diagonalizations.jl` to perform the approximate joint diagonalization.

# Arguments
- X is an m × N matrix (N is number of samples, m is number of sensors)

- A is the mixing matrix, n × m, that maps the matrix of independent sources to X along with
additive noise

- (S ∈ R^n × N) is the matrix of independent sources
"""
module SOBI

using Diagonalizations
using StatsBase
using LinearAlgebra
export sobi

function sobi(X :: Matrix{Float64}) :: Tuple{Matrix{Float64}, Matrix{Float64}}
  m,N = size(X)
  n = m
  defaultLags = 100
  #standardize and whiten
  @info "Standardizing data matrix..."
  X = standardize(X)
  @info "Whitening transformation calculated..."
  W = pcaWhitenTransformation(X)
  X = W*X
  
  #estimate delayed time cov matrices
  @info "Estimating lagged covariance matrices..."
  M = estTimeDelayedCov(X, defaultLags)
  
  #conduct approx joint diagonalization
  @info "Approximate joint diagonalization starting..."
  MVec = [M[:,i*m+1:i*m+m] for i in 0:n-1] #make vector of matrices
  U = ajd(MVec).F

  #estimate mixing matrix A
  @info "Estimating mixing matrix..."
  A = pinv(W)*U[1:n,1:n]
  
  #estimate source activities
  @info "Estimating source activities..."
  S = A*X
  return A,S
end #function

"detrends a signal by subtracting the mean"
function standardize(X :: Matrix{Float64}) :: Matrix{Float64}
  return X .- mean(X, dims=2)
end

function zcaWhitenTransformation(X :: Matrix{Float64})
  #Whitening transform W = inverse sqrt of cov matrix
  R0 = cov(X, dims=2)
  return R0^(-1/2)
end

function pcaWhitenTransformation(X :: Matrix{Float64})
    #whitening W = Λ^-1/2 U' from eigendecomposition
    covm = cov(X, dims=2)
    vals, vecs = eigen(covm)
    return Diagonal(vals)^(-1/2) * vecs'
end

function estTimeDelayedCov(X :: Matrix{Float64}, lags=100)
  m,N = size(X)
  n = m
  k = 1
  p = Int(min(lags, ceil(N/3)))
  pn = p*n
  M = zeros(Float64, m,pn)
  for u in 1:m:pn
    k += 1
    Rxp = X[:,k:N]*X[:,1:N-k+1]'/(N-k+1) #m x m matrix
    M[:,u:u+m-1] = norm(Rxp)*Rxp
    
  end
  return M
end

end #module
