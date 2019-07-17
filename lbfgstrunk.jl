export lbfgsTrunk

include("lbfgs.jl")

"""The L-BFGS method to solve the symmetric linear system Ax=b.
The method does _not_ abort if A is not definite.
A preconditioner M may be provided in the form of a linear operator and is
assumed to be symmetric and positive definite.
"""
function lbfgsTrunk(A :: AbstractLinearOperator, b :: AbstractVector{T};
            M :: AbstractLinearOperator=opEye(), atol :: T=√eps(T),
            rtol :: T=√eps(T), itmax :: Int=0, radius :: T=zero(T),
            verbose :: Bool=false, m :: Int=1,
            scale :: Bool=false) where T <: AbstractFloat

  n = size(b, 1);
  (size(A, 1) == n & size(A, 2) == n) || error("Inconsistent problem size");
  verbose && @printf("LBFGS: system of %d equations in %d variables\n", n, n);

  # Initial state.
  p = zeros(T, n);
  y = zeros(T, n);
  s = zeros(T, n);
  H = InverseLBFGSOperator(n, m, scaling = scale)
  g = -copy(b)
  d = copy(b)
  γ = @kdot(n, g, g)
  γ == 0 && return p, SimpleStats(true, false, [zero(T)], T[], "p = 0 is a zero-residual solution")

  iter = 0;
  itmax == 0 && (itmax = 2 * n);

  gNorm = sqrt(γ);
  gNorms = [gNorm;];
  ε = atol + rtol * gNorm;
  verbose && @printf("%5d  %8.1e  ", iter, gNorm);

  solved = gNorm <= ε;
  tired = iter >= itmax;
  on_boundary = false;
  status = "unknown";

  while ! (solved || tired)
    Ad = A * d;
    dAd = @kdot(n, d, Ad)

    α = γ / dAd;

    # Compute step size to boundary if applicable.
    σ = radius > 0 ? maximum(to_boundary(p, d, radius)) : α

    verbose && @printf("%8.1e  %7.1e  %7.1e\n", dAd, α, σ);

    # Move along p from x to the boundary if either
    # the next step leads outside the trust region or
    # we have nonpositive curvature.
    if (radius > 0) & ((dAd <= 0) | (α > σ))
      α = σ
      on_boundary = true
    end

    @kaxpy!(n, α,  d, p)
    @kaxpy!(n, α, Ad, g)
    y = α * Ad
    s = α * d

    rNorm = sqrt(@kdot(n, g, g));
    push!(rNorms, rNorm);

    solved = (rNorm <= ε) | on_boundary;
    if !solved
      push!(H, s, y)
      d = - H * g
      γ = -@kdot(n, g, d)
    end

    iter = iter + 1;
    tired = iter >= itmax;
    verbose && @printf("%5d  %8.1e  ", iter, rNorm);
  end
  verbose && @printf("\n");

  status = on_boundary ? "on trust-region boundary" : (tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol")
  stats = SimpleStats(solved, false, rNorms, T[], status);
  return (x, stats);
end
