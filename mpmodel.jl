# a multi-precision NLPModel class with automatic differentiation

using NLPModels

using FastClosures
using ForwardDiff
using Quadmath
using ReverseDiff

builtin_fps = [Float16, Float32, Float64, Float128]
costs = Dict{DataType,Int}(Float16 => 1, Float32 => 4, Float64 => 16, Float128 => 64)  # TODO: generalize to BigFloat, etc.

mutable struct MPCounters
  neval_obj    :: Dict{DataType,Int}  # Number of objective evaluations.
  neval_grad   :: Dict{DataType,Int}  # Number of objective gradient evaluations.
  neval_cons   :: Dict{DataType,Int}  # Number of constraint vector evaluations.
  neval_jcon   :: Dict{DataType,Int}  # Number of individual constraint evaluations.
  neval_jgrad  :: Dict{DataType,Int}  # Number of individual constraint gradient evaluations.
  neval_jac    :: Dict{DataType,Int}  # Number of constraint Jacobian evaluations.
  neval_jprod  :: Dict{DataType,Int}  # Number of Jacobian-vector products.
  neval_jtprod :: Dict{DataType,Int}  # Number of transposed Jacobian-vector products.
  neval_hess   :: Dict{DataType,Int}  # Number of Lagrangian/objective Hessian evaluations.
  neval_hprod  :: Dict{DataType,Int}  # Number of Lagrangian/objective Hessian-vector products.
  neval_jhprod :: Dict{DataType,Int}  # Number of individual constraint Hessian-vector products.
end

"""
    counters = MPCounters(precisions)

Initialize a `MPCounters` instance for precisions specified in `precisions`.
Examples:

    counters = MPCounters()  # use `builtin_fps` by default
    counters = MPCounters([Float32, Float64, Float128])
"""
function MPCounters(precisions::Vector{DataType})
  fields = (:n_obj, :n_grad, :n_cons, :n_jcon, :n_jgrad, :n_jac, :n_jprod,
            :n_jtprod, :n_hess, :n_hprod, :n_jhprod)
  for field in fields
    @eval begin
      $field = Dict{DataType,Int}()
      for p in $precisions
        $field[p] = 0
      end
    end
  end
  return MPCounters((eval(field) for field in fields)...)
end

# shortcut constructor for built-in floating-point types
MPCounters() = MPCounters(builtin_fps)

function NLPModels.reset!(mpcounters::MPCounters)
  fields = (:neval_obj, :neval_grad, :neval_cons, :neval_jcon, :neval_jgrad,
            :neval_jac, :neval_jprod, :neval_jtprod, :neval_hess, :neval_hprod,
            :neval_jhprod)
	precisions = keys(mpcounters.neval_obj)
  for field in fields
    for p in precisions
				getproperty(mpcounters, field)[p] = 0
    end
  end
  return mpcounters
end

"""
		neval(d::Dict{DataType,Int})

Return the number of evaluations in `d` normalized with respect to `Float16`.
Example:
"""
neval(d::Dict{DataType,Int}) = sum(d[typ] * costs[typ] for typ in keys(d))

"""
		neval(d::Dict{DataType,Int}, T::DataType)

Return the number of evaluations in `d` normalized with respect to `T`, where
`T` is a floating-point type appearing in the global `costs` dictionary.
"""
function neval(d::Dict{DataType,Int}, T::DataType)
	T in keys(costs) || error("don't know how to normalize with respect to $T")
	return neval(d) / costs[T]
end

mutable struct MPModel{Ff,Fg,FHv} <: AbstractNLPModel where {Ff <: Function, Fg <: Function, FHv <: Function}
  meta::NLPModelMeta
  counters::Counters

  precisions::Vector{DataType}
  mpcounters::MPCounters
  obj::Ff  # a pure Julia function representing the objective
  grad::Fg
	hessvec!::FHv
end

function NLPModels.reset!(nlp::MPModel)
  reset!(nlp.counters)
  reset!(nlp.mpcounters)
end

function MPModel(nvar::Int,
                 obj::F,
                 x0::Vector{<:AbstractFloat}=zeros(Float64, nvar),
                 precisions::Vector{DataType}=builtin_fps;
								 name::String="noname") where F <: Function
  meta = NLPModelMeta(nvar, x0=x0, name=name)
  counters = Counters()
  mpcounters = MPCounters(precisions)

	# build ∇f
	# TODO: use gradient! with f_tape
  ∇f = @closure x -> ReverseDiff.gradient(obj, x)
  Fg = typeof(∇f)

	# build v -> ∇²f(x)*v
	function hessvec!(x::AbstractVector{<:AbstractFloat},
										v::AbstractVector{<:AbstractFloat},
										Hv::AbstractVector{<:AbstractFloat})
		z = map(ForwardDiff.Dual, x, v)  # x + ε * v
		∇f_z = ∇f(z)                     # ∇f(x + ε * v) = ∇f(x) + ε * ∇²f(x)ᵀv
		Hv = ForwardDiff.extract_derivative!(Nothing, Hv, ∇f_z)  # ∇²f(x)ᵀv
		return Hv
	end
	FHv = typeof(hessvec!)

  return MPModel{F,Fg,FHv}(meta, counters, precisions, mpcounters, obj, ∇f, hessvec!)
end

function NLPModels.obj(model::MPModel, x::AbstractVector{<: AbstractFloat})
  T = eltype(x)
  T in model.precisions || error("not prepared to evaluate objective in $T")
  model.counters.neval_obj += 1
  model.mpcounters.neval_obj[T] += 1
  return model.obj(x)
end

function NLPModels.grad(model::MPModel, x::AbstractVector{<: AbstractFloat})
  T = eltype(x)
  T in model.precisions || error("not prepared to evaluate gradient in $T")
  model.counters.neval_grad += 1
  model.mpcounters.neval_grad[T] += 1
  return model.grad(x)
end

function NLPModels.hprod!(model::MPModel,
												  x::AbstractVector{T},
													v::AbstractVector{T},
													Hv::AbstractVector{T}; kwargs...) where T <: AbstractFloat
  T in model.precisions || error("not prepared to evaluate Hessian-vector product in $T")
	model.counters.neval_hprod += 1
	model.mpcounters.neval_hprod[T] += 1
	return model.hessvec!(x, v, Hv)
end

function NLPModels.grad!(model::MPModel,
												  x::AbstractVector{<: AbstractFloat},
													∇f::AbstractVector{<: AbstractFloat})
	T = eltype(x)
	T in model.precisions || error("not prepared to evaluate gradient in $T")
	model.counters.neval_grad += 1
	model.mpcounters.neval_grad[T] += 1
	return model.grad!(x, ∇f)
end
