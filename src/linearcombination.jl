struct LinearCombination{T, As<:Tuple{Vararg{LinearMap}}} <: LinearMap{T}
    maps::As
    function LinearCombination{T, As}(maps::As) where {T, As}
        N = length(maps)
        sz = size(maps[1])
        for n in 1:N
            size(maps[n]) == sz || throw(DimensionMismatch("LinearCombination"))
            promote_type(T, eltype(maps[n])) == T || throw(InexactError())
        end
        new{T, As}(maps)
    end
end

LinearCombination{T}(maps::As) where {T, As} = LinearCombination{T, As}(maps)

# basic methods
Base.size(A::LinearCombination) = size(A.maps[1])
LinearAlgebra.issymmetric(A::LinearCombination) = all(issymmetric, A.maps) # sufficient but not necessary
LinearAlgebra.ishermitian(A::LinearCombination) = all(ishermitian, A.maps) # sufficient but not necessary
LinearAlgebra.isposdef(A::LinearCombination) = all(isposdef, A.maps) # sufficient but not necessary

# adding linear maps
"""
    A::LinearMap + B::LinearMap

Construct a `LinearCombination <: LinearMap`, a (lazy) representation of the sum
of the two operators. Sums of `LinearMap`/`LinearCombination` objects and
`LinearMap`/`LinearCombination` objects are reduced to a single `LinearCombination`.
In sums of `LinearMap`s and `AbstractMatrix`/`UniformScaling` objects, the latter
get promoted to `LinearMap`s automatically.

# Examples
```jldoctest; setup=(using LinearAlgebra, LinearMaps)
julia> CS = LinearMap{Int}(cumsum, 3)::LinearMaps.FunctionMap;

julia> LinearMap(ones(Int, 3, 3)) + CS + I + rand(3, 3);
```
"""
function Base.:(+)(A₁::LinearMap, A₂::LinearMap)
    size(A₁) == size(A₂) || throw(DimensionMismatch("+"))
    T = promote_type(eltype(A₁), eltype(A₂))
    return LinearCombination{T}(tuple(A₁, A₂))
end
function Base.:(+)(A₁::LinearMap, A₂::LinearCombination)
    size(A₁) == size(A₂) || throw(DimensionMismatch("+"))
    T = promote_type(eltype(A₁), eltype(A₂))
    return LinearCombination{T}(tuple(A₁, A₂.maps...))
end
Base.:(+)(A₁::LinearCombination, A₂::LinearMap) = +(A₂, A₁)
function Base.:(+)(A₁::LinearCombination, A₂::LinearCombination)
    size(A₁) == size(A₂) || throw(DimensionMismatch("+"))
    T = promote_type(eltype(A₁), eltype(A₂))
    return LinearCombination{T}(tuple(A₁.maps..., A₂.maps...))
end
Base.:(-)(A₁::LinearMap, A₂::LinearMap) = +(A₁, -A₂)

# comparison of LinearCombination objects, sufficient but not necessary
Base.:(==)(A::LinearCombination, B::LinearCombination) = (eltype(A) == eltype(B) && A.maps == B.maps)

# special transposition behavior
LinearAlgebra.transpose(A::LinearCombination) = LinearCombination{eltype(A)}(map(transpose, A.maps))
LinearAlgebra.adjoint(A::LinearCombination)   = LinearCombination{eltype(A)}(map(adjoint, A.maps))

# multiplication with vectors

# map types that have an allocation-free 5-arg mul! implementation
const FreeMap = Union{MatrixMap,UniformScalingMap}

if VERSION < v"1.3.0-alpha.115"

Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, A::LinearCombination, x::AbstractVector,
                    α::Number=true, β::Number=false)
    @boundscheck check_dim_mul(y, A, x)
    mul!(y, A.maps[1], x, α, β)
    @inbounds begin
        l = length(A.maps)
        if l>1
            z = similar(y)
            for n in 2:l
                mul!(z, A.maps[n], x, α, false)
                y .+= z
            end
        end
    end
    return y
end

else # 5-arg mul! is available for matrices
    for Atype in (AbstractVector, AbstractMatrix)
        @eval Base.@propagate_inbounds function LinearAlgebra.mul!(y::$Atype, A::LinearCombination, x::$Atype,
                            α::Number=true, β::Number=false)
            @boundscheck check_dim_mul(y, A, x)
            return _lincombmul!(y, A, x, α, β)
        end
    end
end # VERSION

############
# multiplication helper functions
############

@inline function _lincombmul!(y, A::LinearCombination{<:Any,<:Tuple{Vararg{FreeMap}}}, x,
                                α::Number, β::Number)
    mul!(y, A.maps[1], x, α, β)
    @inbounds for n in 2:length(A.maps)
        mul!(y, A.maps[n], x, α, true)
    end
    return y
end

@inline function _lincombmul!(y, A::LinearCombination, x, α::Number, β::Number)
    mul!(y, first(A.maps), x, α, β)
    l = length(A.maps)
    if l>1
        z = similar(y)
        for n in 2:l
            @inbounds An = A.maps[n]
            if An isa FreeMap
                mul!(y, An, x, α, true)
            else
                mul!(z, An, x, α, false)
                y .+= z
            end
        end
    end
    return y
end
