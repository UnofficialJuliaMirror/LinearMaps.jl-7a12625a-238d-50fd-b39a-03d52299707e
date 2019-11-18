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

# map types that have an allocation-free 5-arg mul! implementation
const FreeMap = Union{MatrixMap,UniformScalingMap}

Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, A::LinearCombination{T,As}, x::AbstractVector,
                    α::Number=true, β::Number=false) where {T, As<:Tuple{Vararg{FreeMap}}}
    @boundscheck check_dim_mul(y, A, x)
    @inbounds begin
        mul!(y, first(A.maps), x, α, β)
        for Ai in Base.tail(A.maps)
            mul!(y, Ai, x, α, true)
        end
    end
    return y
end
Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, A::LinearCombination, x::AbstractVector,
                    α::Number=true, β::Number=false)
    @boundscheck check_dim_mul(y, A, x)
    @inbounds begin
        mul!(y, first(A.maps), x, α, β)
        l = length(A.maps)
        if l>1
            z = similar(y)
            for An in Base.tail(A.maps)
                if An isa FreeMap
                    mul!(y, An, x, α, true)
                else
                    mul!(z, An, x, α, false)
                    y .+= z
                end
            end
        end
    end
    return y
end

# function LinearAlgebra.mul!(y::AbstractVector, A::LinearCombination{T,As}, x::AbstractVector, α::Number=true, β::Number=false) where {T, As<:Tuple{Vararg{FreeMap}}}
#     length(y) == size(A, 1) || throw(DimensionMismatch("mul!"))
#     if isone(α)
#         iszero(β) && (A_mul_B!(y, A, x); return y)
#         !isone(β) && rmul!(y, β)
#     elseif iszero(α)
#         iszero(β) && (fill!(y, zero(eltype(y))); return y)
#         isone(β) && return y
#         # β != 0, 1
#         rmul!(y, β)
#         return y
#     else # α != 0, 1
#         if iszero(β)
#             A_mul_B!(y, A, x)
#             rmul!(y, α)
#             return y
#         elseif !isone(β)
#             rmul!(y, β)
#         end # β-cases
#     end # α-cases
#
#     for An in A.maps
#         mul!(y, An, x, α, true)
#     end
#     return y
# end

end # VERSION

Base.@propagate_inbounds LinearAlgebra.mul!(y::AbstractVector, A::TransposeMap{<:Any,<:LinearCombination}, x::AbstractVector,
                α::Number=true, β::Number=false) = mul!(y, transpose(A), x, α, β)

Base.@propagate_inbounds LinearAlgebra.mul!(y::AbstractVector, A::AdjointMap{<:Any,<:LinearCombination}, x::AbstractVector,
                α::Number=true, β::Number=false) = mul!(y, adjoint(A), x, α, β)
