module BlockMatrices

import Base:+,*,-,\,^,getindex,setindex!,show,print

# ****************************************************************
# Square Block Diagonal Matrix
# ****************************************************************

export Block, size, block_idx, broadcastf, full,
    copy, getindex, setindex!, +, -, *, \, inv, square, cache;

type Block <: AbstractMatrix{Real}

  Blocks::Array{Any}
  
  # This class is inefficient with too many blocks. If your block 
  # diagonal matrix has many small blocks, it may be wise to 
  # cache the blocks as a sparse matrix. this makes multiplication
  # an order of magnitude faster. 
  mcache::SparseMatrixCSC

  Block(size::Int) = new(Array(Any,size), spzeros(0,0))
  Block(Blk::Array{Any}) = new(Blk, spzeros(0,0))
  Block(Blk::Array) = new(convert(Array{Any}, Blk), spzeros(0,0))

end

function Base.size(A::Block)
  n = sum([size(B,1) for B in A.Blocks])
  return (n,n)
end

Base.size(A::Block, i::Integer)    = (i == 1 || i == 2) ? size(A)[1] : 1
getindex(A::Block, i::Integer)     = A.Blocks[i]
setindex!(A::Block, B, i::Integer) = begin; A.Blocks[i] = B; mcache = spzeros(0,0); end

function block_idx(A::Block)

  #block_sizes = [size(B,1) for B in A.Blocks]

  k = size(A.Blocks,1)
  IColl = Array(UnitRange, k)

  cum_count = 1
  for i = 1:k
    blk_size = size(A.Blocks[i],1);
    IColl[i] = cum_count:(cum_count + blk_size - 1)
    cum_count += blk_size
  end

  return IColl

end


function blockIter(A::Block)

  k = size(A.Blocks,1)
  cum_count = 1
  for i = 1:k
    blk_size = size(A.Blocks[i],1)
    produce(cum_count:(cum_count + blk_size - 1))
    cum_count += blk_size
  end

end


function broadcastf(op::Function, A::Block)

  B = copy(A)
  for i = 1:length(A.Blocks)
    B[i] = op(A[i])
  end
  return B

end

function broadcastf(op::Function, A::Block, B::Block)

  C = copy(A)
  for i = 1:length(A.Blocks)
    C[i] = op(A[i], B[i])
  end
  return C

end

function broadcastf(op::Function, A::Block, x::Vector)

  y = similar(x)
  i = 1
  for I = Task( () -> blockIter(A) )
    xI = view(x,I);
    y[I] = op(A.Blocks[i])
    i += 1;
  end
  return y;

end

function broadcastf(op::Function, A::Block, X::Matrix)

  Y = similar(X)
  i = 1
  for I = Task( () -> blockIter(A) )
    XI = sub(X,I,:);
    Y[I,:] = op(A.Blocks[i],XI)
    i += 1;
  end
  return Y

end

function reduce(op::Function, A::Block)

  α = 0;
  for I = Task( () -> blockIter(A) )
    α = α + op(A.Blocks[i])
  end
  return α;

end

function Base.sparse(A::Block)

  I₊, J₊, V₊ = Int[], Int[], Float64[]
  for (I,Blk) = zip(block_idx(A), A.Blocks)
    Aᵢ = sparse(Blk)
    # for i = 1:size(Aᵢ,1), j = 1:size(Aᵢ,2)
    #   push!(I₊, I[i])
    #   push!(J₊, I[j])
    #   push!(V₊, Aᵢ[i,j])
    # end
    rows = rowvals(Aᵢ)
    vals = nonzeros(Aᵢ)
    m, n = size(Aᵢ)
    for i = 1:n
       for j in nzrange(Aᵢ, i)
          row = rows[j]; val = vals[j]
          push!(I₊, i); push!(J₊, row); push!(V₊, val)
       end
    end
  end
  return sparse(I₊,J₊,V₊);

end

function cache(A::Block)
  A.mcache = sparse(A)
end


function Base.full(A::Block)

  O = zeros(size(A))
  for (I,Blk) = zip(block_idx(A), A.Blocks)
    O[I,I] = full(Blk)
  end
  return O;

end


function *(A::Block, X::Array{Float64,2})

  if length(A.mcache) != 0; return A.mcache*X; end
  Y = similar(X)
  i = 1
  for I = Task( () -> blockIter(A) )
    XI = sub(X,I,:)
    #mult!(A.Blocks[i],XI,sub(Y,I,:))
    Y[I,:] = *(A.Blocks[i],XI)
    i += 1;
  end
  return Y

end

Base.copy(A::Block)        = Block(copy(A.Blocks))
Base.deepcopy(A::Block)    = Block(deepcopy(A.Blocks))
+(A::Block, B::Block)      = Block(A.Blocks + B.Blocks)
-(A::Block, B::Block)      = A + (-B)
Base.inv(A::Block)         = broadcastf(inv, A)
-(A::Block)                = broadcastf(-, A)
*(A::Block, B::Block)      = broadcastf(*, A, B)
Base.norm(A::Block)        = reduce(norm, A)

ViewTypes = Union{SubArray}
VectorTypes = Union{Matrix, Vector, ViewTypes}

# Extra functions for dealing with views and stuff

function +(A::Diagonal, B::Block)
  i = 1
  B0 = Block(length(B.Blocks))
  for I = block_idx(B)
    dI = A.diag[I];
    B0[i] = B[i] + Diagonal(dI);
    i = i + 1
  end
  return B0
end

function mult!(A::Diagonal, x::VectorTypes, s::VectorTypes)
  for i = 1:length(x)
    s[i] = x[i]*A.diag[i];
  end
end

function mult!(A::UniformScaling, x::VectorTypes, s::VectorTypes)
  for i = 1:length(x)
    s[i] = x[i]*A.λ;
  end
end

^(A::Block,n::Integer) = broadcastf(x -> ^(x,n), A);

function show(io::IO, M::Block)
    print(io, "BLock")
end

end