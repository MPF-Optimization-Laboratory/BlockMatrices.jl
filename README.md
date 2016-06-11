# Block.jl

Repersentation of a Block Matrix in Julia

### Construction

```julia
using BlockMatrices

#     ┌               ┐
#     │ 1  1          │
#     │ 1  1          │
# A = │       1  1  1 │
#     │       1  1  1 │
#     │       1  1  1 │
#     └               ┘

A = Block(2)
A[1] = ones(2,2)
A[2] = ones(3,3)

```

### Usage

```
julia> size(A)           
(5,5)                    

julia> A*ones(5,1)       
5x1 Array{Float64,2}:    
 2.0                     
 2.0                     
 3.0                     
 3.0                     
 3.0                     
```

etc.
