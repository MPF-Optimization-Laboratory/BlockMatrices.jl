using SpecialMatrices

k = 4000
B = Block(k)
for i = 1:k
  B[i] = randn(2,2)
end
#BF = sparse(full(B))

x = randn(k*2,1)

B*x;
Profile.clear()
@time B*x;

#println("Sparse")
#@time BF*x;
