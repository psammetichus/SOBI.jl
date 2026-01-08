using Test

@testset "SOBI" begin
    m = 5
    n = 2048
    a = randn(m,n)
    A,S = sobi(a)
    @test size(A) == (m,m)
    @test size(S) == (m,n)
end