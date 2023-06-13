using Optimisers
using Zygote

x = randn(10)

opt = Optimisers.setup(Descent(0.1), x)

function loss(x)
    return sum(x.^2)
end
for i in 1:100
    l, g = Zygote.withgradient(loss, x)
    Optimisers.update!(opt, x, g[1])
    println(l)
end