using Interpolations
using ExtractMacro

mutable struct Interp
    itp
    r
    f
    xmin
    dx
    xmax

    function Interp(f, r::Range)
        xmin, dx, xmax = r[1], r[2] - r[1], r[end]
        A = [f(x) for x in r]
        itp = interpolate(A, BSpline(Cubic(Line())), OnGrid())
        new(itp, r, f, [xmin], [dx], [xmax])
    end


    function Interp(f, r1::Range, r2::Range)
        xmin = [r1[1], r2[1]] 
        dx   = [r1[2] - r1[1], r2[2] - r2[1]]
        xmax   = [r1[end], r2[end]]
        A = [f(x1,x2) for x1 in r1, x2 in r2]
        itp = interpolate(A, BSpline(Cubic(Line())), OnGrid())
        new(itp, [r1,r2], f, xmin, dx, xmax)
    end

    function Interp(f, r1::Range, r2::Range, r3::Range)
        xmin = [r1[1], r2[1], r3[1]] 
        dx   = [r1[2] - r1[1], r2[2] - r2[1], r3[2] - r3[1]]
        xmax   = [r1[end], r2[end], r3[end]]
        A = [f(x1,x2,x3) for x1 in r1, x2 in r2, x3 in r3]
        itp = interpolate(A, BSpline(Cubic(Line())), OnGrid())
        new(itp, [r1,r2,r3], f, xmin, dx, xmax)
    end
end

function (g::Interp)(x...)
    @extract g: xmin dx itp
    i = 1 .+ (x .- xmin) ./ dx
    itp[i...]
end
