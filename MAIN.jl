include("src/Fabrics.jl")
using Makie

function main()
    # Get Eulerian mesh
    M = EulerianMesh()
    # Get Lagrangian particles
    P = LagrangianParticles(M.Δx,M.Δy)
    # Save initial particles
    P0 = deepcopy(P)
    #Eulerian velocity field
    U = EulerianVelocity(M)
    ∇U = VelocityGradient(M)
    # Allocate gradiend of deformation
    F = [Matrix(1.0I,3,3) for i in 1:P.n]

    # Prepare Loop
    t = 0.0
    sec2year = 365.25*86400
    timemax = 1e6*sec2year
    Δt = min(M.Δx/2/maximum(abs.(U.x)), M.Δy/2/maximum(abs.(U.y)))
    cyclemax = div(timemax, Δt);

    # Start loop
    l = fill(0.0,3,3) 
    RK4 = RK(fill(0.0,3,3), fill(0.0,3,3), fill(0.0,3,3), fill(0.0,3,3))
    for cycle in 1:cyclemax
        for m in 1:P.n
            interp∇U!(l,m,∇U,P,M)
            F[m] = updateFSE(F[m],l,Δt,RK4)
            advectparticle!(P,M,U,Δt,m)
        end
        t += Δt
    end

    return U,∇U,F,P,P0
end

U,∇U,F3D,P,P0 = main()
scatter(P.x/1e3, P.y/1e3, color=P0.y, markersize=2)
