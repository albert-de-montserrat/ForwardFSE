import LinearAlgebra: I

struct MESH
    Δx::Number
    Δy::Number
    x::Array{Float64,1}
    y::Array{Float64,1}
    X::Array{Float64,2}
    Y::Array{Float64,2}
    xsize::Int64
    ysize::Int64
end

#Eulerian Mesh
function EulerianMesh(xsize=Int(3e5), ysize=Int(2e5), xnum=61, ynum=41)
    Δx = xsize/(xnum-1)
    Δy = ysize/(ynum-1)
    x = 0:Δx:xsize
    y = 0:Δy:ysize
    X = x'.*ones(length(y))
    Y = y.*ones(1,length(x))
    return MESH(Δx,Δy,x,y,X,Y,Int(xsize),Int(ysize))
end

mutable struct Particles
    x::Array{Float64,1}
    y::Array{Float64,1}
    n::Int64 # number of particles
end

#Lagrangian Particles
function LagrangianParticles(Δx,Δy,mxcell=2, mycell=2, xnum=61, ynum=41)
    mΔx = Δx/mxcell
    mΔy = Δy/mycell
    mxnum = (xnum-1)*mxcell
    mynum = (ynum-1)*mycell
    marknum = mxnum*mynum
    mx = Vector{Float64}(undef,marknum)
    my = similar(mx)
    @inbounds for i=1:mxnum, j=1:mynum
        m = i + (j-1)*mxnum
        mx[m] = mΔx*(i-0.5)
        my[m] = mΔy*(j-0.5)
    end
    return Particles(mx,my,marknum)
end


struct Velocity
    x::Array{Float64,2}
    y::Array{Float64,2}
end

# Eulerian velocity field
function EulerianVelocity(M)
    Vscale = 1
    xnum, ynum = size(M.X,2), size(M.X,1)
    Vx = Array{Float64,2}(undef,ynum,xnum)
    Vy = similar(Vx)
    kx, ky = 1, 1 # wavenumber
    vx0 = 1e-9/Vscale
    vy0 = vx0*kx/ky*M.ysize/M.xsize # Incompressibility
    cte₁ = kx*π/M.xsize
    cte₂ = ky*π/M.ysize
    @inbounds for i=1:xnum, j=1:ynum
        Vx[j,i]=-vx0*sin(cte₁*M.x[i])*cos(cte₂*M.y[j])
        Vy[j,i]= vy0*cos(cte₁*M.x[i])*sin(cte₂*M.y[j])
    end
    return Velocity(Vx, Vy)
end

struct Gradient
    ∂x∂x::Array{Float64,2}
    ∂x∂y::Array{Float64,2}
    ∂y∂y::Array{Float64,2}
    ∂y∂x::Array{Float64,2}
end

# Velocity gradient
function VelocityGradient(M)
    xnum, ynum = size(M.X,2), size(M.X,1)
    dVxdx = Array{Float64,2}(undef,ynum,xnum)
    dVxdy, dVydx, dVydy = similar(dVxdx), similar(dVxdx), similar(dVxdx)
    Vscale = 1
    vx0 = 1e-9/Vscale
    kx, ky = 1, 1 # wavenumber
    vy0 = vx0*kx/ky*M.ysize/M.xsize # Incompressibility
    cte₁ = kx*π/M.xsize
    cte₂ = ky*π/M.ysize
    @inbounds for i=1:xnum, j=1:ynum
        dVxdx[j,i] =-vx0*cos(cte₁*M.x[i])*cte₁*cos(cte₂*M.y[j]);
        dVxdy[j,i] = vx0*sin(cte₁*M.x[i])*sin(cte₂*M.y[j])*cte₂;
        dVydx[j,i] =-vy0*sin(cte₁*M.x[i])*cte₁*sin(cte₂*M.y[j]);
        dVydy[j,i] = vy0*cos(cte₁*M.x[i])*cos(cte₂*M.y[j])*cte₂;
    end
    return Gradient(dVxdx,dVxdy,dVydy,dVydx)
end

function interp∇U!(l,m,∇U,P,M)
    xnum, ynum = size(M.X,2), size(M.X,1)
    i = floor(Int,P.x[m]/M.Δx)+1
    j = floor(Int,P.y[m]/M.Δy)+1

    if i<1
        i = 1
        Δxx = 0
    elseif i > xnum-1
        i = xnum-1
        Δxx = 1
    else
        Δxx = (P.x[m]-M.x[i])/M.Δx
    end
    if j<1
        j = 1
        Δyy = 0
    elseif j > ynum-1
        j = ynum-1
        Δyy = 1
    else
        Δyy = (P.y[m]-M.y[j])/M.Δy
    end
    
    w1 = (1-Δxx)*(1-Δyy)
    w2 = (1-Δxx)*Δyy
    w3 = Δxx*(1-Δyy)
    w4 = Δxx*Δyy

    @inbounds l[1,1] = ∇U.∂x∂x[j,i]*w1 + ∇U.∂x∂x[j+1,i]*w2 + ∇U.∂x∂x[j,i+1]*w3 + ∇U.∂x∂x[j+1,i+1]*w4
    @inbounds l[1,2] = ∇U.∂x∂y[j,i]*w1 + ∇U.∂x∂y[j+1,i]*w2 + ∇U.∂x∂y[j,i+1]*w3 + ∇U.∂x∂y[j+1,i+1]*w4
    @inbounds l[2,1] = ∇U.∂y∂y[j,i]*w1 + ∇U.∂y∂y[j+1,i]*w2 + ∇U.∂y∂y[j,i+1]*w3 + ∇U.∂y∂y[j+1,i+1]*w4
    @inbounds l[2,2] = ∇U.∂y∂x[j,i]*w1 + ∇U.∂y∂x[j+1,i]*w2 + ∇U.∂y∂x[j,i+1]*w3 + ∇U.∂y∂x[j+1,i+1]*w4

end

function interpVel(mxA,myA,M,U)
    xnum, ynum = size(M.X,2), size(M.X,1)
    
    i = floor(Int, mxA/M.Δx)+1
    j = floor(Int, myA/M.Δy)+1

    if i<1
        i = 1
        Δxx = 0
    elseif i > xnum-1
        i = xnum-1
        Δxx = 1
    else
        Δxx = (mxA-M.x[i])/M.Δx
    end
    if j<1
        j = 1
        Δyy = 0
    elseif j > ynum-1
        j = ynum-1
        Δyy = 1
    else
        Δyy = (myA-M.y[j])/M.Δy
    end
    
    w1 = (1-Δxx)*(1-Δyy)
    w2 = (1-Δxx)*Δyy
    w3 = Δxx*(1-Δyy)
    w4 = Δxx*Δyy

    @inbounds Ux = U.x[j,i]*w1 + U.x[j+1,i]*w2 + U.x[j,i+1]*w3 + U.x[j+1,i+1]*w4
    @inbounds Uz = U.y[j,i]*w1 + U.y[j+1,i]*w2 + U.y[j,i+1]*w3 + U.y[j+1,i+1]*w4
    return Ux, Uz
end

function advectparticle!(P,M,U,Δt,m)
    mxA = P.x[m]
    myA = P.y[m]
    # 4th order RK
    mVxA, mVyA = interpVel(mxA,myA,M,U)
    mxB = mxA + 0.5*mVxA*Δt; myB = myA + 0.5*mVyA*Δt
    mVxB, mVyB = interpVel(mxB,myB,M,U)
    mxC = mxA + 0.5*mVxB*Δt; myC = myA + 0.5*mVyB*Δt
    mVxC, mVyC = interpVel(mxC,myC,M,U)
    mxD = mxA + mVxC*Δt; myD = myA + mVyC*Δt
    mVxD, mVyD = interpVel(mxD,myD,M,U)
    P.x[m] += (mVxA + 2*(mVxB + mVxC) + mVxD)/6*Δt
    P.y[m] += (mVyA + 2*(mVyB + mVyC) + mVyD)/6*Δt
end

function updateFSE!(FSE0,l,Δt)
    FSEᵢ = deepcopy(FSE0)
    # 4th-order Runge-Kutta
    Δ1 = l*FSEᵢ*Δt
    FSEᵢ = FSE0 + 0.5*Δ1
    Δ2 = l*FSEᵢ*Δt
    FSEᵢ = FSE0 + 0.5*Δ2
    Δ3 = l*FSEᵢ*Δt
    FSEᵢ = FSE0 + Δ3
    Δ4 = l*FSEᵢ*Δt

    FSE0 += (Δ1 + 2*(Δ2 + Δ3) + Δ4)/6;
end
