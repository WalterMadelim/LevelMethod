"""
Global Optimization of 2SSP, using Lagrangian cuts
"""
module Go
import ..Settings, Gurobi, JuMP

function gen_lag_cuts(grg, aug, lcg, N, stopit)
    proceed = Go.cg_and_load_frac_check(grg, aug, lcg)
    while proceed
        has_one_cut = false
        for (s, θ_che)=enumerate(grg.Θ)
            aug_s = aug[s]
            hasvio = Go.load_pi_and_Cn_in_lag_cut(grg.Xl, θ_che, aug_s, lcg[s])
            hasvio ? (has_one_cut = true) : continue
            aCd = aug_s.Cd
            Obn, aCd[N+1] = aCd[N+1], -1.
            Gurobi.GRBaddconstr(grg.o,N+1,aug_s.Ci,aCd,Cchar('<'),-Obn,C_NULL)
            proceed = Go.cg_and_load_frac_check(grg, aug, lcg)
            proceed || return
        end
        has_one_cut || return
        stopit.value && return
    end
end

"""
Given the current CTPLN master,
generate a pool of bottom Integer solutions,
record them via self-column-generation,
propagate them via local-column-generation.
then generate a (restricted) frac check point as a convex combination.
"""
function cg_and_load_frac_check(grg, aug, lcg)
    (; Θ, o, Xl, Xn, Bv, Cd, Ci) = grg
    S, N, si, genv, frv = length(Θ), length(Xl), 0, Gurobi.GRBgetenv(o), Cd; l=N+1
    config_Bin_noCG_solve_grg(grg, Bv, frv, S, N) # 🟦 master small-scaled MIP
    lb = Settings.getmodeldblattr(grg, "ObjBound")
    while si < Settings.getmodelintattr(grg, "SolCount")
        Gurobi.GRBsetintparam(genv, "SolutionNumber", si)
        si+=1; Gurobi.GRBgetdblattrarray(o, "PoolNX", 1+S, N, Xn[si]) # primal-side Int-Feasible solution
    end # Now si === SolCount
    frv .= 0;   Gurobi.GRBsetdblattrarray(grg.o, "LB", 1+S+N, N+1, frv)
                Gurobi.GRBsetdblattrarray(grg.o, "UB", 1+S+N, N+1, frv) # turn on the CG part
    Bv .= 'C';  Gurobi.GRBsetcharattrarray(grg.o, "VType", 1+S, N, Bv)
    Cgˈd = fill(false, si) # Column generation, properly
    for (z_sol, j) = zip(Xn, eachindex(Cgˈd))
        Gurobi.GRBsetdblattrarray(o, "LB", 1+S, N, z_sol)
        Gurobi.GRBsetdblattrarray(o, "UB", 1+S, N, z_sol)
        3 ≤ Settings.opt_and_ter(grg) ≤ 4 || continue
        (@. Cd[1:N] = round(z_sol); Cd[l] = 1; Cgˈd[j] = true)
        Gurobi.GRBaddvar(o,l,Ci,Cd,0.,0.,1.,Cchar('C'),C_NULL)
    end; NCgˈd = count(Cgˈd)
    println("global> lb = $lb, NCgˈd = $NCgˈd")
    Xl .= 0; Gurobi.GRBsetdblattrarray(o, "LB", 1+S, N, Xl)
    Xl .= 1; Gurobi.GRBsetdblattrarray(o, "UB", 1+S, N, Xl) # unfix the `x` variable (it's C-VType here)
    Settings.opt_ass_opt(grg)
    Gurobi.GRBgetdblattrarray(o, "X", 1+S, N, Xl) # The frac trial point
    Gurobi.GRBgetdblattrarray(o, "X",   1, S,  Θ)
    any(Settings.isfrac, Xl) || (printstyled("All trial entries from master are Integer."; color = :cyan); return false)
    for (z_sol, Cgˈd)=zip(Xn, Cgˈd) # To let lcg be Bounded Above
        Cgˈd && for (aug, lcg) = zip(aug, lcg) # for all scenarios
            lCd = lcg.Cd; @. z_sol = round(z_sol)
            lCd[1:N] .= z_sol
            is_z_sol_new_to_lcg(lcg) || continue
            solve_subMIP_no_bias(aug, z_sol) # 🟧
            qy_sol = Settings.getmodeldblattr(aug, "ObjVal")
            Gurobi.GRBaddconstr(lcg.o,l,lcg.Ci,lCd,Cchar('<'),qy_sol, C_NULL)
        end
    end
    true # we can generate lag-cut towards this trial point
end

"for one specific scene, using level method"
function load_pi_and_Cn_in_lag_cut(x_che, θ_che, aug, lcg)
    pi_hat, pi_cen = aug.Xl, aug.Xl2; N=length(x_che);  l=N+1;  o=lcg.o;   lCd, lCi = lcg.Cd, lcg.Ci; UB, gvio = Inf, -Inf
    lCd[1:N] .= x_che; lCd[l]=1; Gurobi.GRBchgcoeffs(o,l,lcg.levelCCi,lCi,lCd) # ✅ set once per x_che
    Settings.setxdblattrelement(lcg, N, "RHS", -1e100) # These two lines configure the level constraint
    pi_cen .= pi_hat = _sb_core(aug, pi_hat, N, x_che) # 🟧 get an initial pi_hat
    Obn_cen = Obn = Settings.getmodeldblattr(aug, "ObjBound")
    qy_sol = Settings.getxdblattrelement(aug, 0, "X")
    z_sol = Gurobi.GRBgetdblattrarray(aug.o, "X", 1, N, lCd)
    is_z_sol_new_to_lcg(lcg) && Gurobi.GRBaddconstr(o,l,lCi,lCd,Cchar('<'),qy_sol,C_NULL) # local column generation
    LB_cen = LB = pi_hat'x_che + Obn
    while true
        println("    local> gvio=$gvio, LB_cen=$LB_cen, UB=$UB")
        gvio > 3.5 && break # This is early break
        UB = get_local_UB(lcg, x_che)
        (UB - LB_cen)/LB_cen < 0.005 && (gvio > 1e-5 ? break : return false)
        Lvl = 0.8 * LB_cen + 0.2 * UB
        pi_hat = activate_QP_solve_get(pi_hat, pi_cen, lcg, Lvl) # pi_hat is updated
        solve_subMIP(aug, pi_hat) # 🟧 to exact-eval the LB
        Obn = Settings.getmodeldblattr(aug, "ObjBound")
        qy_sol = Settings.getxdblattrelement(aug, 0, "X")
        z_sol = Gurobi.GRBgetdblattrarray(aug.o, "X", 1, N, lCd)
        No_Progress = true
        if is_z_sol_new_to_lcg(lcg)
            Gurobi.GRBaddconstr(o,l,lCi,lCd,Cchar('<'),qy_sol,C_NULL) # local column generation
            No_Progress = false
        end
        LB = pi_hat'x_che + Obn
        if LB > LB_cen
            No_Progress = false
            pi_cen .= pi_hat; LB_cen, Obn_cen = LB, Obn # record the best primal-side solution
        end
        No_Progress && return false
        gvio = LB_cen - θ_che # global vio
    end
    aug.Cd[1:N] .= pi_cen; aug.Cd[l] = Obn_cen # temp storage
    true # has Lag cut
end

function config_Bin_noCG_solve_grg(grg, Bv, frv, S, N)
    Bv .= 'B'; Gurobi.GRBsetcharattrarray(grg.o, "VType", 1+S, N, Bv)
    frv .= -1e100; Gurobi.GRBsetdblattrarray(grg.o, "LB", 1+S+N, N+1, frv)
    frv .=  1e100; Gurobi.GRBsetdblattrarray(grg.o, "UB", 1+S+N, N+1, frv)
    Settings.opt_ass_opt(grg)
end
function exact_eval(grg, aug, N, S)
    printstyled("omit the common part!"; color = :yellow, bold = true)
    frv, Bv, o, x_che, Θ_che = grg.Cd, grg.Bv, grg.o, grg.Xl, grg.Θ
    config_Bin_noCG_solve_grg(grg, Bv, frv, S, N)
    lb, ub = Settings.getmodeldblattr(grg, "ObjBound"), Inf
    ge, si = Gurobi.GRBgetenv(o), 0
    while si < Settings.getmodelintattr(grg, "SolCount")
        Gurobi.GRBsetintparam(ge, "SolutionNumber", si)
        Gurobi.GRBgetdblattrarray(o, "PoolNX", 1+S, N, x_che)
        ubi = 0. # ❌ lazy to write the correct form
        for n = aug
            solve_subMIP_no_bias(n, x_che)
            ubi += Settings.getmodeldblattr(n, "ObjVal")/S
        end
        ub = min(ub, ubi)
        si+=1
    end
    agap = ub - lb; rgap = agap / ub
    printstyled("exact_eval> lb = $lb, agap = $agap, rgap = $rgap"; color = :magenta)
end

function add_sb_cut_once(grg, aug, lcg, N, S)
    frv, Bv, o, x_che, Θ_che = grg.Cd, grg.Bv, grg.o, grg.Xl, grg.Θ
    Bv .= 'C'; Gurobi.GRBsetcharattrarray(o, "VType", 1+S, N, Bv)
    frv .= -1e100; Gurobi.GRBsetdblattrarray(o, "LB", 1+S+N, N+1, frv)
    frv .=  1e100; Gurobi.GRBsetdblattrarray(o, "UB", 1+S+N, N+1, frv)
    gened = false; l = N+1
    for (s, n) = enumerate(aug)
        m = lcg[s]; lCd = m.Cd
        Settings.opt_ass_opt(grg)
        lb = Settings.getmodeldblattr(grg, "ObjBound")
        Gurobi.GRBgetdblattrarray(o, "X",   1, S, Θ_che)
        Gurobi.GRBgetdblattrarray(o, "X", 1+S, N, x_che)
        pi_hat, aCd = n.Xl, n.Cd
        pi_hat = _sb_core(n, pi_hat, N, x_che)
        Obn = Settings.getmodeldblattr(n, "ObjBound")
        qy_sol = Settings.getxdblattrelement(n, 0, "X")
        z_sol = Gurobi.GRBgetdblattrarray(n.o, "X", 1, N, lCd)
        is_z_sol_new_to_lcg(m) && Gurobi.GRBaddconstr(m.o, l, m.Ci, lCd, Cchar('<'), qy_sol, C_NULL)
        vio = pi_hat'x_che + Obn - Θ_che[s]
        println("sb_vio = $vio, global lb = $lb")
        if vio > 5e-5
            aCd[1:N] .= pi_hat; aCd[l]=-1; gened = true
            Gurobi.GRBaddconstr(o, l, n.Ci, aCd, Cchar('<'), -Obn, C_NULL)
        end
    end
    gened
end
function initialize(grg, aug, lcg, N, S)# This should be executed immediately after grg's construction
    _, x_che, Θ_che, o = Settings.opt_ass_opt(grg), grg.Xl, grg.Θ, grg.o
    Gurobi.GRBgetdblattrarray(o, "X", 1+S, N, x_che); l = N+1
    for (n, m)=zip(aug, lcg)
        pi_hat, aCd, lCd = n.Xl, n.Cd, m.Cd
        pi_hat = _sb_core(n, pi_hat, N, x_che)
        Obn = Settings.getmodeldblattr(n, "ObjBound")
        qy_sol = Settings.getxdblattrelement(n, 0, "X")
        z_sol = Gurobi.GRBgetdblattrarray(n.o, "X", 1, N, lCd)
        aCd[1:N] .= pi_hat
        aCd[l]=-1; Gurobi.GRBaddconstr(o, l, n.Ci, aCd, Cchar('<'), -Obn, C_NULL)
        lCd[l]=1; Gurobi.GRBaddconstr(m.o, l, m.Ci, lCd, Cchar('<'), qy_sol, C_NULL)
    end
    Θ_che .= 1/S; Gurobi.GRBsetdblattrarray(o, "Obj", 1, S, Θ_che) # 🟢 Set Once
    Settings.opt_ass_opt(grg)# 🟢 check OPTIMAL Once
end
function get_grg(m, N, S; NX)
    common = JuMP.@variable(m, lower_bound=0, upper_bound=0) # index 0
    θ = JuMP.@variable(m, [1:S]); Θ = similar(θ, Cdouble) # index 1:S
    x = JuMP.@variable(m, [1:N], lower_bound=0, upper_bound=1) # 1st-stage Bin Linking Var `x`
    ax = JuMP.@variable(m, [1:N+1]) # free or ==(0)
    JuMP.@constraint(m, [i=1:N], 0 == x[i] + ax[i]) # RowIndex range(0; length=N)
    JuMP.@constraint(m,          0 ==  1 + ax[N+1]) # RowIndex = N
    JuMP.@objective(m, Min, common) # sum(Prob[s]θ[s])
    o,refi,refd = m.moi_backend,Ref{Cint}(),Ref{Cdouble}(); ge = Gurobi.GRBgetenv(o)
    Gurobi.GRBsetintparam(ge, "PoolSolutions", NX)
    Gurobi.GRBsetintparam(ge, "PoolSearchMode", 2)
    Ci = Cint.(0:N); Cd = similar(Ci, Cdouble) # an aux container
    Bv = similar(x, Cchar); Xl = similar(Bv, Cdouble); Xn = [similar(Xl) for _=1:NX]
    (; m, o, refi, refd, Θ, Bv, Xl, Xn, Ci, Cd)
end

function solve_subLP_no_bias(n, x_che)
    v, Bv = n.Xl2, n.Bv; N = length(x_che); o = n.o
    v .= 0; Gurobi.GRBsetdblattrarray(o, "Obj", 1, N, v)
    Gurobi.GRBsetdblattrarray(o, "LB", 1, N, x_che)
    Gurobi.GRBsetdblattrarray(o, "UB", 1, N, x_che)
    Bv .= 'C'; Gurobi.GRBsetcharattrarray(o, "VType", 1, 2 * N, Bv) # relax all Int Constrs
    Settings.opt_ass_opt(n)
end
function solve_subMIP_no_bias(n, x_che)
    v, Bv = n.Xl2, n.Bv; N = length(x_che); o = n.o
    v .= 0; Gurobi.GRBsetdblattrarray(o, "Obj", 1, N, v)
    Gurobi.GRBsetdblattrarray(o, "LB", 1, N, x_che)
    Gurobi.GRBsetdblattrarray(o, "UB", 1, N, x_che)
    Bv .= 'B'; Gurobi.GRBsetcharattrarray(o, "VType", 1+N, N, Bv) # since copy vars are fixed
    Settings.opt_ass_opt(n)
end
function solve_subMIP(n, pi_hat)
    v, Bv = n.Xl2, n.Bv; N = length(n.Xl); o = n.o
    (@. v = -1. * pi_hat; Gurobi.GRBsetdblattrarray(o, "Obj", 1, N, v)) # price it
    v .= 0; Gurobi.GRBsetdblattrarray(o, "LB", 1, N, v)
    v .= 1; Gurobi.GRBsetdblattrarray(o, "UB", 1, N, v) # unfix it
    Bv .= 'B'; Gurobi.GRBsetcharattrarray(o, "VType", 1, 2 * N, Bv) # enforce all Bin constr
    Settings.opt_ass_opt(n)
end
function _sb_core(n, pi_hat, N, x_che)
    solve_subLP_no_bias(n, x_che)
    Gurobi.GRBgetdblattrarray(n.o, "RC", 1, N, pi_hat)
    solve_subMIP(n, pi_hat)
    pi_hat
end
function aug!(m, s, S; N, D)
    qy = JuMP.@variable(m) # GRBindex 0
    z, Xl2 = JuMP.@variable(m, [1:N], lower_bound=0, upper_bound=1), fill(NaN, N) # linking binary
    y = JuMP.@variable(m, [1:N], lower_bound=0, upper_bound=1) # inner binary (Bounds should be READ-ONLY)
    w = JuMP.@variable(m, [1:N, 1:N], lower_bound=0) # inner continuous
    JuMP.@constraint(m, sum(y[1:(N÷2)]) >= 2 * D)
    JuMP.@constraint(m, sum(y[(N÷2)+1:N]) >= 2 * D)
    JuMP.@constraint(m, [i=1:N, j=1:N], w[i, j] <= z[i])
    JuMP.@constraint(m, [i=1:N, j=1:N], w[i, j] <= y[j])
    JuMP.@constraint(m, [i=1:N, j=1:N], w[i, j] >= z[i] + y[j] - 1)
    Qy = JuMP.@expression(m, sum(D - rand(-1:.0017:1)i for i=w))
    JuMP.@constraint(m, Qy == qy)
    JuMP.@objective(m, Min, Qy)
    o,refi,refd = m.moi_backend,Ref{Cint}(),Ref{Cdouble}(); ge = Gurobi.GRBgetenv(o)
    Xl, Bv = fill(NaN, N), fill(Cchar('B'), 2 * N) # both linking and 2nd-inner
    Ci = Cint[range(1+S; length=N); s] # Read Only
    Cd = similar(Ci, Cdouble)
    (; m, o, refi, refd, Xl, Xl2, Ci, Cd, Bv)
end

function get_local_UB(lcg, x_che)
    o, lCd = lcg.o, lcg.Cd; l = length(lCd); N = l-1
    Settings.setxdblattrelement(lcg, N, "RHS", -1e100) # relax the level requirement
    lCd[1:N] .= x_che; lCd[l] = 1; Gurobi.GRBsetdblattrarray(o, "Obj", 0, l, lCd); Gurobi.GRBdelq(o)
    Settings.opt_ass_opt(lcg)
    Settings.getmodeldblattr(lcg, "ObjBound")
end
function is_z_sol_new_to_lcg(lcg) # 🟧 suppose z_sol has been loaded into lCd
    o, lCd = lcg.o, lcg.Cd; l = length(lCd); lCd[l] = 1; N = l-1
    Settings.setxdblattrelement(lcg, N, "RHS", -1e100)
    Gurobi.GRBsetdblattrarray(o, "Obj", 0, l, lCd); Gurobi.GRBdelq(o)
    Settings.opt_and_ter(lcg) != 2 # then (z_sol, qy_sol) is worth being added
end
function activate_QP_solve_get(pi_hat, pi_cen, lcg, Lvl)
    o, lCd = lcg.o, lcg.Cd; l = length(lCd); N = l-1; QCi, QObj= lcg.QCi, lcg.QObj
    Gurobi.GRBsetdblattrarray(o, "RHS", 0, N, pi_cen)
    Settings.setxdblattrelement(lcg, N, "RHS", Lvl) # 🟧 Note that the Matrix part was set beforehand
    lCd .= 0; Gurobi.GRBsetdblattrarray(o, "Obj", 0, l, lCd)
    QObj .= -1; Gurobi.GRBaddqpterms(o, N, QCi, QCi, QObj)
    lt = Settings.opt_and_ter(lcg) # 🟢 solve a QP, where only the pi_hat is relevant
    lt == 2 || error("local lcg terminate = $lt")
    Gurobi.GRBgetdblattrarray(o, "X", 0, N, pi_hat) # get the new "stable" pi_hat (ONLY this info is relevant)
    pi_hat
end
function lcg!(m, N)
    π = JuMP.@variable(m, [1:N])
    h = JuMP.@variable(m)
    d, QCi, QObj = JuMP.@variable(m, [1:N]), Cint.(range(N+1; length=N)), fill(-1., N)
    JuMP.@constraint(m, [i=1:N], d[i] + π[i] == 0) # RHS to be modified, being π_center
    JuMP.@constraint(m, #= x_che'π + h =# 0 >= -1e100 #=Level=#)
    JuMP.@objective(m, Max, 0.) # Always maximize, Linear/Quad to be set JIT
    Gurobi.GRBsetintparam(Gurobi.GRBgetenv(m.moi_backend), "Method", 1)
    levelCCi = fill(Cint(N), N+1) # READ-ONLY
    levelVCi = Ci = Cint.(0:N)    # READ-ONLY
    Cd = fill(NaN, N+1) # an aux container
    o,refi,refd = m.moi_backend,Ref{Cint}(),Ref{Cdouble}()
    (; m, o, refi, refd, Ci, Cd, QCi, QObj, levelCCi)
end
function get_model_vectors(N, S)
    genvs, mm = Settings.Env(S), Vector{JuMP.Model}(undef, S)
    Settings.Model!(mm, genvs)
    aug = [aug!(m, s, S; N=N, D=s) for (s,m)=enumerate(mm)]
    Settings.Model!(mm, genvs)
    lcg = [lcg!(m, N) for m=mm]
    aug, lcg
end

end
