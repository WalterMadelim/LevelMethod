module Settings
import JuMP, Gurobi

# ✅ Suboptimal
# Settings.getmodelintattr(m, "SolCount")
# Gurobi.GRBsetintparam(Gurobi.GRBgetenv(m.o), "SolutionNumber", 0)
# "X" -> "PoolNX"

# ✅ param
# Gurobi.GRBsetintparam(Gurobi.GRBgetenv(mst.o), "OutputFlag", 1)
# Gurobi.GRBsetdblparam(Gurobi.GRBgetenv(mst.o), "TimeLimit", 20.4)
# Gurobi.GRBsetdblparam(Gurobi.GRBgetenv(mst.o), "NodeLimit", 0 or 1e100) # 0 is compute only at root node
# Gurobi.GRBsetdblparam(Gurobi.GRBgetenv(mst.o), "MIPGap", 0) # This can be written as 0::Int

"""
create Envs
"""
const C = Dict{String,Any}("Threads"=>1,"OutputFlag"=>0)
Env() = Gurobi.Env(C)
function Env(N::Int)
    v = Vector{Gurobi.Env}(undef, N)
    Threads.@threads for i=eachindex(v)
        v[i] = Env()
    end
    v
end

"""
create Models
"""
function Model(e::Gurobi.Env)
    m = JuMP.direct_model(Gurobi.Optimizer(e))
    JuMP.set_string_names_on_creation(m, false)
    m
end
function Model!(mv::Vector{JuMP.Model}, i, #=with existing ones=# ev::Vector{Gurobi.Env})
    m = JuMP.direct_model(Gurobi.Optimizer(ev[i]))
    JuMP.set_string_names_on_creation(m, false)
    mv[i] = m
end
Model!(mv::Vector{JuMP.Model}, #=with existing ones=# ev::Vector{Gurobi.Env}) = Threads.@threads for i=eachindex(mv)
    Model!(mv, i, ev)
end

# m::MyNamedTuple
# |-m::Model
# |-o::Optimizer
# |-refd::ref_Cdouble
# |-refi::ref_Cint

# ✅ e.g. setting `x::JuMP.VariableRef`'s "Obj" Attribute
getxdblattrelement(m, i::Integer, str) = (r=m.refd;Gurobi.GRBgetdblattrelement(m.o, str,i,r);r.x)
getxdblattrelement(m, x::JuMP.VariableRef, str) = getxdblattrelement(m, _gcc(m.o, x), str)
setxdblattrelement(m, i::Integer, str, v) = Gurobi.GRBsetdblattrelement(m.o, str, i, v)
setxdblattrelement(m, x::JuMP.VariableRef, str, v) = setxdblattrelement(m, _gcc(m.o, x), str, v)
setxcharattrelement(m, i::Integer, str, v) = Gurobi.GRBsetcharattrelement(m.o, str, i, v) # e.g. Cchar('B')
# GRBgetcharattrelement(m.o, "VType", 0, &first_one);

# ✅ e.g. add a row
# Gurobi.GRBaddconstr(o, read_len, ci, cd, Cchar('=' or '>'), Gn, C_NULL)

# ✅ e.g. add a column
# Gurobi.GRBaddvar(o, 1+1, Cint[0,1], Cd, Qy_col, 0., 1e100, Cchar('C'), C_NULL)

# ✅ e.g. query "X" of the trial vec in 1st-stage, and then fix the copy vec in 2nd-stage
# Gurobi.GRBgetdblattrarray(o, "X", start, len, Ptr)
# Gurobi.GRBsetdblattrarray(o, "LB", start, len, Ptr)
# Gurobi.GRBsetcharattrarray(o, "VType", start, len, fill(Cchar('B'), len)) # ✅ e.g. set variables to binary

# ✅ e.g. query ObjVal/Runtime
function getmodeldblattr(m, str)
    r = m.refd
    Gurobi.GRBgetdblattr(m.o, str, r)
    r.x
end
# ✅ e.g. query NumConstrs/NumVars
function getmodelintattr(m, str)
    r = m.refi
    Gurobi.GRBgetintattr(m.o, str, r)
    r.x
end

opt_ass_opt(m) = (opt_and_ter(m)==2 || error())
function opt_and_ter(m)
    o, r = m.o, m.refi
    Gurobi.GRBoptimize(o)
    Gurobi.GRBgetintattr(o, "Status", r)
    r.x
end

isfrac(x) = ≈(x, 0.5; atol = 0.49999)
printinfo() = (th = map(Threads.nthreads, (:default, :interactive)); println("Settings> Threads=$th"))
_gcc(o,x) = Gurobi.c_column(o,JuMP.index(x))
end

# ✅ Status: 3==INFEASIBLE, 5==UNBOUNDED, 4==(3||5), 9==TimeLimit, 1==Not_solved_yet
