import Random
SEED = rand(1:typemax(Int32));
Random.seed!(34)
include("src/Settings.jl");
include("Seq1/Go.jl");

N, S = 18, 2; # currently S is restricted at 2
aug, lcg = Go.get_model_vectors(N, S);
grg = Go.get_grg(Settings.Model(Settings.Env()), N, S; NX = 5);
Go.initialize(grg, aug, lcg, N, S)
for _=1:40 Go.add_sb_cut_once(grg, aug, lcg, N, S) || break end
Go.exact_eval(grg, aug, N, S)

stopit = Threads.Atomic{Bool}(false);
tsk = Threads.@spawn(Go.gen_lag_cuts(grg, aug, lcg, N, stopit))

##########################
stopit.value = true
wait(tsk)
Go.exact_eval(grg, aug, N, S)
