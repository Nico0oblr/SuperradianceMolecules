using DataStructures
using OrdinaryDiffEq
using SparseArrays
using PyPlot
using LinearAlgebra

# Collective factors (used also in the individual channels):
"""
    A_JM_minus(J, M)

Collective lowering amplitude:
  A_{JM}⁻ = √[(J + M)(J – M + 1)]
"""
A_JM_minus(J, M) = sqrt((J + M) * (J - M + 1))

"""
    A_JM_plus(J, M)

Collective raising amplitude:
  A_{JM}⁺ = √[(J – M)(J + M + 1)]
"""
A_JM_plus(J, M) = sqrt((J - M) * (J + M + 1))

# ---------------------------
# Individual Decay (lowering M)
# ---------------------------

"""
    P_JM_minus_0(J, M, N)

Individual decay channel with no change in J:
  P_{JM}^{-,0} = √[(2+N)/(4J(J+1))] A_{JM}⁻.
"""
function P_JM_minus_0(J, M, N)
    if J == 0
        return 0.0
    end
    return sqrt((2 + N) / (4 * J * (J + 1))) * A_JM_minus(J, M)
end

"""
    P_JM_minus_minus(J, M, N)

Individual decay channel lowering J by 1 (“s = -1”):
  P_{JM}^{-,-} = -√[(N+2J+2)(J+M)(J+M-1)/(4J(2J+1))].
"""
function P_JM_minus_minus(J, M, N) 
    if J == 0
        return 0.0
    end
    return -sqrt((N + 2 * J + 2) * (J + M) * (J + M - 1) / (4 * J * (2 * J + 1)))
end

"""
    P_JM_minus_plus(J, M, N)

Individual decay channel increasing J by 1 (“s = +1”):
  P_{JM}^{-,+} = √[(N-2J)(J-M+1)(J-M+2)/(4(J+1)(2J+1))].
"""
P_JM_minus_plus(J, M, N) = sqrt((N - 2 * J) * (J - M + 1) * (J - M + 2) / (4 * (J + 1) * (2 * J + 1)))

# ---------------------------
# Individual Pumping (raising M)
# ---------------------------

"""
    P_JM_plus_0(J, M, N)

Individual pumping channel with no change in J:
  P_{JM}^{+,0} = √[(2+N)/(4J(J+1))] A_{JM}⁺.
"""
function P_JM_plus_0(J, M, N)
    if J == 0
        return 0.0
    end
    return sqrt((2 + N) / (4 * J * (J + 1))) * A_JM_plus(J, M)
end

"""
    P_JM_plus_minus(J, M, N)

Individual pumping channel lowering J by 1 (“s = -1”):
  P_{JM}^{+,-} = √[(N+2J+2)(J-M)(J-M-1)/(4J(2J+1))].
"""
function P_JM_plus_minus(J, M, N)
    if J == 0
        return 0.0
    end
    return sqrt((N + 2 * J + 2) * (J - M) * (J - M - 1) / (4 * J * (2 * J + 1)))
end

"""
    P_JM_plus_plus(J, M, N)

Individual pumping channel increasing J by 1 (“s = +1”):
  P_{JM}^{+,+} = -√[(N-2J)(J+M+1)(J+M+2)/(4(J+1)(2J+1))].
"""
P_JM_plus_plus(J, M, N) = -sqrt((N - 2 * J) * (J + M + 1) * (J + M + 2) / (4 * (J + 1) * (2 * J + 1)))

# ---------------------------
# Individual Dephasing
# ---------------------------

"""
    P_JM_z_0(J, M, N)

Individual dephasing channel that leaves M unchanged:
  P_{JM}^{z,0} = √[(2+N)/(4J(J+1))] M.
"""
function P_JM_z_0(J, M, N)
    if J == 0
        return 0.0
    else
        return sqrt((2 + N) / (4 * J * (J + 1))) * M
    end
end
#P_JM_z_0(J, M, N) = sqrt((2 + N) / (4 * J * (J + 1))) * M

"""
    P_JM_z_minus(J, M, N)

Individual dephasing channel lowering J by 1 (“s = -”):
  P_{JM}^{z,-} = √[(N+2J+2)(J-M)(J+M)/(4J(2J+1))].
"""
function P_JM_z_minus(J, M, N)
    if J == 0
        return 0.0
    end
    sqrt((N + 2 * J + 2) * (J - M) * (J + M) / (4 * J * (2 * J + 1)))
end

"""
    P_JM_z_plus(J, M, N)

Individual dephasing channel increasing J by 1 (“s = +”):
  P_{JM}^{z,+} = √[(N-2J)(J+1-M)(J+1+M)/(4(J+1)(2J+1))].
"""
P_JM_z_plus(J, M, N) = sqrt((N - 2 * J) * (J + 1 - M) * (J + 1 + M) / (4 * (J + 1) * (2 * J + 1)))

function build_basis(J)
    S0 = iseven(2*J) ? 0 : 1/2
    return [(S,M) for S=S0:J for M=-S:S]
end

function sane(J,M,Jmax)
    if J > Jmax return false end
    if abs(M) > J return false end
    if J < 0 return false end
    return true
end 

"""
state is a tuple (S,M)
params is a dictionary (global_decay, global_pump, local_pump, local_decay, local_dephasing)
gives the outgoing rates and the states they point to
"""
function outgoing_for_state(state, params, Jmax)
    J, M = state
    out = DefaultDict{Tuple{Rational, Rational}, Float64}(0.0)
    # global_decay M -> M-1 and J->J
    if sane(J,M-1,Jmax) out[(J, M-1)] += A_JM_minus(J,M) * sqrt(params["global_decay"]) end
    # global_decay M -> M+1 and J->J
    if sane(J, M+1,Jmax) out[(J, M+1)] += A_JM_plus(J,M) * sqrt(params["global_pump"]) end
    # local_decay M -> M-1 and J->(J+1,J,J-1)
    if sane(J+1, M-1,Jmax) out[(J+1, M-1)] += P_JM_minus_plus(J,M,2*J) * sqrt(params["local_decay"]) end
    if sane(J-1,M-1,Jmax) out[(J-1,M-1)] += P_JM_minus_minus(J,M,2*Jmax) * sqrt(params["local_decay"]) end
    if sane(J,M-1,Jmax) out[(J,M-1)] += P_JM_minus_0(J,M,2*Jmax) * sqrt(params["local_decay"]) end
    # local_pump M -> M+1 and J->(J+1,J,J-1)
    if sane(J+1,M+1,Jmax) out[(J+1,M+1)] += P_JM_plus_plus(J,M,2*Jmax) * sqrt(params["local_pump"]) end
    if sane(J-1,M+1,Jmax) out[(J-1,M+1)] += P_JM_plus_minus(J,M,2*Jmax) * sqrt(params["local_pump"]) end
    if sane(J,M+1,Jmax) out[(J,M+1)] += P_JM_plus_0(J,M,2*Jmax) * sqrt(params["local_pump"]) end 
    # local_dephasing M -> M+1 and J->(J+1,J,J-1)
    if sane(J+1, M,Jmax) out[(J+1, M)] += P_JM_z_plus(J,M,2*Jmax) * sqrt(params["local_dephasing"]) end
    if sane(J-1, M,Jmax) out[(J-1, M)] += P_JM_z_minus(J,M,2*Jmax) * sqrt(params["local_dephasing"]) end 
    if sane(J, M,Jmax) out[(J, M)] += P_JM_z_0(J,M,2*Jmax) * sqrt(params["local_dephasing"]) end
    return out
end

function incoming_for_state(state, params, Jmax)
    J, M = state
    inp = DefaultDict{Tuple{Rational, Rational}, Float64}(0.0)

    if sane(J,M+1, Jmax); inp[(J,M+1)] += A_JM_minus(J,M+1)* sqrt(params["global_decay"]); end
    if sane(J,M-1, Jmax); inp[(J,M-1)] += A_JM_plus(J, M-1) * sqrt(params["global_pump"]); end

    if sane(J-1, M+1, Jmax); inp[(J-1,M+1)] += P_JM_minus_plus(J-1,M+1, 2*(J-1))  * sqrt(params["local_decay"]); end
    if sane(J+1, M+1, Jmax); inp[(J+1,M+1)] += P_JM_minus_minus(J+1,M+1, 2*Jmax)  * sqrt(params["local_decay"]); end
    if sane(J, M+1, Jmax); inp[(J, M+1)] += P_JM_minus_0(J,M+1, 2*Jmax)   * sqrt(params["local_decay"]); end

    if sane(J-1, M-1, Jmax); inp[(J-1, M-1)] += P_JM_plus_plus(J-1, M-1, 2*Jmax) * sqrt(params["local_pump"]); end
    if sane(J+1, M-1, Jmax); inp[(J+1, M-1)] += P_JM_plus_minus(J+1,M-1, 2*Jmax) * sqrt(params["local_pump"]); end
    if sane(J, M-1, Jmax); inp[(J, M-1)] += P_JM_plus_0(J,    M-1, 2*Jmax)* sqrt(params["local_pump"]); end

    if sane(J-1, M, Jmax); inp[(J-1, M)] += P_JM_z_plus(J-1, M, 2*Jmax) * sqrt(params["local_dephasing"]); end
    if sane(J+1, M, Jmax); inp[(J+1, M)] += P_JM_z_minus(J+1, M, 2*Jmax) * sqrt(params["local_dephasing"]); end
    if sane(J, M, Jmax); inp[(J, M)] += P_JM_z_0(J, M, 2*Jmax) * sqrt(params["local_dephasing"]); end

    return inp
end

function sum_incoming_for_state(state, params, Jmax)
    tmp = incoming_for_state(state, params, Jmax)
    return sum(x[2] ^ 2 for x in tmp)
end

function sum_outgoing_for_state(state, params, Jmax)
    tmp = outgoing_for_state(state, params, Jmax)
    return sum(x[2] ^ 2 for x in tmp)
end

function check_inversion(target, params, Jmax; atol=1e-10)
    incoming = incoming_for_state(target, params, Jmax)
    ok = true
    for (src, w_in) in incoming
        out = outgoing_for_state(src, params, Jmax)
        w_out = get(out, target, 0.0)
        ok &= isapprox(w_in, w_out; atol=atol)
        if !ok
            @warn "Mismatch" src=src target=target w_in=w_in w_out=w_out
        end
    end
    return ok
end

"""
For diagonal evolution, define the rate matrix
"""
function set_up_rate_matrix(Jmax, params)
    basis = build_basis(Jmax)
    index_mapping = Dict(basis .=> 1:length(basis))
    sparse_constructor = DefaultDict{Tuple{Int64, Int64}, Float64}(0.0)

    for state in basis
        out_rate = sum_outgoing_for_state(state, params, Jmax)
        in_rates = incoming_for_state(state, params, Jmax)
        is = index_mapping[state]
        sparse_constructor[(is, is)] -= out_rate
        for (src, rate) in in_rates
            sparse_constructor[(is, index_mapping[src])] += rate ^ 2
        end
    end


    I = Int[]; J = Int[]; V = Float64[]
    sizehint!(I, length(sparse_constructor)); sizehint!(J, length(sparse_constructor)); sizehint!(V, length(sparse_constructor))
    for ((i, j), v) in sparse_constructor
        push!(I, i); push!(J, j); push!(V, v)
    end
        
    N = length(basis)
    out = sparse(I,J,V,N,N)
    dropzeros!(out)
    return basis, index_mapping, out, N
end

function initial_state(index_mapping, state, nstates)
    u = zeros(Float64, nstates)
    u[index_mapping[state]] = 1.0
    return u
end