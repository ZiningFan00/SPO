using Distributions
using LinearAlgebra
using Random
using JuMP
using GLPK
using Gurobi
using Parameters, DecisionTree
using ScikitLearn.CrossValidation: cross_val_score
using LightGraphs
using SimpleWeightedGraphs
using DelimitedFiles

d = 40;
p = 5;

function generate_B(d,p)
    B = zeros((d,p));
    N = Normal();
    ran = rand(N,d*p);
    for i in 1:(d*p)
        if ran[i] >= 0
            B[i] = 1
        else
            B[i] = 0
        end
    end
    return B
end

function generate_data(B, n, deg, noiseWidth)
    I_p = Matrix{Float64}(1I, p, p);
    mu = [0, 0, 0, 0, 0];
    MV = MvNormal(mu, I_p);
    X = rand(MV,n);
    X = reshape(X,(n,p));
    epsilon = rand(Uniform(1-noiseWidth,1+noiseWidth),(n,d))
    C = ((X*B'/sqrt(p).+3).^deg.+1).*epsilon
    return X, C
end

function dijkstra(C)
    grid = 5
    g = SimpleWeightedDiGraph(grid^2)
    k = 1
    dict = Dict()
    for i in 1:grid
        for j in 1:grid
            index = (i-1)*grid+j
            if i != 1
                add_edge!(g, index-grid, index, C[k])
                dict[(index-grid, index)] = k
                k = k+1
            end
            if j != 1
                add_edge!(g, index-1, index, C[k])
                dict[(index-1, index)] = k
                k = k+1
            end
        end
    end
    result = dijkstra_shortest_paths(g, 1)
    node_path = enumerate_paths(result, grid^2)
    w = zeros(40)
    for i in 1:length(node_path)-1
        src = node_path[i]
        dst = node_path[i+1]
        w[dict[src,dst]] =1
    end
    return w, result.dists[grid^2]
end

function Normalizedloss(C_hat, C)
    n = size(C)[1]
    num = 0
    den = 0
    for i in 1:n
        w_hat, z_hat = dijkstra(C_hat[i,:])
        w, z = dijkstra(C[i,:])
        num += C[i,:]'*w_hat-z
        den += z;
    end
    return num/den
end

function outerProduct(A,B)
    m = size(A)[1];
    n = size(B)[1];
    
    res=zeros(m,n)
    for i in 1:m
        for j in 1:n
            res[i,j] = A[i]*B[j]
        end
    end
    return res
end   

function Normalizedloss(C_hat, C)
    n = size(C)[1]
    num = 0
    den = 0
    for i in 1:n
        w_hat, z_hat = dijkstra(C_hat[i,:])
        w, z = dijkstra(C[i,:])
        num += C[i,:]'*w_hat-z
        den += z;
    end
    return num/den
end

function outerProduct(A,B)
    m = size(A)[1];
    n = size(B)[1];
    
    res=zeros(m,n)
    for i in 1:m
        for j in 1:n
            res[i,j] = A[i]*B[j]
        end
    end
    return res
end   

function find_lambda(X, C, X_test, C_test, loss_type)
        
    lambdas = [10.0^(i) for i in -6:2]
    push!(lambdas, 0)
    loss = zeros(10)
    for i in 1:10
        lambda = lambdas[i]
        accuracy = optimiztion(X, C, X_test, C_test, lambda, loss_type)
        loss[i] = accuracy
    end
    println(loss)
    return lambdas[argmin(loss)]
end

function RandomForest(X, C, X_test, C_test,)
    
    C_hat=zeros((10000,d))
    
    for i in 1:d
        # model = DecisionTreeRegressor()
        model = RandomForestRegressor(n_trees=100)
        fit!(model, X, C[:,i])
        c = predict(model, X_test[:,rand(1:5,2)]);
        C_hat[:,i]=c
    end
    
    accuracy=Normalizedloss(C_hat, C_test);
    return accuracy
end

function optimiztion(X, C, X_test, C_test, lambda,loss_type)
    n = size(C)[1]
    model = Model(Gurobi.Optimizer)
    @variable(model, B_var[1:d, 1:p])
    @variable(model, reg[1:d, 1:p])
    @constraint(model, B_var .<= reg)
    @constraint(model, -B_var .<= reg)
    
    if loss_type == "ls"
        @objective(model, Min, sum(dot((B_var*X[i,:]-C[i,:]),(B_var*X[i,:]-C[i,:])) for i in 1:n)/(2*n)+lambda*sum(reg[i] for i in 1:d*p));
    elseif loss_type == "absolute"
        @variable(model, abs[1:n, 1:d])
        @constraint(model, X*B_var'-C .<= abs)
        @constraint(model, C-X*B_var'.<= abs)
        @objective(model, Min, sum(abs[i] for i in 1:n*d)/n+lambda*sum(reg[i] for i in 1:d*p))
    elseif loss_type == "spo"
        m = 90
        A = zeros(m, d)
        b = zeros(m)
        k = 0
        for i in 0:4
            for j in 0:4
                k = k+1
                if i != 0
                    A[2k-1, 4*j+i] = 1
                    A[2k, 4*j+i] = -1
                end
                if j != 0
                    A[2k-1, 20+4*i+j] = 1
                    A[2k, 20+4*i+j] = -1
                end
                if i != 4
                    A[2k-1, 4*j+i+1] = -1
                    A[2k, 4*j+i+1] = 1
                end
                if j != 4
                    A[2k-1, 20+4*i+j+1] = -1
                    A[2k, 20+4*i+j+1] = 1
                end
                if (i==0) && (j==0)
                    b[2k-1] = -1
                    b[2k] = 1
                end
                if (i==4) && (j==4)
                    b[2k-1] = 1
                    b[2k] = -1
                end
            end
        end

        for i in 1:d
            A[i+50, i] = 1
        end
        
        @variable(model, p_var[1:n, 1:m] >=0)
        @constraint(model, con[i = 1:n], A'*p_var[i,:] .== 2*B_var*X[i,:]-C[i,:])
        @objective(model, Min, 
#             sum(-b'*p_var[i,:] .+ dijkstra(C[i,:])[1]*X[i,:]'.*B_var.*2
#                  .- dijkstra(C[i,:])[2] for i in 1:n)/n
#             + lambda*sum(reg[i] for i in 1:d*p))
            sum(-b'*p_var[i,:] .+ 2*(B_var*X[i,:])'*dijkstra(C[i,:])[1]
                 .- dijkstra(C[i,:])[2] for i in 1:n)/n
            + lambda*sum(reg[i] for i in 1:d*p))
    end
    
    optimize!(model)
    
    C_hat = X_test * value.(B_var)';
    
    accuracy=Normalizedloss(C_hat, C_test);
    return accuracy
end 

n = 100
deg = 1
noiseWidth = 0.5
# loss=zeros((5,50))
loss=[]
for t in 1:50
    B = generate_B(d,p);
    X, C = generate_data(B, n, deg, noiseWidth);
    X_test, C_test = generate_data(B, 10000, deg, noiseWidth);
    lambda=find_lambda(X, C, X_test, C_test, "ls");
    res=optimiztion(X, C, X_test, C_test, lambda,"ls");
    append!(loss,res)
end
# i = 1
# for d in deg
#     loss=[]
#     for t in 1:50
#         B = generate_B(d,p);
#         X, C = generate_data(B, n, deg, noiseWidth);
#         X_test, C_test = generate_data(B, 10000, deg, noiseWidth);
#         lambda=find_lambda(X, C, X_test, C_test, "ls");
#         res=optimiztion(X, C, X_test, C_test, lambda,"ls");
#         append!(loss,res)
#     end
#     i = i+1
# end

writedlm( "ls_1_100",  loss, ',')



