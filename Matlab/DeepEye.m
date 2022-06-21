function [adj_est] = DeepEye(adj_seq,hid_dim,win_size,lambd,max_iter,min_error,init_op,num_run)
%Function to implement DeepEye
%adj_seq: sequence of adjacnecy matrices (i.e., topology input)
%hid_dim: dimensionality of latent space
%win_size: window size (#historical snapshots)
%lambd: hyper-parameter
%max_iter: maximum number of iterations
%min_error: minimum relative error to determine convergence
%init_op: option for initialization, 0 for NNDSVD (recommended), 1 for random init.
%num_run: number of independent runs for random init. 
%adj_est: prediciton result

    %====================
    [num_nodes, ~] = size(adj_seq{1});
    if init_op==0 %NNDSVD initialization
        U_init_seq = cell(win_size);
        V_init_seq = cell(win_size);
        U_init = zeros(num_nodes, hid_dim);
        V_init = zeros(num_nodes, hid_dim);
        dec_sum = 0;
        for t=1:win_size
            [U_t_init, V_t_init] = NNDSVD(adj_seq{t}, hid_dim, 0);
            V_t_init = V_t_init';
            decay = lambd^(win_size-t);
            dec_sum = dec_sum + decay;
            U_init = U_init + decay*U_t_init;
            V_init = V_init + decay*V_t_init;
            U_init_seq{t} = U_t_init;
            V_init_seq{t} = V_t_init;
        end
        U_init = U_init/dec_sum;
        V_init = V_init/dec_sum;
        %Encoder
        [U,V,~] = DeepEye_Enc(U_init_seq,V_init_seq,U_init,V_init,adj_seq,win_size,lambd,max_iter,min_error);
        %Decoder
        adj_est = DeepEye_Dec(U,V);
    %====================
    else %Random initialization w/ multiple independent runs
        U_init_seq = cell(win_size);
        V_init_seq = cell(win_size);
        U_init = rand(num_nodes, hid_dim);
        V_init = rand(num_nodes, hid_dim);
        for t=1:win_size
            U_t_init = rand(num_nodes, hid_dim);
            V_t_init = rand(num_nodes, hid_dim);
            U_init_seq{t} = U_t_init;
            V_init_seq{t} = V_t_init;
        end
        %Encoder
        [U,V,loss] = DeepEye_Enc(U_init_seq,V_init_seq,U_init,V_init,adj_seq,win_size,lambd,max_iter,min_error);
        %==========
        for r=2:num_run
            U_init_seq = cell(win_size);
            V_init_seq = cell(win_size);
            U_init = rand(num_nodes, hid_dim);
            V_init = rand(num_nodes, hid_dim);
            for t=1:win_size
                U_t_init = rand(num_nodes, hid_dim);
                V_t_init = rand(num_nodes, hid_dim);
                U_init_seq{t} = U_t_init;
                V_init_seq{t} = V_t_init;
            end
            %Encoder
            [c_U,c_V,c_loss] = DeepEye_Enc(U_init_seq,V_init_seq,U_init,V_init,adj_seq,win_size,lambd,max_iter,min_error);
            if c_loss<loss
                loss = c_loss;
                U = c_U;
                V = c_V;
            end
        end
        %Decoder
        adj_est = DeepEye_Dec(U,V);
    end

end

function [U,V,loss] = DeepEye_Enc(U_seq,V_seq,U,V,adj_seq,win_size,lambd,max_iter,min_error)
%Encoder of DeepEye
%U_seq,V_seq: sequences of snapshot-induced latent representations
%U, V: shared (global) latent representations
%adj_seq: sequence of historical adjacency matrices
%win_size: window size (i.e., #historical snapshots)
%lambd: hyper-parameter
%max_iter: maximum number of iterations
%min_error: minimum relative error to determine convergence

    %====================
    %Compute loss function
    loss = 0;
    dec_sum = 0; %Sum of decaying factors
    for t=1:win_size
        decay = lambd^(win_size-t); %Decaying factor
        dec_sum = dec_sum + decay;
        loss = loss + decay*norm(adj_seq{t} - U_seq{t}*V_seq{t}', 'fro')^2;
        loss = loss + decay*norm(U_seq{t} - U, 'fro')^2;
        loss = loss + decay*norm(V_seq{t} - V, 'fro')^2;
    end
    %==========
    [num_nodes, hid_dim] = size(U);
    iter_cnt = 0; %Counter of iteration
    error = 0; %Relative error of loss function
    while iter_cnt==0 || (error>=min_error && iter_cnt<=max_iter)
        pre_loss = loss; %Loss function in previous iteraton
        %===========
        %Update {U_t} matrices
        for t=1:win_size
            numer = adj_seq{t}*V_seq{t} + U;
            denom = U_seq{t}*(V_seq{t}'*V_seq{t}) + U_seq{t};
            U_seq{t} = U_seq{t}.*(numer./max(denom, realmin));
        end
        %===========
        %Update {V_t} matrics
        for t=1:win_size
            numer = adj_seq{t}*U_seq{t} + V;
            denom = V_seq{t}*(U_seq{t}'*U_seq{t}) + V_seq{t};
            V_seq{t} = V_seq{t}.*(numer./max(denom, realmin));
        end
        %===========
        %Update U
        aux = zeros(num_nodes, hid_dim);
        for t=1:win_size
            aux = aux + lambd^(win_size-t)*U_seq{t};
        end
        U = aux/dec_sum;
        %===========
        %Update V
        aux = zeros(num_nodes, hid_dim);
        for t=1:win_size
            aux = aux + lambd^(win_size-t)*V_seq{t};
        end
        V = aux/dec_sum;
        
        %====================
        %Calculate the objective value
        loss = 0;
        for t=1:win_size
            decay = lambd^(win_size-t);
            loss = loss + decay*norm(adj_seq{t} - U_seq{t}*V_seq{t}', 'fro')^2;
            loss = loss + decay*norm(U_seq{t} - U, 'fro')^2;
            loss = loss + decay*norm(V_seq{t} - V, 'fro')^2;
        end
        %==========
        error = abs(loss-pre_loss)/pre_loss;
        iter_cnt = iter_cnt+1;
        %fprintf('Iteration: %d; Loss: %8.4f Error: %8.4f\n', [iter_cnt, loss, error]);
    end

end

function [adj_est] = DeepEye_Dec(U,V)
%Decoder of DeepEye
%U, V: learned shared (global) latent representations
%adj_est: prediction result

    %====================
    adj_est = U*V';

end
