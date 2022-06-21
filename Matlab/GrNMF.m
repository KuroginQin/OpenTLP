function [adj_est] = GrNMF(adj_seq,hid_dim,win_size,alpha,theta,max_iter,min_error,init_op,num_run)
%Function to implement GrNMF
%adj_seq: sequence of adjacnecy matrices (i.e., topology input)
%hid_dim: dimensionality of latent space
%win_size: window size (#historical snapshots)
%alpha, theta: hyper-parameters
%max_iter: maximum number of iterations
%min_error: minimum relative error to determine convergence
%init_op: option for initialization, 0 for NNDSVD (recommended), 1 for random init.
%num_run: number of independent runs for random init. 
%adj_est: prediciton result

    %====================
    adj = adj_seq{win_size}; %Adjacency matrix of current time step
    [num_nodes, ~] = size(adj);
    his_adj = cell(win_size-1); %Sequence of historical adj. mat. (except current snapshot)
    his_deg = cell(win_size-1); %Sequence of histroical degree diag mat. (except current snapshot)
    for t=1:win_size-1
        his_adj{t} = adj_seq{t};
        his_deg{t} = diag(sum(adj_seq{t}, 2));
    end
    
    %====================
    if init_op==0 % NNDSVD initiliazation
        [aux_init, rep_init] = NNDSVD(adj_seq{win_size}, hid_dim, 0);
        rep_init = rep_init';
        %Encoder
        [aux,rep,~] = GrNMF_Enc(aux_init,rep_init,adj,his_adj,his_deg,win_size,alpha,theta,max_iter,min_error);
        %Decoder
        adj_est = GrNMF_Dec(aux, rep);
    %====================
    else % random initilization (w/ multiple independent runs)
        rep_init = rand(num_nodes, hid_dim);
        aux_init = rand(num_nodes, hid_dim);
        %Encoder
        [aux,rep,loss] = GrNMF_Enc(aux_init,rep_init,adj,his_adj,his_deg,win_size,alpha,theta,max_iter,min_error);
        for r=2:num_run
            rep_init = rand(num_nodes, hid_dim);
            aux_init = rand(num_nodes, hid_dim);
            %Encoder
            [c_aux,c_rep,c_loss] = GrNMF_Enc(aux_init,rep_init,adj,his_adj,his_deg,win_size,alpha,theta,max_iter,min_error);
            if c_loss<loss
                loss = c_loss;
                aux = c_aux;
                rep = c_rep;
            end
        end
        %Decoder
        adj_est = GrNMF_Dec(aux, rep);
    end

end

function [aux,rep,loss] = GrNMF_Enc(aux,rep,adj,his_adj,his_deg,win_size,alpha,theta,max_iter,min_error)
%Encoder of GrNMF
%aux: auxiliary latent representation
%rep: latent representation to be optimized
%obj: value of loss fucntion
%adj: adjacency matrix of current time step
%his_adj: sequence of historical adjacency matrices (except current snapshot)
%his_deg: sequence of histroical degree diagonal matrices (except current snapshot)
%win_size: window size (i.e., #historical snapshots)
%alpha, theta: hyper-parameters
%max_iter: maximum number of iterations
%min_error: minimum relative error to determine convergence

    %====================
    %Compute loss function
    loss = norm(adj - aux*rep', 'fro')^2;
    for t=1:win_size-1
        loss = loss + alpha*theta^(win_size-t)*trace(rep'*(his_deg{t}-his_adj{t})*rep);
    end
    %==========
    iter_cnt = 0; %Counter of iteration
    error = 0; %Relative error of loss function
    %==========
    while iter_cnt==0 || (error>=min_error && iter_cnt<=max_iter)
        pre_loss = loss; %Loss in previous iteraton
        %====================
        %Update rep
        adj_sum = 0;
        deg_sum = 0;
        for t=1:win_size-1
           adj_sum = adj_sum + theta^(win_size-t)*his_adj{t};
           deg_sum = deg_sum + theta^(win_size-t)*his_deg{t};
        end
        numer = adj*aux + alpha*adj_sum*rep;
        denom = rep*(aux'*aux) + alpha*deg_sum*rep;
        rep = rep.*(numer./max(denom, realmin));
        %==========
        %Update aux
        aux = aux.*((adj*rep)./max(aux*(rep'*rep), realmin));
        
        %====================
        %Compute loss function
        loss = norm(adj-aux*rep', 'fro')^2;
        for t=1:win_size-1
            loss = loss + alpha*theta^(win_size-t)*trace(rep'*(his_deg{t}-his_adj{t})*rep);
        end
        %==========
        error = abs(loss-pre_loss)/pre_loss; %Relative error
        iter_cnt = iter_cnt+1;
        %fprintf('Iteration: %d; Loss: %8.4f Error: %8.4f\n', [iter_cnt, loss, error]);
    end
    
end

function [adj_est] = GrNMF_Dec(aux,rep)
%Decoder of GrNMF
%aux: learned auxiliary latent representation
%rep: learned latent representation
%adj_est: prediction result

    %====================
    adj_est = aux*rep';

end
