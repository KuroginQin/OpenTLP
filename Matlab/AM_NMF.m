function [adj_est] = AM_NMF(adj_seq,hid_dim,win_size,alpha,min_error,max_iter,dec_op,theta,beta)
%Function to implement AM-NMF
%adj_seq: sequence of historical adjacency matrices
%hid_dim: dimensionality of latent space
%win_size: window size (#historical snapshots)
%alpha: hyper-parameter
%min_error: minimum relative error to determine convergence
%max_iter: maximum number of iterations
%dec_op: option of decoder, 0 for standard version, 1 for refined version
%theta, beta: hyper-parameters of refined decoder
%adj_est: prediction result

    %====================
    rep_init_seq = cell(win_size); %Sequence of independent represetation matrices
    aux_init_seq = cell(win_size); %Sequence of auxiliary matrices
    for t=1:win_size
        adj = adj_seq{t};
        [aux_init,rep_init,~] = comp_init(adj,hid_dim,min_error,max_iter);
        rep_init_seq{t} = rep_init;
        aux_init_seq{t} = aux_init;
    end
    rep_init = rep_init_seq{win_size};
    %Encoder
    [rep,aux_seq,~] = AM_NMF_Enc(adj_seq,rep_init,rep_init_seq,aux_init_seq,win_size,alpha,min_error,max_iter);
    %Decoder
    if dec_op==0
        adj_est = AM_NMF_Dec(rep,aux_seq);
    else
        adj_est = AM_NMF_Dec_Ref(rep,aux_seq,theta,beta);
    end
    
end

function [rep,aux_seq,loss] = AM_NMF_Enc(adj_seq,rep,rep_seq,aux_seq,win_size,alpha,min_error,max_iter)
%Encoder of AM-NMF
%adj_seq: sequence of historical adjacency matrices
%rep: shared representation matrix
%rep_seq: sequence of independent representation matrices (w.r.t. each snapshot)
%aux_seq: sequence of auxiliary matrices
%win_size: window size (#historical snapshots)
%alpha: hyper-parameter
%min_error: minimum relative error to determine convergence
%max_iter: maximum number of iterations
%loss: loss function

    %====================
	%Compute loss function
	%计算目标函数的值
    loss = norm(adj_seq{win_size} - aux_seq{win_size}*rep', 'fro')^2;
    for t=1:win_size-1
        adp_param = get_adp_param(rep, rep_seq{t}); %Adpative parameter
        loss = loss + alpha*adp_param^(win_size-t)*norm(adj_seq{t} - aux_seq{t}*rep', 'fro')^2;
    end
    %==========
	iter_cnt = 0; %Counter of iteration
    error = 0; %Relative error of loss function
    while iter_cnt==0 || (error>=min_error && iter_cnt<=max_iter)
        pre_loss = loss; %Loss function in previous iteraton
        %==========
        %Y-Process
        for t=1:win_size-1
            aux_seq{t} = aux_seq{t}.*((adj_seq{t}*rep)./max(aux_seq{t}*(rep'*rep), 1e-100));
        end
        aux_seq{win_size} = aux_seq{win_size}.*((adj_seq{win_size}*rep)./max(aux_seq{win_size}*(rep'*rep), 1e-100));
        %==========
        %X-Process
        numer = adj_seq{win_size}*aux_seq{win_size};
        denom = aux_seq{win_size}'*aux_seq{win_size};
        for t=1:win_size-1
            adp_param = get_adp_param(rep, rep_seq{t});
            numer = numer + alpha*adp_param^(win_size-t)*adj_seq{t}*aux_seq{t};
            denom = denom + alpha*adp_param^(win_size-t)*aux_seq{t}'*aux_seq{t};
        end
        denom = rep*denom;
        rep = rep.*(numer./max(denom, 1e-100));
        
        %==========
        %Compute loss function
        %计算目标函数的值
        loss = norm(adj_seq{win_size} - aux_seq{win_size}*rep', 'fro')^2;
        for t=1:win_size-1
            adp_param = get_adp_param(rep, rep_seq{t}); %Adpative parameter
            loss = loss + alpha*adp_param^(win_size-t)*norm(adj_seq{t} - aux_seq{t}*rep', 'fro')^2;
        end
        %===========
        error = abs(loss-pre_loss)/pre_loss;
        iter_cnt = iter_cnt+1;
        %fprintf('Iteration: %d; Obj. Value: %8.4f Error: %8.4f\n', [iter_cnt, obj, error]);
    end

end

function [adp_param] = get_adp_param(rep, rep_ind)
%Function to get adaptive parameter
%rep: shared representation matrix
%rep_ind: independent representation matrix (w.r.t. each snapshot)
%adp_param: adaptive parameter

    %====================
    [num_nodes, ~] = size(rep); 
    avg_sim = 0.0; %Average similarity
    for i=1:num_nodes
        a = rep(i,:);
        a = (a-min(a))/max(max(a)-min(a), realmin);
        b = rep_ind(i,:);
        b = (b-min(b))/max(max(b)-min(b), realmin);
        avg_sim = avg_sim + 1/(1.0 + norm(a-b)); 
    end
    avg_sim = avg_sim/num_nodes;
    adp_param = avg_sim;
    
end

function [aux,rep,loss] = comp_init(adj,hid_dim,min_error,max_iter)
%Function to initialize the representation & auxiliary matrices w.r.t. a NMF component
%adj: adjacency matrix of current snapshot
%hid_dim: dimensionality of latent space
%rep: initialized representation matrix
%aux: initialized auxiliary matrix
%min_error: minimum relative error to determine convergence
%max_iter: maximum number of iterations
%loss: loss function of initialization

	%====================
    [aux, rep] = NNDSVD(adj, hid_dim, 0); %NNDSVD init.
    rep = rep';
    %==========
    %Compute loss function for the initialization
    loss = norm(adj - aux*rep', 'fro')^2;
    %==========
    iter_cnt = 0; %Counter of iteration
    error = 0; %Relative error of loss function
    while iter_cnt==0 || (error>=min_error && iter_cnt<=max_iter)
        pre_loss = loss; %Loss function in previous iteraton
        %==========
        %Y-Process: update auxiliary matrix
        aux = aux.*((adj*rep)./max(aux*(rep'*rep), realmin));
        %==========
        %X-Process: update representation matrix
        rep = rep.*((adj'*aux)./max(rep*(aux'*aux), realmin));
        %==========
        loss = norm(adj - aux*rep', 'fro')^2;
        error = abs(loss-pre_loss)/pre_loss;
        iter_cnt = iter_cnt+1;
        %fprintf('Init. Iteration: %d; Loss: %8.4f Error: %8.4f\n', [iter_cnt, loss, error]);
    end
    
end

function [adj_est] = AM_NMF_Dec(rep,aux_seq)
%Decoder of AM-NMF
%rep: learned shared representation matrix
%aux_seq: sequence of learned auxiliary matrices
%adj_est: prediction result

    %====================
    [win_size, ~] = size(aux_seq);
    adj_est = aux_seq{win_size}*rep';

end

function [adj_est] = AM_NMF_Dec_Ref(rep,aux_seq,theta,beta)
%Decoder of AM-NMF (refined version for unweighted graphs)
%rep: learned shared representation matrix
%aux_seq: sequence of learned auxiliary matrices
%adj_est: prediction result

    %====================
    [num_nodes, ~] = size(rep);
    [win_size, ~] = size(aux_seq);
    adj_est = zeros(num_nodes, num_nodes);
    for t=1:win_size
        adj_est = adj_est + (1-theta)^(win_size-t)*aux_seq{t}*rep';
    end
    adj_est = inv(eye(num_nodes) - beta*adj_est) - eye(num_nodes);

end