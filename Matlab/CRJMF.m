function [adj_est] = CRJMF(col,att,prox,hid_dim,alpha,lambd,min_error,max_iter,num_run)
%Function to implement CRJMF
%col: adjaceny matrix of the collapsed graph
%att: node attribute matrix
%prox: graph proximity matrix
%hid_dim: dimensionality of latent space
%alpha, lambd: hyper-parameters
%max_iter: maximum number of iterations
%min_error: minimum relative error to determine convergence
%num_run: number of independent runs
%adj_est: prediction result

    %====================
    %Random initialization w/ multiple independent runs
    [num_nodes, feat_dim] = size(att);
    U_init = rand(num_nodes, hid_dim);
    S_init = rand(hid_dim, hid_dim);
    V_init = rand(feat_dim, hid_dim);
    %Encoder
    [U,S,~,loss] = CRJMF_Enc(col,att,prox,U_init,S_init,V_init,alpha,lambd,min_error,max_iter);
    for r=2:num_run
        U_init = rand(num_nodes, hid_dim);
        S_init = rand(hid_dim, hid_dim);
        V_init = rand(feat_dim, hid_dim);
        [c_U,c_S,~,c_loss] = CRJMF_Enc(col,att,prox,U_init,S_init,V_init,alpha,lambd,min_error,max_iter);
        if c_loss<loss
            loss = c_loss;
            U = c_U;
            S = c_S;
        end
    end
    %Decoder
    adj_est = CRJMF_Dec(U,S);
end

function [U,S,V,loss] = CRJMF_Enc(col,att,prox,U,S,V,alpha,lambd,min_error,max_iter)
%Encoder of CRJMF
%col: adjaceny matrix of the collapsed graph
%att: node attribute matrix
%prox: graph proximity matrix
%U, S, V: matrices (latent representations) to be optimized
%alpha, lambd: hyper-parameters
%min_error: minimum relative error to determien the convergence
%max_iter: maximum number iterations
%loss: loss function

    %====================
    prox_deg = diag(sum(prox, 1)); %Degree diag. matrix of graph prox. matrix
    %Compute loss function
    loss = norm(col - U*S*U');
    loss = loss + alpha*norm(att - U*V');
    loss = loss + lambd*trace(U'*(prox_deg - prox)*U);
    %==========
    iter_cnt = 0; %Counter of iteration
    error = 0; %Relative error of loss function
    while iter_cnt==0 || (error>=min_error && iter_cnt<=max_iter)
        pre_loss = loss; %Loss function in previous iteraton
        %==========
        %Update U
        numer = col*U*S' + col'*U*S + alpha*att*V + lambd*prox*U;
        denom = U*S*(U'*U)*S' + U*S'*(U'*U)*S + alpha*U*(V'*V) + lambd*prox_deg*U;
        U = U.*sqrt(numer./max(denom, realmin));
        %==========
        %Update S
        numer = U'*col*U;
        denom = (U'*U)*S*(U'*U);
        S = S.*sqrt(numer./max(denom, realmin));
        %==========
        %Update V
        numer = att'*U;
        denom = V*(U'*U);
        V = V.*sqrt(numer./max(denom, realmin));
        
        %==========
        %Compute loss function
        loss = norm(col - U*S*U');
        loss = loss + alpha*norm(att - U*V');
        loss = loss + lambd*trace(U'*(prox_deg - prox)*U);
        %==========
        error = abs(loss-pre_loss)/pre_loss;
        iter_cnt = iter_cnt+1;
        %fprintf('Iteration: %d; Loss: %8.4f Error: %8.4f\n', [iter_cnt, loss, error]);
    end

end

function [adj_est] = CRJMF_Dec(U,S)
%Decoder of CRJMF
%%U, S: learned latent representations
%adj_est: prediction result

    %====================
    adj_est = U*S*U';

end

