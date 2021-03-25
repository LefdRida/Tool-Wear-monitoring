function theta = fTrain_LogisticReg(X_train, Y_train, alpha)
% This function implements the training of a logistic regression classifier
% using the training data (X_train) and its classes (Y_train).  
%
% INPUT
%   - X_train: Matrix with dimensions (m x n) with the training data, where
%   m is the number of training patterns (i.e. elements) and n is the
%   number of features (i.e. the length of the feature vector which
%   characterizes the object).
%   - Y_train: Vector that contains the classes of the training patterns.
%   Its length is n.
%   - alpha: Learning rate for the gradient descent algorithm.
%
% OUTPUT
%   theta: Vector with length n (i.e, the same length as the number of
%   features on each pattern). It contains the parameters theta of the
%   hypothesis function obtained after the training.
%

    % CONSTANTS
    % =================
    VERBOSE = true;
    max_iter = 100; % Try with a different number of iterations
    % =================

    % Number of training patterns.
    m = size(X_train,1);

    % Allocate space for the outputs of the hypothesis function for each
    % training pattern
    h_train = zeros(1,m);
    
    % Allocate spaces for the values of the cost function on each iteration
    J = zeros(1,max_iter);
    
    % Initialize the vector to store the parameters of the hypothesis
    % function
    theta = zeros(1, 1+size(X_train,2));

% *************************************************************************
% CALCULATE THE VALUE OF THE COST FUNCTION FOR THE INITIAL THETAS
    
    total_cost = 0;
    for i=1:m
        x_i = [1, X_train(i, :)]; 
        
        % Expected output (i.e. result of the sigmoid function) for i-th pattern
        % ============================================================
        h_train(i) = fun_sigmoid(theta,x_i);
        % ============================================================
        
        % Calculate the cost for the i-the pattern and add the cost of the last patterns
        % ============================================================
        total_cost = total_cost + fCalculateCostLogReg(Y_train(i),h_train(i));
        % ============================================================

    end
    
    % b. Calculate the total cost
    % ============================================================
    total_cost = -total_cost/m;
    % ============================================================

% *************************************************************************
% GRADIENT DESCENT ALGORITHM TO UPDATE THE THETAS
    for num_iter=1:max_iter
        
        % *********************
        for i=1:m
            x_i = [1, X_train(i,:)]; 
            % ============================================================
             h_train(i) = fun_sigmoid(theta,x_i);
            % ============================================================

        end
        
        % *********************
        % ============================================================
        gradJ = (h_train - Y_train)*[ones(m,1),X_train];
        theta = theta - alpha/m*gradJ;
        % ============================================================

        
        % *********************
  
        % ============================================================
        J(num_iter) = -1/m*sum(fCalculateCostLogReg(Y_train,h_train));
        % ============================================================

    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if VERBOSE
        figure;
        plot(1:max_iter, J, '-')
        title(['Cost function over the iterations with alfa=', num2str(alpha)]);
        xlabel('Number of iterations');
        ylabel('Cost J');
    end

end