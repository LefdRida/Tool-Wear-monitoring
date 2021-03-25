%------------------------------------
%   LOGISTIC REGRESSION CLASSIFIER
%------------------------------------

% Warning:  in this project some parts of this code are written by my Teacher.

clear
clc
close all

%% STEP 1: DESCRIPTION

% Load the images and labels
load Horizontal_edges;

% Number of images of cutting edges
num_edges = length(horiz_edges);

% Number of features (in the case of ShapeFeat it is 10)
num_features = 10;

% Initialiation of matrix of descriptors. It will have a size (m x n), where
% m is the number of training patterns (i.e. elements) and n is the number 
% of features (i.e. the length of the feature vector which characterizes 
% the cutting edge).
X = zeros(num_edges, num_features);

% Describe the images of the horizontal edges by calling the fGetShapeFeat 
% function
for i=1:num_edges
    disp(['Describing image number ' num2str(i)]);
    
    % Get the i-th cutting edge
    edge = logical(horiz_edges{i}); % DON'T REMOVE
    
    % Compute the descriptors of the cutting edge usign the fGetShapeFeat
    % function
    desc_edge_i = fGetShapeFeat(edge);
    
    % Store the feature vector into the matrix X.
    X(i,:) = desc_edge_i;
end

% Create the vector of labels Y. Y(j) will store the label of the curring
% edge represented by the feature vector contained in the j-th row of the 
% matrix X.
% The problem will be binary: class 0 correspond to a low or medium wear
% level, whereas class 1 correspond to a high wear level.
Y = labels(:,2)'>=2;

save('tool_descriptors.mat', 'X', 'Y');

%% STEP 2: CLASSIFICATION

%% PRELIMINARY: LOAD DATASET AND PARTITION TRAIN-TEST SETS 

clear
clc
close all

load tool_descriptors;
% X contains the training patterns (dimension 10)
% Y contains the class label of the patterns (i.e. Y(37) contains the label
% of the pattern X(37,:) ).

% Number of patterns (i.e., elements) and variables per pattern in this
% dataset
[num_patterns, num_features] = size(X);

% Normalization of the data
mu_data = mean(X);
std_data = std(X);
X = (X-mu_data)./std_data;

% Parameter that indicates the percentage of patterns that will be used for
% the training
p_train = 0.7;

% SPLIT DATA INTO TRAINING AND TEST SETS

num_patterns_train = round(p_train*num_patterns);

indx_permutation = randperm(num_patterns);

indxs_train = indx_permutation(1:num_patterns_train);
indxs_test = indx_permutation(num_patterns_train+1:end);

X_train = X(indxs_train, :);
Y_train = Y(indxs_train);

X_test= X(indxs_test, :);
Y_test = Y(indxs_test);


%% PART 2.1: TRAINING OF THE CLASSIFIER AND CLASSIFICATION OF THE TEST SET

% Learning rate. Change it accordingly, depending on how the cost function
% evolve along the iterations
alpha = 0.01;

% The function fTrain_LogisticReg implements the logistic regression 
% classifier. Open it and complete the code.

% TRAINING
theta = fTrain_LogisticReg(X_train, Y_train, alpha);

% CLASSIFICATION OF THE TEST SET
Y_test_hat = fClassify_LogisticReg(X_test, theta);

% Assignation of the class
Y_test_asig = Y_test_hat>=0.5;

%% PART 2.2: PERFORMANCE OF THE CLASSIFIER: CALCULATION OF THE ACCURACY AND FSCORE

% Show confusion matrix
figure;
plotconfusion(Y_test, Y_test_asig');


% ACCURACY AND F-SCORE
% ==========================================================
C = confusionmat(Y_test,Y_test_asig');
C = C';
accuracy = (C(1,1) + C(2,2))/sum(sum(C));
precision  = C(2,2)/(C(2,2)+C(2,1));
recall = C(2,2)/(C(1,2)+C(2,2));
FScore = 2*precision*recall/(precision + recall);

figure;
I = 0:0.1:1;
[FPR, TPR] = ROC(Y_test,Y_test_hat);
hold all
plot(FPR,TPR)
plot(I,I)
xlabel("FRP(1 - specificity)")
ylabel("TPR")
legend('ROC')
auc = AUC(FPR,TPR);
% ============================================================

fprintf('\n******\nAccuracy = %1.4f%% (classification)\n', accuracy*100);
fprintf('\n******\nFScore = %1.4f (classification)\n', FScore);
fprintf('\n******\nAUC = %1.4f (classification)\n', auc);


%% SVM  

%==================== the optimal hyperparameters ==========================================

Md1 = fitcsvm(X, Y, 'kernelFunction','polynomial',...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions', ...
    struct('AcquisitionFunctionName',...
    'expected-improvement-plus','ShowPlots',true));
%==================== test the model =======================================================


Mdfinal = fitcsvm(X_train, Y_train, 'kernelFunction','polynomial',...
    'KernelScale',  8.7883  ,'BoxConstraint',338.21);

[prediction,score] = predict(fitPosterior(compact(Mdfinal),X_train,Y_train),X_test);

%=====================performance==========================================================

% Show confusion matrix
figure;
plotconfusion(Y_test, Y_test_asig');

Y_test_asig = prediction;
Y_test_hat = score(:,2);
C = confusionmat(Y_test,Y_test_asig');
C = C';
accuracy = (C(1,1) + C(2,2))/sum(sum(C));
precision  = C(2,2)/(C(2,2)+C(2,1));
recall = C(2,2)/(C(1,2)+C(2,2));
FScore = 2*precision*recall/(precision + recall);

figure;
I = 0:0.1:1;
[FPR, TPR] = ROC(Y_test,Y_test_hat);
hold all
plot(FPR,TPR)
plot(I,I)
xlabel("FRP(1 - specificity)")
ylabel("TPR")
legend('ROC')
auc = AUC(FPR,TPR);
% ============================================================

fprintf('\n******\nAccuracy = %1.4f%% (classification)\n', accuracy*100);
fprintf('\n******\nFScore = %1.4f (classification)\n', FScore);
fprintf('\n******\nAUC = %1.4f (classification)\n', auc);

%% Neural Network 
load tool_descriptors;
inputs = X';
targets = Y;

% Create a Fitting Network
hiddenLayerSize = [64, 32, 16, 8];
net = fitnet(hiddenLayerSize);


% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the Network
[net,tr] = train(net,inputs,targets);

% Test the Network
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);

plotconfusion(targets,outputs)
 
