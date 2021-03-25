function [FPR, TPR] = ROC(Y_test,Y_test_hat)

thresholds = 0:0.001:1;
n = length(thresholds);
for i=1:n
    Y_test_asig = Y_test_hat >= thresholds(i);
    C = confusionmat(Y_test,Y_test_asig');
    C = C';
    TPR(i) = C(2,2)/(C(1,2)+C(2,2));
    specificity =  C(1,1)/(C(1,1)+C(2,1));
    FPR(i) = 1 - specificity;
end
