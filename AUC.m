function auc = AUC(FPR,TPR)
auc  = 0;
n = length(FPR);
for i=1:n-1 
    auc = auc + (FPR(i) - FPR(i+1))*TPR(i);
end
