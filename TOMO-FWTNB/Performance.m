function [ PD,PF,Precision, F1,AUC,Accuracy,G_measure,MCC ] = Performance(actual_label, probPos)
%PERFORMANCE Summary of this function goes here
%   Detailed explanation goes here
% INPUTS:
%   (1) actual_label - The actual label, a column vetor, each row is an instance's class label.
%   (2) probPos - The probability of being predicted as postive class.
% OUTPUTS:
%   PF,PF,..,MCC - A total of eight performance measures.

if numel(unique(actual_label)) < 1
    error('Please make sure that the true label ''actual_label'' must has at least two different kinds of values.');
    
end

predict_label = double(probPos>=0.5);

cf=confusionmat(actual_label,predict_label);
TP=cf(2,2);
TN=cf(1,1);
FP=cf(1,2);
FN=cf(2,1);

Accuracy = (TP+TN)/(FP+FN+TP+TN);
PD=TP/(TP+FN);
PF=FP/(FP+TN);
Precision=TP/(TP+FP);
F1=2*Precision*PD/(Precision+PD);
[X,Y,T,AUC]=perfcurve(actual_label, probPos, '1');% 
G_measure = (2*PD*(1-PF))/(PD+1-PF);
MCC = (TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)));

end

