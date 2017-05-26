function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    
	
	cvPredictions = pval < epsilon;
	tp = sum((cvPredictions == 1) & (yval == 1));
	fp = sum((cvPredictions == 1) & (yval == 0));
	fn = sum((cvPredictions == 0) & (yval == 1));
	
	prec = tp / (tp + fp + 1e-10);
	rec = tp / (tp + fn + 1e-10);
	F1 = 2 * prec * rec / (prec + rec + 1e-10);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
