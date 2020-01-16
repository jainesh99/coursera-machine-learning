function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
arr = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30]; 
old_pred_error = Inf;
new_C = 0;
new_sigma = 0;

for i=1:size(arr,1)
  C = arr(i);
  for j = 1:size(arr,1)
    sigma = arr(j);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    new_pred_error = mean(double(predictions ~= yval));
    
    if new_pred_error < old_pred_error
      old_pred_error = new_pred_error;
      new_C = C;
      new_sigma = sigma;
    endif
  endfor
endfor

C = new_C;
sigma = new_sigma;



% =========================================================================

end
