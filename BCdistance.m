function BC =  BCdistance(X,Y)
% X is the data matrix associated with the first distribution. The rows
% are samples and the columns are features.
% Y is the data matrix associated with the second distribution. The rows
% are samples and the columns are features.
% BC is a measure of distance between the data sets X and Y. When X and Y
% are completely overlapping, this distance measure is 0. When X and Y and
% completely separable, this distance measure is 1.

mu1 = mean(X)';
mu2 = mean(Y)';
S1 = cov(X);
S2 = cov(Y);
S = (S1+S2)/2;

Db = (mu1 - mu2)'*inv(S)*(mu1-mu2) + 0.5*log(det(S)/(sqrt(det(S1))*sqrt(det(S2))));

BC = 1-exp(-Db);


