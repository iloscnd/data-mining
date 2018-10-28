function y = dists(X,Y)
%[d1, N] = size(X);
%[d2, M] = size(Y);
%if d1 ~= d2
%    error("Dimensions do not match")
%end
%d = d1;
y = bsxfun(@minus, X, permute(Y, [1,3,2])); % X: dxN Y:dxM -> permute: dx1XM y:dxNxM
y = squeeze(sqrt(sum(y.^2, 1))); % was 1 x N x M
    