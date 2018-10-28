function y = task3(X, Y, k)
if nargin < 3
    k = 1;
end
[~,i] = sort(dists(X,Y),2);
y = i(:, 1:k);


