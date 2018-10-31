function y = task3(X, Y, k)
if nargin < 3
    k = 1;
end
[~,n] = size(X);
if k > log2(n)/4
    [~,i] = sort(dists(X,Y),2);
    y = i(:, 1:k);
else
    y = zeros(n,k);
    d = dists(X,Y);
    
    for i=1:(k)
        [~, id] = min(d, [], 2);
        y(:,i) = id;
        d(sub2ind(size(d), 1:length(id), id')) = inf;
    end
    
end



