a = [1:100];
b = [1:2:100];
c = [-1:0.01:1]*pi;
d = c;
d(101) = [];
e = max(sin(a),0);

A = reshape(a,[10,10]);
B = diag(a,0) + diag(a(99:-1:1),-1) + diag(a(99:-1:1),1);
C = triu(ones(10,10));
D = [cumsum(a);cumprod(a)];
E = bsxfun(@mod, a, a') == 0;
