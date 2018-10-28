d = 100;

x = rand(1,d);
y = rand(1,d);
w = rand(1,d);

disp( sqrt(x*x'));
disp( sum(x .* w)/sum(w));
disp( sqrt( (x-y) * (x-y)') );
disp( x*y' )

N  = 1000;

X = rand(d,N);
y = rand(d,1);
z = rand(d,1);
disp(sqrt(sum(X .* X)));
disp( sum(bsxfun(@times, X, w'))/sum(w) );
disp( sqrt(sum(bsxfun(@minus, X, y).^2)) );
disp( sum(bsxfun(@minus, X, y).^2) );