function p = psnr(x,y, vmax)

m1 = max( abs(x(:)) );
m2 = max( abs(y(:)) );
vmax = 1;

d = mean( (x(:)-y(:)).^2 );
p = 10*log10( vmax^2/d );