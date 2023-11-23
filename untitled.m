clear all;
global K y v

K=30;

d=linspace(-15,15,500);
x=d(1:5:end);
y=sin(x)./x;

w20=2*(rand(1,1)-0.5);
w2=2*(rand(1,K)-0.5);
w10=2*(rand(K,1)-0.5);
w1=2*(rand(K,1)-0.5);

for p=1:length(x)
    v(:,p)=tanh(w10+w1*x(:,p));
end

v(K+1,:)=1;
P = inv(v'*v+0.000001*eye(length());
wagi2=y*pinv(v);

for i=1:K
    w2(1,i)=wagi2(i);
end

w20=wagi2(K+1);
ym=w20+w2*v(1:K,:);

Eucz=(y-ym)*(y-ym)';

figure;
subplot(2,1,1)
plot(y,'.b');
hold on;
plot(ym,'or');
xlabel('x')
ylabel('y')
txt=sprintf('K=%d, Eucz=%e', K, Eucz);
title(txt);
subplot(2,1,2)
plot(y,ym,'.b');
xlabel('Dane')
ylabel('Model');








