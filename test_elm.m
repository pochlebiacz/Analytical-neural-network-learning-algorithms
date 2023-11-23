clear all;
global K y v

K=30;

zadanie=2;
if zadanie==1
    x=linspace(-15,15,500);
    %y=sin(2*x)./exp(x/5);
    y=sin(x)./x;
    ilwe=1;
end

if zadanie==2
    ilpktx=20; ilpkty=20;
    xmin=-2; xmax=2; ymin=-2; ymax=2;
    dx=(xmax-xmin)/(ilpktx-1); dy=(ymax-ymin)/(ilpkty-1);
    l=1;
    for i=1:ilpktx
        for j=1:ilpkty
            x(1,l)=xmin+(i-1)*dx;
            x(2,l)=ymin+(j-1)*dy;
            y(l)=x(1,l)*exp(-x(1,l)^2-x(2,l)^2);
            l=l+1;
        end
    end
    ilwe=2;
end

P=length(y);

w20=2*(rand(1,1)-0.5);
w2=2*(rand(1,K)-0.5);
w10=2*(rand(K,1)-0.5);
w1=2*(rand(K,ilwe)-0.5);
%pause

%w20+w2*tanh(w10+w1*x)
for p=1:length(x)
    v(:,p)=tanh(w10+w1*x(:,p));
end
%ym=w20+w2*v;
%size(ym);
%E0=(y-ym)*(y-ym)';

v(K+1,:)=1;

wagi2=y*pinv(v);

for i=1:K
    w2(1,i)=wagi2(i);
end
w20=wagi2(K+1);
ym=w20+w2*v(1:K,:);

Eucz=(y-ym)*(y-ym)';

if zadanie==1
    figure;
    subplot(2,1,1)
    plot(y,'.b');
    hold on;
    plot(ym,'or');
    xlabel('x')
    ylabel('y')
    txt=sprintf('K=%d, Eucz=%e',K,Eucz);
    title(txt);
    subplot(2,1,2)
    plot(y,ym,'.b');
    xlabel('Dane')
    ylabel('Model');
end
if zadanie==2
    figure;
    plot3(x(1,:),x(2,:),y,'.b')
    hold on;
    plot3(x(1,:),x(2,:),ym,'or')
end



% weryfikacja
clear x y v ym
if zadanie==1
    x=linspace(-15,15,300);
    %y=sin(2*x)./exp(x/5);
    y=sin(x)./x;
end

if zadanie==2
    ilpktx=15; ilpkty=15;
    xmin=-2; xmax=2; ymin=-2; ymax=2;
    dx=(xmax-xmin)/(ilpktx-1); dy=(ymax-ymin)/(ilpkty-1);
    l=1;
    for i=1:ilpktx
        for j=1:ilpkty
            x(1,l)=xmin+(i-1)*dx;
            x(2,l)=ymin+(j-1)*dy;
            y(l)=x(1,l)*exp(-x(1,l)^2-x(2,l)^2);
            l=l+1;
        end
    end
    ilwe=2;
end

for p=1:length(x)
    v(:,p)=tanh(w10+w1*x(:,p));
end
ym=w20+w2*v;

v(K+1,:)=1;

wagi2=y*pinv(v);

ym=w20+w2*v(1:K,:);

Ewer=(y-ym)*(y-ym)';

if zadanie==1
    figure;
    subplot(2,1,1)
    plot(y,'.b');
    hold on;
    plot(ym,'or');
    xlabel('x')
    ylabel('y')
    txt=sprintf('K=%d, Ewer=%e',K,Ewer);
    title(txt);
    subplot(2,1,2)
    plot(y,ym,'.b');
    xlabel('Dane')
    ylabel('Model');
end
if zadanie==2
    figure;
    plot3(x(1,:),x(2,:),y,'.b')
    hold on;
    plot3(x(1,:),x(2,:),ym,'or')
end






