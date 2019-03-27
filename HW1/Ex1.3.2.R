x<-rnorm(10,0,1);# X

x2=x^2;
x3=x^3;
sigma=2;#σ=0.5,2
e=rnorm(10,0,sigma);#ϵ

y=3*x+6+e;

# 用于预测
xp<- rnorm(100,0,1);
e=rnorm(100,0,sigma);#ϵ
xp2=xp^2;
xp3=xp^3;
ys=3*xp+6+e;

# 线性
model<- lm (y~x);
yp<-predict(model,data.frame(x=xp));

plot(xp,ys);
points(xp,yp,pch=17,col="red")
sum1=sum((ys-yp)^2)

# 二次
model2<- lm(y~x+x2)
yp<-predict(model2,data.frame(x=xp,x2=xp2))
plot(xp,ys)
points(xp,yp,pch=17,col="red")
sum2=sum((ys-yp)^2)

# 三次
model3<- lm(y~x+x2+x3)
yp<-predict(model3,data.frame(x=xp,x2=xp2,x3=xp3))
plot(xp,ys)
points(xp,yp,pch=17,col="red")
sum3=sum((ys-yp)^2)


