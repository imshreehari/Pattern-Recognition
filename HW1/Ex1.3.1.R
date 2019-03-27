x<-rnorm(10,0,1);# X

x2=x^2;
x3=x^3;
sigma=2;#σ=0.5,2
e=rnorm(10,0,sigma);#ϵ

y=3*x+6+e;


# 用于绘图
xs<- seq(min(x)-1,max(x)+1,length.out = 1000)
xs2=xs^2;
xs3=xs^3;


# 线性
model<- lm (y~x);
plot(x,y);
abline(model);

res<-model$residuals;
rss1=sum(res^2);

# 二次
model2<- lm(y~x+x2)
ys<-predict(model2,data.frame(x=xs,x2=xs2))
plot(x,y);
lines(xs,ys);

res<-model2$residuals;
rss2=sum(res^2);

# 三次
model3<- lm(y~x+x2+x3)
ys<-predict(model3,data.frame(x=xs,x2=xs2,x3=xs3))
plot(x,y);
lines(xs,ys);

res<-model3$residuals;
rss3=sum(res^2);

