# 读入数据
testSet<-read.table("prostate_test.txt",head=TRUE);
trainSet<-read.table("prostate_train.txt",head=TRUE);

# 回归
model<-lm(lpsa~lcavol+lweight+lbph+svi,data = trainSet);
predicted<-predict.lm(model,newdata=testSet);

# 模型评价

par(mfrow=c(2,2))
plot(model)

summary(model)

# 交叉项
model1<-lm(lpsa~lcavol+lweight+lbph+lbph:lweight+svi,data = trainSet)
summary(model1)