
rm(list=ls())

# data preparation ####

train <- read.csv("data/train.csv",header=TRUE,dec = ".",sep=",",stringsAsFactors=FALSE,fileEncoding="UTF8")
test <- read.csv("data/test.csv",header=TRUE,dec = ".",sep=",",stringsAsFactors=FALSE,fileEncoding="UTF8")

library(lubridate)

train$City.Group<-as.factor(train$City.Group)
test$City.Group<-as.factor(test$City.Group)

train$day<-as.numeric(day(as.POSIXlt(train$Open.Date, format="%m/%d/%Y")))
train$month<-as.numeric(month(as.POSIXlt(train$Open.Date, format="%m/%d/%Y")))
train$year<-as.numeric(year(as.POSIXlt(train$Open.Date, format="%m/%d/%Y")))
train$opendate<-as.Date(train$Open.Date, format="%m/%d/%Y")
train$diffdate<-as.numeric(as.Date("2015-01-01")-train$opendate)

test$day<-as.numeric(day(as.POSIXlt(test$Open.Date, format="%m/%d/%Y")))
test$month<-as.numeric(month(as.POSIXlt(test$Open.Date, format="%m/%d/%Y")))
test$year<-as.numeric(year(as.POSIXlt(test$Open.Date, format="%m/%d/%Y")))
test$opendate<-as.Date(test$Open.Date, format="%m/%d/%Y")
test$diffdate<-as.numeric(as.Date("2015-01-01")-test$opendate)

train$logrevenue<-log(train$revenue)
train$logdiffdate <- log(train$diffdate)
test$logdiffdate <- log(test$diffdate)

save(list=c("train","test"), file = "data.Rda")

# clean script with caret ####

library(caret)

rm(list=ls())
load("data.rda")

# data partition ####

set.seed(12345)
trainIndex <- createDataPartition(train$revenue, p = 0.80,list=FALSE)

# inject random data
set.seed(12345)
train$rndvec<-rnorm(n=nrow(train))

# nah, nevermind partifion for validation, we will use Leaderboard for that, and caret will do cross-validation
datatrain <- train
datatest  <- test

# exploratory charts ####

library(lattice)

bwplot(train$logrevenue ~ train$City.Group)
bwplot(train$logrevenue ~ train$Type)
xyplot(train$logrevenue ~ as.numeric(train$diffdate))
xyplot(train$logrevenue ~ train$P28)
xyplot(train$logrevenue ~ train$P29)

par(mfrow=c(3,3))
off <- which(names(train)=="P1")
for (i in 0:36) {
        print(xyplot(train$logrevenue ~ train[,off+i],xlab=names(train)[off+i]))
}
par(mfrow=c(1,1))

# log-log ?

hist(datatrain$diffdate)
hist(log(datatrain$diffdate))
xyplot(datatrain$diffdate ~ datatrain$logrevenue)
xyplot(log(datatrain$diffdate) ~ datatrain$logrevenue)

# CARET ####

library(doSNOW)
library(foreach)

cl<-makeCluster(3)
#cl<-makeCluster(4)
registerDoSNOW(cl)

# lasso for predictor selection

set.seed(12345)
# all P- variables and logdiffdate
fit.lasso <- train(y=datatrain$logrevenue,x=datatrain[,c(6:42,50)],
                   method="lasso",
                   trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.lasso
predictors(fit.lasso)

# chosen predictors:
#[1] "P2"          "P8"          "P17"         "P21"         "P22"         "P23"         "P27"        
#[8] "P28"         "P30"         "P34"         "P37"         "logdiffdate"

# model with random forest
set.seed(12345)
fit.rf <- train(y=datatrain$logrevenue,x=datatrain[,which(names(datatrain) %in% predictors(fit.lasso))],
                 method="rf",
                 trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.rf
varImpPlot(fit.rf$finalModel)

stopCluster(cl)

predictions<-as.data.frame(predict(fit.rf,newdata=test))
submit<-as.data.frame(cbind(test[,1],exp(predictions)))
colnames(submit)<-c("Id","Prediction")
write.csv(submit,"submission.csv",row.names=FALSE,quote=FALSE)
