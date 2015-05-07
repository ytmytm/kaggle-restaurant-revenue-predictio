
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

# CARET ####

library(doSNOW)
library(foreach)

cl<-makeCluster(3)
#cl<-makeCluster(4)
registerDoSNOW(cl)

names(datatrain)[c(4,6:42,45,48,50)]
set.seed(12345)
fit.rf <- train(y=datatrain$logrevenue,x=datatrain[,c(4,6:42,45,48,50)],
                method="rf",
                trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.rf
predictors(fit.rf)
varImpPlot(fit.rf$finalModel)

set.seed(12345)
fit.glm <- train(y=datatrain$logrevenue,x=datatrain[,c(4,6:42,45,48,50)],
                method="glm",
                trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.glm
predictors(fit.glm)

set.seed(12345)
fit.gbm <- train(y=datatrain$logrevenue,x=datatrain[,c(4,6:42,45,48)],
                method="gbm",
                trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.gbm
predictors(fit.gbm)

set.seed(12345)
fit.rf2 <- train(y=datatrain$logrevenue,x=datatrain[,which(names(datatrain) %in% predictors(fit.gbm))],
                method="rf",
                trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.rf2
varImpPlot(fit.rf2$finalModel)


fit.svm <- train(y=datatrain$logrevenue,x=datatrain[,c(6:42,45,48)],
                method="svmLinear",
                trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.svm

fit.pls <- train(y=datatrain$logrevenue,x=datatrain[,c(4,6:42,45,48)],
                 method="pls",
                 trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.pls

fit.plscs <- train(y=datatrain$logrevenue,x=datatrain[,c(4,6:42,45,48)],
                 method="pls",
                 preproc=c("center","scale"),
                 trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.plscs

set.seed(12345)
fit.lasso <- train(y=datatrain$logrevenue,x=datatrain[,c(6:42,45,48)],
                   method="lasso",
                   trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.lasso
predictors(fit.lasso)

set.seed(12345)
fit.rf3 <- train(y=datatrain$logrevenue,x=datatrain[,which(names(datatrain) %in% predictors(fit.lasso))],
                 method="rf",
                 trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.rf3
varImpPlot(fit.rf3$finalModel)

set.seed(12345)
fit.gamboost <- train(y=datatrain$logrevenue,x=datatrain[,c(6:42,45,48)],
                   method="gamboost",
                   trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.gamboost

set.seed(12345)
fit.glmboost <- train(y=datatrain$logrevenue,x=datatrain[,c(4,6:42,45,48)],
                      method="glmboost",
                      trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.glmboost

set.seed(12345)
fit.cubist <- train(y=datatrain$logrevenue,x=datatrain[,c(6:42,45,48)],
                      method="cubist",
                      trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.cubist

set.seed(12345)
fit.elm <- train(y=datatrain$logrevenue,x=datatrain[,c(6:42,45,48)],
                    method="elm",
                    trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.elm

set.seed(12345)
fit.glmnet <- train(y=datatrain$logrevenue,x=datatrain[,c(6:42,45,48)],
                    method="glmnet",
                    trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.glmnet

set.seed(12345)
fit.kknn <- train(y=datatrain$logrevenue,x=datatrain[,c(6:42,45,48)],
                    method="kknn",
                    trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.kknn

set.seed(12345)
fit.knn <- train(y=datatrain$logrevenue,x=datatrain[,c(6:42,45,48)],
                  method="knn",
                  trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.knn

set.seed(12345)
fit.lmstep <- train(y=datatrain$logrevenue,x=datatrain[,c(6:42,45,48)],
                 method="leapForward",
                 trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.lmstep

set.seed(12345)
fit.lmforward <- train(y=datatrain$logrevenue,x=datatrain[,c(6:42,45,48)],
                    method="leapSeq",
                    trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.lmforward

# teraz zmieniamy typy zmiennych

datatrain$month<-as.factor(datatrain$month)

set.seed(12345)
fit.rf4 <- train(y=datatrain$logrevenue,x=datatrain[,which(names(datatrain) %in% unique(c("P3","P5","P21",predictors(fit.lasso))))],
                 method="rf",
                 trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.rf4
varImpPlot(fit.rf4$finalModel)

preds<-data.frame(var=rownames(fit.rf4$finalModel$importance),imp=fit.rf4$finalModel$importance)
predsnames<-rownames(preds[order(preds$IncNodePurity,decreasing = TRUE),])[1:10]

set.seed(12345)
fit.rf5 <- train(y=datatrain$logrevenue,x=datatrain[,which(names(datatrain) %in% predsnames)],
                 method="rf",
                 trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.rf5
varImpPlot(fit.rf5$finalModel)

stopCluster(cl)

plot(datatest$logrevenue,datatest$logrevenue,xlim=c(14,18),ylim=c(14,18))
lines(c(0,20),c(0,20),col="gray")
points(datatest$logrevenue,predict(fit.rf2,datatest),col="yellow")
points(datatest$logrevenue,predict(fit.rf3,datatest),col="purple")

points(datatest$logrevenue,predict(fit.kknn,datatest[,c(6:42,45,48)]),col="purple")

points(datatest$logrevenue,predict(fit.pls,datatest),col="yellow")
points(datatest$logrevenue,predict(fit.plscs,datatest),col="purple")
points(datatest$logrevenue,predict(fit.svm,datatest[,c(6:42)]),col="red")
points(datatest$logrevenue,predict(fit.gbm,datatest[,c(6:42)]),col="blue")
points(datatest$logrevenue,predict(fit.glm,datatest),col="green")
points(datatest$logrevenue,predict(fit.lasso,datatest[,c(6:42,45,48)]),col="purple")

# submissions ####

predictions.0<-as.data.frame(predict(fit.rf2,newdata=test))
predictions.1<-as.data.frame(predict(fit.rf3,newdata=test))
predictions.2<-as.data.frame(predict(fit.gbm,newdata=test[,which(names(test) %in% names(datatest)[c(4,6:42,45,48)])]))

predictions.3<-as.data.frame(predict(fit.lmstep,newdata=test[,which(names(test) %in% names(datatest)[c(6:42,45,48)])]))
predictions.4<-as.data.frame(predict(fit.cubist,newdata=test[,which(names(test) %in% names(datatest)[c(6:42,45,48)])]))
predictions.5<-as.data.frame(predict(fit.lasso,newdata=test[,which(names(test) %in% names(datatest)[c(6:42,45,48)])]))
predictions.6<-as.data.frame(predict(fit.knn,newdata=test[,which(names(test) %in% names(datatest)[c(6:42,45,48)])]))

predictions.7<-(predictions.0+predictions.1+predictions.5+predictions.6)/4

predictions.8<-as.data.frame(predict(fit.rf4,newdata=test))
predictions.9<-as.data.frame(predict(fit.rf5,newdata=test))
predictions.10<-(predictions.1+predictions.8+predictions.9)/3


submit.0<-as.data.frame(cbind(test[,1],exp(predictions.0)))
submit.1<-as.data.frame(cbind(test[,1],exp(predictions.1)))

submit.2<-as.data.frame(cbind(test[,1],exp(predictions.2)))
submit.3<-as.data.frame(cbind(test[,1],exp(predictions.3)))
submit.4<-as.data.frame(cbind(test[,1],exp(predictions.4)))

submit.5<-as.data.frame(cbind(test[,1],exp(predictions.5)))
submit.6<-as.data.frame(cbind(test[,1],exp(predictions.6)))
submit.7<-as.data.frame(cbind(test[,1],exp(predictions.7)))

submit.8<-as.data.frame(cbind(test[,1],exp(predictions.8)))
submit.9<-as.data.frame(cbind(test[,1],exp(predictions.9)))
submit.10<-as.data.frame(cbind(test[,1],exp(predictions.10)))


colnames(submit.0)<-c("Id","Prediction")
colnames(submit.1)<-c("Id","Prediction")
colnames(submit.2)<-c("Id","Prediction")
colnames(submit.3)<-c("Id","Prediction")
colnames(submit.4)<-c("Id","Prediction")
colnames(submit.5)<-c("Id","Prediction")
colnames(submit.6)<-c("Id","Prediction")
colnames(submit.7)<-c("Id","Prediction")
colnames(submit.8)<-c("Id","Prediction")
colnames(submit.9)<-c("Id","Prediction")
colnames(submit.10)<-c("Id","Prediction")

write.csv(submit.0,"submissions/rf2.2.csv",row.names=FALSE,quote=FALSE)
write.csv(submit.1,"submissions/rf3.1.csv",row.names=FALSE,quote=FALSE)
write.csv(submit.2,"submissions/gbm.3.csv",row.names=FALSE,quote=FALSE)
write.csv(submit.3,"submissions/lmstep.0.csv",row.names=FALSE,quote=FALSE)
write.csv(submit.4,"submissions/cubist.0.csv",row.names=FALSE,quote=FALSE)
write.csv(submit.5,"submissions/lasso.0.csv",row.names=FALSE,quote=FALSE)
write.csv(submit.6,"submissions/knn.1.csv",row.names=FALSE,quote=FALSE)
write.csv(submit.7,"submissions/ensemble.0.csv",row.names=FALSE,quote=FALSE)
write.csv(submit.8,"submissions/rf4.0.csv",row.names=FALSE,quote=FALSE)
write.csv(submit.9,"submissions/rf5.0.csv",row.names=FALSE,quote=FALSE)
write.csv(submit.10,"submissions/ensemble.1.csv",row.names=FALSE,quote=FALSE)
smoothScatter(predictions.0[,1]~predictions.1[,1])
par(mfrow=c(3,2))
smoothScatter(predictions.2[,1]~predictions.1[,1],xlim=c(14,16),ylim=c(14,16))
smoothScatter(predictions.3[,1]~predictions.1[,1],xlim=c(14,16),ylim=c(14,16))
smoothScatter(predictions.4[,1]~predictions.1[,1],xlim=c(14,16),ylim=c(14,16))
smoothScatter(predictions.5[,1]~predictions.1[,1],xlim=c(14,16),ylim=c(14,16))
smoothScatter(predictions.6[,1]~predictions.1[,1],xlim=c(14,16),ylim=c(14,16))
smoothScatter(predictions.7[,1]~predictions.1[,1],xlim=c(14,16),ylim=c(14,16))
par(mfrow=c(1,1))

# podsumowanie 25.03 ####

# najlepszy wynik dał rf związany z lasso (rf3)
# - może teraz użyć lasso bezpośrednio?
# spróbować podsumowania pca P1-P37 - ale jeśli okaże się, że to kategorie, to porażka
# widać, że kluczem do rf jest wybór zmiennych

# podsumowanie 26.03 ####

# dopasowanych wiele różnych modeli, wszystkie na pełnych danych
# różnice są nie wielkie, najlepsze rezultaty dały lmstep/lmforward, cubist, knn

# podsumowanie 27.03 ####
# rf.2 wysłane na nowo - po ustawieniu set.seed gbm zwróciło inny zestaw predyktorów
# ensemble 4 najlepszych (do tej pory) modeli

# podsumowanie 28.03 ####
# rezygnacja z wlasnego train/test, uruchomienie modeli na całym zbiorze train, testem będzie kaggle
# najlepsze nadal rf2/rf3/gbm, ale mają inny zbiór predyktorów

# podsumowanie 29.03 ####
# uzyc lm ~ diffdate, modelami (rf i inne) wyjaśniać resiuda

datatrain$logdiffdate <- log(datatrain$diffdate)
set.seed(12345)
fit.lasso <- train(y=datatrain$logrevenue,x=datatrain[,c(6:42,45,48)],
                   method="lasso",
                   trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.lasso
predictors(fit.lasso)

set.seed(12345)
fit.rf3 <- train(y=datatrain$logrevenue,x=datatrain[,which(names(datatrain) %in% predictors(fit.lasso))],
                 method="rf",
                 trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.rf3
varImpPlot(fit.rf3$finalModel)


# loglog+residua

fit.lmday <- lm(logrevenue ~ diffdate,data=datatrain)
summary(fit.lmday)
datatrain$residlm <- fit.lmday$residuals

set.seed(12345)
fit.lasso <- train(y=datatrain$residlm,x=datatrain[,c(6:42,45)],
                   method="lasso",
                   trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.lasso
predictors(fit.lasso)

set.seed(12345)
fit.rf6 <- train(y=datatrain$residlm,x=datatrain[,which(names(datatrain) %in% predictors(fit.lasso))],
                 method="rf",
                 trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.rf6
varImpPlot(fit.rf6$finalModel)

set.seed(12345)
fit.rf7 <- train(y=datatrain$residlm,x=datatrain[,c(6:42,45)],
                 method="rf",
                 trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.rf7
varImpPlot(fit.rf7$finalModel)

test$predlm <- predict(fit.lmday,newdata=test)

predictions.11<-as.data.frame(predict(fit.rf6,newdata=test))
predictions.12<-as.data.frame(predict(fit.rf7,newdata=test))
predictions.13<-as.data.frame(predict(fit.lasso,newdata=test[,which(names(test) %in% names(datatest)[c(6:42,45)])]))

submit.11<-as.data.frame(cbind(test[,1],exp(test$predlm+predictions.11)))
submit.12<-as.data.frame(cbind(test[,1],exp(test$predlm+predictions.12)))
submit.13<-as.data.frame(cbind(test[,1],exp(test$predlm+predictions.13)))

colnames(submit.11)<-c("Id","Prediction")
colnames(submit.12)<-c("Id","Prediction")
colnames(submit.13)<-c("Id","Prediction")

write.csv(submit.11,"submissions/rf6.0.csv",row.names=FALSE,quote=FALSE)
write.csv(submit.12,"submissions/rf7.0.csv",row.names=FALSE,quote=FALSE)
write.csv(submit.13,"submissions/lasso.1.csv",row.names=FALSE,quote=FALSE)

# plan 03.04 ####
# log-log ?

hist(datatrain$diffdate)
hist(log(datatrain$diffdate))
xyplot(datatrain$diffdate ~ datatrain$logrevenue)
xyplot(log(datatrain$diffdate) ~ datatrain$logrevenue)

datatrain$logdiffdate <- log(datatrain$diffdate)

fit.lmday <- lm(logrevenue ~ logdiffdate,data=datatrain)
summary(fit.lmday)
datatrain$residlm <- fit.lmday$residuals

# lasso for predictor selection

set.seed(12345)
fit.lasso <- train(y=datatrain$residlm,x=datatrain[,c(6:42,51)],
                   method="lasso",
                   trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.lasso
predictors(fit.lasso)

set.seed(12345)
fit.rf8 <- train(y=datatrain$residlm,x=datatrain[,which(names(datatrain) %in% predictors(fit.lasso))],
                 method="rf",
                 trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.rf8
varImpPlot(fit.rf8$finalModel)

# lasso for predictor selection

set.seed(12345)
fit.lasso <- train(y=datatrain$logrevenue,x=datatrain[,c(6:42,51)],
                   method="lasso",
                   trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.lasso
predictors(fit.lasso)

set.seed(12345)
fit.rf9 <- train(y=datatrain$logrevenue,x=datatrain[,which(names(datatrain) %in% predictors(fit.lasso))],
                 method="rf",
                 trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.rf9
varImpPlot(fit.rf9$finalModel)

set.seed(12345)
fit.rf10 <- train(y=datatrain$residlm,x=datatrain[,c(6:42,45,51)],
                 method="rf",
                 trControl=trainControl(number=5,method="cv",verboseIter=TRUE,allowParallel=TRUE))
fit.rf10
varImpPlot(fit.rf10$finalModel)

test$predlm <- predict(fit.lmday,newdata=test)
test$logdiffdate <- log(test$diffdate)


predictions.14<-as.data.frame(predict(fit.rf8,newdata=test))
predictions.15<-as.data.frame(predict(fit.rf9,newdata=test))
predictions.16<-as.data.frame(predict(fit.rf10,newdata=test))

submit.14<-as.data.frame(cbind(test[,1],exp(test$predlm+predictions.14)))
submit.15<-as.data.frame(cbind(test[,1],exp(predictions.15)))
submit.16<-as.data.frame(cbind(test[,1],exp(test$predlm+predictions.16)))

colnames(submit.14)<-c("Id","Prediction")
colnames(submit.15)<-c("Id","Prediction")
colnames(submit.16)<-c("Id","Prediction")

write.csv(submit.14,"submissions/rf8.0.csv",row.names=FALSE,quote=FALSE)
write.csv(submit.15,"submissions/rf9.0.csv",row.names=FALSE,quote=FALSE)
write.csv(submit.16,"submissions/rf10.csv",row.names=FALSE,quote=FALSE)

# podsumowanie 03.04 ####
# rf9 dał znacząco lepszy wynik - metoda analogiczna do poprzedniego rekordzisty (rf3)
# tylko teraz jeszcze z log(diffdates)
# całkiem nieźle wypadł też rf8 - dopasowanie do residuów po dopasowaniu log-log
