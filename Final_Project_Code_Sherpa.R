library(tidyverse)
library(broom)
library(janitor)
library(skimr)
library(dplyr)
library(readr)
library(mosaic)
library(FactoMineR)
library(GGally)
library(leaflet)
library(car)
library(ggfortify)
library(corrplot)
library(rsample)
library(caret)
library(patchwork) #to combine graphs together
library(class)
library(randomForest)
library(kernlab)
library(ROCR)
library(adabag)
library(rpart.plot)
library(party)
library(pROC)
library(gbm)
library(ada)
library(xgboost)

####################################    Loading Dataset    ##########################################

Churn_Modelling <- read.csv("~/Syracuse Courses/Third Semester/PredictiveAnalytics/project/Churn_Modelling.csv")
str(Churn_Modelling)
summary(Churn_Modelling)
#data types
sapply(Churn_Modelling, class)
#checking for class imbalance
cbind(freq=table(Churn_Modelling$Exited), percentage=prop.table(table(Churn_Modelling$Exited))*100)
# the dataset is balanced
######################################  Corelation   ############################################
#convert variables to numeric to check corelation
churn_new = data.matrix(Churn_Modelling)
str(churn_new)

complete_cases <- complete.cases(churn_new)
churn_new = cor(churn_new[complete_cases,4:14])
library(reshape2)
cormap<-cor(churn_new)
cormap_melted<-melt(cormap)
#creating heatmap
ggplot(data = cormap_melted, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()+scale_fill_gradient2(low="red",high="blue",mid="white")+theme(axis.text.x = element_text(angle = 90))

## sorting correlation in descending order
cormap_melted_Death<-cormap_melted[cormap_melted$Var2=='Exited',]
cormap_melted_Death$value<-abs(cormap_melted_Death$value)
cormap_melted_Death <-cormap_melted_Death %>%
  arrange(desc(value))
cormap_melted_Death

#### Now we choose the top 8 variables with cor above 0.1
#### Age,isactive member,gender,numofproducts,creditscore,balance,Tenure,hascrcard
### maybe we won't consider balance, since it highly corelated with num of producs
cor(Churn_Modelling$NumOfProducts,Churn_Modelling$Balance) #-0.3

######################################  Data Wrangling   ############################################
########choose variables to be in model and convert variables into correct data type
dataset = Churn_Modelling[,-c(1,2,3,5,13)] # geography and salary 
                                        # is removed since it is least corelated
str(dataset)
dataset <- Churn_Modelling %>%
  summarise(Exited = as.factor(Exited),
            Gender = as.factor(Gender),
            NumOfProducts = as.factor(NumOfProducts),
            HasCrCard = as.factor(HasCrCard),
            IsActiveMember = as.factor(IsActiveMember),
            Age,CreditScore,Tenure,Balance,EstimatedSalary)

y=dataset$Exited
cbind(freq=table(y), percentage=prop.table(table(y))*100) # proportion of values

######################################  Data Partition   ############################################
ind <- sample(2, nrow(dataset), replace = TRUE, prob = c(0.75, 0.25))
train <- dataset[ind==1,]
test <- dataset[ind==2,]
#remove these variables since they have low correlation with Y and 
# are not  significant as proved below in log reg(creditcard,salary)
#it also leads to better accuracy

######################################  Model baseline   ############################################
summary(train$Exited)
(5985/(5985+1528))*100
#79.66% for predicting not exiting

(1528/(5985+1528))*100
#20% for predicting exiting customer


######################################  Logistic Regression   ############################################
#using backward selection to choose best logistic regression model
set.seed(116)
cv_model1 <- train(
  Exited~., 
  data = train, 
  method = "glm",
  family = binomial(link="logit"),
  trControl = trainControl(method = "cv", number = 10)
)
summary(cv_model1)

set.seed(116)
cv_model2 <- train(
  Exited~Gender+NumOfProducts+IsActiveMember+Age+Balance,
  data = train, 
  method = "glm",
  family = binomial(link="logit"),
  trControl = trainControl(method = "cv", number = 10)
)
summary(cv_model2)

set.seed(116)
cv_model3 <- train(
  Exited~Gender+IsActiveMember+Age+Balance, 
  data = train, 
  method = "glm",
  family = binomial(link="logit"),
  trControl = trainControl(method = "cv", number = 10)
)
summary(cv_model3)

summary(resamples(
  list(model1=cv_model1,
       model2=cv_model2,
       model3=cv_model3)))$statistics$Accuracy
## for training set, model 1 performs best at 83.86%. 
## We choose model 2 against model 1, because it has lesser variables and 
#  decreases model complexity , also there is notmuch difference in acuracy

########## Accuracy - predictions
glmPredict = predict(cv_model1,newdata = test)
confusionMatrix(glmPredict, test$Exited)

glmPredict = predict(cv_model2,newdata = test)
confusionMatrix(glmPredict, test$Exited)

glmPredict = predict(cv_model3,newdata = test)
confusionMatrix(glmPredict, test$Exited)

## we chose model 2 with a accuracy of 84.12% i.e. higher than the baseline and prevalece of 79.20%.
## it also has higher specifity(34.26%) than the other logistic models

########### AUC - Prediction
p.lr <- predict(cv_model2,newdata=test,type="prob")
roc.lr <- roc(test$Exited,p.lr[,1])
plot(roc.lr,col=c(2))
auc(roc.lr)
#0.8374

######################################  KNN   ############################################
str(train)
set.seed(116)
knnmodel <- train(
  Exited ~ ., 
  data = train, 
  method = "knn",
  trControl = trainControl(method="cv",number = 5),
  preProcess = c("center","scale"), #to standarlize the data
  tuneLength = 5
)
knnmodel

set.seed(117)
knnmodel2 <- train(
  Exited ~ ., 
  data = train, 
  method = "knn",
  trControl = trainControl(method="cv",number = 10),
  preProcess = c("center","scale"), #to standarlize the data
  tuneLength = 5
)
knnmodel2
##knnmodel 1 gives better accuracy 84.36% with a cross validation of 5, which is better than 10
# when it comes to prediction

## Accuracy - predictions
knnPredict <- predict(knnmodel,newdata = test )
#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(knnPredict, test$Exited )
#Accuracy is 84.76%
#specificity =is 34.86%

#AUC - Prediction
p.knn <- predict(knnmodel2,newdata=test,type="prob")
roc.knn <- roc(test$Exited,p.knn[,1])
plot(roc.knn,col=c(2))
auc(roc.knn)
#0.8192


######################################  Random Forest   ############################################
ctrl <- trainControl(method="cv",number = 10)
rfmodel <- train(
  Exited ~ ., 
  data = train, 
  method = "rf", 
  preProc = c("center","scale"),
  trControl = ctrl
) #to standarlize the data)
rfmodel
#training accuracy of 85.09%

## Accuracy - predictions
rfpredict <- predict(rfmodel,newdata = test)
#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(rfpredict, test$Exited )
### Accuracy is 86.24%, higher specificity(38.25%) and higher accuracy than knn. 
##  Therefore, better for predicting customers who exited the bank.
?train

#AUC Prediction
p.rf <- predict(rfmodel,newdata=test,type="prob")
roc.rf <- roc(test$Exited,p.rf[,1])
plot(roc.rf,col=c(2))
auc(roc.rf)
#85.5



######################################  Trees and Ensembles   ############################################
### setting cross validation
cvcontrol <- trainControl(method="repeatedcv",  
                          number = 10,
                          repeats = 5,
                          allowParallel=TRUE)


#########Decision Tree
set.seed(1234) 
tree <- train(Exited ~ ., 
              data=train,
              method="rpart",  
              trControl=cvcontrol,
              tuneLength = 10)  

#Accuracy - Predictions
p <-  predict(tree, newdata = test,type="raw")
confusionMatrix(p, test$Exited)
### Accuracy is 85.96%
## Specificity is 44.42%

#AUC - Prediction
p.dt <- predict(tree,newdata=test,type="prob")
roc.dt <- roc(test$Exited,p.dt[,1])
plot(roc.dt,col=c(2))
auc(roc.dt)
## 82.37%

######## Bagging

set.seed(1234)
bag <- train(Exited~., 
             data=train,
             method="treebag",
             trControl=cvcontrol,
             importance=TRUE)
plot(varImp(bag))

#Accuracy - Predictions
p1 <-  predict(bag, newdata = test, type="raw")
confusionMatrix(p1,test$Exited)
#accuracy is 84.96%
#specificity is 45.02%

#AUC - Prediction
p.bag <- predict(bag,newdata=test,type="prob")
roc.bag <- roc(test$Exited,p.bag[,1])
plot(roc.bag,col=c(2), main = 'ROC Curve with Test Data')
auc(roc.bag)
#82.67%

# Boosting
set.seed(1234)
boo <- train(Exited ~ ., 
             data=train,
             method="xgbTree",   
             trControl=cvcontrol,
             tuneGrid = expand.grid(nrounds = 500,
                                    max_depth = 3,
                                    eta = 0.3,
                                    gamma = 0.01,
                                    colsample_bytree = 1,
                                    min_child_weight = 1,
                                    subsample = 1))

#Accuracy - Predictions
p3 <-  predict(boo, newdata = test, type="raw")
confusionMatrix(p3,test$Exited)
#accuracy is 84.92% ,specificity is 47.21% (Highest)

#AUC - Predictions
p.boo <- predict(boo,newdata=test,type="prob")
roc.boo <- roc(test$Exited,p.boo[,1])
auc(roc.boo)
#0.8234

######################################  AUC Chart(All models)   ############################################
plot(roc.boo,col=c(5), main = 'ROC Curve with Test Data')
plot(roc.lr,add = T, col=c(1))
plot(roc.knn,add = T, col=c(2))
plot(roc.rf,add = T, col=c(3))
plot(roc.dt,add = T, col=c(6))
plot(roc.bag,add = T, col=c(7))
legend(x = "bottomright", 
       legend = c("Log", "KNN", "RF","SVM","DT","Bagging","XGBoost"),
       fill = c(1,2,3, 4,6,7),
       cex = 0.6)
##AUC is highest for Random Forest Model at 0.855 i.e. 85.5%
##AUC is second highest for LR 83.16%
##AUC is third highest for Boosting 81.76%


####################################  Overall Accuracy(All models)   ############################################
model <- c("Log","KNN", "RF","DT","Bagging","XGBoost")
accuracy <- c(84.12,84.76,86.24,85.96,84.96,84.92)
model <- data.frame(model, accuracy)
model
models <- ggplot(model, aes(x=model, y=accuracy,fill=model))+
  geom_col()+
  labs(title = "Accuracy of Six different models",
       xlab = "")+
  geom_text(aes(label = accuracy), vjust = 0.5)
models
### Accuracy also shows Random Forest as the Best Model with an accuracy of 85.81%

######################################  Specificity   ############################################
#Accuracy for all models
model2 <- c("Log","KNN", "RF","DT","Bagging","XGBoost")
specificity <- c(34.26,34.86,38.25,44.42,45.02,47.21)
model2 <- data.frame(model2, specificity)
model2
models2 <- ggplot(model2, aes(x=model2, y=specificity,fill=model2))+
  geom_col()+
  labs(title = "Accuracy of Six different models",
       xlab = "")+
  geom_text(aes(label = specificity), vjust = 0.5)
models2
### Accuracy also shows Random Forest as the Best Model with an accuracy of 85.81%

######################################  Balanced model   ############################################
library(dplyr)
summary(train$Exited)
Stayed <- train %>% filter(Exited == 0)
Churned <- train %>% filter(Exited == 1)
Stayed <- Stayed[1:1535,]
t <- rbind(Stayed,Churned)
set.seed(1234)
boo.balanced <- train(Exited ~ ., 
                      data=t,
                      method="xgbTree",   
                      trControl=cvcontrol,
                      tuneGrid = expand.grid(nrounds = 500,
                                             max_depth = 2,
                                             eta = 0.3,
                                             gamma = 0.01,
                                             colsample_bytree = 1,
                                             min_child_weight = 1,
                                             subsample = 1))

p.balanced <-  predict(boo.balanced, newdata = test, type="raw")
confusionMatrix(p.balanced,test$Exited)
p.balanced <- predict(boo.balanced,newdata=test,type="prob")
roc.balanced <- roc(test$Exited,p.balanced[,1])
plot(roc.balanced,col=c(6))
auc(roc.balanced)

