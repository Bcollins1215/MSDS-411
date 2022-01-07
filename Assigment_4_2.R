library(cvTools) # explicit creation of folds for cross-validation
library(ModelMetrics) # used for precision-recall evaluation of classifiers
library(car) # for recode function
library(dplyr)
library(fossil)
library(foreign)
library(psych)
library(lessR)
library(corrplot)
library(survey)
library(ggplot2)
library(reshape2)
library(randomForest)
library(caret)
library(caTools)
library(mclust)
library(pracma)
library(gridExtra)

setwd("~/MSDS 411/assignment #4")

iris<-read.csv(file = 'assignment-4-option-2-iris.csv')

summary(iris)
iris$Species = factor(iris$Species)

# Let's plot all the variables in a single visualization that will contain all the boxplots


BpSl <- ggplot(iris, aes(Species, Sepal.Length, fill=Species)) + 
  geom_boxplot()+
  scale_y_continuous("Sepal Length (cm)", breaks= seq(0,30, by=.5))+
  theme(legend.position="none")



BpSw <-  ggplot(iris, aes(Species, Sepal.Width, fill=Species)) + 
  geom_boxplot()+
  scale_y_continuous("Sepal Width (cm)", breaks= seq(0,30, by=.5))+
  theme(legend.position="none")

BpPl <- ggplot(iris, aes(Species, Petal.Length, fill=Species)) + 
  geom_boxplot()+
  scale_y_continuous("Petal Length (cm)", breaks= seq(0,30, by=.5))+
  theme(legend.position="none")



BpPw <-  ggplot(iris, aes(Species, Petal.Width, fill=Species)) + 
  geom_boxplot()+
  scale_y_continuous("Petal Width (cm)", breaks= seq(0,30, by=.5))+
  labs(title = "Iris Box Plot", x = "Species")

# Plot all visualizations
grid.arrange(BpSl  + ggtitle(""),
             BpSw  + ggtitle(""),
             BpPl + ggtitle(""),
             BpPw + ggtitle(""),
             nrow = 2,
             top = textGrob("Sepal and Petal Box Plot", 
                            gp=gpar(fontsize=15))
)

DhistPl <-    ggplot(iris, aes(x=Petal.Length, colour=Species, fill=Species)) +
  geom_density(alpha=.3) +
  geom_vline(aes(xintercept=mean(Petal.Length),  colour=Species),linetype="dashed",color="grey", size=1)+
  xlab("Petal Length (cm)") +  
  ylab("Density")+
  theme(legend.position="none")

DhistPw <- ggplot(iris, aes(x=Petal.Width, colour=Species, fill=Species)) +
  geom_density(alpha=.3) +
  geom_vline(aes(xintercept=mean(Petal.Width),  colour=Species),linetype="dashed",color="grey", size=1)+
  xlab("Petal Width (cm)") +  
  ylab("Density")



DhistSw <- ggplot(iris, aes(x=Sepal.Width, colour=Species, fill=Species)) +
  geom_density(alpha=.3) +
  geom_vline(aes(xintercept=mean(Sepal.Width),  colour=Species), linetype="dashed",color="grey", size=1)+
  xlab("Sepal Width (cm)") +  
  ylab("Density")+
  theme(legend.position="none")


DhistSl <- ggplot(iris, aes(x=Sepal.Length, colour=Species, fill=Species)) +
  geom_density(alpha=.3) +
  geom_vline(aes(xintercept=mean(Sepal.Length),  colour=Species),linetype="dashed", color="grey", size=1)+
  xlab("Sepal Length (cm)") +  
  ylab("Density")+
  theme(legend.position="none")
ggplot(iris,aes(Species,Petal.Length,fill=Species))+geom_boxplot()

# Plot all density visualizations
grid.arrange(DhistSl + ggtitle(""),
             DhistSw  + ggtitle(""),
             DhistPl + ggtitle(""),
             DhistPw  + ggtitle(""),
             nrow = 2,
             top = textGrob("Iris Density Plot", 
                            gp=gpar(fontsize=15))
)

#split into train and test

index <- sample(2,nrow(iris),replace=TRUE,prob=c(.7,.3))

train <- iris[index==1,]
test <- iris[index==2,]


#build RFM

RFM <- randomForest(Species~.,data=train)

#importance

importance(RFM)

varImpPlot(RFM)

#EVALUATE 

PRED<-predict(RFM,test)

test$pred=PRED

#confusion matrix

CFM<- table(test$Species,test$pred)

#CFM_accuracy

#f1 score function
f1_score <- function(predicted, expected, positive.class="1") {
  predicted <- factor(as.character(predicted), levels=unique(as.character(expected)))
  expected  <- as.factor(expected)
  cm = as.matrix(table(expected, predicted))
  
  precision <- diag(cm) / colSums(cm)
  recall <- diag(cm) / rowSums(cm)
  f1 <-  ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
  
  #Assuming that F1 is zero when it's not possible compute it
  f1[is.na(f1)] <- 0
  
  #Binary F1 or Multi-class macro-averaged F1
  ifelse(nlevels(expected) == 2, f1[positive.class], mean(f1))
}

all_features_f1<-f1_score(test$pred,test$Species)

all_features

f_1_data_frame<- data.frame(all_features)

#2

# Compute k-means with k = 3

set.seed(240) # Setting seed

icluster <- kmeans(iris[,1:4],3,nstart = 20)
table(icluster$cluster,iris$Species)

test<-ifelse(iris$Species=="setosa",1,ifelse(iris$Species=="versicolor",2,3))

#randindex

all_features<-rand.index(icluster$cluster,test)

rand_dataframe<-data.frame(all_features)

rand_dataframe

#3 drop sepal width 

iris_3<-subset(iris,select=-c(Sepal.Width))

#split into train and test

index <- sample(2,nrow(iris_3),replace=TRUE,prob=c(.7,.3))

train <- iris_3[index==1,]
test <- iris_3[index==2,]


RFM <- randomForest(Species~.,data=train)

PRED<-predict(RFM,test)


test$pred=PRED

drop_sepal.width<-f1_score(test$pred,test$Species)

drop_sepal.width

f_1_data_frame<- data.frame(all_features,drop_sepal.width)

f_1_data_frame


#3 drop sepal width and sepal length

iris_4<-subset(iris,select=-c(Sepal.Width,Sepal.Length))

#split into train and test

index <- sample(2,nrow(iris_4),replace=TRUE,prob=c(.7,.3))

train <- iris_4[index==1,]
test <- iris_4[index==2,]


RFM <- randomForest(Species~.,data=train)

PRED<-predict(RFM,test)


test$pred=PRED

drop_sepal<-f1_score(test$pred,test$Species)

drop_sepal

f_1_data_frame<- data.frame(all_features,drop_sepal.width,drop_sepal)

f_1_data_frame

#4 add bogus feature
iris_bogus<-iris%>%mutate(stem = sample(n()))

index <- sample(2,nrow(iris_bogus),replace=TRUE,prob=c(.7,.3))

train <- iris_bogus[index==1,]
test <- iris_bogus[index==2,]


#build RFM

RFM <- randomForest(Species~.,data=train)

#importance

importance(RFM)

varImpPlot(RFM)

#EVALUATE 

PRED<-predict(RFM,test)

test$pred=PRED

#confusion matrix

CFM<- table(test$Species,test$pred)

bogus<-f1_score(test$pred,test$Species)
f_1_data_frame<- data.frame(all_features,drop_sepal.width,drop_sepal,bogus)

f_1_data_frame


# eliminate lowest importance
iris_4<-subset(iris_bogus,select=-c(stem))

index <- sample(2,nrow(iris_4),replace=TRUE,prob=c(.7,.3))

train <- iris_4[index==1,]
test <- iris_4[index==2,]


#build RFM

RFM <- randomForest(Species~.,data=train)

PRED<-predict(RFM,test)

test$pred=PRED

CFM<- table(test$Species,test$pred)

f1_score(test$pred,test$Species)

drop_bogus<-f1_score(test$pred,test$Species)

f_1_data_frame<- data.frame(all_features,drop_sepal.width,drop_sepal,bogus,drop_bogus)

f_1_data_frame

#5 
# Compute k-means with k = 3

set.seed(240) # Setting seed

iris_bogus2<-subset(iris_bogus,select=-c(Species))

icluster <- kmeans(iris_bogus2[,1:4],3,nstart = 20)
table(icluster$cluster,iris_bogus$Species)

#randindex

bogus<-adjustedRandIndex(icluster$cluster,iris$Species)

rand_dataframe<-data.frame(all_features,bogus)

#6

iris_bogus$Class<-"A"


A<- iris_bogus


#transform each column in A 

iris_bogus$Class<-"B"

B<-iris_bogus

# set seed
set.seed(23)

B$Sepal.Length<-sample(B$Sepal.Length)

set.seed(56)

B$Sepal.Width<-sample(B$Sepal.Width)

set.seed(123)
B$Petal.Length<-sample(B$Petal.Length)

set.seed(456)

B$Petal.Width<-sample(B$Petal.Width)

set.seed(458)

B$stem<-sample(B$stem)

#random forest for A and B

#create RFM

AB<-rbind(A,B)

AB$Class = factor(AB$Class)

index <- sample(2,nrow(AB),replace=TRUE,prob=c(.7,.3))

train <- AB[index==1,]
test <- AB[index==2,]


#build RFM


RFM<- randomForest(Class~.,data=train)

importance(RFM)

varImpPlot(RFM)

#EVALUATE 

PRED<-predict(RFM,test)

test$pred=PRED

#confusion matrix

CFM<- table(test$Class,test$pred)

CFM

precision<- CFM[1,1]/(CFM[1,1]+CFM[1,2])
recall<- CFM[1,1]/(CFM[1,1]+CFM[2,1])

class_pred<-((precision*recall)/(precision+recall))*2

f_1_data_frame<- data.frame(all_features,drop_sepal.width,drop_sepal,bogus,drop_bogus,class_pred)

f_1_data_frame

#drop stem and Sepal Width

AB<-rbind(A,B)

AB_RF<-subset(AB,select=-c(stem,Sepal.Width))

AB_RF$Class = factor(AB_RF$Class)

index <- sample(2,nrow(AB_RF),replace=TRUE,prob=c(.7,.3))

train <- AB_RF[index==1,]
test <- AB_RF[index==2,]


#build RFM


RFM <- randomForest(Class~.,data=train)

importance(RFM)

varImpPlot(RFM)

#EVALUATE 

PRED<-predict(RFM,test)

test$pred=PRED

#confusion matrix

CFM<- table(test$Class,test$pred)

CFM

precision<- CFM[1,1]/(CFM[1,1]+CFM[1,2])
recall<- CFM[1,1]/(CFM[1,1]+CFM[2,1])

drop_class<-((precision*recall)/(precision+recall))*2
f_1_data_frame<- data.frame(all_features_f1,drop_sepal.width)

f_1_data_frame


#8 compute cluster with A 

iris_A<-subset(iris,select=-c(Sepal.Width))

Acluster<-kmeans(iris_A[,1:3],3,nstart = 20)

Bcluster<-kmeans(iris_A[,2:3],3,nstart = 20)

table(Acluster$cluster,iris_A$Species)

table(Bcluster$cluster,iris_A$Species)

iris_A$test<-ifelse(iris_A$Species=="setosa",2,ifelse(iris_A$Species=="versicolor",3,1))

breiman<-rand.index(iris_A$test,Acluster$cluster)

iris_A$test<-ifelse(iris_A$Species=="setosa",1,ifelse(iris_A$Species=="versicolor",3,2))

drop_sepal<-rand.index(iris_A$test,Bcluster$cluster)

#cluster from 2

cluster<- kmeans(iris[,1:4],3,nstart=20)

table(cluster$cluster,iris$Species)
iris$test<-ifelse(iris$Species=="setosa",1,ifelse(iris$Species=="versicolor",3,2))

all<-rand.index(cluster$cluster,iris$test)
all

#randindex


rand_dataframe<-data.frame(all,breiman,drop_sepal)

rand_dataframe


percent <- function(x, digits = 2, format = "f", ...) {      # Create user-defined function
  paste0(formatC(x * 100, format = format, digits = digits, ...), "%")
}

x<-colnames(f_1_data_frame)
value<-c(f_1_data_frame$all_features_f1,f_1_data_frame$drop_sepal.width)

value<-percent(value)

ggplot(data=NULL, aes(x, value)) + 
geom_bar(aes(fill = x), stat = "identity", position = "dodge")+geom_text(aes(label=value), position=position_dodge(width=0.9), vjust=-0.25)


x<-colnames(rand_dataframe)
value<-c(rand_dataframe$all, rand_dataframe$breiman,rand_dataframe$drop_sepal)

value<-percent(value)

ggplot(data=NULL, aes(x, value)) + 
  geom_bar(aes(fill = x), stat = "identity", position = "dodge")+geom_text(aes(label=value), position=position_dodge(width=0.9), vjust=-0.25)


