library(readxl)
library(tidyverse)
library(corrplot)
library(FactoMineR)
library(factoextra)
library(Rtsne)

setwd("~/MSDS 411/assignment #2")

alldata = read.csv("Melbourne_housing_FULL.csv",header=TRUE)
houses = alldata[complete.cases(alldata),]
print(str(houses))
workdata = houses[,c("Rooms","Price","Type","Distance","Bedroom2",
                     "Bathroom","Car","Landsize","BuildingArea","YearBuilt","Regionname")]
print(str(workdata))



#add price per building size so different fixed properties can be compared and age



workdata$BuildingArea[workdata$BuildingArea==0] <- NA
workdata<-workdata[complete.cases(workdata),]

workdata<- workdata %>% mutate(workdata,price_size=Price/BuildingArea)


#EDA

ggplot(houses,aes(Type))+geom_bar()
ggplot(houses,aes(Price))+geom_histogram(col="blue")
ggplot(houses,aes(Type,Price))+geom_boxplot()+coord_flip()
ggplot(workdata,aes(Regionname,Price))+geom_boxplot()+coord_flip()
ggplot(houses,aes(Type,Rooms))+geom_boxplot()+coord_flip()

houses %>% count(Regionname)

#find Q1, Q3, and interquartile range for values in column A
Q1 <- quantile(workdata$Price, .25)
Q3 <- quantile(workdata$Price, .75)
IQR <- IQR(workdata$Price)

#only keep rows in dataframe that have values within 1.5*IQR of Q1 and Q3
workdata <- subset(workdata, workdata$Price> (Q1 - 1.5*IQR) & workdata$Price< (Q3 + 1.5*IQR))



#EDA

ggplot(workdata,aes(Regionname,Price))+geom_boxplot()+coord_flip()



#add variables

x1<-workdata$Rooms
x2<-workdata$Price
x3<-workdata$Distance
x4<-workdata$Bedroom2
x5<-workdata$Bathroom
x6<-workdata$Car
x7<-workdata$Landsize
x8<-workdata$BuildingArea
x9<-workdata$YearBuilt

x11<-workdata$price_size
x12<-workdata$Type

workdata<-cbind.data.frame(x1,x4,x5,x6,x7,x8,x11)




#hierarchical clustering

x<-as.matrix(workdata)
y<-dist(x)



xsc<-scale(x)
y<-dist(xsc)



#hierarchical clustering

hc.complete<-hclust(y,method="complete")
hc.average<-hclust(y,method="average")
hc.single<-hclust(y,method="single")

plot(hc.complete,main="Complete Linkage", xlab=" ",sub=" ", cex=0.9)
plot(hc.average,main="Average Linkage", xlab=" ",sub=" ", cex=0.9)
plot(hc.single,main="Single Linkage", xlab=" ", sub=" ",cex=0.9)


# find the number of clusters in kmeans
set.seed(123)
# Compute and plot wss for k = 2 to k = 15.
k.max <- 15
data <- xsc
wss <- sapply(1:k.max, 
              function(k){kmeans(data, k, nstart=50,iter.max = 15 )$tot.withinss})
wss
plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")




km.out2<-kmeans(xsc,3,nstart=50)
km.out2


km_cluster<-km.out2$cluster


fviz_cluster(km.out2, data = workdata,
             palette = c("#2E9FDF", "#00AFBB", "#E7B800"), 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw()
)


#append cut tree to data frame

HC<-cutree(hc.complete, 3)
AC<-cutree(hc.average, 3)
SC<-cutree(hc.single, 3)


workdata<-cbind.data.frame(workdata,x12,km_cluster,HC,AC,SC)

#TSNE

tsne <- Rtsne(workdata %>% select(-x12,-HC,-AC,-SC), dims = 2, perplexity=30, verbose=TRUE, max_iter = 5000, learning = 200,check_duplicates=FALSE)

# visualizing
colors = rainbow(length(unique(workdata$km_cluster)))
names(colors) = unique(workdata$km_cluster)
par(mgp=c(2.5,1,0))
plot(tsne$Y, t='n', main="tSNE", xlab="tSNE dimension 1", ylab="tSNE dimension 2", cex.main=2, cex.lab=1.5)
text(tsne$Y, labels = workdata$km_cluster, col = colors[workdata$km_cluster])

#check for accuracy and combine h and t

workdata$x12<-ifelse(workdata$x12=="h",1,ifelse(workdata$x12=="u",2,1))
workdata$km_cluster<-ifelse(workdata$km_cluster==1,1,ifelse(workdata$km_cluster==3,1,2))

k_means_check<-workdata$x12==workdata$km_cluster
#accuary

sum(k_means_check==TRUE)/nrow(workdata)









