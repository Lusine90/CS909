# CS909
Data Mining 
Week 10 Assignment Lusine Shirvanyan (1457676)

---
title: "Week10"
author: "Lusine Shirvanyan"
date: "Tuesday, April 21, 2015"
output: word_document

---
```{r}
setwd("C:/Users/Lusine/Desktop/Term2/CS909/assignments/AssLast")

# Installing necessary packages

#install.packages("tm")
require("tm")

library(stringr) #for str_count

#install.packages("SnowballC")
library(SnowballC) #for stemming

#install.packages("topicmodels")
require("topicmodels")

```

```{r}

# Reading data from file
reutersDataset = read.csv(file="reutersCSV.csv",header=T,sep=",")

# Remove rows with 'empty' texts (texts, which length is less than 15 characers)

toBeRemoved <- numeric(0)

for(i in 1:nrow(reutersDataset)) 
{
    nwords <- length(strsplit(toString(reutersDataset[i,140]),' ')[[1]])

    if(nwords < 5) 
    {
      toBeRemoved <- c(toBeRemoved, i)
    }
}
dataset <- reutersDataset[-toBeRemoved,]


#Remove all documents with class different from 10  most  populous	classes,	namely:	(earn,	
#acquisitions,	money-fx,	grain,	crude,	trade,	interest,	ship,	wheat,	corn).
#datasetFiltered <- subset(reutersDataset, reutersDataset$topic.earn > 0 
                    #    | reutersDataset$topic.acq> 0
                    #    | reutersDataset$topic.money.fx > 0 | reutersDataset$topic.grain > 0
                  #      | reutersDataset$topic.crude > 0    | reutersDataset$topic.trade > 0
                  #     | reutersDataset$topic.interest > 0 | reutersDataset$topic.ship > 0
                    #    | reutersDataset$topic.wheat > 0    | reutersDataset$topic.corn > 0)

topTenTopics <- c("topic.earn","topic.acq","topic.money.fx","topic.grain","topic.crude","topic.trade","topic.interest","topic.ship","topic.wheat","topic.corn")

#Remove all documents with class different from 10  most  populous	classes,	namely:	(earn,	
#acquisitions,	money-fx,	grain,	crude,	trade,	interest,	ship,	wheat,	corn).

topTopicsDataset <- subset(dataset, select=c(topTenTopics)) 

topTopicsIndexes <- which(rowSums(topTopicsDataset) > 0)

datasetFiltered <- topTopicsDataset[topTopicsIndexes,]

# Get list of values from purpose column to use it for train'test dataset separation
purposeList <- dataset[topTopicsIndexes, ]$purpose

# Get list of classes for documents 
# Since some documents are assigned multiple topics, we will assign only majority class
classList <- as.factor(colnames(datasetFiltered)[max.col(t(t(datasetFiltered) * colSums(datasetFiltered)))])

#classList <- factor(classList)    # this line drops the empty level in the factor 

```

```{r}
# Getting doc texts from the last column of dataset and creating corpus
reuters <- dataset[topTopicsIndexes, 140]

reutersCorpus <- Corpus(VectorSource(reuters))

# Data preprocessing
reutersCorpus <- tm_map(reutersCorpus, content_transformer(tolower))
reutersCorpus <- tm_map(reutersCorpus, removeNumbers)
reutersCorpus <- tm_map(reutersCorpus, removePunctuation)
reutersCorpus <- tm_map(reutersCorpus, removeWords, stopwords("english"))
reutersCorpus <- tm_map(reutersCorpus, stripWhitespace)
reutersCorpus <- tm_map(reutersCorpus, PlainTextDocument)
reutersCorpus <- tm_map(reutersCorpus, removeWords, c("reuter"))
reutersCorpus <- tm_map(reutersCorpus, stemDocument, language = "english")


#2  Preparing BOW and topic models for classification

#Create document term matrix from corpus
reutersDtm <- DocumentTermMatrix(reutersCorpus,
                                 control = list(minWordLength = 2))

# Reduce the matrix to words which occur in at least 15 documents
reutersDtmFiltered <- reutersDtm[ , which(table(reutersDtm$j) >= 15)]

#reutersDtmFiltered <- removeSparseTerms(reutersDtmFiltered, 0.999)
reutersDtmFiltered <- reutersDtmFiltered[rowSums(as.matrix(reutersDtmFiltered)) > 0,]

reutersDtmTfIdf <- weightTfIdf(reutersDtmFiltered)

reutersDtmTfIdf

#As we can see from the above result, the document-term matrix is composed of 2676 terms and
#8649 documents. It is very sparse, with 98% of the entries being zero.

# Split train and test data using purpose column list
trainIndexList <- which(purposeList == 'train')
testIndexList <- which(purposeList == 'test')

length(trainIndexList)
length(testIndexList)


# We will use top N terms from BOW for classification
nTerm = 100    
# Sort by decreasing order of column sums so that top terms will be first
reutersDTMTfIdfTrain <- reutersDtmTfIdf[trainIndexList, ]
reutersTermsSorted <- sort(colSums(as.matrix(reutersDTMTfIdfTrain)), decreasing=T)
reutersTopNTerms <-names(reutersTermsSorted)[1:nTerm]


# Make a data frame with top N terms as cols, docs as rows and
# cell values as tfXIDF for each document and class column at the end
reutersTrainSetBOW <- as.data.frame(inspect((reutersDtmTfIdf[trainIndexList, reutersTopNTerms])))
reutersTrainSetBOW$class<-classList[trainIndexList]

reutersTestSetBOW <- as.data.frame(inspect((reutersDtmTfIdf[trainIndexList, reutersTopNTerms])))
reutersTestSetBOW$class<-classList[trainIndexList]

# We will need whole dataset for k-cross validation
reutersDFBOW <- rbind(reutersTrainSetBOW, reutersTestSetBOW)

```
```{r}
# Topic Models
k <- 20 # set number of topics
# generate model
reutersDTMTrainAndTest <- rbind(reutersDtmFiltered[trainIndexList,], reutersDtmFiltered[testIndexList,])
reutersLdaModel <- LDA(reutersDTMTrainAndTest, k)
# Now we have a topic model with 5960 docs and 20 topics

# Make a data frame with topics as cols, docs as rows and
# cell values as posterior topic distribution for each document
reutersDFTMWhole <- as.data.frame(reutersLdaModel@gamma) 

# Split into train and test sets
reutersTrainSetTM <- as.data.frame(reutersDFTM[trainIndexList, ])
reutersTrainSetTM$class<-classList[trainIndexList]

reutersTestSetTM <- as.data.frame(reutersDFTM[testIndexList, ])
reutersTestSetTM$class<-classList[testIndexList]

reutersDFTM <- rbind(reutersTrainSetTM, reutersTestSetTM)

# add topic models to BOW
#reutersDF <- cbind(reutersDFBOW[, -(nTerm+1)], reutersDFTM)
#reutersTrainSet <- cbind(reutersTrainSetBOW[, -(nTerm+1)], reutersTrainSetTM)
#reutersTestSet <- cbind(reutersTestSetBOW[, -(nTerm+1)], reutersTestSetTM)

```




```{r}

#3 Classification

require(e1071)
require(randomForest)

# NAIVE BAYES on BOW
reutersNBModelBOW <- naiveBayes(class ~ ., reutersTrainSetBOW)
reutersNBPred <- predict(reutersNBModelBOW, reutersTestSetBOW[, -(nTerm + 1)])

#Compute confusion matrix
(reutersNBConfMatrix <- table(Predicted = reutersNBPred, Real = reutersTestSetBOW[, nTerm + 1]))

#Compare classifier performance using 10 cross validation
(reutersNBError <- tune(naiveBayes, class ~ ., data = reutersDFBOW, cross = 10, best.model = T))

#Naive Bayes - topic Models

reutersNBModelTM <- naiveBayes(class ~ ., reutersTrainSetTM)
reutersNBPredTM <- predict(reutersNBModelTM, reutersTestSetTM[, -(k + 1)])

#Compute confusion matrix
(reutersNBConfMatrixTM <- table(Predicted = reutersNBPredTM, Real = reutersTestSetTM[, k + 1]))

#Compare classifier performance using 10 cross validation
(reutersNBErrorTM <- tune(naiveBayes, class ~ ., data = reutersDFTM, cross = 10, best.model = T))


#Naive Bayes - Bow and topic Models

#reutersNBModel <- naiveBayes(class ~ ., reutersTrainSet)
#reutersNBPred <- predict(reutersNBModel, reutersTestSet[, -(nTerm + k + 1)])

#Compute confusion matrix
#(reutersNBConfMatrix <- table(Predicted = reutersNBPred, Real = reutersTestSet[, nTerm + k + 1]))

#Compare classifier performance using 10 cross validation
#(reutersNBError <- tune(naiveBayes, class ~ ., data = reutersDFTM, cross = 10, best.model = T))

# As we can see, topic models perform better

#Random forest - topic models
#set.seed(100)
#reutersRFModel <- randomForest(class~.,importance=T, data=reutersTrainSetTM,  ntree = 100)
#reutersRFPred <- predict(reutersRFModel, reutersTestSetTM[, -(k + 1)])

#Confusion matrix
#(reutersRFConfMatrix <- table(Predicted = reutersRFPred, Real = reutersTestSetTM[, k + 1]))

#Compare classifier performance using 10 cross validation
#(tune(randomForest, class ~ ., data = reutersDFTM, cross = 10, best.model = TRUE))

# SUPPORT VECTOR MACHINES - topic models
reutersSVMModel <- svm(class ~ ., data = reutersTrainSetTM, probability=T)
reutersSVMPred <- predict(reutersSVMModel, reutersTestSetTM[, -(k + 1)])

(reutersSVMConfMatrix <- table(Predicted = reutersSVMPred, Real = reutersTestSetTM[, k + 1]))

#Compare classifier performance using 10 cross validation
(tune(svm, class ~ ., data = reutersDFTM, cross = 10, best.model = TRUE))


```
```{r}
# Measures
Measures = function(mydata){
  D = sum(mydata)   # number of instances
  n <- nrow(mydata) 
  TP = array()       
  TN = array()       
  FP = array()       
  FN = array()      
  Accuracy = array()         
  Recall = array()         
  Precision = array()       
  MacroAverage = list()
  MicroAverage = list()
  for(i in 1:n){
    TP[i] = mydata[i,i]
    FP[i] = sum(mydata[i,]) - TP[i]
    FN[i] = sum(mydata[,i])- TP[i] 
    Accuracy[i] = TP[i]/colSums(mydata)[i]
    Recall[i] = TP[i]/(TP[i]+FN[i])
    Precision[i] = TP[i]/(TP[i]+FP[i])
  }
  Mtable <- data.frame(TP, FP, FN, Accuracy, Recall, Precision, row.names = row.names(mydata))
  MicroAverage$Recall <- sum(Mtable[,1])/(sum(Mtable[,c(1,3)]))
  MicroAverage$Precision <- sum(Mtable[,1])/(sum(Mtable[,c(1,2)]))
  MacroAverage$Recall <- sum(Mtable[,5], na.rm = TRUE)/n
  MacroAverage$Precision <- sum(Mtable[,6], na.rm = TRUE)/n
  output.list <- list("Performance table:" = Mtable, "MicroAveraging" = MicroAverage, "MacroAveraging" = MacroAverage)
  return(output.list)
}

PerfMeasures <- function(confMatrix)
{
        nInstance = sum(confMatrix)
        TP = array()
        TN = array()
        FP = array()
        FN = array()
        
        accuracy = array()
        recall = array()
        precision = array()
        f_measure = array()
        
        for(k in 1:nrow(confMatrix)) 
        {
            TP[k] = confMatrix[k,k]
            TN[k] = sum(confMatrix[-k,-k])
            FP[k] = sum(confMatrix[,k])
            FN[k] = sum(confMatrix[k,])
            
            accuracy[k]  = (TP[k]+TN[k]) / nInstance
            recall[k]    = TP[k] / (TP[k] + FN[k])
            precision[k] = TP[k] / (TP[k] + FP[k])
            f_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])    
        }
        TP[nrow(confMatrix)+1] = mean(TP[1:nrow(confMatrix)], na.rm=TRUE)
        TN[nrow(confMatrix)+1] = mean(TN[1:nrow(confMatrix)], na.rm=TRUE)
        FP[nrow(confMatrix)+1] = mean(FP[1:nrow(confMatrix)], na.rm=TRUE)
        FN[nrow(confMatrix)+1] = mean(FN[1:nrow(confMatrix)], na.rm=TRUE)
        
        avgAcc = mean(accuracy[1:nrow(confMatrix)],  na.rm=TRUE)
        accuracy[nrow(confMatrix)+1]  = avgAcc
        recall[nrow(confMatrix)+1]    = mean(recall[1:nrow(confMatrix)],    na.rm=TRUE)
        precision[nrow(confMatrix)+1] = mean(precision[1:nrow(confMatrix)] ,na.rm=TRUE)
        f_measure[nrow(confMatrix)+1] = mean(f_measure[1:nrow(confMatrix)], na.rm=TRUE)
        
     
        result = data.frame(TP, TN , FP, FN, accuracy, 
                            recall, precision, f_measure,
                            row.names=c(row.names(confMatrix), "Model Average"))
        
        return(list("Performance measures" =result, "AvgAccuracy" = avgAcc ))
}

PerfMeasures(reuterstNBModel)
PerfMeasures(reuterstNBModelTM)
PerfMeasures(reutersSVMModel)
