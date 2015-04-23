---
title: "Week10"
author: "Lusine Shirvanyan"
date: "Tuesday, April 21, 2015"
output: word_document

---
```{r}
#setwd("C:/Users/Lusine/Desktop/Term2/CS909/assignments/AssLast")

# Installing necessary packages

install.packages("tm")
require("tm")

library(stringr) #for str_count

install.packages("SnowballC")
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
#reutersRFModel <- randomForest(class ~ . , data = reutersTrainSetTM,  ntree = 100)
#reutersRFPred <- predict(reutersRFModel, reutersTestSetTM[, -(k + 1)])

#Confusion matrix
#(reutersRFConfMatrix <- table(Predicted = reutersRFPred, Real = reutersTestSetTM[, k + 1]))

#Compare classifier performance using 10 cross validation
#(tune(randomForest, class ~ ., data = reutersDFTM, cross = 10, best.model = TRUE))

# SUPPORT VECTOR MACHINES - topic models
reutersSVMModel <- svm(class ~ ., data = reutersTrainSetTM,  cost = 100, gamma = 1)
reutersSVMPred <- predict(reutersSVMModel,  reutersTestSetTM[, -(k+1)])

(reutersSVMConfMatrix <- table(Predicted = reutersSVMPred, Real = reutersTestSetTM[1:length(reutersSVMPred), k+1]))

#Compare classifier performance using 10 cross validation
(tune(svm, class ~ ., data = reutersDFTM, cross = 10, best.model = TRUE))
#Error estimation of ‘svm’ using 10-fold cross validation: 0.2620144


```
```{r}


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
        macroAverage = list()
        microAverage = list()
        
        for(k in 1:nrow(confMatrix)) 
        {
            TP[k] = confMatrix[k,k]
            FP[k] = sum(confMatrix[k,]) - TP[k]
            FN[k] = sum(confMatrix[,k]) - TP[k]
           
            
            accuracy[k]  = TP[k] / colSums(confMatrix)[k]
            recall[k]    = TP[k] / (TP[k] + FN[k])
            precision[k] = TP[k] / (TP[k] + FP[k])
            f_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])    
        }
       
       perfMeasures <- data.frame(TP,FP, FN, accuracy, recall, precision,f_measure, row.names = row.names(confMatrix))
       
       microAverage$Recall <- sum(perfMeasures[,1])/(sum(perfMeasures[,c(1,3)]))
       microAverage$Precision <- sum(perfMeasures[,1])/(sum(perfMeasures[,c(1,2)]))
       macroAverage$Recall <- sum(perfMeasures[,5], na.rm = TRUE)/ nInstance
       macroAverage$Precision <- sum(perfMeasures[,6], na.rm = TRUE)/ nInstance
       
  return(list("Performance measures" = perfMeasures, "MicroAveraging" = microAverage, "Macro Averaging" = macroAverage))
 
}

PerfMeasures(reutersNBConfMatrix)
PerfMeasures(reutersNBConfMatrixTM)
PerfMeasures(reutersSVMConfMatrix)


#4. Clustering
#install.packages('cluster')
#install.packages('flexclust')
#install.packages('fpc')
require(cluster)
require(flexclust)
require(fpc)


reutersClustering <- reutersDFTM
reutersClustering$class <- NULL

reutersScale <- scale(reutersClustering)
reutersDist <- dist(reutersScale, method = "euclidean") 

# Clustering using CLARA
reutersClara <- clara(na.omit(reutersScale), 10, samples=50)
(claraKm <- table(reutersDFTM[1:length(reutersClara),]$class, reutersClara$clustering))
randIndex(claraKm)
plotcluster(na.omit(reutersScale), reutersClara$clustering)

# k-means Clustering
reuterskMean <- kmeans(na.omit(reutersScale), 10)
(reuterskMeanKm <- table(reutersDFTM$class, reuterskMean$cluster[1:nrow(reutersDFTM)]))
randIndex(reuterskMeanKm)
plotcluster(na.omit(reutersScale), reuterskMean$cluster)


# Comparing the similarity of two clusterings
#cluster.stats(reutersDist, reutersClara$clustering, reuterskMean$cluster)


```
