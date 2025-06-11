## ----setup, include=FALSE, warning=FALSE, echo=FALSE, message = FALSE--------------------------------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = FALSE)

##########################################################
# Begin CYO Project
##########################################################

# As part of initial setup, Check if required libraries are installed already, if installed, load them, if not installed, install and load them.

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(OpenML)) install.packages("openML", repos = "http://cran.us.r-project.org")
if(!require(farff)) install.packages("farff", repos = "http://cran.us.r-project.org")
if(!require(devtools)) install.packages("devtools", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(class)) install.packages("class", repos = "http://cran.us.r-project.org")
if(!require(naivebayes)) install.packages("naivebayes", repos = "http://cran.us.r-project.org")
if(!require(reticulate)) install.packages("reticulate", repos = "http://cran.us.r-project.org")
if(!require(tensorflow)) install.packages("tensorflow", repos = "http://cran.us.r-project.org")
if(!require(keras3)) install.packages("keras3", repos = "http://cran.us.r-project.org")
if(!require(mice)) install.packages("mice", repos = "http://cran.us.r-project.org")
if(!require(ranger)) install.packages("ranger", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages('xgboost', repos = c('https://dmlc.r-universe.dev', 'https://cloud.r-project.org'))
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(adabag)) install.packages("adabag", repos = "http://cran.us.r-project.org")
if(!require(mltools)) install.packages("mltools", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
#keras3::install_keras(backend = c("tensorflow", "jax")) #Comment Out after initial installation

library(tidyverse)
library(caret)
library(ggthemes)
library(gridExtra)
library(purrr)
library(readr)
library(OpenML)
library(farff)
library(devtools)
library(randomForest)
library(class)
library(naivebayes)
library(nnet)
library(reticulate)
library(tensorflow)
library(keras3)
library(mice)
library(ranger)
library(ggplot2)
library(ggthemes)
library(xgboost)
library(rpart)
library(adabag)
library(mltools)
library(data.table)
library(matrixStats)
library(devtools)
library(kableExtra)

# Set Up Environment to disable GPU computing and use JAX
Sys.setenv("CUDA_VISIBLE_DEVICES" = "-1")
Sys.setenv("JAX_PLATFORMS" = "cpu")
use_backend("jax")

# Initialise Parallelisation
#### Use Parallel Library #####

library(parallel)               # Load parallel package, a core R package
library(doParallel)             # Load doparallel backend for functions that use it
nCores <- 8                     # Register 8 Threads (4 Cores with Hyperthreading)
registerDoParallel(nCores)      # Register Threads with doParallel
makeCluster(nCores)             # Turn on Parallel Processing for Registered number of Threads


## ----Download HFP Dataset, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE------------------------------------------------------------------------------------------------------------------------------

##########################################################
# Begin Analysis of HFP Dataset
##########################################################

##########################################################
# Download the Raw Data from the respective repositories as the source for truth for the Datasets
##########################################################

# Heart_Failure_Prediction
# https://www.openml.org/search?type=data&status=active&id=45950&sort=runs
# https://api.openml.org/data/download/22120391/dataset - Since dataset is in arff format, we will use the openML tools to download and save the data locally
# openML saves the Data Automatically to a local Cache and reads from the Cache within the same R Session. However the Cache is not persistent between R sessions and we can instead store a binary value of the Dataset if required for offline working. 

hfp_rdata_file <- "heart_failure_prediction.RData"

if(file.exists(hfp_rdata_file)){
      load(hfp_rdata_file)
      rm(hfp_rdata_file)
  
} else {
  hfp <- getOMLDataSet(data.id = 45950L)
  save(hfp, file = "heart_failure_prediction.RData")
  rm(hfp_rdata_file)
}

# Extract Data from the Downloaded OpenML Dataset 

hfp_data <- hfp$data

# Print for Visualisation
print("Structure of the HFP dataset", quote=FALSE)
str(hfp_data)


# Reecord & print Survival Rate for the whole dataset. This will serve as our reference to evaluate the accuracy of our algorithms
hfp_survival_rate <- 1 - mean(hfp_data$DEATH_EVENT)

# Print survival rate for reference
print(c("The Survival Rate of patients in the HFP dataset is :", hfp_survival_rate), quote=FALSE)



## ----Load HFP Datasets for initial Analysis, Use 80% for Training and 20% for Testing, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE------------------------------------------------------------------


# Prepare Data for further analysis taking advantage of R's capabilities. Convert Binary Variables as factors. 



hfp_data <- hfp_data %>% 
              mutate(anaemia = as.factor(anaemia), diabetes = as.factor(diabetes), high_blood_pressure = as.factor(high_blood_pressure), sex=as.factor(sex), smoking=as.factor(smoking), DEATH_EVENT = as.factor(DEATH_EVENT)) 

# Create Datasets for Training and Testing

set.seed(1024)

hfp_test_index <- createDataPartition(y = hfp_data$DEATH_EVENT, times = 1, p = 0.2, list = FALSE)
hfp_train <- hfp_data[-hfp_test_index,]
hfp_test <- hfp_data[hfp_test_index,]

# Create Datasets for Cross Validation 

set.seed(1024)

hfp_test_index_cv <- createDataPartition(y = hfp_train$DEATH_EVENT, times = 1, p = 0.2, list = FALSE)
hfp_cv_train_set <- hfp_train[-hfp_test_index_cv,]
hfp_cv_test_set <- hfp_train[hfp_test_index_cv,]




## ----Visualise HFP Datasets for initial Analysis, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE-------------------------------------------------------------------------------------------------------

# Generate Histograms to visualise the distribution of values across various ranges. Arrange two graphs per row 

grid.arrange(histogram(hfp_train$age, type="count", xlab = "Age"), histogram(hfp_train$anaemia, type="count", xlab = "Anaemia"), histogram(hfp_train$creatinine_phosphokinase, type="count", xlab = "Creatinine Phospokinase"), histogram(hfp_train$diabetes, type="count", xlab = "Diabetes"), ncol = 2)

grid.arrange(histogram(hfp_train$ejection_fraction, type="count", xlab = "Ejection Fraction"), histogram(hfp_train$high_blood_pressure, type="count", xlab = "High Blood Pressure"), histogram(hfp_train$platelets, type="count", xlab = "Platelets"),histogram(hfp_train$serum_creatinine, type="count", xlab = "Serum Creatinine"),ncol =2)

grid.arrange(histogram(hfp_train$serum_sodium, type="count", xlab = "Serum Sodium"), histogram(hfp_train$sex, type="count", xlab = "Sex"), histogram(hfp_train$smoking, type="count", xlab = "Smoking"),histogram(hfp_train$time, type="count", xlab = "Time"),ncol =2)




## ----Perform Initial Analysis for HFP Dataset using Random Forest, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE, out.width="50%"---------------------------------------------------------------------


# Use the inbuilt tuneRF functionality to find the best mtry value to reduce additional computation

# Use Matrix Notation rather than Formula Notation as it is a lot more efficient

print("=====================================",quote = FALSE)
print("The Random Forest mtry values and error rates are",quote = FALSE) 

set.seed(1024)
best_mtry_rf_tune_hfp <- tuneRF(hfp_cv_train_set[,1:12], hfp_cv_train_set[,13],stepFactor = 0.5, improve = 0.00001,trace = TRUE, plot = TRUE, doBest = TRUE) 

# Print summary about Random Forest. Only print what is necessary and easy to comprehend
print("=====================================",quote = FALSE)
print( c(" Details for Random Forest for the HFP Dataset are: "),quote = FALSE, justify = "left") 
print(c("Prediction Type :",best_mtry_rf_tune_hfp$type),quote = FALSE)
print(c("Number of Trees (ntree) :",best_mtry_rf_tune_hfp$ntree),quote = FALSE)
print(c("mtry value :",best_mtry_rf_tune_hfp$mtry),quote = FALSE)
print("=====================================",quote = FALSE)

# Extract and print the variables in order of decreasing importance
imp <- as.data.frame(randomForest::importance(best_mtry_rf_tune_hfp, type=2))
imp <- data.frame(Importance = imp$MeanDecreaseGini,
           names   = rownames(imp))
imp <- imp[order(imp$Importance, decreasing = TRUE),]

print("The Variables in order of decreasing importance in prediction are:")
print("=============================================",quote = FALSE)
knitr::kable(x = imp, col.names = c("Col Id", "Importance", "Names"), caption = "Heart Failure Prediction - Variable Importance (MeanDecreaseGini)")
print("=============================================",quote = FALSE)

# Generate Predictions for the Test Set
pred_rf_hfp <- predict(best_mtry_rf_tune_hfp, newdata = hfp_cv_test_set[,1:12], type = "response" )

# Print overall accuracy
print("The Accuracy of the Predictions are:",quote = FALSE)
print("=============================================",quote = FALSE)
mean(pred_rf_hfp == hfp_cv_test_set$DEATH_EVENT)
print("=============================================",quote = FALSE)

# Print confusion matrix
print("The confusion Matrix for the predictions is:",quote = FALSE)
print("=============================================")
confusionMatrix(data = pred_rf_hfp,reference =  hfp_cv_test_set$DEATH_EVENT)
print("=============================================",quote = FALSE)

# Remove data and variables that are not required anymore
rm(best_mtry_rf_tune_hfp, pred_rf_hfp,imp)


## ----Perform Initial Analysis for HFP Dataset using Naive Bayes, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE----------------------------------------------------------------------------------------

# Naive Bayes

# Train Naive Bayes on the Training Set      
fit_nb_train_hfp <- naive_bayes(x = hfp_cv_train_set[,1:12], y = hfp_cv_train_set[,13], usekernel = TRUE, usepoisson = FALSE, laplace = 1)
summary(fit_nb_train_hfp)

# Generate Predictions for the Test Set
pred_nb_test_hfp <- predict(fit_nb_train_hfp, newdata = hfp_cv_test_set, type = c("class"))

# Print overall accuracy
print("The Accuracy of the Predictions are:",quote=FALSE)
print("=============================================",quote = FALSE)
mean(pred_nb_test_hfp == hfp_cv_test_set$DEATH_EVENT)

# Print confusion matrix
print("The confusion Matrix for the predictions is:",quote = FALSE)
print("=============================================",quote = FALSE)
confusionMatrix(data = pred_nb_test_hfp,reference =  hfp_cv_test_set$DEATH_EVENT)
print("=============================================",quote = FALSE)


# Remove data and variables that are not required anymore
rm(fit_nb_train_hfp, pred_nb_test_hfp)



## ----Perform Initial Analysis for HFP Dataset using Neural Networks using Keras, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE, out.width="75%"-------------------------------------------------------

########## Hack to achieve Consistent results between Keras Runs ############
# Unload and Reload Keras
unload(package = "keras3", quiet = TRUE)
library(keras3)
# Set Up Environment
set.seed(1024)
keras3::set_random_seed(1024)
#tensorflow::set_random_seed(1024, disable_gpu = TRUE) #Uncomment if using TensorFlow
reticulate::py_set_seed(0)
#############################################################################

# Since Neural Networks do not understand Factors, we will need to use the Original Numeric Values. As we have only Binary Factors, One-Hot-Encoding or other methods are not required. 

hfp_data_orig <- hfp$data

# Create Datasets for Training and Testing

hfp_train_orig <- hfp_data_orig[-hfp_test_index,]
hfp_test_orig <- hfp_data_orig[hfp_test_index,]

# Create Datasets for Cross Validation 

hfp_cv_train_orig_set <- hfp_train_orig[-hfp_test_index_cv,]
hfp_cv_test_orig_set <- hfp_train_orig[hfp_test_index_cv,]

# Remove Variables that are not needed anymore
rm(hfp_test_index, hfp_test_index_cv)

# We also need to ensure that the input variables are scaled & centered so that they we do not have any one factor unnecessarily influencing the outcome and all have adequate representation

# Because there are no missing values or single value variables in the datset, it is not necessary to check for columns that can cause NaN generation during scaling

feature_names <- colnames(hfp_cv_train_orig_set) %>% setdiff("DEATH_EVENT")

train_features <- as.matrix(hfp_cv_train_orig_set[feature_names])
train_targets <- as.matrix(as.numeric(hfp_cv_train_orig_set$DEATH_EVENT))

val_features <- as.matrix(hfp_cv_test_orig_set[feature_names])
val_targets <- as.matrix(as.numeric(hfp_cv_test_orig_set$DEATH_EVENT))

train_features %<>% scale()
val_features %<>% scale(center = attr(train_features, "scaled:center"),
                        scale = attr(train_features, "scaled:scale"))


# Let us build the Nueral Network 

model <-
  keras_model_sequential(input_shape = ncol(train_features)) |>
  layer_dense(192, activation = "relu") |>
  layer_dense(192, activation = "relu") |>
  layer_dropout(0.3) |>
  layer_dense(192, activation = "relu") |>
  layer_dense(192, activation = "relu") |>
  layer_dropout(0.3) |>
  layer_dense(96, activation = "relu") |>
  layer_dense(96, activation = "relu") |>
  layer_dropout(0.3) |>
  layer_dense(1, activation = "sigmoid")

# Let us print the model for visualisation. 
# commented out for Report creation
# print("The Keras Sequential API model is",quote = FALSE)
# print("=============================================",quote = FALSE)
# model
# print("=============================================",quote = FALSE)

# Collect counts to derive initial weights
counts <- table(hfp_cv_train_orig_set$DEATH_EVENT) # Counts for Training Set 

# Used for Diagnostic purposes. Commented out for Report creation
# print("counts for training set",quote = FALSE)
# counts
# print("counts for testing set", qoute=FALSE)
# table(hfp_cv_test_orig_set$DEATH_EVENT) # Counts for Validation Set

# Setup weights. Weights are modified manually. Changes are not updated automatically 
weight_for_0 = (1 / counts["0"]) 
weight_for_1 = (1 / counts["1"]) 

################# Train the Model #################

# Setup Metrics
metrics <- list(
  metric_false_negatives(name = "fn"),
  metric_false_positives(name = "fp"),
  metric_true_negatives(name = "tn"),
  metric_true_positives(name = "tp"),
  metric_precision(name = "precision"),
  metric_recall(name = "recall")
)

# Compile Model
model |> compile(
  optimizer = optimizer_adam(1e-2),
  loss = "binary_crossentropy",
  metrics = metrics
)


# Fit Model
class_weight <- list("0" = weight_for_0,
                     "1" = weight_for_1)

plot_seq_api_model <- model |> fit(
  train_features, train_targets,
  validation_data = list(val_features, val_targets),
  class_weight = class_weight,
  batch_size = 512,
  epochs = 30,
  verbose = 0 # Set verbose=2 during the tuning stage
)

# Print metrics for visualisation
print("The model metrics and trend during training are",quote = FALSE)
print("=============================================",quote = FALSE)
plot(plot_seq_api_model)
print("=============================================",quote = FALSE)

# Prepare Predictions

val_pred <- model %>%
  predict(val_features) %>%
  { as.integer(. > 0.5) }

pred_correct <- hfp_cv_test_orig_set$DEATH_EVENT == val_pred

# Print overall accuracy
print("=============================================",quote = FALSE)
print(c("The Accuracy of the Predictions is: ", mean(pred_correct)), quote=FALSE)
print("=============================================",quote = FALSE)


# Collect death events
deaths <- hfp_cv_test_orig_set$DEATH_EVENT == 1

# Prepare and print summary for  death events 
n_deaths_detected <- sum(deaths & pred_correct)
n_deaths_missed <- sum(deaths & !pred_correct)
n_live_flagged <- sum(!deaths & !pred_correct)

print("=============================================",quote = FALSE)
print(c("deaths detected :", n_deaths_detected),quote = FALSE)
print(c("deaths missed :", n_deaths_missed),quote = FALSE)
print(c("live cases flagged :", n_live_flagged),quote = FALSE)
print("=============================================",quote = FALSE)

# Print confusion matrix
print("The confusion Matrix for the predictions is:",quote = FALSE)
print("=============================================",quote = FALSE)
confusionMatrix(data = as.factor(val_pred), reference =  as.factor(hfp_cv_test_orig_set$DEATH_EVENT))
print("=============================================",quote = FALSE)

# Remove data that is no longer required
rm(callbacks,class_weight,hfp_cv_train_orig_set, hfp_cv_test_orig_set,  metrics, train_features, train_targets, val_features, val_targets, counts, deaths, feature_names, model, n_deaths_detected, n_deaths_missed, n_live_flagged, pred_correct, val_pred, weight_for_0, weight_for_1, plot_seq_api_model)


## ----Perform Final Analysis for HFP Dataset using Random Forest, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE, out.width= "50%"----------------------------------------------------------------------

# Perform Finaly Analysis on Holdout set
# Use the inbuilt tuneRF functionality to find the best mtry value to reduce additional computation

# Use Matrix Notation rather than Formula Notation as it is a lot more efficient

print("=====================================",quote = FALSE)
print("The Random Forest mtry values and error rates are",quote = FALSE) 
set.seed(1024)
best_mtry_rf_tune_hfp_final <- tuneRF(hfp_train[,1:12], hfp_train[,13],stepFactor = 0.5, improve = 0.00001,trace = TRUE, plot = TRUE, doBest = TRUE) 


# Print summary about Random Forest. Only print what is necessary and easy to comprehend
print("=====================================",quote = FALSE)
print( c(" Details for Random Forest  for the HFP Dataset are: "),quote = FALSE, justify = "left") 
print(c("Prediction Type: ",best_mtry_rf_tune_hfp_final$type),quote = FALSE)
print(c("Number of Trees (ntree): ",best_mtry_rf_tune_hfp_final$ntree),quote = FALSE)
print(c("mtry value: ",best_mtry_rf_tune_hfp_final$mtry),quote = FALSE)
print("=====================================",quote = FALSE)

# Extract, Sort and Print Random Forest Variable importance in order of decreasing importance 

imp <- as.data.frame(randomForest::importance(best_mtry_rf_tune_hfp_final))
imp <- data.frame(Importance = imp$MeanDecreaseGini,
           names   = rownames(imp))
imp <- imp[order(imp$Importance, decreasing = TRUE),]

print("The Variables in order of decreasing importance in prediction are:")
print("=============================================",quote = FALSE)
knitr::kable(x = imp, col.names = c("Col Id", "Importance", "Names"), caption = "Heart Failure Prediction - Variable Importance (MeanDecreaseGini)")
print("=============================================",quote = FALSE)

# Generate Predictions for the Test Set
pred_rf_hfp_final <- predict(best_mtry_rf_tune_hfp_final, newdata = hfp_test[,1:12], type = "response" )

# Print overall accuracy
print("The Accuracy of the predictions is:",quote = FALSE)
mean(pred_rf_hfp_final == hfp_test$DEATH_EVENT)

# Print confusion matrix
print("The confusion Matrix for the predictions is:",quote = FALSE)
print("=============================================", quote= FALSE)
hfp_cm_rf_final <- confusionMatrix(data = pred_rf_hfp_final,reference =  hfp_test$DEATH_EVENT)
hfp_cm_rf_final
print("=============================================",quote = FALSE)

# Remove data that is no longer required
rm(best_mtry_rf_tune_hfp_final, pred_rf_hfp_final, imp)


## ----Perform Final Analysis for HFP Dataset using Neural Networks for the Holdout set, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE------------------------------------------------------------------

########## Hack to achieve Consistent results between Keras Runs ############
# Unload and Reload Keras
unload(package = "keras3", quiet = TRUE)
library(keras3)
# Set Up Environment
set.seed(1024)
keras3::set_random_seed(1024)
#tensorflow::set_random_seed(1024, disable_gpu = TRUE) #Uncomment if using TensorFlow
reticulate::py_set_seed(0)
#############################################################################

# Since Neural Networks do not understand Factors, we will need to use the Original Numeric Values. As we have only Binary Factors, One-Hot-Encoding or other methods are not required. 


# We also need to ensure that the input variables are scaled & centered so that they we do not have any one factor unnecessarily influencing the outcome and all have adequate representation

feature_names <- colnames(hfp_train_orig) %>% setdiff("DEATH_EVENT")

train_features <- as.matrix(hfp_train_orig[feature_names])
train_targets <- as.matrix(as.numeric(hfp_train_orig$DEATH_EVENT))

val_features <- as.matrix(hfp_test_orig[feature_names])
val_targets <- as.matrix(as.numeric(hfp_test_orig$DEATH_EVENT))

train_features %<>% scale()
val_features %<>% scale(center = attr(train_features, "scaled:center"),
                        scale = attr(train_features, "scaled:scale"))

# Let us build the Nueral Network 

model <-
  keras_model_sequential(input_shape = ncol(train_features)) |>
  layer_dense(192, activation = "relu") |>
  layer_dense(192, activation = "relu") |>
  layer_dropout(0.3) |>
  layer_dense(192, activation = "relu") |>
  layer_dense(192, activation = "relu") |>
  layer_dropout(0.3) |>
  layer_dense(96, activation = "relu") |>
  layer_dense(96, activation = "relu") |>
  layer_dropout(0.3) |>
  layer_dense(1, activation = "sigmoid")

# Print model
# Commented out for Report creation
# model

# Collect counts to generate initial weights
counts <- table(hfp_train_orig$DEATH_EVENT) # Counts for Training Set

# Used for Diagnostic purposes. Commented out for Report creation
# print("counts for training set")
# counts
# print("counts for testing set")
# table(hfp_test_orig$DEATH_EVENT) # Counts for Validation Set

# Setup weights. Weights are modified manually. Changes are not updated automatically 
weight_for_0 = (1 / counts["0"]) 
weight_for_1 = (1 / counts["1"])

# Train the Model 

metrics <- list(
  metric_false_negatives(name = "fn"),
  metric_false_positives(name = "fp"),
  metric_true_negatives(name = "tn"),
  metric_true_positives(name = "tp"),
  metric_precision(name = "precision"),
  metric_recall(name = "recall")
)
model |> compile(
  optimizer = optimizer_adam(1e-2),
  loss = "binary_crossentropy",
  metrics = metrics
)

class_weight <- list("0" = weight_for_0,
                     "1" = weight_for_1)


model |> fit(
  train_features, train_targets,
  validation_data = list(val_features, val_targets),
  class_weight = class_weight,
  batch_size = 512,
  epochs = 30,
  verbose = 0 # Set verbose = 2 during tuning
)


# Prepare Predictions

val_pred_final <- model %>%
  predict(val_features) %>%
  { as.integer(. > 0.5) }

# Record and print overall accuracy
pred_correct_final <- hfp_test_orig$DEATH_EVENT == val_pred_final
print("=============================================",quote = FALSE)
print("The Accuracy of the Predictions is:",quote = FALSE)
mean(pred_correct_final)
print("=============================================",quote = FALSE)

# Collect death events
deaths_final <- hfp_test_orig$DEATH_EVENT == 1

# Prepare and print summary for death events
n_deaths_detected <- sum(deaths_final & pred_correct_final)
n_deaths_missed <- sum(deaths_final & !pred_correct_final)
n_live_flagged <- sum(!deaths_final & !pred_correct_final)
print("=============================================", quote= FALSE)
print(c("deaths detected", n_deaths_detected))
print(c("deaths missed", n_deaths_missed))
print(c("live cases flagged", n_live_flagged))
print("=============================================", quote= FALSE)

# Print confusion matrix
print("The confusion Matrix for the predictions is:",quote = FALSE)
print("=============================================", quote= FALSE)
hfp_cm_ann_final <- confusionMatrix(data = as.factor(val_pred_final), reference =  as.factor(hfp_test_orig$DEATH_EVENT))
hfp_cm_ann_final
print("=============================================", quote= FALSE)

# Remove data that is no longer required
rm(callbacks,class_weight,hfp_cv_train_orig_set, hfp_cv_test_orig_set,  metrics, train_features, train_targets, val_features, val_targets, counts, deaths, feature_names, model, n_deaths_detected, n_deaths_missed, n_live_flagged, pred_correct_final, val_pred_final, weight_for_0, weight_for_1, deaths_final)



## ----HFP Results Summary, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE-------------------------------------------------------------------------------------------------------------------------------

# Print Survival Rate for the whole dataset for reference 
print(c("The Survival Rate of patients in the HFP dataset is :", hfp_survival_rate), quote=FALSE)


# Prepare final results table

hfp_summary <- data.frame(c("Random Forest", "ANN-SequentialAPI"), c(hfp_cm_rf_final$overall["Accuracy"], hfp_cm_ann_final$overall["Accuracy"]), c(hfp_cm_rf_final$byClass["Balanced Accuracy"], hfp_cm_ann_final$byClass["Balanced Accuracy"]), c(hfp_cm_rf_final$table[2,2], hfp_cm_ann_final$table[2,2]), c(hfp_cm_rf_final$table[1,2], hfp_cm_ann_final$table[1,2]),c( hfp_cm_rf_final$table[2,1],  hfp_cm_ann_final$table[2,1]))

knitr::kable(x = hfp_summary, col.names = c("Model", "overall accuracy", "balanced accuracy", "deaths detected", "deaths missed", "live flagged"), caption = "Heart Failure Prediction - Results Summary ", digits = 4) %>% kable_styling(font_size = 10)


## ----HFP Data Cleanup, include=FALSE, warning=FALSE, echo=FALSE, message = FALSE---------------------------------------------------------------------------------------------------------------------------------

# Remove residual data related to the HFP dataset and run the garbage collector

rm(hfp,hfp_data, hfp_test, hfp_train, hfp_data_orig, hfp_test_orig, hfp_train_orig, hfp_cv_train_set, hfp_cv_test_set, hfp_cm_rf_final, hfp_cm_ann_final, hfp_summary, hfp_survival_rate)

gc()

##########################################################
# End Analysis of HFP Dataset
##########################################################


## ----Download MIC Dataset, include=FALSE, warning=FALSE, echo=FALSE, message = FALSE-----------------------------------------------------------------------------------------------------------------------------

##########################################################
# Begin Analysis of MIC Dataset
##########################################################

##########################################################
# Download the Raw Data from the respective repositories as the source for truth for the Datasets
##########################################################


# Myocardial infarction complications:
# https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications
# https://www.doi.org/10.25392/leicester.data.12045261.v3
# https://figshare.le.ac.uk/articles/dataset/Myocardial_infarction_complications_Database/12045261/3
# https://figshare.le.ac.uk/ndownloader/files/23581310


options(timeout = 120)

mic_data_file_csv <- "Myocardial infarction complications Database.csv"
if (!file.exists(mic_data_file_csv))
  download.file("https://figshare.le.ac.uk/ndownloader/files/23581310")

mic_data <- read.csv(mic_data_file_csv)


####### 
# Remove Variables used to hold filenames as they are not required anymore
rm(dl_mic, mic_data_file_csv)



## ----Prepare MIC data for Analysis - create variable sets, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE----------------------------------------------------------------------------------------------

# Structure of MIC data
print("================================================", quote=FALSE)
print("The structure of the MIC dataset as imported is")
print("================================================", quote = FALSE)
str(mic_data)
print("================================================", quote=FALSE)

# Compute Survival Rate which will act as the basic benchmark for all our Accuracy reports
survival_rate <- round(mean(mic_data$LET_IS == 0),digits = 4)

print("================================================", quote=FALSE)
print(c("The Survival Rate for patients affected by MIC on this datset is",survival_rate),quote = FALSE)
print("================================================", quote=FALSE)

# Create Lists of Continuous, Ordinal (Categorical) and Nominal (Binary) Features so that it is easier to process them later. Though using Column numbers is easier for cross reference,they become very unweildy soon enough. 


# Exclude ID from the list of Variables as it is not to be used for prediction

mic_continuous_variables <- c("AGE","S_AD_KBRIG","D_AD_KBRIG","S_AD_ORIT","D_AD_ORIT","K_BLOOD","NA_BLOOD", "ALT_BLOOD", "AST_BLOOD", "KFK_BLOOD","L_BLOOD", "ROE")

mic_ordinal_variables <- c("INF_ANAM","STENOK_AN","FK_STENOK","IBS_POST","GB","DLIT_AG","ant_im","lat_im","inf_im","post_im","TIME_B_S","R_AB_1_n","R_AB_2_n","R_AB_3_n","NA_R_1_n","NA_R_2_n","NA_R_3_n","NOT_NA_1_n","NOT_NA_2_n","NOT_NA_3_n" )

mic_part_ordinal_variables <- c("ZSN_A")

mic_nominal_variables <- c("SEX","IBS_NASL","SIM_GIPERT","nr_11","nr_01","nr_02","nr_03","nr_04","nr_07","nr_08","np_01","np_04","np_05","np_07","np_08","np_09","np_10", "endocr_01", "endocr_02", "endocr_03", "zab_leg_01", "zab_leg_02", "zab_leg_03", "zab_leg_04", "zab_leg_06", "O_L_POST", "K_SH_POST",     "MP_TP_POST", "SVT_POST", "GT_POST", "FIB_G_POST", "IM_PG_P", "ritm_ecg_p_01", "ritm_ecg_p_02", "ritm_ecg_p_04", "ritm_ecg_p_06", "ritm_ecg_p_07", "ritm_ecg_p_08", "n_r_ecg_p_01", "n_r_ecg_p_02",  "n_r_ecg_p_03", "n_r_ecg_p_04", "n_r_ecg_p_05", "n_r_ecg_p_06", "n_r_ecg_p_08", "n_r_ecg_p_09", "n_r_ecg_p_10", "n_p_ecg_p_01", "n_p_ecg_p_03", "n_p_ecg_p_04", "n_p_ecg_p_05", "n_p_ecg_p_06",  "n_p_ecg_p_07" , "n_p_ecg_p_08", "n_p_ecg_p_09", "n_p_ecg_p_10" , "n_p_ecg_p_11", "n_p_ecg_p_12",  "fibr_ter_01", "fibr_ter_02", "fibr_ter_03", "fibr_ter_05", "fibr_ter_06" , "fibr_ter_07", "fibr_ter_08", "GIPO_K", "GIPER_NA", "NA_KB", "NOT_NA_KB",  "LID_KB", "NITR_S", "LID_S_n", "B_BLOK_S_n", "ANT_CA_S_n", "GEPAR_S_n", "ASP_S_n", "TIKL_S_n", "TRENT_S_n")

# Exclude LET_IS from the list of complications similar to how we have excluded ID. We will handle it separately
mic_complications <- c("FIBR_PREDS", "PREDS_TAH", "JELUD_TAH", "FIBR_JELUD", "A_V_BLOK" , "OTEK_LANC" , "RAZRIV", "DRESSLER", "ZSN", "REC_IM", "P_IM_STEN")


######

# Create Feature sets for Functional API

mic_demographic_history_features <- c("AGE","SEX", "STENOK_AN","GB","SIM_GIPERT","DLIT_AG","IBS_NASL", "endocr_01", "endocr_02", "endocr_03","zab_leg_01", "zab_leg_02", "zab_leg_03", "zab_leg_04", "zab_leg_06", "O_L_POST" )


# Do not include np_09"
mic_infarction_features <- c("INF_ANAM","FK_STENOK", "IBS_POST", "IM_PG_P", "ZSN_A", "nr_11", "nr_01", "nr_02", "nr_03", "nr_04", "nr_07", "nr_08","K_SH_POST","MP_TP_POST","SVT_POST","GT_POST","FIB_G_POST", "np_01","np_04","np_05", "np_07", "np_08", "np_09", "np_10")

#Include  "TIME_B_S" here
mic_emergency_icu_features <- c("S_AD_KBRIG","D_AD_KBRIG","S_AD_ORIT","D_AD_ORIT", "O_L_POST","K_SH_POST","MP_TP_POST","SVT_POST","GT_POST","FIB_G_POST","TIME_B_S") 

mic_ecg_features <- c("ant_im","lat_im","inf_im","post_im","ritm_ecg_p_01","ritm_ecg_p_02","ritm_ecg_p_04","ritm_ecg_p_07","ritm_ecg_p_08","n_r_ecg_p_01","n_r_ecg_p_02","n_r_ecg_p_03","n_r_ecg_p_04","n_r_ecg_p_05","n_r_ecg_p_06", "n_p_ecg_p_03", "n_p_ecg_p_06","n_p_ecg_p_07","n_p_ecg_p_08","n_p_ecg_p_09","n_p_ecg_p_10","n_p_ecg_p_11","n_p_ecg_p_12")

mic_ft_features <- c("fibr_ter_01","fibr_ter_02","fibr_ter_03","fibr_ter_05","fibr_ter_06","fibr_ter_07","fibr_ter_08")

# Do not include "KFK_BLOOD"
mic_serum_features <- c("GIPO_K","K_BLOOD","GIPER_Na","Na_BLOOD","ALT_BLOOD","AST_BLOOD","L_BLOOD","ROE")

mic_relapse_features <- c("R_AB_1_n","R_AB_2_n","R_AB_3_n")

mic_medicine_features <- c("NA_KB","NOT_NA_KB","LID_KB","NITR_S","NA_R_1_n","NA_R_2_n","NA_R_3_n","NOT_NA_1_n","NOT_NA_2_n","NOT_NA_3_n","LID_S_n","B_BLOK_S_n","ANT_CA_S_n","GEPAR_S_n","ASP_S_n","TIKL_S_n","TRENT_S_n")




## ----Prepare MIC data - Create Training, Testing and CV Partitions, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE-------------------------------------------------------------------------------------

# We will modify the mic_data dataset
# We will configure all Ordinal and Binary Variables as Factors.
# Continuous variables are already set as integers or numeric(float) during the import operation.
# We will also set KFK_BLOOD = 0 for all observations. 
# We will retain LET_IS as it is. Do not convert to Factors as we will lose the integer values. Look at note for partition creation

mic_data_orig <- mic_data %>% 
    mutate_at(c(mic_nominal_variables, mic_complications), ~as.factor(.)) %>%
    mutate_at(c(mic_ordinal_variables), ~as.ordered(.)) %>%
    mutate_at(c(mic_part_ordinal_variables), ~as.factor(.)) %>%    
    mutate_at(c("KFK_BLOOD"), ~(. = 0)) 



# Structure of MIC data after modification of variable types
print("================================================", quote = FALSE)
print("The structure of the MIC dataset after modification of variable types and before imputation is: ", quote = FALSE)
print("================================================", quote = FALSE)
str(mic_data_orig)
print("================================================", quote = FALSE)

# Split into Training and Testing Sets. For partition creation treat LET_IS as vector of factors

set.seed(1024)

mic_test_index <- createDataPartition(y = as.factor(mic_data_orig$LET_IS), times = 1, p = 0.2, list = FALSE)
mic_orig_train <- mic_data_orig[-mic_test_index,]
mic_orig_test <- mic_data_orig[mic_test_index,]


# Create Datasets for Cross Validation, For partition creation treat LET_IS as vector of factors
# Retain Indices for the rest of the analysis as they provide consistency in partitioning
set.seed(1024)

mic_test_index_cv <- createDataPartition(y = as.factor(mic_orig_train$LET_IS), times = 1, p = 0.2, list = FALSE)
mic_orig_cv_train_set <- mic_orig_train[-mic_test_index_cv,]
mic_orig_cv_test_set <- mic_orig_train[mic_test_index_cv,]





## ----Prepare MIC data - Perform Imputation , include=TRUE, warning=FALSE, echo=FALSE, message = FALSE------------------------------------------------------------------------------------------------------------

# Imputation is CPU intensive, single threaded and takes time. Retrieve from disk if an imputed dataset is available. Else impute and save the dataset to disk for quick retrieval next time. 

################ Perform Imputation for Whole MIC dataSet ########################

# Use only Features excluding ID for imputation. 

mic_orig_imputed_rdata_file <- "mic_orig_imputed_mice.RData"

if(file.exists(mic_orig_imputed_rdata_file)){
  load(mic_orig_imputed_rdata_file)
  rm(mic_orig_imputed_rdata_file)
} else{
  mic_orig_imputed_mice <- complete(mice(mic_data_orig[,2:112], method = "rf", m=95, seed = 1024, printFlag = FALSE))
  save(mic_orig_imputed_mice, file = mic_orig_imputed_rdata_file)
  rm(mic_orig_imputed_rdata_file)
}

# Recombine Features/Predictors with unique key (ID), complications and outcome ("LET_IS)

# Broken up deliberately for ease of understanding
mic_orig_imputed_mice <- cbind(mic_data_orig[,1], mic_orig_imputed_mice, mic_data_orig[,113:123], mic_data_orig[,124] )

# ID and LET_IS specified explicitly, List of complications excluding LET_IS was created already
colnames(mic_orig_imputed_mice) = c("ID", colnames(mic_orig_imputed_mice[,2:112]), mic_complications, "LET_IS")

############## Split MIC dataset with imputed values using the same partion scheme ##############

# Split into Training and Testing Sets

mic_orig_train_imputed_mice <- mic_orig_imputed_mice[-mic_test_index,]
mic_orig_test_imputed_mice <- mic_orig_imputed_mice[mic_test_index,]


# Create Datasets for Cross Validation 

mic_orig_cv_train_imputed_mice <- mic_orig_train_imputed_mice[-mic_test_index_cv,]
mic_orig_cv_test_imputed_mice <- mic_orig_train_imputed_mice[mic_test_index_cv,]






## ----Visualise MIC continuous variables before and after imputation using the mice Package, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE-------------------------------------------------------------

# Compare Original and Imputed values of Continuous Distributions to check if there is any major distortion

grid.arrange (histogram(mic_orig_train[,"AGE"], type="count", xlab = "AGE", ylab = "before imputation"), histogram(mic_orig_train_imputed_mice[,"AGE"], type="count", xlab = "AGE", ylab = "after imputation"),	histogram(mic_orig_train[,"S_AD_KBRIG"], type="count", xlab = "S_AD_KBRIG",ylab = "before imputation"), histogram(mic_orig_train_imputed_mice[,"S_AD_KBRIG"], type="count", xlab="S_AD_KBRIG", ylab = "after imputation"), ncol =2, nrow =2, name = "MIC - Histogram of distributions before and after imputation - 1" )


grid.arrange (histogram(mic_orig_train[,"D_AD_KBRIG"], type="count", xlab = "D_AD_KBRIG",ylab = "before imputation"), histogram(mic_orig_train_imputed_mice[,"D_AD_KBRIG"], type="count", xlab="D_AD_KBRIG", ylab = "after imputation"), histogram(mic_orig_train[,"S_AD_ORIT"], type="count", xlab = "S_AD_ORIT",ylab = "before imputation"), histogram(mic_orig_train_imputed_mice[,"S_AD_ORIT"], type="count", xlab="S_AD_ORIT", ylab = "after imputation"),	ncol =2, nrow =2, name = "MIC - Histogram of distributions before and after imputation - 2")


grid.arrange (histogram(mic_orig_train[,"D_AD_ORIT"], type="count", xlab = "D_AD_ORIT",ylab = "before imputation"), histogram(mic_orig_train_imputed_mice[,"D_AD_ORIT"], type="count", xlab="D_AD_ORIT", ylab = "after imputation"),	histogram(mic_orig_train[,"K_BLOOD"], type="count", xlab = "K_BLOOD",ylab = "before imputation"), histogram(mic_orig_train_imputed_mice[,"K_BLOOD"], type="count", xlab="K_BLOOD", ylab = "after imputation"), ncol =2, nrow =2, name = "MIC - Histogram of distributions before and after imputation - 3"	)


grid.arrange (histogram(mic_orig_train[,"NA_BLOOD"], type="count", xlab = "NA_BLOOD",ylab = "before imputation"), histogram(mic_orig_train_imputed_mice[,"NA_BLOOD"], type="count", xlab="NA_BLOOD", ylab = "after imputation"), histogram(mic_orig_train[,"ALT_BLOOD"], type="count", xlab = "ALT_BLOOD",ylab = "before imputation"), histogram(mic_orig_train_imputed_mice[,"ALT_BLOOD"], type="count", xlab="ALT_BLOOD", ylab = "after imputation"),	ncol =2, nrow =2, name = "MIC - Histogram of distributions before and after imputation - 4"	)



grid.arrange (histogram(mic_orig_train[,"AST_BLOOD"], type="count", xlab = "AST_BLOOD",ylab = "before imputation"), histogram(mic_orig_train_imputed_mice[,"AST_BLOOD"], type="count", xlab="AST_BLOOD", ylab = "after imputation"),	histogram(mic_orig_train[,"L_BLOOD"], type="count", xlab = "L_BLOOD",ylab = "before imputation"), histogram(mic_orig_train_imputed_mice[,"L_BLOOD"], type="count", xlab="L_BLOOD", ylab = "after imputation"), ncol =2, nrow =2, name = "MIC - Histogram of distributions before and after imputation - 5"	)


grid.arrange (histogram(mic_orig_train[, "ROE"], type="count", xlab = "ROE",ylab = "before imputation"), histogram(mic_orig_train_imputed_mice[, "ROE"], type="count", xlab="ROE", ylab = "after imputation"),	histogram(mic_orig_train[, "IBS_NASL"], type="count", xlab = "IBS_NASL",ylab = "before imputation"), histogram(mic_orig_train_imputed_mice[, "IBS_NASL"], type="count", xlab="IBS_NASL", ylab = "after imputation"), ncol =2, nrow =2, name = "MIC - Histogram of distributions before and after imputation - 6"	)



## ----Prepare MIC data - Modify LET_IS for binary outcome, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE-----------------------------------------------------------------------------------------------


# Change LET_IS to have a Single Category for Deaths rather than multiple 

mic_modified_cv_train_set <- mic_orig_cv_train_set %>% 
      mutate( LET_IS = ifelse(LET_IS==1 | LET_IS==2 | LET_IS==3 | LET_IS==4 | LET_IS==5 | LET_IS==6 | LET_IS==7, 1, 0)) 

mic_modified_cv_test_set <- mic_orig_cv_test_set %>% 
      mutate( LET_IS = ifelse(LET_IS==1 | LET_IS==2 | LET_IS==3 | LET_IS==4 | LET_IS==5 | LET_IS==6 | LET_IS==7, 1, 0))




## ----Check Naive Bayes with Original MIC Dataset without imputation - 2, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE--------------------------------------------------------------------------------

# For the sake of convenience and reproducibility, we are reusing variables after deleting them. However, for this to work, the variables need to be sanitised or deleted after use. If variables are not  sanitised or deleted, data could leak inadvertently.

# We will configure 
# S_AD_KBRIG, S_AD_ORIT, NA_BLOOD, D_AD_KBRIG, D_AD_ORIT and ROE as Poisson
# AGE, K_BLOOD, ALT_BLOOD, AST_BLOOD and L_BLOOD as Gaussian
# KFK_BLOOD = 0 is set already for all observations

# Revert S_AD_KBRIG, S_AD_ORIT, NA_BLOOD as integers

mic_modified_cv_train_set <- mic_modified_cv_train_set %>% 
            mutate_at(c("S_AD_KBRIG", "S_AD_ORIT", "NA_BLOOD"), ~(.= as.integer(.)))  %>% 
            mutate(LET_IS = as.factor(LET_IS))

mic_modified_cv_test_set <- mic_modified_cv_test_set %>% 
            mutate_at(c("S_AD_KBRIG", "S_AD_ORIT", "NA_BLOOD"), ~(.= as.integer(.))) %>% 
            mutate(LET_IS = as.factor(LET_IS))


###########
# Fit Naive Bayes
fit_nb_native_LET_IS <- naive_bayes(x = mic_modified_cv_train_set[,2:112], y = mic_modified_cv_train_set$LET_IS, laplace = 1, usekernel = FALSE, usepoisson = TRUE)

# Print Summary
summary(fit_nb_native_LET_IS)

# Prepare Predictions
pred_nb_test_native_LET_IS <-predict(object = fit_nb_native_LET_IS, newdata = mic_modified_cv_test_set[,2:112])

# Print table of predictions
print("=====================================", quote = FALSE)
print("The table of predictions is: ", quote = FALSE)
table(as.factor(pred_nb_test_native_LET_IS))
print("=====================================", quote = FALSE)

# Print table of actual values
print("=====================================", quote = FALSE)
print("The table of actual values is: ", quote = FALSE)
table(mic_modified_cv_test_set$LET_IS)
print("=====================================", quote = FALSE)

# Print overall accuracy
print("The accuracy of predictions is: ", quote = FALSE)
mean(pred_nb_test_native_LET_IS == mic_modified_cv_test_set$LET_IS)
print("=====================================", quote = FALSE)

# Print confusion matrix
print("The confusion matrix is: ")
print("=====================================", quote = FALSE)
confusionMatrix(data = pred_nb_test_native_LET_IS, reference = mic_modified_cv_test_set$LET_IS)
print("=====================================", quote = FALSE)

# Remove Data that is no longer required

rm(fit_nb_native_LET_IS, pred_nb_test_native_LET_IS, cm)

rm(mic_modified_cv_train_set, mic_modified_cv_test_set)


## ----Prepare Predictions for RF imputed MIC Dataset with Naive Bayes - 1, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE-------------------------------------------------------------------------------

# Modify Datasets for Cross Validation using binary outcomes 

mic_modified_cv_train_set <- mic_orig_cv_train_imputed_mice %>% 
      mutate( LET_IS = ifelse(LET_IS==1 | LET_IS==2 | LET_IS==3 | LET_IS==4 | LET_IS==5 | LET_IS==6 | LET_IS==7, 1, 0))

mic_modified_cv_test_set <- mic_orig_cv_test_imputed_mice %>% 
      mutate( LET_IS = ifelse(LET_IS==1 | LET_IS==2 | LET_IS==3 | LET_IS==4 | LET_IS==5 | LET_IS==6 | LET_IS==7, 1, 0))

# We will configure
# S_AD_KBRIG, S_AD_ORIT, NA_BLOOD, D_AD_KBRIG, D_AD_ORIT and ROE as Poisson
# AGE, K_BLOOD, ALT_BLOOD, AST_BLOOD and L_BLOOD as Gaussian
# KFK_BLOOD = 0 is set already for all observations

# Configure S_AD_KBRIG, S_AD_ORIT, NA_BLOOD as integers

mic_modified_cv_train_set <- mic_modified_cv_train_set %>% 
            mutate_at(c("S_AD_KBRIG", "S_AD_ORIT", "NA_BLOOD"), ~(.= as.integer(.)))  %>% 
            mutate(LET_IS = as.factor(LET_IS))

mic_modified_cv_test_set <- mic_modified_cv_test_set %>% 
            mutate_at(c("S_AD_KBRIG", "S_AD_ORIT", "NA_BLOOD"), ~(.= as.integer(.))) %>% 
            mutate(LET_IS = as.factor(LET_IS))


###########
# Fit Naive Bayes
fit_nb_native_LET_IS <- naive_bayes(x = mic_modified_cv_train_set[,c(2:112)], y = mic_modified_cv_train_set$LET_IS, laplace = 1, usekernel = FALSE, usepoisson = TRUE)

# Print Summary
summary(fit_nb_native_LET_IS)

# Prepare Predictions
pred_nb_test_native_LET_IS <-predict(object = fit_nb_native_LET_IS, newdata = mic_modified_cv_test_set)

# Print table of predictions
print("=====================================", quote = FALSE)
print("The table of predictions is", quote = FALSE)
table(as.factor(pred_nb_test_native_LET_IS))
print("=====================================", quote = FALSE)

# Print table of actual values
print("=====================================", quote = FALSE)
print("The table of actual values is", quote = FALSE)
table(mic_modified_cv_test_set$LET_IS)
print("=====================================", quote = FALSE)

# Print overall accuracy
print("The accuracy of predictions is", quote = FALSE)
mean(pred_nb_test_native_LET_IS == mic_modified_cv_test_set$LET_IS)
print("=====================================", quote = FALSE)

# Print confusion matrix
print("The confusion matrix is")
print("=====================================", quote = FALSE)
confusionMatrix(data = pred_nb_test_native_LET_IS, reference = mic_modified_cv_test_set$LET_IS)
print("=====================================", quote = FALSE)


# Remove Data that is no longer required

rm(fit_nb_native_LET_IS, pred_nb_test_native_LET_IS)




## ----Prepare Predictions for RF imputed MIC Dataset with Random Forest, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE, out.width= "50%"---------------------------------------------------------------

# Modify Datasets for Cross Validation using binary outcomes 

mic_modified_cv_train_set_rf <- mic_orig_cv_train_imputed_mice %>% 
      mutate( LET_IS = ifelse(LET_IS==1 | LET_IS==2 | LET_IS==3 | LET_IS==4 | LET_IS==5 | LET_IS==6 | LET_IS==7, 1, 0))

mic_modified_cv_test_set_rf <- mic_orig_cv_test_imputed_mice %>% 
      mutate( LET_IS = ifelse(LET_IS==1 | LET_IS==2 | LET_IS==3 | LET_IS==4 | LET_IS==5 | LET_IS==6 | LET_IS==7, 1, 0))


# Ensure NA Values are removed as Random Forest generates errors otherwise
# Use na.omit() to remove reamining values if any

mic_modified_cv_train_set_rf <- mic_modified_cv_train_set_rf %>% 
        mutate(LET_IS = as.factor(LET_IS))


mic_modified_cv_test_set_rf <- mic_modified_cv_test_set_rf %>% 
        mutate(LET_IS = as.factor(LET_IS))

# Use the inbuilt tuneRF functionality to find the best mtry value to reduce additional computation

# Use Matrix Notation rather than Formula Notation as it is a lot more efficient

print("=====================================",quote = FALSE)
print("The Random Forest mtry values and error rates are",quote = FALSE) 

set.seed(1024)

best_mtry_rf_tune <- tuneRF(x = mic_modified_cv_train_set_rf[,c(2:112)], y = mic_modified_cv_train_set_rf[,124], stepFactor = 0.5, improve = 0.000001,trace = TRUE, plot = TRUE, doBest = TRUE)

# Prepare predictions
pred_rf <- predict(best_mtry_rf_tune, newdata = mic_modified_cv_test_set_rf[,c(2:112)], type = "response")

# Print summary about Random Forest. Only print what is necessary and easy to comprehend
print("=====================================",quote = FALSE)
print("The details for Random Forest are",quote = FALSE)
print(c("Prediction Type :",best_mtry_rf_tune$type),quote = FALSE)
print(c("Number of Trees (ntree) :",best_mtry_rf_tune$ntree),quote = FALSE)
print(c("mtry value :",best_mtry_rf_tune$mtry),quote = FALSE)
print("=====================================",quote = FALSE)

# Print table of predictions
print("=====================================",quote = FALSE)
print("The table of predictions is")
table(as.factor(pred_rf))
print("=====================================",quote = FALSE)

# Print table of actual values
print("=====================================",quote = FALSE)
print("The table of actual values is")
table (mic_modified_cv_test_set_rf[,124])
print("=====================================",quote = FALSE)

# Print overall accuracy
print("The accuracy of predictions is")
mean(pred_rf == mic_modified_cv_test_set_rf[,124])
print("=====================================",quote = FALSE)

# Print confusion matrix
print("The confusion matrix is")
print("=====================================",quote = FALSE)
confusionMatrix(data = pred_rf,reference = mic_modified_cv_test_set_rf[,124])
print("=====================================",quote = FALSE)

# Extract and Sort Random Forest Variable importance in order of decreasing importance and print 10 most important values

imp <- as.data.frame(randomForest::importance(best_mtry_rf_tune))
imp <- data.frame(Importance = imp$MeanDecreaseGini,
           names   = rownames(imp))
imp <- imp[order(imp$Importance, decreasing = TRUE),]

print("The first 20 Features in order of decreasing importance in prediction are:",quote = FALSE)
print("=============================================",quote = FALSE)
knitr::kable(x = imp[1:20,], col.names = c("Col Id", "Importance", "Names"), caption = "MIC Prediction - Variable Importance (MeanDecreaseGini)")
print("=============================================",quote = FALSE)

# Remove Data that is no longer required

rm(best_mtry_rf_tune, pred_rf, imp)



## ----MIC Initial Analysis Naive Bayes & RF Cleanup , include=FALSE, warning=FALSE, echo=FALSE, message = FALSE---------------------------------------------------------------------------------------------------

# Remove Data that is no longer required and run the garbage collector
rm( mic_modified_cv_train_set, mic_modified_cv_test_set)

rm(mic_modified_cv_train_set_rf,mic_modified_cv_test_set_rf)

gc()



## ----Perform initial analysis of MIC dataset using XGBoost for binary outcome, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE--------------------------------------------------------------------------

# For the sake of convenience and reproducibility, we are reusing variables after deleting them. However, for this to work, the variables need to be sanitised or deleted after use. If variables are not  sanitised or deleted, data could leak inadvertently.

# Use Original Dataset without Imputation

# Modify LET_IS to have a Single Category for Deaths rather than multiple> Use original dataset which has the integer values

mic_data_xgboost <- mic_data %>% mutate( LET_IS = ifelse(LET_IS==1 | LET_IS==2 | LET_IS==3 | LET_IS==4 | LET_IS==5 | LET_IS==6 | LET_IS==7, as.integer(1), as.integer(0)))

# SINCE VARIABLES ARE ORDINAL, WE USE THEM AS-IS TO INDICATE THEIR VALUES. NO ONE-HOT CODING IS NECESSARY
# XGBOOST FOR R ONLY SUPPORTS FACTORS EXPERIMENTALLY. NOT USED HERE TO AVOID COMPLEXITY

# DO NOT CONVERT CATEGORICAL(ORDINAL) AND BINARY(NOMINAL) VARIABLES TO FACTORS
# CONVERSION USING as.matrix() CAUSES THEM TO BE COERCED TO CHARACTER VECTORS


mic_data_xgboost <- mic_data_xgboost %>% 
    mutate_at(mic_continuous_variables, ~as.numeric(.))

# Partition into Training and Testing Sets. Use index created earlier, Create only Training set. Testing is not required for now

mic_modified_train_xgboost <- mic_data_xgboost[-mic_test_index,]

######################

# Create Datasets for Cross Validation. 


mic_modified_cv_train_set_xgboost <- mic_modified_train_xgboost[-mic_test_index_cv,]
mic_modified_cv_test_set_xgboost <- mic_modified_train_xgboost[mic_test_index_cv,]

#######################

# Create a matrix that can be used by XGBoost for Training
dtrain <- xgb.DMatrix(data = as.matrix(mic_modified_cv_train_set_xgboost[,c(2:112)]), label= mic_modified_cv_train_set_xgboost$LET_IS, nthread = 8)

# Fit XGBoost
fit_xgboost_mic <- xgb.train(
    data = dtrain,
    seed = 1024,
    booster	= "gbtree",
    max_depth = 10,
    eta = 0.15,
    nthread = 8,
    nrounds = 200,
    objective = "binary:logistic",
    verbose = 0, # set verbose=1 during development, tuning and testing
)

# Print Summary of Parameters

print("=====================================", quote = FALSE)
print("The details for XGBoost are")
fit_xgboost_mic
print("=====================================", quote = FALSE)

# Generate probabilities for predictions
pred_xgboost_mic <-predict(object = fit_xgboost_mic, newdata = as.matrix(mic_modified_cv_test_set_xgboost[,c(2:112)]))


#######################


# Convert probabilities to binary values 0 and 1.
pred_xgboost_mic_binary <- as.numeric(pred_xgboost_mic > 0.5)

# Print table of predictions
print("=====================================", quote = FALSE)
print("The table of predictions is: ", quote = FALSE)
table(as.factor(pred_xgboost_mic_binary ))
print("=====================================", quote = FALSE)

# Print table of actual values
print("=====================================", quote = FALSE)
print("The table of actual values is: ", quote = FALSE)
table(mic_modified_cv_test_set_xgboost$LET_IS)
print("=====================================", quote = FALSE)

# Print table of overall accuracy
print("The accuracy of predictions is", quote = FALSE)
mean(pred_xgboost_mic_binary == mic_modified_cv_test_set_xgboost$LET_IS)

# Print table of confusion matrix
print("The confusion matrix is", quote = FALSE)
print("=====================================", quote = FALSE)
confusionMatrix(data = as.factor(pred_xgboost_mic_binary), reference = as.factor(mic_modified_cv_test_set_xgboost$LET_IS))
print("=====================================", quote = FALSE)


rm(fit_xgboost_mic, pred_xgboost_mic ,pred_xgboost_mic_binary, dtrain)




## ----MIC XGBoost Binary Outcome - CV Set Cleanup , include=FALSE, warning=FALSE, echo=FALSE, message = FALSE-----------------------------------------------------------------------------------------------------

# Remove data that is no longer required and run the garbage collector 
rm(mic_modified_cv_test_set_xgboost, mic_modified_cv_train_set_xgboost, mic_modified_train_xgboost, mic_data_xgboost)

gc()



## ----Prepare MIC Dataset for expansion of Categorical Variables and binary outcome, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE---------------------------------------------------------------------

# As previously we will use only a Single Category to classify all Deaths

# Let us use the imputed dataset as ANN cannot work with missing values 

# The code for ANN is sligthly more complex than for the other cases. Please follow comments carefully. More than one reading may be required to comprehend the flow

########

# Convert nominal variables from factors to integers. Since they are binary, changing them from factors to numerics does not alter their behaviour 

mic_modified_cv_train_set_ann <- mic_orig_cv_train_imputed_mice %>% 
          mutate_at(c(mic_nominal_variables),~(.=as.integer(.))) 

mic_modified_cv_test_set_ann <- mic_orig_cv_test_imputed_mice %>% 
          mutate_at(c(mic_nominal_variables),~(.=as.integer(.))) 


# Expand Imputed values using Dummy Variables.

##########

# Convert LET_IS to binary values

mic_modified_cv_train_set_ann  <- mic_modified_cv_train_set_ann  %>% 
        mutate( LET_IS = ifelse(LET_IS==1 | LET_IS==2 | LET_IS==3 | LET_IS==4 | LET_IS==5 | LET_IS==6 | LET_IS==7, as.integer(1), as.integer(0))) 

mic_modified_cv_test_set_ann  <- mic_modified_cv_test_set_ann  %>% 
        mutate( LET_IS = ifelse(LET_IS==1 | LET_IS==2 | LET_IS==3 | LET_IS==4 | LET_IS==5 | LET_IS==6 | LET_IS==7, as.integer(1), as.integer(0))) 

# Retain "LET_IS" column as it will be removed during conversion

mic_cv_train_let_is_bin <- mic_modified_cv_train_set_ann$LET_IS

mic_cv_test_let_is_bin <- mic_modified_cv_test_set_ann$LET_IS

########## ONE-HOT ENCODING USING CARET #################################

# Caret is used for one-hot encoding as it supports ordinal variables (ordered factors) 

# Use formula interface to create template for dummy vars. Create dummy vars for Ordinal and Partially Ordinal variables
dumms_vars_cv_template <- dummyVars(formula = "~.", data = mic_modified_cv_train_set_ann[,c(mic_ordinal_variables, mic_part_ordinal_variables)], sep = "_", levelsOnly = FALSE, fullRank = FALSE)

# Create dummy vars for Training CV and Testing CV datasets. 
mic_modified_cv_train_set_ann_dummy <- as.data.frame(predict(dumms_vars_cv_template, mic_modified_cv_train_set_ann[,c(mic_ordinal_variables, mic_part_ordinal_variables)]))

mic_modified_cv_test_set_ann_dummy <- as.data.frame(predict(dumms_vars_cv_template, mic_modified_cv_test_set_ann[,c(mic_ordinal_variables, mic_part_ordinal_variables)]))

# Collect Training CV dummy vars names. Same applies for Testing CV  
dummy_vars_names <- colnames(mic_modified_cv_train_set_ann_dummy)

# Modify Training CV and Testing CV Datasets to remove the existing columns for variables related to  "ID", Ordinal,  Partially Ordinal and Complications as they are not used in Predictions anymore.
# Remove "LET_IS" to ensure that the same logic works for both binary and categorical predictions of the outcome

mic_modified_cv_train_set_ann %<>% select(-c("ID",mic_ordinal_variables,mic_part_ordinal_variables, mic_complications,"LET_IS"))

mic_modified_cv_test_set_ann %<>% select(-c("ID",mic_ordinal_variables,mic_part_ordinal_variables, mic_complications,"LET_IS"))

# Collect Modified Training CV Column names. Same applies for Testing CV
mic_modified_cv_col_names <- colnames(mic_modified_cv_train_set_ann)

# Combine Collected Modified Training CV column names with Training CV dummy vars names
mic_modified_cv_col_names <- c(mic_modified_cv_col_names, dummy_vars_names)

# Create combined Training CV and Testing CV datasets by binding together the Modified CV Training and Testing datasets with their respective dummy vars daatasets
mic_modified_cv_train_set_ann <- cbind(mic_modified_cv_train_set_ann, mic_modified_cv_train_set_ann_dummy)

mic_modified_cv_test_set_ann <- cbind(mic_modified_cv_test_set_ann, mic_modified_cv_test_set_ann_dummy)

# Assign column names to the newly created combined Training CV and Testing CV datasets
colnames(mic_modified_cv_train_set_ann) <- mic_modified_cv_col_names

colnames(mic_modified_cv_test_set_ann) <- mic_modified_cv_col_names

# Remove data that is not required anymore

rm(dumms_vars_cv_template, mic_modified_cv_train_set_ann_dummy, mic_modified_cv_test_set_ann_dummy, mic_modified_cv_col_names)


## ----Perform Initial Analysis for MIC Dataset using Neural Networks after expansion using Sequential API, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE, out.width="75%", fig.align='center'----------

########## Hack to achieve Consistent results between Keras Runs ############
# Unload and Reload Keras
unload(package = "keras3", quiet = TRUE)
library(keras3)
# Set Up Environment
set.seed(1024)
keras3::set_random_seed(1024)
#tensorflow::set_random_seed(1024, disable_gpu = TRUE) #Uncomment if using TensorFlow
reticulate::py_set_seed(0)
#############################################################################

########################## WARNING WARNING WARNING ##########################
############## DOUBLE CHECK INPUT FEATURES TO PREVENT LEAKAGE ###############

feature_names <- c(mic_nominal_variables, mic_continuous_variables, dummy_vars_names)

train_features <- as.matrix(mic_modified_cv_train_set_ann[feature_names])
train_targets <- as.matrix(mic_cv_train_let_is_bin)


val_features <- as.matrix(mic_modified_cv_test_set_ann[feature_names])
val_targets <- as.matrix(mic_cv_test_let_is_bin)

####################################
# We need to ensure that the input variables are scaled & centered so that they we do not have any one factor unnecessarily influencing the outcome and all have adequate representation
# Remove Columns from the list that produce NA when scaled

####### Normal Scaling ############
feature_names <- colnames(train_features)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features) == 0)

index_features_scaling <- which(!feature_names  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features[,index_features_scaling])

train_features <- train_features[,c(feature_names_for_scaling)]
val_features <- val_features[,c(feature_names_for_scaling)]

train_features %<>% scale()
val_features %<>% scale(center = attr(train_features, "scaled:center"),
                        scale = attr(train_features, "scaled:scale"))

# Let us build the Nueral Network 


model <-
  keras3::keras_model_sequential(input_shape = ncol(train_features)) |> 
  layer_dense(units = 256, activation = "relu") |> 
  layer_dropout(rate = 0.3, seed = 1024) |>
  layer_dense(units = 256, activation = "relu") |> 
  layer_dropout(rate = 0.3, seed = 1024) |>
  layer_dense(units = 192, activation = "relu") |> 
  layer_dropout(rate = 0.3, seed = 1024) |>
  layer_dense(units = 192, activation = "relu") |> 
  layer_dropout(rate = 0.3, seed = 1024) |>
  layer_dense(1, activation = "sigmoid")


# Let us print the model for visualisation. 
# Commented out for Report Creation
#print("The Keras Sequential API model is",quote = FALSE)
#print("=============================================",quote = FALSE)
#model
#print("=============================================",quote = FALSE)

# Collect counts for configuration of initial weights
counts <- table(mic_cv_train_let_is_bin) # Counts for Training Set

# Used for Diagnostic purposes. Commented out for Report creation
# print("counts for training set")
# counts
# print("counts for testing set")
# table(mic_cv_test_let_is_bin) # Counts for Validation Set

# Setup weights. Weights are modified manually. Changes are not updated automatically 
weight_for_0 = as.numeric(1 / counts["0"]) 
weight_for_1 = as.numeric(1 / counts["1"]) *0.98

# Train the Model 

metrics <- list(
  metric_false_negatives(name = "fn"),
  metric_false_positives(name = "fp"),
  metric_true_negatives(name = "tn"),
  metric_true_positives(name = "tp"),
  metric_precision(name = "precision"),
  metric_recall(name = "recall")
)
model |> compile(
  optimizer = optimizer_adam(1e-3),
  loss = "binary_crossentropy",
  metrics = metrics
)
# Add callbacks if required

class_weight <- list("0" = weight_for_0,
                     "1" = weight_for_1)



plot_sequential_api_model <- model |> fit(
  train_features, train_targets,
  validation_data = list(val_features, val_targets),
  class_weight = class_weight,
  batch_size = 2048,
  epochs = 30,
  verbose = 0 # Set verbose = 2 during development and tuning
)

# Let us plot the metrics 
print("Sequential API - Metrics and Trends during Training", quote = FALSE)
print("=============================================", quote = FALSE)
plot(plot_sequential_api_model,smooth = TRUE)
print("=============================================", quote = FALSE)

# Prepare Predictions

val_pred <- model %>%
  predict(val_features) %>%
  { as.integer(. > 0.5) }

pred_correct <- mic_cv_test_let_is_bin == val_pred

# Print table of predicted values
print("================================", quote = FALSE)
print("The Table of Predicted values is", quote = FALSE)
table(as.numeric(!pred_correct))
print("================================", quote = FALSE)

# Print table of actual values
print("=====================================", quote = FALSE)
print("The table of actual values is")
table(as.factor(mic_cv_test_let_is_bin))
print("=====================================", quote = FALSE)

# Print overall accuracy. Method is slightly different from previous Algorithms 
print("================================", quote = FALSE)
print(c("Validation accuracy is : ", round(mean(pred_correct), digits = 4)))
print("================================", quote = FALSE)

# Collect death events
deaths <-mic_cv_test_let_is_bin == 1

# Prepare and print summary for death events
n_deaths_detected <- sum(deaths & pred_correct)
n_deaths_missed <- sum(deaths & !pred_correct)
n_live_flagged <- sum(!deaths & !pred_correct)

print(c("deaths detected",n_deaths_detected))
print(c("deaths missed",n_deaths_missed))
print(c("survival cases flagged as deaths",n_live_flagged))

# Print confusion matrix
print("================================", quote = FALSE)
print("The Confusion Matrix is")
confusionMatrix(data = as.factor(val_pred), reference = as.factor(mic_cv_test_let_is_bin))
print("================================", quote = FALSE)

# Remove data that is no longer required

rm(class_weight, metrics, train_features, train_targets, val_features, val_targets, counts, deaths, feature_names, model, n_deaths_detected, n_deaths_missed, n_live_flagged, pred_correct, val_pred, weight_for_0, weight_for_1, plot_sequential_api_model)

rm(index_features_scaling, feature_names_for_scaling)




## ----Prepare MIC Dataset for expansion of Categorical Variables and binary outcome using Functional API, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE------------------------------------------------

# Code for Functional API is more complex and will require following the code and comments carefully

# Initialise Empty Vectors to collect and store respective column indices for all feature sets.
mic_demographic_history_col_indices <- c() 
mic_infarction_col_indices <- c()
mic_emergency_icu_col_indices <- c()
mic_ecg_col_indices <- c()
mic_ft_col_indices <- c()
mic_serum_col_indices <- c()
mic_relapse_col_indices <- c()
mic_medicine_col_indices <- c() 

# Use string detection to identify and extract the respective column indices for each feature set
# Extract the Column indices and store them in respective vectors. 

##### 
for (i in 1:(length(mic_demographic_history_features))) {
mic_demographic_history_col <- str_which( string = colnames(mic_modified_cv_train_set_ann), pattern = mic_demographic_history_features[i])
 mic_demographic_history_col_indices <- c(mic_demographic_history_col_indices, mic_demographic_history_col)
}
rm(mic_demographic_history_col,i)


##### 
for (i in 1:(length(mic_infarction_features))) {
mic_infarction_col <- str_which( string = colnames(mic_modified_cv_train_set_ann), pattern = mic_infarction_features[i])
 mic_infarction_col_indices <- c(mic_infarction_col_indices, mic_infarction_col)
}
rm(mic_infarction_col,i)


##### 
for (i in 1:(length(mic_emergency_icu_features))) {
mic_emergency_icu_col <- str_which( string = colnames(mic_modified_cv_train_set_ann), pattern = mic_emergency_icu_features[i])
 mic_emergency_icu_col_indices <- c(mic_emergency_icu_col_indices, mic_emergency_icu_col)
}
rm(mic_emergency_icu_col,i)

##### 
for (i in 1:(length(mic_ecg_features))) {
mic_ecg_col <- str_which( string = colnames(mic_modified_cv_train_set_ann), pattern = mic_ecg_features[i])
 mic_ecg_col_indices <- c(mic_ecg_col_indices, mic_ecg_col)
}
rm(mic_ecg_col,i)

##### 
for (i in 1:(length(mic_ft_features))) {
mic_ft_col <- str_which( string = colnames(mic_modified_cv_train_set_ann), pattern = mic_ft_features[i])
 mic_ft_col_indices <- c(mic_ft_col_indices, mic_ft_col)
}
rm(mic_ft_col,i)

##### 
for (i in 1:(length(mic_serum_features))) {
mic_serum_col <- str_which( string = colnames(mic_modified_cv_train_set_ann), pattern = mic_serum_features[i])
 mic_serum_col_indices <- c(mic_serum_col_indices, mic_serum_col)
}
rm(mic_serum_col,i)

##### 
for (i in 1:(length(mic_relapse_features))) {
mic_relapse_col <- str_which( string = colnames(mic_modified_cv_train_set_ann), pattern = mic_relapse_features[i])
 mic_relapse_col_indices <- c(mic_relapse_col_indices, mic_relapse_col)
}
rm(mic_relapse_col,i)

##### 
for (i in 1:(length(mic_medicine_features))) {
mic_medicine_col <- str_which( string = colnames(mic_modified_cv_train_set_ann), pattern = mic_medicine_features[i])
 mic_medicine_col_indices <- c(mic_medicine_col_indices, mic_medicine_col)
}
rm(mic_medicine_col,i)




## ----Perform Initial Analysis of MIC Dataset after expansion of Categorical Variables for binary outcome using Functional API, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE, out.width="75%", fig.align='center'----

########## Hack to achieve Consistent results between Keras Runs ############
# Unload and Reload Keras
unload(package = "keras3", quiet = TRUE)
library(keras3)
# Set Up Environment
set.seed(1024)
keras3::set_random_seed(1024)
#tensorflow::set_random_seed(1024, disable_gpu = TRUE) #Uncomment if using TensorFlow
reticulate::py_set_seed(0)
#############################################################################


############# Create the Input Data #########################################


########## Demographic & History Features ###########################

train_features_demographic_history <- as.matrix(mic_modified_cv_train_set_ann[,c(mic_demographic_history_col_indices)])
val_features_demographic_history <- as.matrix(mic_modified_cv_test_set_ann[,c(mic_demographic_history_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_demographic_history <- colnames(train_features_demographic_history)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_demographic_history) == 0)

index_features_scaling <- which(!feature_names_demographic_history  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_demographic_history[,index_features_scaling])

train_features_demographic_history <- train_features_demographic_history[,c(feature_names_for_scaling)]
val_features_demographic_history <- val_features_demographic_history[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_demographic_history %<>% scale()
val_features_demographic_history %<>% 
        scale(center = attr(train_features_demographic_history, "scaled:center"),
        scale = attr(train_features_demographic_history, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## Infarction Features ###########################

train_features_infarction <- as.matrix(mic_modified_cv_train_set_ann[,c(mic_infarction_col_indices)])
val_features_infarction <- as.matrix(mic_modified_cv_test_set_ann[,c(mic_infarction_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_infarction <- colnames(train_features_infarction)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_infarction) == 0)

index_features_scaling <- which(!feature_names_infarction  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_infarction[,index_features_scaling])

train_features_infarction <- train_features_infarction[,c(feature_names_for_scaling)]
val_features_infarction <- val_features_infarction[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_infarction %<>% scale()
val_features_infarction %<>% 
        scale(center = attr(train_features_infarction, "scaled:center"),
        scale = attr(train_features_infarction, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## Emergency ICU Features ###########################

train_features_emergency_icu <- as.matrix(mic_modified_cv_train_set_ann[,c(mic_emergency_icu_col_indices)])
val_features_emergency_icu <- as.matrix(mic_modified_cv_test_set_ann[,c(mic_emergency_icu_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_emergency_icu <- colnames(train_features_emergency_icu)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_emergency_icu) == 0)

index_features_scaling <- which(!feature_names_emergency_icu  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_emergency_icu[,index_features_scaling])

train_features_emergency_icu <- train_features_emergency_icu[,c(feature_names_for_scaling)]
val_features_emergency_icu <- val_features_emergency_icu[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_emergency_icu %<>% scale()
val_features_emergency_icu %<>% 
        scale(center = attr(train_features_emergency_icu, "scaled:center"),
        scale = attr(train_features_emergency_icu, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## ECG Features ###########################

train_features_ecg <- as.matrix(mic_modified_cv_train_set_ann[,c(mic_ecg_col_indices)])
val_features_ecg <- as.matrix(mic_modified_cv_test_set_ann[,c(mic_ecg_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_ecg <- colnames(train_features_ecg)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_ecg) == 0)

index_features_scaling <- which(!feature_names_ecg  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_ecg[,index_features_scaling])

train_features_ecg <- train_features_ecg[,c(feature_names_for_scaling)]
val_features_ecg <- val_features_ecg[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_ecg %<>% scale()
val_features_ecg %<>% 
        scale(center = attr(train_features_ecg, "scaled:center"),
        scale = attr(train_features_ecg, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )

########## FT Features ###########################

train_features_ft <- as.matrix(mic_modified_cv_train_set_ann[,c(mic_ft_col_indices)])
val_features_ft <- as.matrix(mic_modified_cv_test_set_ann[,c(mic_ft_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_ft <- colnames(train_features_ft)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_ft) == 0)

index_features_scaling <- which(!feature_names_ft  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_ft[,index_features_scaling])

train_features_ft <- train_features_ft[,c(feature_names_for_scaling)]
val_features_ft <- val_features_ft[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_ft %<>% scale()
val_features_ft %<>% 
        scale(center = attr(train_features_ft, "scaled:center"),
        scale = attr(train_features_ft, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## Serum Features ###########################

train_features_serum <- as.matrix(mic_modified_cv_train_set_ann[,c(mic_serum_col_indices)])
val_features_serum <- as.matrix(mic_modified_cv_test_set_ann[,c(mic_serum_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_serum <- colnames(train_features_serum)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_serum) == 0)

index_features_scaling <- which(!feature_names_serum  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_serum[,index_features_scaling])

train_features_serum <- train_features_serum[,c(feature_names_for_scaling)]
val_features_serum <- val_features_serum[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_serum %<>% scale()
val_features_serum %<>% 
        scale(center = attr(train_features_serum, "scaled:center"),
        scale = attr(train_features_serum, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## Relpase Features ###########################

train_features_relapse <- as.matrix(mic_modified_cv_train_set_ann[,c(mic_relapse_col_indices)])
val_features_relapse <- as.matrix(mic_modified_cv_test_set_ann[,c(mic_relapse_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_relapse <- colnames(train_features_relapse)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_relapse) == 0)

index_features_scaling <- which(!feature_names_relapse  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_relapse[,index_features_scaling])

train_features_relapse <- train_features_relapse[,c(feature_names_for_scaling)]
val_features_relapse <- val_features_relapse[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_relapse %<>% scale()
val_features_relapse %<>% 
        scale(center = attr(train_features_relapse, "scaled:center"),
        scale = attr(train_features_relapse, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## Medicine Features ###########################

train_features_medicine <- as.matrix(mic_modified_cv_train_set_ann[,c(mic_medicine_col_indices)])
val_features_medicine <- as.matrix(mic_modified_cv_test_set_ann[,c(mic_medicine_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_medicine <- colnames(train_features_medicine)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_medicine) == 0)

index_features_scaling <- which(!feature_names_medicine  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_medicine[,index_features_scaling])

train_features_medicine <- train_features_medicine[,c(feature_names_for_scaling)]
val_features_medicine <- val_features_medicine[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_medicine %<>% scale()
val_features_medicine %<>% 
        scale(center = attr(train_features_medicine, "scaled:center"),
        scale = attr(train_features_medicine, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


###################################################################################

####### Create Training & Validation Targets #######################

train_targets <- as.matrix(mic_cv_train_let_is_bin)
val_targets <- as.matrix(mic_cv_test_let_is_bin)


###################################################################################

# Let us define the input shapes. 
input_shape_demographic_history  <- ncol(train_features_demographic_history)
input_shape_infarction <- ncol(train_features_infarction)
input_shape_emergency_icu <- ncol(train_features_emergency_icu)
input_shape_ecg <- ncol(train_features_ecg)
input_shape_ft <- ncol(train_features_ft)
input_shape_serum <- ncol(train_features_serum)
input_shape_relapse <- ncol(train_features_relapse)
input_shape_medicine <- ncol(train_features_medicine)


# Let us build the Keras Inputs & Feature sets
input_demographic_history <- keras_input(shape(input_shape_demographic_history), name = "demographic_history")
input_infarction <- keras_input(shape(input_shape_infarction), name = "infarction")
input_emergency_icu <- keras_input(shape(input_shape_emergency_icu), name = "emergency_icu")
input_ecg <- keras_input(shape(input_shape_ecg), name = "ecg")
input_ft <- keras_input(shape(input_shape_ft), name = "ft")
input_serum <- keras_input(shape(input_shape_serum), name = "serum")
input_relapse <- keras_input(shape(input_shape_relapse), name = "relapse")
input_medicine <- keras_input(shape(input_shape_medicine), name = "medicine")


demographic_history_features <- 
    layer_dense(object = input_demographic_history, units = 256) |> 
    layer_dropout(rate = 0.3, seed = 1024) |>
    layer_dense(units = 128, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024) 

infarction_features <- 
    layer_dense(object = input_infarction, units = 1536, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024) 


emergency_icu_features <- 
    layer_dense(object = input_emergency_icu, units = 1216, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024) 

ecg_features <- 
    layer_dense(object = input_ecg, units = 576, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024) 

ft_features <- 
    layer_dense(object = input_ft, units = 64, activation = "relu") |>  
    layer_dropout(rate = 0.3, seed = 1024)

serum_features <- 
    layer_dense(object = input_serum, units = 64, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024) 

relapse_features <- 
    layer_dense(object = input_relapse, units = 72, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024)

medicine_features <- 
    layer_dense(object = input_medicine, units = 240, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024) |>
    layer_dense(units = 120, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024)

# Let us combine the Feature Layers together

combined_features <- layer_concatenate(list(demographic_history_features,infarction_features, emergency_icu_features, ecg_features, ft_features, serum_features, relapse_features, medicine_features))


pred_functional_api <- layer_dense(object = combined_features, units = 1, activation = "sigmoid")

# Instantiate an end-to-end model 

functional_api_model <- keras_model(
  inputs = list(input_demographic_history, input_infarction, input_emergency_icu, input_ecg, input_ft, input_serum, input_relapse, input_medicine),
  outputs = list(pred_functional_api)
)

# Let us Plot the model for Visualisation
# Commented Out for Report Creation
#print("The Plot of the Keras Functional API model is",quote = FALSE)
#print("=============================================",quote = FALSE)
#plot(functional_api_model, show_shapes = TRUE)
#print("=============================================",quote = FALSE)

#print("The summary of the Keras Functional API model is",quote = FALSE)
#print("=============================================",quote = FALSE)
#summary(functional_api_model)
#print("=============================================",quote = FALSE)

# Collect counts for initial weight generation
counts <- table(mic_cv_train_let_is_bin) # Counts for Training Set

# Used for Diagnostic purposes. Commented out for Report creation
# print("counts for training set")
# counts
# print("counts for testing set")
# table(mic_cv_test_let_is_bin) # Counts for Validation Set

# Configure weights. Weights are updated manually. 
weight_for_0 = as.numeric(1 / counts["0"])
weight_for_1 = as.numeric(1 / counts["1"]) 

# Compile Model

metrics <- list(
  metric_false_negatives(name = "fn"),
  metric_false_positives(name = "fp"),
  metric_true_negatives(name = "tn"),
  metric_true_positives(name = "tp"),
  metric_precision(name = "precision"),
  metric_recall(name = "recall")
)
functional_api_model |> compile(
  optimizer = optimizer_adam(1e-3),
  loss = "binary_crossentropy",
  metrics = metrics
)


class_weight <- list("0" = weight_for_0,
                     "1" = weight_for_1)


######  Fit model ##############


plot_functional_api_model <- functional_api_model |> 
  fit(
  x = list(demographic_history = train_features_demographic_history, infarction = train_features_infarction, emergency_icu = train_features_emergency_icu, ecg = train_features_ecg, ft = train_features_ft, serum = train_features_serum, relapse = train_features_relapse, medicine = train_features_medicine),
  y = train_targets,
  validation_data = list(list(val_features_demographic_history, val_features_infarction, val_features_emergency_icu, val_features_ecg, val_features_ft, val_features_serum, val_features_relapse, val_features_medicine),val_targets),
  class_weight = class_weight,
  batch_size = 2048,
  epochs = 30,
  verbose = 0 #set verbose=2 during development, tuning and testing
)

# Let us plot the metrics 
print("Functional API - Metrics and Trends during Training",quote = FALSE)
print("=============================================",quote = FALSE)
plot(plot_functional_api_model,smooth = TRUE)
print("=============================================",quote = FALSE)

# Prepare Predictions

val_pred_functional_api <- functional_api_model %>%
  predict(list(val_features_demographic_history, val_features_infarction, val_features_emergency_icu, val_features_ecg, val_features_ft, val_features_serum, val_features_relapse, val_features_medicine)) %>%
  { as.integer(. > 0.5) }

pred_correct_functional_api <- mic_cv_test_let_is_bin == val_pred_functional_api

# Print table of predictions
print("================================",quote=FALSE)
print("The Table of Predicted values is: ",quote = FALSE)
table(as.numeric(!pred_correct_functional_api))
print("================================",quote=FALSE)

# Print table of actual values
print("=====================================", quote = FALSE)
print("The table of actual values is: ", quote = FALSE)
table(as.factor(mic_cv_test_let_is_bin))
print("=====================================", quote = FALSE)

# Print overall accuracy. 
print("================================", quote = FALSE)
print(c("Validation accuracy is: ", round(mean(pred_correct_functional_api), digits = 4)),quote = FALSE)
print("================================", quote = FALSE)

# Collect death events
deaths_functional_api <- mic_cv_test_let_is_bin == 1

# Prepare and print summary of death events
n_deaths_detected <- sum(deaths_functional_api & pred_correct_functional_api)
n_deaths_missed <- sum(deaths_functional_api & !pred_correct_functional_api)
n_live_flagged <- sum(!deaths_functional_api & !pred_correct_functional_api)

print(c("deaths detected",n_deaths_detected))
print(c("deaths missed",n_deaths_missed))
print(c("survival cases flagged as deaths",n_live_flagged))

# Print confusion matrix
print("================================", quote=FALSE)
print("The Confusion Matrix is",quote=FALSE)
confusionMatrix(data = as.factor(val_pred_functional_api), reference = as.factor(mic_cv_test_let_is_bin))
print("================================", quote=FALSE)



######## Remove Variables that are not required anymore ############

rm(metrics, counts, deaths_functional_api, feature_names, model, n_deaths_detected, n_deaths_missed, n_live_flagged, pred_correct_functional_api, val_pred_functional_api, weight_for_0, weight_for_1)

rm( demographic_history_features, ecg_features, emergency_icu_features,  ft_features,   infarction_features, medicine_features, relapse_features, serum_features, combined_features)

rm( input_demographic_history, input_ecg, input_emergency_icu,  input_ft,  input_infarction, input_medicine, input_relapse, input_serum)

rm( train_features_demographic_history, train_features_ecg, train_features_emergency_icu, train_features_ft,  train_features_infarction, train_features_medicine, train_features_relapse, train_features_serum, train_features_all)

rm( val_features_demographic_history, val_features_ecg, val_features_emergency_icu, val_features_ft,  val_features_infarction, val_features_medicine, val_features_relapse, val_features_serum, val_features_all)

rm( feature_names_demographic_history, feature_names_ecg, feature_names_emergency_icu,  feature_names_ft,  feature_names_infarction, feature_names_medicine, feature_names_relapse, feature_names_serum)

rm(input_shape_demographic_history, input_shape_ecg, input_shape_emergency_icu,  input_shape_ft,  input_shape_infarction, input_shape_medicine, input_shape_relapse, input_shape_serum)

rm(functional_api_model, pred_functional_api,train_targets, val_targets, class_weight, plot_functional_api_model)

rm(mic_cv_train_let_is_bin, mic_cv_test_let_is_bin)


## ----MIC Data Clean up Cross Validation Sets Keras ANN Binary outcome, include=FALSE, warning=FALSE, echo=FALSE, message = FALSE---------------------------------------------------------------------------------

# Remove data that is no longer required and run the garbage collector
rm(mic_modified_cv_train_set_ann, mic_modified_cv_test_set_ann, mic_modified_train_set_ann) 

gc()



## ----Perform initial analysis of MIC dataset using XGBoost for categorical outcome, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE---------------------------------------------------------------------

# For the sake of convenience and reproducibility, we are reusing variables after deleting them. However, for this to work, the variables need to be sanitised or deleted after use. If variables are not  sanitised or deleted, data could leak inadvertently.

# Use Original Dataset without Imputation

# Retain LET_IS with multiple categories for deaths. It is already set as an integer during the import process

# DO NOT CONVERT CATEGORICAL(ORDINAL) AND BINARY(NOMINAL) VARIABLES TO FACTORS
# CONVERSION USING as.matrix() CAUSES THEM TO BE COERCED TO CHARACTER VECTORS

mic_data_xgboost <- mic_data %>% 
    mutate_at(mic_continuous_variables, ~as.numeric(.))


# Split into Training and Testing Sets. Use index created earlier. Create only Training Set. Testing Set is not required for now

mic_modified_train_xgboost <- mic_data_xgboost[-mic_test_index,]

######################

# Create Datasets for Cross Validation. 

mic_modified_cv_train_set_xgboost <- mic_modified_train_xgboost[-mic_test_index_cv,]
mic_modified_cv_test_set_xgboost <- mic_modified_train_xgboost[mic_test_index_cv,]

###########


###########

# Create a matrix that can be used by XGBoost for Training
dtrain_categorical <- xgb.DMatrix(data = as.matrix(mic_modified_cv_train_set_xgboost[,c(2:112)]), label= mic_modified_cv_train_set_xgboost$LET_IS, nthread = 8)

# Fit XGBoost
fit_xgboost_mic_categorical <- xgb.train(
    data = dtrain_categorical,
    max_depth = 10,
    eta = 0.15,
    nthread = 8,
    nrounds = 200,
    objective = "multi:softmax", 
    params = list("num_class" = 8, "booster" = "gbtree"),
    verbose = 0 # set verbose=2 during development, tuning and testing
)

# Print Summary of Parameters

print("=====================================", quote = FALSE)
print("The details for XGBoost are: ", quote = FALSE)
fit_xgboost_mic_categorical
print("=====================================")

# Generate probabilities for predictions
pred_xgboost_mic_categorical <-predict(object = fit_xgboost_mic_categorical, newdata = as.matrix(mic_modified_cv_test_set_xgboost[,c(2:112)]))

# Print table of predictions
print("=====================================", quote = FALSE)
print("The table of predictions is: ", quote = FALSE)
table(as.factor(pred_xgboost_mic_categorical ))
print("=====================================", quote = FALSE)

# Print table of actual values
print("=====================================", quote = FALSE)
print("The table of actual values is: ", quote = FALSE)
table(mic_modified_cv_test_set_xgboost$LET_IS)
print("=====================================", quote = FALSE)

# Print overall accuracy
print("The accuracy of predictions is: ", quote = FALSE)
mean(pred_xgboost_mic_categorical == mic_modified_cv_test_set_xgboost$LET_IS)
print("=====================================", quote = FALSE)

print("The confusion matrix is")
print("=====================================", quote = FALSE)
cm_xgboost_multi <- confusionMatrix(data = as.factor(pred_xgboost_mic_categorical), reference = as.factor(mic_modified_cv_test_set_xgboost$LET_IS))
cm_xgboost_multi
print("=====================================", quote = FALSE)

# Extract and Print summary of death events

print("================================",quote=FALSE)
print("The Number of Deaths correctly categorised is :",quote=FALSE)
sum(cm_xgboost_multi$table[2,2], cm_xgboost_multi$table[3,3], cm_xgboost_multi$table[4,4], cm_xgboost_multi$table[5,5], cm_xgboost_multi$table[6,6], cm_xgboost_multi$table[7,7], cm_xgboost_multi$table[8,8])
print("================================",quote=FALSE)

print("================================",quote=FALSE)
print("The Number of Deaths correctly detected is :",quote=FALSE)
sum(cm_xgboost_multi$table[2:8,2:8])
print("================================",quote=FALSE)

# Remove data that is no longer required
rm(fit_xgboost_mic_categorical, pred_xgboost_mic_categorical , dtrain_categorical)

rm(cm_xgboost_multi)


## ----MIC XGBoost Categorical Outcome - CV Set Cleanup , include=FALSE, warning=FALSE, echo=FALSE, message = FALSE------------------------------------------------------------------------------------------------

# Remove data that is no longer required and run the garbage collector

rm(mic_modified_cv_test_set_xgboost, mic_modified_cv_train_set_xgboost, dtrain, mic_modified_train_xgboost, mic_data_xgboost)

rm(mic_modified_cv_test_set, mic_modified_cv_train_set, mic_data_modified, mic_data_xgboost)

gc()



## ----Prepare MIC Dataset for expansion of Categorical Variables and multiple outcomes, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE------------------------------------------------------------------

# For the sake of convenience and reproducibility, we are reusing variables after deleting them. However, for this to work, the variables need to be sanitised or deleted after use. If variables are not  sanitised or deleted, data could leak inadvertently.

# Use prviously created Lists of Continuous, Ordinal (Categorical) and Nominal (Binary) Features to avoid duplication and errors. 


########

# Convert nominal variables from factors to integers. Since they are binary, changing them from factors to integers does not alter their behaviour 

mic_modified_cv_train_set_ann <- mic_orig_cv_train_imputed_mice %>% 
          mutate_at(c(mic_nominal_variables),~(.=as.integer(.))) 

mic_modified_cv_test_set_ann <- mic_orig_cv_test_imputed_mice %>% 
          mutate_at(c(mic_nominal_variables),~(.=as.integer(.))) 


# Expand Imputed values using Dummy Variables.

##########

# Retain LET_IS values as they will be lost during conversion

mic_cv_train_let_is_multi <- mic_modified_cv_train_set_ann$LET_IS

mic_cv_test_let_is_multi <- mic_modified_cv_test_set_ann$LET_IS

# Convert LET_IS as factor for one-hot encoding

mic_modified_cv_train_set_ann %<>% mutate(LET_IS = as.factor(LET_IS))

mic_modified_cv_test_set_ann %<>% mutate(LET_IS = as.factor(LET_IS))

########## ONE-HOT ENCODING USING CARET FOR PREDICTORS AND OUTCOME ###################

# We will use CARET for creating the dummy variables for all the predictors and the outcome as it supports both ordered and unordered factors 

# We will use two seperate instances of dummyVars for creations of the dummy variables for the features and the outcome respectively. We can do it using a single instance but the code is very confusing and difficult to follow.

################## Use formula interface to create template for dummy vars. ################## 

########### Features ###########
dummy_vars_cv_features <- dummyVars(formula = "~.", data = mic_modified_cv_train_set_ann[,c(mic_ordinal_variables, mic_part_ordinal_variables)], sep = "_", levelsOnly = FALSE, fullRank = FALSE)
########### Outcome ###########
dummy_vars_cv_outcome <- dummyVars(formula = "~.", data = mic_modified_cv_train_set_ann["LET_IS"],  sep = "_", levelsOnly = FALSE, fullRank = FALSE)

################## Create dummy vars for Training CV and Testing CV datasets. ##################
mic_modified_cv_train_set_ann_dummy <- as.data.frame(predict(dummy_vars_cv_features, mic_modified_cv_train_set_ann[,c(mic_ordinal_variables, mic_part_ordinal_variables)]))

mic_modified_cv_test_set_ann_dummy <- as.data.frame(predict(dummy_vars_cv_features, mic_modified_cv_test_set_ann[,c(mic_ordinal_variables, mic_part_ordinal_variables)]))

mic_modified_cv_train_set_ann_outcome <-  as.data.frame(predict(dummy_vars_cv_outcome, mic_modified_cv_train_set_ann["LET_IS"]))

mic_modified_cv_test_set_ann_outcome <- as.data.frame(predict(dummy_vars_cv_outcome, mic_modified_cv_test_set_ann["LET_IS"]))

##############################################################################################

# Collect Training CV dummy vars names. Same applies for Testing CV. 
dummy_vars_names <- colnames(mic_modified_cv_train_set_ann_dummy)

# Modify Training CV and Testing CV Datasets to remove the existing columns for variables related to  "ID", Ordinal,  Partially Ordinal, Complications and "LET_IS".

mic_modified_cv_train_set_ann %<>% select(-c("ID",mic_ordinal_variables,mic_part_ordinal_variables, mic_complications,"LET_IS"))

mic_modified_cv_test_set_ann %<>% select(-c("ID",mic_ordinal_variables,mic_part_ordinal_variables, mic_complications,"LET_IS"))


# Collect Modified Training CV Column names. Same applies for Testing CV
mic_modified_cv_col_names <- colnames(mic_modified_cv_train_set_ann)

# Combine Collected Modified Training CV column names with Training CV dummy vars names for predictors.
mic_modified_cv_col_names <- c(mic_modified_cv_col_names, dummy_vars_names)

# Create combined Training CV and Testing CV datasets by binding together the Modified CV Training and Testing datasets with the respective dummy vars daatasets
mic_modified_cv_train_set_ann <- cbind(mic_modified_cv_train_set_ann, mic_modified_cv_train_set_ann_dummy)

mic_modified_cv_test_set_ann <- cbind(mic_modified_cv_test_set_ann, mic_modified_cv_test_set_ann_dummy)

# Assign column names to the newly created combined Training CV, Testing CV datasets and the outcome datasets.

colnames(mic_modified_cv_train_set_ann) <- mic_modified_cv_col_names

colnames(mic_modified_cv_test_set_ann) <- mic_modified_cv_col_names

# Remove data that is not required anymore

rm(dummy_vars_cv_features, dummy_vars_cv_outcome, mic_modified_cv_train_set_ann_dummy, mic_modified_cv_test_set_ann_dummy, mic_modified_cv_col_names)


## ----Perform Initial Analysis for MIC Dataset using Neural Networks with Categories for outcome LET_IS using Sequential API, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE----------------------------

########## Hack to achieve Consistent results between Keras Runs ############
# Unload and Reload Keras
unload(package = "keras3", quiet = TRUE)
library(keras3)
# Set Up Environment

set.seed(1024)
keras3::set_random_seed(1024)
#tensorflow::set_random_seed(1024, disable_gpu = TRUE) #Uncomment if using TensorFlow
reticulate::py_set_seed(0)
#############################################################################

########################## WARNING WARNING WARNING ##########################
############## DOUBLE CHECK INPUT FEATURES TO PREVENT LEAKAGE ###############


feature_names <- c(mic_nominal_variables, mic_continuous_variables, dummy_vars_names)


train_features <- as.matrix(mic_modified_cv_train_set_ann[feature_names])
train_targets <- as.matrix(mic_modified_cv_train_set_ann_outcome)

val_features <- as.matrix(mic_modified_cv_test_set_ann[feature_names])
val_targets <- as.matrix(mic_modified_cv_test_set_ann_outcome)

####################################
# As earlier, We need to ensure that the input variables are scaled & centered so that they we do not have any one factor unnecessarily influencing the outcome and all have adequate representation
# As earlier, we also need to remove Columns from the list that produce NA when scaled

####### Normal Scaling ############
feature_names <- colnames(train_features)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features) == 0)

index_features_scaling <- which(!feature_names  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features[,index_features_scaling])

train_features <- train_features[,c(feature_names_for_scaling)]
val_features <- val_features[,c(feature_names_for_scaling)]

train_features %<>% scale()
val_features %<>% scale(center = attr(train_features, "scaled:center"),
                        scale = attr(train_features, "scaled:scale"))

####### Normal Scaling ############
train_features %<>% scale()
val_features %<>% scale(center = attr(train_features, "scaled:center"),
                        scale = attr(train_features, "scaled:scale"))




model <-
  keras3::keras_model_sequential(input_shape = ncol(train_features)) |>
  layer_dense(units = 3072) |>
  layer_dropout(rate = 0.3, seed = 1024) |>
  layer_dense(units = 3072) |>
  layer_dropout(rate = 0.3, seed = 1024) |>
  layer_dense(units = 2304) |>
  layer_dropout(rate = 0.3, seed = 1024) |>
  layer_dense(units = 8, activation = 'softmax')

# Print model for visualisation
# Commented out for report creation
# summary(model)

# Collect counts for generation of initial weights
counts <- table(mic_cv_train_let_is_multi) # Counts for Training Set

# Used for Diagnostic purposes. Commented out for Report creation
# print("counts for training set")
# counts
# print("counts for testing set")
# table(mic_cv_test_let_is_multi) # Counts for Validation Set

# Configure Weights. Weights updated manually with multipliers. 

weight_for_0 = as.numeric(1 / counts["0"])*2.6
weight_for_1 = as.numeric(1 / counts["1"])*0.225
weight_for_2 = as.numeric(1 / counts["2"])*0.05
weight_for_3 = as.numeric(1 / counts["3"])*0.25
weight_for_4 = as.numeric(1 / counts["4"])*0.05
weight_for_5 = as.numeric(1 / counts["5"])*0.0750
weight_for_6 = as.numeric(1 / counts["6"])*0.0875
weight_for_7 = as.numeric(1 / counts["7"])*0.06125

# Train the Model 

model |> compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy','categorical_accuracy')
)


class_weight <- list("0" = weight_for_0,
                     "1" = weight_for_1,
                     "2" = weight_for_2,
                     "3" = weight_for_3,
                     "4" = weight_for_4,
                     "5" = weight_for_5,
                     "6" = weight_for_6,
                     "7" = weight_for_7)



model |> fit(
  train_features, train_targets,
  validation_data = list(val_features, val_targets),
  batch_size = 2048,
  epochs = 30,
  class_weight = class_weight,
  verbose = 0 # set verbose=2 for development and testing
)



# Evaluate Model
# Commented out for Report creation
# model |> evaluate(val_features, val_targets)

# Prepare Predictions. Method is different from those used earlier for Binary outcomes

probs <- model |> predict(val_features)

pred_ann <- max.col(probs) - 1L

# Print table of predictions 
print("================================", quote = FALSE)
print("The Table of Predicted values is", quote= FALSE)
table(as.factor(pred_ann))

# Print table of actual values
print("=====================================", quote = FALSE)
print("The table of actual values is: ", quote = FALSE)
table(as.factor(mic_cv_test_let_is_multi))

# Print overall accuracy
print("================================", quote = FALSE)
print("Validation accuracy is : ",quote = FALSE)
mean(mic_cv_test_let_is_multi == pred_ann)
print("================================", quote = FALSE)

# Print confusion matrix
print("================================",quote=FALSE)
print("The Confusion Matrix is",quote=FALSE)
cm_ann_seq_api_multi <-confusionMatrix(data = as.factor(pred_ann), reference = as.factor(mic_cv_test_let_is_multi))
cm_ann_seq_api_multi 
print("================================",quote=FALSE)

# Extract and print summary of death events 
print("================================",quote=FALSE)
print("The Number of Deaths correctly categorised is :",quote=FALSE)
sum(cm_ann_seq_api_multi$table[2,2], cm_ann_seq_api_multi$table[3,3], cm_ann_seq_api_multi$table[4,4], cm_ann_seq_api_multi$table[5,5], cm_ann_seq_api_multi$table[6,6], cm_ann_seq_api_multi$table[7,7], cm_ann_seq_api_multi$table[8,8])
print("================================",quote=FALSE)

print("================================",quote=FALSE)
print("The Number of Deaths correctly detected is :",quote=FALSE)
sum(cm_ann_seq_api_multi$table[2:8,2:8])
print("================================",quote=FALSE)

# Remove unwanted variables

rm(class_weight, metrics, train_features, train_targets, val_features, val_targets, counts, deaths, feature_names, model, n_deaths_detected, n_deaths_missed, n_live_flagged, pred_correct, val_pred, weight_for_0, weight_for_1, weight_for_2, weight_for_3, weight_for_4, weight_for_5, weight_for_6, weight_for_7)

rm(index_features_scaling, feature_names_for_scaling)

rm(cm_ann_seq_api_multi)



## ----Perform Initial Analysis of MIC Dataset after expansion of Categorical Variables for Multiple Category outcome using Functional API, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE, fig.align='center'----

########## Hack to achieve Consistent results between Keras Runs ############
# Unload and Reload Keras
unload(package = "keras3", quiet = TRUE)
library(keras3)
# Set Up Environment
set.seed(1024)
keras3::set_random_seed(1024)
#tensorflow::set_random_seed(1024, disable_gpu = TRUE) #Uncomment if using TensorFlow
reticulate::py_set_seed(0)
#############################################################################
########################## WARNING WARNING WARNING ##########################
############## DOUBLE CHECK INPUT FEATURES TO PREVENT LEAKAGE ###############



############# Create the Input Data #########################################

########## Demographic & History Features ###########################

train_features_demographic_history <- as.matrix(mic_modified_cv_train_set_ann[,c(mic_demographic_history_col_indices)])
val_features_demographic_history <- as.matrix(mic_modified_cv_test_set_ann[,c(mic_demographic_history_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_demographic_history <- colnames(train_features_demographic_history)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_demographic_history) == 0)

index_features_scaling <- which(!feature_names_demographic_history  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_demographic_history[,index_features_scaling])

train_features_demographic_history <- train_features_demographic_history[,c(feature_names_for_scaling)]
val_features_demographic_history <- val_features_demographic_history[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_demographic_history %<>% scale()
val_features_demographic_history %<>% 
        scale(center = attr(train_features_demographic_history, "scaled:center"),
        scale = attr(train_features_demographic_history, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )



########## Infarction Features ###########################

train_features_infarction <- as.matrix(mic_modified_cv_train_set_ann[,c(mic_infarction_col_indices)])
val_features_infarction <- as.matrix(mic_modified_cv_test_set_ann[,c(mic_infarction_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_infarction <- colnames(train_features_infarction)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_infarction) == 0)

index_features_scaling <- which(!feature_names_infarction  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_infarction[,index_features_scaling])

train_features_infarction <- train_features_infarction[,c(feature_names_for_scaling)]
val_features_infarction <- val_features_infarction[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_infarction %<>% scale()
val_features_infarction %<>% 
        scale(center = attr(train_features_infarction, "scaled:center"),
        scale = attr(train_features_infarction, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## Emergency ICU Features ###########################

train_features_emergency_icu <- as.matrix(mic_modified_cv_train_set_ann[,c(mic_emergency_icu_col_indices)])
val_features_emergency_icu <- as.matrix(mic_modified_cv_test_set_ann[,c(mic_emergency_icu_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_emergency_icu <- colnames(train_features_emergency_icu)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_emergency_icu) == 0)

index_features_scaling <- which(!feature_names_emergency_icu  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_emergency_icu[,index_features_scaling])

train_features_emergency_icu <- train_features_emergency_icu[,c(feature_names_for_scaling)]
val_features_emergency_icu <- val_features_emergency_icu[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_emergency_icu %<>% scale()
val_features_emergency_icu %<>% 
        scale(center = attr(train_features_emergency_icu, "scaled:center"),
        scale = attr(train_features_emergency_icu, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## ECG Features ###########################

train_features_ecg <- as.matrix(mic_modified_cv_train_set_ann[,c(mic_ecg_col_indices)])
val_features_ecg <- as.matrix(mic_modified_cv_test_set_ann[,c(mic_ecg_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_ecg <- colnames(train_features_ecg)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_ecg) == 0)

index_features_scaling <- which(!feature_names_ecg  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_ecg[,index_features_scaling])

train_features_ecg <- train_features_ecg[,c(feature_names_for_scaling)]
val_features_ecg <- val_features_ecg[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_ecg %<>% scale()
val_features_ecg %<>% 
        scale(center = attr(train_features_ecg, "scaled:center"),
        scale = attr(train_features_ecg, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )

########## FT Features ###########################

train_features_ft <- as.matrix(mic_modified_cv_train_set_ann[,c(mic_ft_col_indices)])
val_features_ft <- as.matrix(mic_modified_cv_test_set_ann[,c(mic_ft_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_ft <- colnames(train_features_ft)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_ft) == 0)

index_features_scaling <- which(!feature_names_ft  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_ft[,index_features_scaling])

train_features_ft <- train_features_ft[,c(feature_names_for_scaling)]
val_features_ft <- val_features_ft[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_ft %<>% scale()
val_features_ft %<>% 
        scale(center = attr(train_features_ft, "scaled:center"),
        scale = attr(train_features_ft, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## Serum Features ###########################

train_features_serum <- as.matrix(mic_modified_cv_train_set_ann[,c(mic_serum_col_indices)])
val_features_serum <- as.matrix(mic_modified_cv_test_set_ann[,c(mic_serum_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_serum <- colnames(train_features_serum)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_serum) == 0)

index_features_scaling <- which(!feature_names_serum  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_serum[,index_features_scaling])

train_features_serum <- train_features_serum[,c(feature_names_for_scaling)]
val_features_serum <- val_features_serum[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_serum %<>% scale()
val_features_serum %<>% 
        scale(center = attr(train_features_serum, "scaled:center"),
        scale = attr(train_features_serum, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## Relapse Features ###########################

train_features_relapse <- as.matrix(mic_modified_cv_train_set_ann[,c(mic_relapse_col_indices)])
val_features_relapse <- as.matrix(mic_modified_cv_test_set_ann[,c(mic_relapse_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_relapse <- colnames(train_features_relapse)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_relapse) == 0)

index_features_scaling <- which(!feature_names_relapse  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_relapse[,index_features_scaling])

train_features_relapse <- train_features_relapse[,c(feature_names_for_scaling)]
val_features_relapse <- val_features_relapse[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_relapse %<>% scale()
val_features_relapse %<>% 
        scale(center = attr(train_features_relapse, "scaled:center"),
        scale = attr(train_features_relapse, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## Medicine Features ###########################

train_features_medicine <- as.matrix(mic_modified_cv_train_set_ann[,c(mic_medicine_col_indices)])
val_features_medicine <- as.matrix(mic_modified_cv_test_set_ann[,c(mic_medicine_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_medicine <- colnames(train_features_medicine)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_medicine) == 0)

index_features_scaling <- which(!feature_names_medicine  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_medicine[,index_features_scaling])

train_features_medicine <- train_features_medicine[,c(feature_names_for_scaling)]
val_features_medicine <- val_features_medicine[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_medicine %<>% scale()
val_features_medicine %<>% 
        scale(center = attr(train_features_medicine, "scaled:center"),
        scale = attr(train_features_medicine, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )



############## CREATE TRAINING AND VALIDATION TARGETS ##############################
train_targets <- as.matrix(mic_modified_cv_train_set_ann_outcome)
val_targets <- as.matrix(mic_modified_cv_test_set_ann_outcome)

###################################################################################

# Let us define the input shapes. 
input_shape_demographic_history  <- ncol(train_features_demographic_history)
input_shape_infarction <- ncol(train_features_infarction)
input_shape_emergency_icu <- ncol(train_features_emergency_icu)
input_shape_ecg <- ncol(train_features_ecg)
input_shape_ft <- ncol(train_features_ft)
input_shape_serum <- ncol(train_features_serum)
input_shape_relapse <- ncol(train_features_relapse)
input_shape_medicine <- ncol(train_features_medicine)



# Let us build the Keras Inputs & Features
input_demographic_history <- keras_input(shape(input_shape_demographic_history), name = "demographic_history")
input_infarction <- keras_input(shape(input_shape_infarction), name = "infarction")
input_emergency_icu <- keras_input(shape(input_shape_emergency_icu), name = "emergency_icu")
input_ecg <- keras_input(shape(input_shape_ecg), name = "ecg")
input_ft <- keras_input(shape(input_shape_ft), name = "ft")
input_serum <- keras_input(shape(input_shape_serum), name = "serum")
input_relapse <- keras_input(shape(input_shape_relapse), name = "relapse")
input_medicine <- keras_input(shape(input_shape_medicine), name = "medicine")

########################################

demographic_history_features <- 
    layer_dense(object = input_demographic_history, units = 1024) |> 
    layer_dense(units = 1024) |>
    layer_dense(units = 1024) |>
    layer_dense(units = 1024) |>
    layer_dropout(rate = 0.3, seed = 1024) 

infarction_features <- 
    layer_dense(object = input_infarction, units = 2048) |> 
    layer_dense(units = 2048) |>
    layer_dropout(rate = 0.3, seed = 1024) 

emergency_icu_features <- 
    layer_dense(object = input_emergency_icu, units = 1280) |> 
    layer_dense(units = 1280) |>
    layer_dense(units = 1280) |>
    layer_dense(units = 1280) |>
    layer_dropout(rate = 0.3, seed = 1024)

ecg_features <- 
    layer_dense(object = input_ecg, units = 1024) |> 
    layer_dense(units = 1024) |>
    layer_dense(units = 1024) |>
    layer_dense(units = 1024) |>
    layer_dropout(rate = 0.3, seed = 1024)
      
ft_features <- 
    layer_dense(object = input_ft, units = 64) |> 
    layer_dropout(rate = 0.3, seed = 1024)    

serum_features <- 
    layer_dense(object = input_serum, units = 64) |> 
    layer_dropout(rate = 0.3, seed = 1024)

relapse_features <- 
    layer_dense(object = input_relapse, units = 96) |>
    layer_dropout(rate = 0.3, seed = 1024) 

medicine_features <- 
    layer_dense(object = input_medicine, units = 960) |> 
    layer_dense(units = 960) |>
    layer_dense(units = 960) |>
    layer_dense(units = 960) |>
    layer_dropout(rate = 0.3, seed = 1024)


########################################

# Let us combine the Feature Layers together

combined_features <- layer_concatenate(list(demographic_history_features, infarction_features, emergency_icu_features, ecg_features, ft_features, serum_features, relapse_features, medicine_features))

pred_functional_api <- layer_dense(object = combined_features, units = 8, activation = "softmax")

# Instantiate an end-to-end model 

functional_api_model <- keras_model(
  inputs = list(input_demographic_history, input_infarction, input_emergency_icu, input_ecg, input_ft, input_serum, input_relapse, input_medicine),
  outputs = list(pred_functional_api)
)

# Let us Plot the model for Visualisation
# Commented out for Report Creation
#print("The Plot of the Keras Functional API model is",quote = FALSE)
#print("=============================================",quote = FALSE)
#plot(functional_api_model, show_shapes = TRUE)
#print("=============================================",quote = FALSE)

#print("The summary of the Keras Functional API model is",quote = FALSE)
#print("=============================================",quote = FALSE)
#summary(functional_api_model)
#print("=============================================",quote = FALSE)

# Collect counts for generation of initial weights
counts <- table(mic_cv_train_let_is_multi) # Counts for Training Set

# Used for Diagnostic purposes. Commented out for Report creation
# print("counts for training set")
# counts
# print("counts for testing set")
# table(mic_cv_test_let_is_multi) # Counts for Validation Set

# Configure weights. Weights are updated manually with multipliers
weight_for_0 = as.numeric(1 / counts["0"])*3.5
weight_for_1 = as.numeric(1 / counts["1"])*1.1
weight_for_2 = as.numeric(1 / counts["2"])*0.2
weight_for_3 = as.numeric(1 / counts["3"])*0.7
weight_for_4 = as.numeric(1 / counts["4"])*0.15
weight_for_5 = as.numeric(1 / counts["5"])*0.3
weight_for_6 = as.numeric(1 / counts["6"])*0.3
weight_for_7 = as.numeric(1 / counts["7"])*0.4


# Train the Model 

functional_api_model |> compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy','categorical_accuracy')
)


##############
class_weight <- list("0" = weight_for_0,
                     "1" = weight_for_1,
                     "2" = weight_for_2,
                     "3" = weight_for_3,
                     "4" = weight_for_4,
                     "5" = weight_for_5,
                     "6" = weight_for_6,
                     "7" = weight_for_7)

######  Fit model ##############


functional_api_model |> 
  fit(
  x = list(demographic_history = train_features_demographic_history, infarction = train_features_infarction, emergency_icu = train_features_emergency_icu, ecg = train_features_ecg, ft = train_features_ft, serum = train_features_serum, relapse = train_features_relapse, medicine = train_features_medicine ),
  y = train_targets,
  validation_data = list(list(val_features_demographic_history,  val_features_infarction,  val_features_emergency_icu, val_features_ecg, val_features_ft, val_features_serum, val_features_relapse, val_features_medicine),val_targets),
  batch_size = 2048,
  epochs = 30,
  class_weight = class_weight,
  verbose = 0 # Set verbose=2 during development, tuning and testing
)


# Evaluate Model
# Commented out for Report creation
# functional_api_model |> evaluate(list(val_features_demographic_history,  val_features_infarction,   val_features_emergency_icu, val_features_ecg, val_features_ft, val_features_serum, val_features_relapse, val_features_medicine),val_targets)

# Prepare Predictions. Method is different from those used earlier for Binary outcomes

probs <- functional_api_model |> predict(list(val_features_demographic_history,  val_features_infarction, val_features_emergency_icu, val_features_ecg, val_features_ft, val_features_serum, val_features_relapse, val_features_medicine))

pred_ann <- max.col(probs) - 1L 

# Print table of predictions 
print("================================", quote = FALSE)
print("The Table of Predicted values is: ", quote= FALSE)
table(as.factor(pred_ann))

# Print table of actual values
print("=====================================", quote = FALSE)
print("The table of actual values is", quote = FALSE)
table(as.factor(mic_cv_test_let_is_multi))

# Print overall accuracy
print("================================", quote = FALSE)
print("Validation accuracy is : ",quote = FALSE)
mean(mic_cv_test_let_is_multi  == pred_ann)

# Print confusion matrix
print("================================",quote=FALSE)
print("The Confusion Matrix is", quote=FALSE)
cm_ann_seq_func_multi <- confusionMatrix(data = as.factor(pred_ann), reference = as.factor(mic_cv_test_let_is_multi ))
cm_ann_seq_func_multi
print("================================",quote=FALSE)

# Extract and Print summary about death events
print("================================", quote=FALSE)
print("The Number of Deaths correctly categorised is :",quote=FALSE)
sum(cm_ann_seq_func_multi$table[2,2], cm_ann_seq_func_multi$table[3,3], cm_ann_seq_func_multi$table[4,4], cm_ann_seq_func_multi$table[5,5], cm_ann_seq_func_multi$table[6,6], cm_ann_seq_func_multi$table[7,7], cm_ann_seq_func_multi$table[8,8])
print("================================", quote=FALSE)

print("================================", quote=FALSE)
print("The Number of Deaths correctly detected is :",quote=FALSE)
sum(cm_ann_seq_func_multi$table[2:8,2:8])
print("================================", quote=FALSE)

# Remove data that is no longer required

rm(metrics, counts, deaths_functional_api, feature_names, model, n_deaths_detected, n_deaths_missed, n_live_flagged, pred_correct_functional_api, val_pred_functional_api)

rm( demographic_history_features, ecg_features, emergency_icu_features, ft_features,  infarction_features, medicine_features, relapse_features, serum_features, combined_features)

rm(input_demographic_history, input_ecg, input_emergency_icu, input_ft,  input_infarction, input_medicine, input_relapse, input_serum)

rm( train_features_demographic_history, train_features_ecg, train_features_emergency_icu,  train_features_ft,   train_features_infarction, train_features_medicine, train_features_relapse, train_features_serum)

rm( val_features_demographic_history, val_features_ecg, val_features_emergency_icu, val_features_ft,  val_features_infarction, val_features_medicine, val_features_relapse, val_features_serum)

rm( feature_names_demographic_history, feature_names_ecg, feature_names_emergency_icu,  feature_names_ft,   feature_names_infarction, feature_names_medicine, feature_names_relapse, feature_names_serum)

rm( input_shape_demographic_history, input_shape_ecg, input_shape_emergency_icu, input_shape_ft,  input_shape_infarction, input_shape_medicine, input_shape_relapse, input_shape_serum)

rm(mic_blockage_col_indices, mic_demographic_history_col_indices, mic_ecg_col_indices, mic_emergency_icu_col_indices, mic_endocrine_col_indices, mic_ft_col_indices, mic_hf_rhythm_col_indices, mic_hypertension_col_indices, mic_infarction_col_indices, mic_medicine_col_indices, mic_pulmonary_col_indices, mic_relapse_col_indices, mic_serum_col_indices, mic_all_col_indices)

rm(functional_api_model, pred_functional_api,probs, pred_ann, mic_train_let_is, mic_cv_test_let_is_multi, mic_cv_train_let_is_multi)

rm(weight_for_0, weight_for_1, weight_for_2, weight_for_3, weight_for_4, weight_for_5, weight_for_6, weight_for_7, class_weight)

rm(train_features, train_targets, val_features, val_targets)

rm(cm_ann_seq_func_multi)


## ----MIC Data Clean up Cross Validation Sets Keras ANN Categorical outcome, include=FALSE, warning=FALSE, echo=FALSE, message = FALSE----------------------------------------------------------------------------

# Remove data that is no longer required 
rm(mic_modified_cv_train_set_ann, mic_modified_cv_test_set_ann, mic_modified_train_set_ann) 

rm(mic_modified_cv_train_set_ann_outcome, mic_modified_cv_test_set_ann_outcome, dummy_vars_names)

gc()



## ----MIC Data Clean up Cross Validation Sets, include=FALSE, warning=FALSE, echo=FALSE, message = FALSE----------------------------------------------------------------------------------------------------------

# Clean up Data and run the garbage collector to free up memory

rm(mic_orig_cv_train_set, mic_orig_cv_test_set, mic_orig_cv_train_imputed_mice, mic_orig_cv_test_imputed_mice)

rm(mic_test_index_cv)

gc()


## ----Predictions using Naive Bayes for MIC Holdout Dataset without imputation - 2, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE----------------------------------------------------------------------

# For the sake of convenience and reproducibility, we are reusing variables after deleting them. However, for this to work, the variables need to be sanitised or deleted after use. If variables are not  sanitised or deleted, data could leak inadvertently.

# Change LET_IS to have a Single Category for Deaths rather than multiple 

mic_modified_train <- mic_orig_train %>% 
    mutate( LET_IS = ifelse(LET_IS==1 | LET_IS==2 | LET_IS==3 | LET_IS==4 | LET_IS==5 | LET_IS==6 | LET_IS==7, 1, 0)) %>%
    mutate(LET_IS = as.factor(LET_IS))

mic_modified_test <- mic_orig_test %>% 
    mutate( LET_IS = ifelse(LET_IS==1 | LET_IS==2 | LET_IS==3 | LET_IS==4 | LET_IS==5 | LET_IS==6 | LET_IS==7, 1, 0)) %>%
    mutate(LET_IS = as.factor(LET_IS))


# We will configure 
# S_AD_KBRIG, S_AD_ORIT, NA_BLOOD, D_AD_KBRIG, D_AD_ORIT and ROE as Poisson
# AGE, K_BLOOD, ALT_BLOOD, AST_BLOOD and L_BLOOD as Gaussian
# KFK_BLOOD = 0 is set already for all observations


mic_modified_train <- mic_modified_train %>% 
      mutate_at(c("S_AD_KBRIG", "S_AD_ORIT", "NA_BLOOD", "D_AD_KBRIG", "D_AD_ORIT","ROE"), ~(.= as.integer(.))) %>%
      mutate(AGE = as.numeric(AGE))

mic_modified_test <- mic_modified_test %>% 
      mutate_at(c("S_AD_KBRIG", "S_AD_ORIT", "NA_BLOOD", "D_AD_KBRIG", "D_AD_ORIT","ROE"), ~(.= as.integer(.)))  %>%
      mutate(AGE = as.numeric(AGE))



###########
# Fit Naive Bayes
fit_nb_native_LET_IS_final <- naive_bayes(x = mic_modified_train[,2:112], y = mic_modified_train$LET_IS, laplace = 1, usekernel = FALSE, usepoisson = TRUE)

# Print Summary
summary(fit_nb_native_LET_IS_final)

# Make Predictions
pred_nb_test_native_LET_IS_final <-predict(object = fit_nb_native_LET_IS_final, newdata = mic_modified_test[,2:112])

# Print table of predictions
print("================================", quote=FALSE)
print("The table of predictions is", quote = FALSE)
table(as.factor(pred_nb_test_native_LET_IS_final))
print("================================", quote=FALSE)

# Print table of actual values
print("================================", quote=FALSE)
print("The table of actual values is", quote = FALSE)
table(mic_modified_test$LET_IS)
print("================================", quote=FALSE)

# Print overall accuracy
print("The accuracy of predictions is", quote = FALSE)
mean(pred_nb_test_native_LET_IS_final == mic_modified_test$LET_IS)

# Record and Print confusion matrix
print("The confusion matrix is", quote = FALSE)
print("================================", quote=FALSE)
mic_nb_cm_bin_2 <- confusionMatrix(data = pred_nb_test_native_LET_IS_final, reference = mic_modified_test$LET_IS)
mic_nb_cm_bin_2
print("================================", quote=FALSE)


# Remove Data that is no longer required

rm(fit_nb_native_LET_IS_final, pred_nb_test_native_LET_IS_final, mic_modified_train, mic_modified_test)




## ----MIC Holdout Set Predictions Naive Bayes - Imputation using the Mice Package and Random Forest as the Imputation Method - 1, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE------------------------

# For the sake of convenience and reproducibility, we are reusing variables after deleting them. However, for this to work, the variables need to be sanitised or deleted after use. If variables are not  sanitised or deleted, data could leak inadvertently.


# Set up new Training set with imputed values. Configure LET_IS for binary prediction

mic_modified_train <- mic_orig_train_imputed_mice %>% 
    mutate( LET_IS = ifelse(LET_IS==1 | LET_IS==2 | LET_IS==3 | LET_IS==4 | LET_IS==5 | LET_IS==6 | LET_IS==7, 1, 0)) %>%
    mutate(LET_IS = as.factor(LET_IS))

mic_modified_test <- mic_orig_test_imputed_mice %>% 
    mutate( LET_IS = ifelse(LET_IS==1 | LET_IS==2 | LET_IS==3 | LET_IS==4 | LET_IS==5 | LET_IS==6 | LET_IS==7, 1, 0)) %>%
    mutate(LET_IS = as.factor(LET_IS))

# Configure variables 

# We will configure 
# S_AD_KBRIG, S_AD_ORIT, NA_BLOOD, D_AD_KBRIG, D_AD_ORIT and ROE as Poisson
# AGE, K_BLOOD, ALT_BLOOD, AST_BLOOD and L_BLOOD as Gaussian
# KFK_BLOOD = 0 is set already for all observations

mic_modified_train <- mic_modified_train %>% 
      mutate_at(c("S_AD_KBRIG", "S_AD_ORIT", "NA_BLOOD"), ~(.= as.integer(.))) 

mic_modified_test <- mic_modified_test %>% 
      mutate_at(c("S_AD_KBRIG", "S_AD_ORIT", "NA_BLOOD"), ~(.= as.integer(.))) 


###########
# Fit Naive Bayes
fit_nb_native_LET_IS_final <- naive_bayes(x = mic_modified_train[,2:112], y = mic_modified_train$LET_IS, laplace = 1, usekernel = FALSE, usepoisson = TRUE)

# Print Summary
summary(fit_nb_native_LET_IS_final)

# Prepare Predictions
pred_nb_test_native_LET_IS_final <-predict(object = fit_nb_native_LET_IS_final, newdata = mic_modified_test)

# Print table of predictions
print("================================", quote=FALSE)
print("The table of predictions is: ", quote = FALSE)
table(as.factor(pred_nb_test_native_LET_IS_final))
print("================================", quote=FALSE)

# Print table of actual values
print("================================", quote=FALSE)
print("The table of actual values is: ", quote = FALSE)
table(mic_modified_test$LET_IS)
print("================================", quote=FALSE)

# Print overall accuracy
print("The accuracy of predictions is: ", quote = FALSE)
mean(pred_nb_test_native_LET_IS_final == mic_modified_test$LET_IS)

# Record and Print confusion matrix
print("The confusion matrix is: ", quote = FALSE)
print("================================",quote=FALSE)
mic_nb_wi_cm_bin_1 <- confusionMatrix(data = pred_nb_test_native_LET_IS_final, reference = mic_modified_test$LET_IS)
mic_nb_wi_cm_bin_1
print("================================",quote=FALSE)



# Remove Data that is no longer required

rm(fit_nb_native_LET_IS_final, pred_nb_test_native_LET_IS_final)




## ----Perform Final analysis of MIC dataset using XGBoost for binary outcome, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE----------------------------------------------------------------------------

# For the sake of convenience and reproducibility, we are reusing variables after deleting them. However, for this to work, the variables need to be sanitised or deleted after use. If variables are not  sanitised or deleted, data could leak inadvertently. 

# Use Original Dataset without Imputation

# Change LET_IS to have a Single Category for Deaths rather than multiple. Set up outcomes as Integers specifically or they could get coerced to character vectors otherwise. 

mic_data_modified_xgboost <- mic_data %>% mutate( LET_IS = ifelse(LET_IS==1 | LET_IS==2 | LET_IS==3 | LET_IS==4 | LET_IS==5 | LET_IS==6 | LET_IS==7, as.integer(1), as.integer(0)))


# DO NOT CONVERT CATEGORICAL(ORDINAL) AND BINARY(NOMINAL) VARIABLES TO FACTORS
# CONVERSION USING as.matrix() CAUSES THEM TO BE COERCED TO CHARACTER VECTORS

mic_data_modified_xgboost <- mic_data_modified_xgboost %>% mutate_at(mic_continuous_variables, ~as.numeric(.))

mic_modified_train_xgboost <- mic_data_modified_xgboost[-mic_test_index,]
mic_modified_test_xgboost <- mic_data_modified_xgboost[mic_test_index,]



###########
# Prepare Dmatrix for XGBoost
dtrain_final <- xgb.DMatrix(data = as.matrix(mic_modified_train_xgboost[,c(2:112)]), label= mic_modified_train_xgboost$LET_IS, nthread = 8)

# Fit XGBoost
fit_xgboost_mic_final <- xgb.train(
    data = dtrain_final,
    max_depth = 10,
    eta = 0.15,
    nthread = 8,
    nrounds = 200,
    objective = "binary:logistic",
    verbose = 0 # Set verbose=2 during development and testing
)

# Print Summary
print("================================",quote=FALSE)
print("The Details for XGBoost are", quote = FALSE)
fit_xgboost_mic_final
print("================================",quote=FALSE)

# Prepare Predictions
pred_xgboost_mic_final <-predict(object = fit_xgboost_mic_final, newdata = as.matrix(mic_modified_test_xgboost[,c(2:112)]))

pred_xgboost_mic_binary_final <- as.numeric(pred_xgboost_mic_final > 0.5)

# Print table of predictions
print("================================",quote=FALSE)
print("The table of predictions is", quote = FALSE)
table(as.factor(pred_xgboost_mic_binary_final ))
print("================================",quote=FALSE)

# Print table of actual values
print("================================",quote=FALSE)
print("The table of actual values is", quote = FALSE)
table(mic_modified_test_xgboost$LET_IS)
print("================================",quote=FALSE)

# Print overall accuracy
print("================================",quote=FALSE)
mean(pred_xgboost_mic_binary_final == mic_modified_test_xgboost$LET_IS)

# Record and print confusion matrix
print("The confusion matrix is", quote = FALSE)
print("================================",quote=FALSE)
mic_xgb_cm_bin <- confusionMatrix(data = as.factor(pred_xgboost_mic_binary_final), reference = as.factor(mic_modified_test_xgboost$LET_IS))
mic_xgb_cm_bin
print("================================",quote=FALSE)

# Remove data that is no longer required

rm(fit_xgboost_mic_final, pred_xgboost_mic_final, pred_xgboost_mic_binary_final)

rm(mic_modified_test, mic_modified_train, mic_data_modified, dtrain_final)




## ----Prepare MIC Holdout Test Dataset for expansion of Categorical Variables and binary outcome,, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE-------------------------------------------------------



# As previously we will use only a Single Category to classify all Deaths

# Let us use the imputed dataset as ANN cannot work with missing values 


########

# Convert nominal variables from factors to integers. Since they are binary, changing them from factors to numerics does not alter their behaviour 

mic_modified_train_set_ann <- mic_orig_train_imputed_mice %>% 
          mutate_at(c(mic_nominal_variables),~(.=as.integer(.))) 

mic_modified_test_set_ann <- mic_orig_test_imputed_mice %>% 
          mutate_at(c(mic_nominal_variables),~(.=as.integer(.))) 


# Expand Imputed values using Dummy Variables.

##########

# Convert LET_IS to binary values

mic_modified_train_set_ann  <- mic_modified_train_set_ann  %>% 
        mutate( LET_IS = ifelse(LET_IS==1 | LET_IS==2 | LET_IS==3 | LET_IS==4 | LET_IS==5 | LET_IS==6 | LET_IS==7, as.integer(1), as.integer(0))) 

mic_modified_test_set_ann  <- mic_modified_test_set_ann  %>% 
        mutate( LET_IS = ifelse(LET_IS==1 | LET_IS==2 | LET_IS==3 | LET_IS==4 | LET_IS==5 | LET_IS==6 | LET_IS==7, as.integer(1), as.integer(0))) 



# Retain "LET_IS" column as we will remove it during conversion

mic_train_let_is_final_bin <- mic_modified_train_set_ann$LET_IS
mic_test_let_is_final_bin <- mic_modified_test_set_ann$LET_IS


########## ONE-HOT ENCODING USING CARET #################################

# Use formula interface to create template for dummy vars. Create dummy vars for Ordinal and Partially Ordinal variables
dumms_vars_template <- dummyVars(formula = "~.", data = mic_modified_train_set_ann[,c(mic_ordinal_variables, mic_part_ordinal_variables)], sep = "_", levelsOnly = FALSE, fullRank = TRUE)

# Create dummy vars for Training  and Testing datasets. 
mic_modified_train_set_ann_dummy <- as.data.frame(predict(dumms_vars_template, mic_modified_train_set_ann[,c(mic_ordinal_variables, mic_part_ordinal_variables)]))

mic_modified_test_set_ann_dummy <- as.data.frame(predict(dumms_vars_template, mic_modified_test_set_ann[,c(mic_ordinal_variables, mic_part_ordinal_variables)]))

# Collect Training dummy vars names. Same applies for Testing  
dummy_vars_names <- colnames(mic_modified_train_set_ann_dummy)

# Modify Training  and Testing  Datasets to remove the existing columns for variables related to  "ID", Ordinal,  Partially Ordinal and Complications.
mic_modified_train_set_ann %<>% select(-c("ID",mic_ordinal_variables,mic_part_ordinal_variables, mic_complications,"LET_IS"))

mic_modified_test_set_ann %<>% select(-c("ID",mic_ordinal_variables,mic_part_ordinal_variables, mic_complications,"LET_IS"))

# Collect Modified Training Column names. Same applies for Testing 
mic_modified_col_names <- colnames(mic_modified_train_set_ann)

# Combine Collected Modified Training column names with Training dummy vars names
mic_modified_col_names <- c(mic_modified_col_names, dummy_vars_names)

# Create combined Training  and Testing  datasets by binding together the Modified Training and Testing datasets with their respective dummy vars daatasets
mic_modified_train_set_ann <- cbind(mic_modified_train_set_ann, mic_modified_train_set_ann_dummy)

mic_modified_test_set_ann <- cbind(mic_modified_test_set_ann, mic_modified_test_set_ann_dummy)

# Assign column names to the newly created combined Training and Testing datasets
colnames(mic_modified_train_set_ann) <- mic_modified_col_names

colnames(mic_modified_test_set_ann) <- mic_modified_col_names

# Remove data that is not required anymore

rm(dumms_vars_template, mic_modified_train_set_ann_dummy, mic_modified_test_set_ann_dummy, mic_modified_col_names)


## ----Perform Final Analysis for MIC Dataset using Neural Networks after expansion using Sequential API and binary outcome, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE------------------------------

########## Hack to achieve Consistent results between Keras Runs ############
# Unload and Reload Keras
unload(package = "keras3", quiet = TRUE)
library(keras3)
# Set Up Environment
set.seed(1024)
keras3::set_random_seed(1024)
#tensorflow::set_random_seed(1024, disable_gpu = TRUE) #Uncomment if using TensorFlow
reticulate::py_set_seed(0)
#############################################################################


########## Create ANN for Predictions #######################################

########################## WARNING WARNING WARNING ##########################
############## DOUBLE CHECK INPUT FEATURES TO PREVENT LEAKAGE ###############


feature_names <- c(mic_nominal_variables, mic_continuous_variables, dummy_vars_names)

train_features <- as.matrix(mic_modified_train_set_ann[feature_names])
train_targets <- as.matrix(mic_train_let_is_final_bin)

val_features <- as.matrix(mic_modified_test_set_ann[feature_names])
val_targets <- as.matrix(mic_test_let_is_final_bin)

####################################
# We need to ensure that the input variables are scaled & centered so that they we do not have any one factor unnecessarily influencing the outcome and all have adequate representation
# Remove Columns from the list that produce NA when scaled

####### Normal Scaling ############
feature_names <- colnames(train_features)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features) == 0)

index_features_scaling <- which(!feature_names  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features[,index_features_scaling])

train_features <- train_features[,c(feature_names_for_scaling)]
val_features <- val_features[,c(feature_names_for_scaling)]

train_features %<>% scale()
val_features %<>% scale(center = attr(train_features, "scaled:center"),
                        scale = attr(train_features, "scaled:scale"))

# Let us build the Nueral Network 


model <-
  keras3::keras_model_sequential(input_shape = ncol(train_features)) |> 
  layer_dense(units = 256, activation = "relu") |> 
  layer_dropout(rate = 0.3, seed = 1024) |>
  layer_dense(units = 256, activation = "relu") |> 
  layer_dropout(rate = 0.3, seed = 1024) |>
  layer_dense(units = 192, activation = "relu") |> 
  layer_dropout(rate = 0.3, seed = 1024) |>
  layer_dense(units = 192, activation = "relu") |> 
  layer_dropout(rate = 0.3, seed = 1024) |>
  layer_dense(1, activation = "sigmoid")

# Collect counts for initial weight generation
counts <- table(mic_train_let_is_final_bin) # Counts for Training Set

# Used for Diagnostic purposes. Commented out for Report creation
# print("counts for training set")
# counts
# print("counts for testing set")
# table(mic_test_let_is_final_bin) # Counts for Validation Set

# Configure Weights. Weights are manually updated with multipliers
weight_for_0 = as.numeric(1 / counts["0"]) 
weight_for_1 = as.numeric(1 / counts["1"]) * 0.98 


# Train the Model 

metrics <- list(
  metric_false_negatives(name = "fn"),
  metric_false_positives(name = "fp"),
  metric_true_negatives(name = "tn"),
  metric_true_positives(name = "tp"),
  metric_precision(name = "precision"),
  metric_recall(name = "recall")
)
model |> compile(
  optimizer = optimizer_adam(1e-3),
  loss = "binary_crossentropy",
  metrics = metrics
)

class_weight <- list("0" = weight_for_0,
                     "1" = weight_for_1)

model |> fit(
  train_features, train_targets,
  validation_data = list(val_features, val_targets),
  class_weight = class_weight,
  batch_size = 2048,
  epochs = 30,
  verbose = 0 # Set verbose=2 during development, tuning and testing
)

# Prepare Predictions

val_pred_final <- model %>%
  predict(val_features) %>%
  { as.integer(. > 0.5) }

pred_correct_final <- mic_test_let_is_final_bin == val_pred_final

# Print table of predictions
print("================================",quote=FALSE)
print("The Table of Predicted values is", quote = FALSE)
table(as.numeric(!pred_correct_final))
print("================================",quote=FALSE)

# Print table of actual values
print("================================",quote=FALSE)
print("The table of actual values is", quote = FALSE)
table(as.factor(mic_test_let_is_final_bin))
print("================================",quote=FALSE)

# Print overall accuracy
print("================================",quote=FALSE)
print(c("The overall accuracy is : ", round(mean(pred_correct_final), digits = 4)))
print("================================",quote=FALSE)

# Record and Print confusion matrix
print("================================",quote=FALSE)
print("The Confusion Matrix is", quote = FALSE)
mic_ann_seq_cm_bin <- confusionMatrix(data = as.factor(val_pred_final), reference = as.factor(mic_test_let_is_final_bin))
mic_ann_seq_cm_bin
print("================================",quote=FALSE)

# Collect death events
deaths_final <- mic_test_let_is_final_bin == 1

# Prepare and print summary of death events
n_deaths_detected <- sum(deaths_final & pred_correct_final)
n_deaths_missed <- sum(deaths_final & !pred_correct_final)
n_live_flagged <- sum(!deaths_final & !pred_correct_final)

print(c("deaths detected: ",n_deaths_detected), quote = FALSE)
print(c("deaths missed: ",n_deaths_missed), quote = FALSE)
print(c("survival cases flagged as deaths: ",n_live_flagged), quote = FALSE)


# Remove Data that is no longer required

rm(class_weight, metrics, train_features, train_targets, val_features, val_targets, counts, deaths_final, feature_names, model, n_deaths_detected, n_deaths_missed, n_live_flagged, pred_correct_final, val_pred_final, mic_binary_seq_api_class_weight)

rm(index_features_scaling, feature_names_for_scaling)

rm(weight_for_0, weight_for_1)

rm(mic_binary_seq_api_class_weight)


## ----Prepare Final MIC Dataset with Holdout Set for expansion of Categorical Variables and binary outcome using Functional API, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE-------------------------


# Initialise Empty Vectors to collect and store respective column indices
mic_demographic_history_col_indices <- c() 
mic_infarction_col_indices <- c()
mic_emergency_icu_col_indices <- c()
mic_ecg_col_indices <- c()
mic_ft_col_indices <- c()
mic_serum_col_indices <- c()
mic_relapse_col_indices <- c()
mic_medicine_col_indices <- c() 


# Extract the Column indices and store them in respective vectors for each feature set. 
##### 
for (i in 1:(length(mic_demographic_history_features))) {
mic_demographic_history_col <- str_which( string = colnames(mic_modified_train_set_ann), pattern = mic_demographic_history_features[i])
 mic_demographic_history_col_indices <- c(mic_demographic_history_col_indices, mic_demographic_history_col)
}
rm(mic_demographic_history_col,i)


##### 
for (i in 1:(length(mic_infarction_features))) {
mic_infarction_col <- str_which( string = colnames(mic_modified_train_set_ann), pattern = mic_infarction_features[i])
 mic_infarction_col_indices <- c(mic_infarction_col_indices, mic_infarction_col)
}
rm(mic_infarction_col,i)


##### 
for (i in 1:(length(mic_emergency_icu_features))) {
mic_emergency_icu_col <- str_which( string = colnames(mic_modified_train_set_ann), pattern = mic_emergency_icu_features[i])
 mic_emergency_icu_col_indices <- c(mic_emergency_icu_col_indices, mic_emergency_icu_col)
}
rm(mic_emergency_icu_col,i)

##### 
for (i in 1:(length(mic_ecg_features))) {
mic_ecg_col <- str_which( string = colnames(mic_modified_train_set_ann), pattern = mic_ecg_features[i])
 mic_ecg_col_indices <- c(mic_ecg_col_indices, mic_ecg_col)
}
rm(mic_ecg_col,i)

##### 
for (i in 1:(length(mic_ft_features))) {
mic_ft_col <- str_which( string = colnames(mic_modified_train_set_ann), pattern = mic_ft_features[i])
 mic_ft_col_indices <- c(mic_ft_col_indices, mic_ft_col)
}
rm(mic_ft_col,i)

##### 
for (i in 1:(length(mic_serum_features))) {
mic_serum_col <- str_which( string = colnames(mic_modified_train_set_ann), pattern = mic_serum_features[i])
 mic_serum_col_indices <- c(mic_serum_col_indices, mic_serum_col)
}
rm(mic_serum_col,i)

##### 
for (i in 1:(length(mic_relapse_features))) {
mic_relapse_col <- str_which( string = colnames(mic_modified_train_set_ann), pattern = mic_relapse_features[i])
 mic_relapse_col_indices <- c(mic_relapse_col_indices, mic_relapse_col)
}
rm(mic_relapse_col,i)

##### 
for (i in 1:(length(mic_medicine_features))) {
mic_medicine_col <- str_which( string = colnames(mic_modified_train_set_ann), pattern = mic_medicine_features[i])
 mic_medicine_col_indices <- c(mic_medicine_col_indices, mic_medicine_col)
}
rm(mic_medicine_col,i)




## ----Perform Final Analysis of MIC Dataset after expansion of Categorical Variables for binary outcome using Functional API, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE----------------------------

########## Hack to achieve Consistent results between Keras Runs ############
# Unload and Reload Keras
unload(package = "keras3", quiet = TRUE)
library(keras3)
# Set Up Environment
set.seed(1024)
keras3::set_random_seed(1024)
#tensorflow::set_random_seed(1024, disable_gpu = TRUE) #Uncomment if using TensorFlow
reticulate::py_set_seed(0)
#############################################################################


############# Create the Input Data #########################################


########## Demographic & History Features ###########################

train_features_demographic_history <- as.matrix(mic_modified_train_set_ann[,c(mic_demographic_history_col_indices)])
val_features_demographic_history <- as.matrix(mic_modified_test_set_ann[,c(mic_demographic_history_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_demographic_history <- colnames(train_features_demographic_history)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_demographic_history) == 0)

index_features_scaling <- which(!feature_names_demographic_history  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_demographic_history[,index_features_scaling])

train_features_demographic_history <- train_features_demographic_history[,c(feature_names_for_scaling)]
val_features_demographic_history <- val_features_demographic_history[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_demographic_history %<>% scale()
val_features_demographic_history %<>% 
        scale(center = attr(train_features_demographic_history, "scaled:center"),
        scale = attr(train_features_demographic_history, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## Infarction Features ###########################

train_features_infarction <- as.matrix(mic_modified_train_set_ann[,c(mic_infarction_col_indices)])
val_features_infarction <- as.matrix(mic_modified_test_set_ann[,c(mic_infarction_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_infarction <- colnames(train_features_infarction)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_infarction) == 0)

index_features_scaling <- which(!feature_names_infarction  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_infarction[,index_features_scaling])

train_features_infarction <- train_features_infarction[,c(feature_names_for_scaling)]
val_features_infarction <- val_features_infarction[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_infarction %<>% scale()
val_features_infarction %<>% 
        scale(center = attr(train_features_infarction, "scaled:center"),
        scale = attr(train_features_infarction, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## Emergency ICU Features ###########################

train_features_emergency_icu <- as.matrix(mic_modified_train_set_ann[,c(mic_emergency_icu_col_indices)])
val_features_emergency_icu <- as.matrix(mic_modified_test_set_ann[,c(mic_emergency_icu_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_emergency_icu <- colnames(train_features_emergency_icu)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_emergency_icu) == 0)

index_features_scaling <- which(!feature_names_emergency_icu  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_emergency_icu[,index_features_scaling])

train_features_emergency_icu <- train_features_emergency_icu[,c(feature_names_for_scaling)]
val_features_emergency_icu <- val_features_emergency_icu[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_emergency_icu %<>% scale()
val_features_emergency_icu %<>% 
        scale(center = attr(train_features_emergency_icu, "scaled:center"),
        scale = attr(train_features_emergency_icu, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## ECG Features ###########################

train_features_ecg <- as.matrix(mic_modified_train_set_ann[,c(mic_ecg_col_indices)])
val_features_ecg <- as.matrix(mic_modified_test_set_ann[,c(mic_ecg_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_ecg <- colnames(train_features_ecg)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_ecg) == 0)

index_features_scaling <- which(!feature_names_ecg  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_ecg[,index_features_scaling])

train_features_ecg <- train_features_ecg[,c(feature_names_for_scaling)]
val_features_ecg <- val_features_ecg[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_ecg %<>% scale()
val_features_ecg %<>% 
        scale(center = attr(train_features_ecg, "scaled:center"),
        scale = attr(train_features_ecg, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )

########## FT Features ###########################

train_features_ft <- as.matrix(mic_modified_train_set_ann[,c(mic_ft_col_indices)])
val_features_ft <- as.matrix(mic_modified_test_set_ann[,c(mic_ft_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_ft <- colnames(train_features_ft)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_ft) == 0)

index_features_scaling <- which(!feature_names_ft  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_ft[,index_features_scaling])

train_features_ft <- train_features_ft[,c(feature_names_for_scaling)]
val_features_ft <- val_features_ft[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_ft %<>% scale()
val_features_ft %<>% 
        scale(center = attr(train_features_ft, "scaled:center"),
        scale = attr(train_features_ft, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## Serum Features ###########################

train_features_serum <- as.matrix(mic_modified_train_set_ann[,c(mic_serum_col_indices)])
val_features_serum <- as.matrix(mic_modified_test_set_ann[,c(mic_serum_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_serum <- colnames(train_features_serum)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_serum) == 0)

index_features_scaling <- which(!feature_names_serum  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_serum[,index_features_scaling])

train_features_serum <- train_features_serum[,c(feature_names_for_scaling)]
val_features_serum <- val_features_serum[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_serum %<>% scale()
val_features_serum %<>% 
        scale(center = attr(train_features_serum, "scaled:center"),
        scale = attr(train_features_serum, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## Relpase Features ###########################

train_features_relapse <- as.matrix(mic_modified_train_set_ann[,c(mic_relapse_col_indices)])
val_features_relapse <- as.matrix(mic_modified_test_set_ann[,c(mic_relapse_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_relapse <- colnames(train_features_relapse)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_relapse) == 0)

index_features_scaling <- which(!feature_names_relapse  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_relapse[,index_features_scaling])

train_features_relapse <- train_features_relapse[,c(feature_names_for_scaling)]
val_features_relapse <- val_features_relapse[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_relapse %<>% scale()
val_features_relapse %<>% 
        scale(center = attr(train_features_relapse, "scaled:center"),
        scale = attr(train_features_relapse, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## Medicine Features ###########################

train_features_medicine <- as.matrix(mic_modified_train_set_ann[,c(mic_medicine_col_indices)])
val_features_medicine <- as.matrix(mic_modified_test_set_ann[,c(mic_medicine_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_medicine <- colnames(train_features_medicine)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_medicine) == 0)

index_features_scaling <- which(!feature_names_medicine  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_medicine[,index_features_scaling])

train_features_medicine <- train_features_medicine[,c(feature_names_for_scaling)]
val_features_medicine <- val_features_medicine[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_medicine %<>% scale()
val_features_medicine %<>% 
        scale(center = attr(train_features_medicine, "scaled:center"),
        scale = attr(train_features_medicine, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


####### Create Training & Validation Targets #######################

train_targets <- as.matrix(mic_train_let_is_final_bin)
val_targets <- as.matrix(mic_test_let_is_final_bin)


###################################################################################

# Let us define the input shapes. 
input_shape_demographic_history  <- ncol(train_features_demographic_history)
input_shape_infarction <- ncol(train_features_infarction)
input_shape_emergency_icu <- ncol(train_features_emergency_icu)
input_shape_ecg <- ncol(train_features_ecg)
input_shape_ft <- ncol(train_features_ft)
input_shape_serum <- ncol(train_features_serum)
input_shape_relapse <- ncol(train_features_relapse)
input_shape_medicine <- ncol(train_features_medicine)


# Let us build the Keras Inputs & Features
input_demographic_history <- keras_input(shape(input_shape_demographic_history), name = "demographic_history")
input_infarction <- keras_input(shape(input_shape_infarction), name = "infarction")
input_emergency_icu <- keras_input(shape(input_shape_emergency_icu), name = "emergency_icu")
input_ecg <- keras_input(shape(input_shape_ecg), name = "ecg")
input_ft <- keras_input(shape(input_shape_ft), name = "ft")
input_serum <- keras_input(shape(input_shape_serum), name = "serum")
input_relapse <- keras_input(shape(input_shape_relapse), name = "relapse")
input_medicine <- keras_input(shape(input_shape_medicine), name = "medicine")


# Let us build the ANN Feature Layers
demographic_history_features <- 
    layer_dense(object = input_demographic_history, units = 256) |> 
    layer_dropout(rate = 0.3, seed = 1024) |>
    layer_dense(units = 128, activation = "relu") |> #128
    layer_dropout(rate = 0.3, seed = 1024) 

infarction_features <- 
    layer_dense(object = input_infarction, units = 1536, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024) 


emergency_icu_features <- 
    layer_dense(object = input_emergency_icu, units = 1216, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024) 

ecg_features <- 
    layer_dense(object = input_ecg, units = 576, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024) 

ft_features <- 
    layer_dense(object = input_ft, units = 64, activation = "relu") |>  
    layer_dropout(rate = 0.3, seed = 1024)

serum_features <- 
    layer_dense(object = input_serum, units = 64, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024) 

relapse_features <- 
    layer_dense(object = input_relapse, units = 72, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024)

medicine_features <- 
    layer_dense(object = input_medicine, units = 240, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024) |>
    layer_dense(units = 120, activation = "relu") |> #320 #512
    layer_dropout(rate = 0.3, seed = 1024)

# Let us combine the Feature Layers together

combined_features <- layer_concatenate(list(demographic_history_features,infarction_features, emergency_icu_features, ecg_features, ft_features, serum_features, relapse_features, medicine_features))

pred_functional_api <- layer_dense(object = combined_features, units = 1, activation = "sigmoid")

# Instantiate an end-to-end model 

functional_api_model <- keras_model(
  inputs = list(input_demographic_history, input_infarction, input_emergency_icu, input_ecg, input_ft, input_serum, input_relapse, input_medicine),
  outputs = list(pred_functional_api)
)


# Collect counts for initial weight generation
counts <- table(mic_test_let_is_final_bin) # Counts for Training Set
# Used for Diagnostic purposes. Commented out for Report creation
# print("counts for training set")
# counts
# print("counts for testing set")
# table(mic_test_let_is_final_bin) # Counts for Validation Set

# Configure Weights. Weights are updated manually with multipliers

weight_for_0 = as.numeric(1 / counts["0"]) 
weight_for_1 = as.numeric(1 / counts["1"]) 


# Compile Model

metrics <- list(
  metric_false_negatives(name = "fn"),
  metric_false_positives(name = "fp"),
  metric_true_negatives(name = "tn"),
  metric_true_positives(name = "tp"),
  metric_precision(name = "precision"),
  metric_recall(name = "recall")
)
functional_api_model |> compile(
  optimizer = optimizer_adam(1e-3),
  loss = "binary_crossentropy",
  metrics = metrics
)


class_weight <- list("0" = weight_for_0,
                     "1" = weight_for_1)



######  Fit model ##############


functional_api_model |> 
  fit(
  x = list(demographic_history = train_features_demographic_history, infarction = train_features_infarction, emergency_icu = train_features_emergency_icu, ecg = train_features_ecg, ft = train_features_ft, serum = train_features_serum, relapse = train_features_relapse, medicine = train_features_medicine),
  y = train_targets,
  validation_data = list(list(val_features_demographic_history, val_features_infarction, val_features_emergency_icu, val_features_ecg, val_features_ft, val_features_serum, val_features_relapse, val_features_medicine),val_targets),
  class_weight = class_weight,
  batch_size = 2048,
  epochs = 30,
  verbose = 0 # Set verbose=2 during development, tuning and testing
)


# Prepare Predictions

val_pred_functional_api_final <- functional_api_model %>%
  predict(list(val_features_demographic_history, val_features_infarction, val_features_emergency_icu, val_features_ecg, val_features_ft, val_features_serum, val_features_relapse, val_features_medicine)) %>%
  { as.integer(. > 0.5) }

pred_correct_functional_api_final <- mic_test_let_is_final_bin == val_pred_functional_api_final

# Print table of predictions
print("================================",quote=FALSE)
print("The Table of Predicted values is", quote = FALSE)
table(as.numeric(!pred_correct_functional_api_final))
print("================================",quote=FALSE)

# Print table of actual values
print("================================",quote=FALSE)
print("The table of actual values is", quote = FALSE)
table(as.factor(mic_test_let_is_final_bin))
print("================================",quote=FALSE)

# Print overall accuracy
print("================================",quote=FALSE)
print(c("The overall accuracy is : ", round(mean(pred_correct_functional_api_final), digits = 4)), quote = FALSE)
print("================================",quote=FALSE)

# Record and print confusion matrix
print("================================",quote=FALSE)
print("The Confusion Matrix is", quote = FALSE)
mic_ann_func_cm_bin <- confusionMatrix(data = as.factor(val_pred_functional_api_final), reference = as.factor(mic_test_let_is_final_bin))
mic_ann_func_cm_bin
print("================================",quote=FALSE)

# Collect death events 
deaths_functional_api_final <- mic_test_let_is_final_bin == 1

# Prepare and print summary of death events
n_deaths_detected <- sum(deaths_functional_api_final & pred_correct_functional_api_final)
n_deaths_missed <- sum(deaths_functional_api_final & !pred_correct_functional_api_final)
n_live_flagged <- sum(!deaths_functional_api_final & !pred_correct_functional_api_final)

print(c("deaths detected: ",n_deaths_detected), quote = FALSE)
print(c("deaths missed: ",n_deaths_missed), quote = FALSE)
print(c("survival cases flagged as deaths: ",n_live_flagged), quote = FALSE)


# Remove data that is no longer required

rm(callbacks,class_weight, metrics, train_features, train_targets, val_features, val_targets, counts, deaths, feature_names, model, n_deaths_detected, n_deaths_missed, n_live_flagged, pred_correct, val_pred, weight_for_0, weight_for_1)


rm(deaths_functional_api_final)

rm(weight_for_0, weight_for_1)
rm(mic_binary_funcational_api_class_weight)

rm(mic_train_let_is_final_bin,mic_test_let_is_final_bin)



## ----MIC Final Analysis ANN Cleanup - Binary , include=FALSE, warning=FALSE, echo=FALSE, message = FALSE---------------------------------------------------------------------------------------------------------

# Remove data that is no longer required and run garbage collector to free up memory

rm(mic_modified_cv_test_set_ann, mic_modified_cv_train_set_ann, dtrain, mic_modified_cv_test_set_ann_outcome, mic_modified_cv_train_set_ann_outcome )

gc()



## ----Perform final analysis of MIC dataset using XGBoost for categorical outcome, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE-----------------------------------------------------------------------

# For the sake of convenience and reproducibility, we are reusing variables after deleting them. However, for this to work, the variables need to be sanitised or deleted after use. If variables are not  sanitised or deleted, data could leak inadvertently. 

# Use Original Dataset without Imputation

# DO NOT CONVERT CATEGORICAL(ORDINAL) AND BINARY(NOMINAL) VARIABLES TO FACTORS
# CONVERSION USING as.matrix() CAUSES THEM TO BE COERCED TO CHARACTER VECTORS

mic_data_xgboost <- mic_data %>% 
    mutate_at(mic_continuous_variables, ~as.numeric(.))

# Create Training and Testing datasets

mic_modified_train_xgboost <- mic_data_xgboost[-mic_test_index,]
mic_modified_test_xgboost <- mic_data_xgboost[mic_test_index,]


###########

# Create a matrix that can be used by XGBoost for Training
dtrain_categorical_final <- xgb.DMatrix(data = as.matrix(mic_modified_train_xgboost[,c(2:112)]), label= mic_modified_train_xgboost$LET_IS, nthread = 8)

# Fit XGBoost
fit_xgboost_mic_categorical_final <- xgb.train(
    data = dtrain_categorical_final,
    max_depth = 10,
    eta = 0.15,
    nthread = 8,
    nrounds = 200,
    objective = "multi:softmax", 
    params = list("num_class" = 8, "booster" = "gbtree"),
    verbose = 0 # set verbose=2 during development, tuning and testing
)

# Print Summary of Parameters

print("=====================================", quote = FALSE)
print("The details for XGBoost are", quote = FALSE)
fit_xgboost_mic_categorical_final
print("=====================================", quote = FALSE)

# Prepare predictions
pred_xgboost_mic_categorical_final <-predict(object = fit_xgboost_mic_categorical_final, newdata = as.matrix(mic_modified_test_xgboost[,c(2:112)]))

# Print table of predictions
print("=====================================", quote = FALSE)
print("The table of predictions is", quote = FALSE)
table(as.factor(pred_xgboost_mic_categorical_final ))
print("=====================================")

# Print table of actual values
print("=====================================", quote = FALSE)
print("The table of actual values is", quote = FALSE)
table(mic_modified_test_xgboost$LET_IS)
print("=====================================", quote = FALSE)

# Print overall accuracy
print("The accuracy of predictions is", quote = FALSE)
mean(pred_xgboost_mic_categorical_final == mic_modified_test_xgboost$LET_IS)
print("=====================================", quote = FALSE)

# Record and print confusion matrix
print("The confusion matrix is", quote = FALSE)
print("=====================================", quote = FALSE)
mic_xgb_cm_multi_final <- confusionMatrix(data = as.factor(pred_xgboost_mic_categorical_final), reference = as.factor(mic_modified_test_xgboost$LET_IS))
mic_xgb_cm_multi_final
print("=====================================", quote = FALSE)

# Extract and Print Summary of Death Events
print("================================",quote=FALSE)
print("The Number of Deaths correctly categorised is :",quote=FALSE)
sum(mic_xgb_cm_multi_final$table[2,2], mic_xgb_cm_multi_final$table[3,3], mic_xgb_cm_multi_final$table[4,4], mic_xgb_cm_multi_final$table[5,5], mic_xgb_cm_multi_final$table[6,6], mic_xgb_cm_multi_final$table[7,7], mic_xgb_cm_multi_final$table[8,8])
print("================================",quote=FALSE)

print("================================",quote=FALSE)
print("The Number of Deaths correctly detected is :",quote=FALSE)
sum(mic_xgb_cm_multi_final$table[2:8,2:8])
print("================================",quote=FALSE)

print("================================",quote=FALSE)
print("The Number of Live cases flagged as Deaths is :",quote=FALSE)
sum(mic_xgb_cm_multi_final$table[2:8,1])
print("================================",quote=FALSE)

# Remove Data that is no longer required
rm(fit_xgboost_mic_categorical_final, dtrain_categorical_final)

rm(mic_data_modified, dtrain, mic_modified_test_xgboost, mic_modified_train_xgboost)

rm(mic_data_modified_xgboost, mic_data_xgboost)


## ----Prepare MIC Dataset for Final Analysis and multiple outcomes, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE--------------------------------------------------------------------------------------

# Use prviously created Lists of Continuous, Ordinal (Categorical) and Nominal (Binary) Features to avoid duplication and errors. 


########

# Convert nominal variables from factors to integers. Since they are binary, changing them from factors to numerics does not alter their behaviour 

mic_modified_train_set_ann <- mic_orig_train_imputed_mice %>% 
          mutate_at(c(mic_nominal_variables),~(.=as.integer(.))) 

mic_modified_test_set_ann <- mic_orig_test_imputed_mice %>% 
          mutate_at(c(mic_nominal_variables),~(.=as.integer(.))) 


#############
# Expand Imputed values using Dummy Variables.

# Retain "LET_IS" column as it will be lost during conversion

mic_train_let_is_final_multi <- mic_modified_train_set_ann$LET_IS
mic_test_let_is_final_multi <- mic_modified_test_set_ann$LET_IS

# Set up Training and Testing sets for one hot encoding. Convert LET_IS to a factor

mic_modified_train_set_ann %<>% mutate(LET_IS = as.factor(LET_IS))

mic_modified_test_set_ann %<>% mutate(LET_IS = as.factor(LET_IS))


########## ONE-HOT ENCODING USING CARET FOR PREDICTORS AND OUTCOME ###################

# We will use CARET for creating the dummy variables for all the predictors and the outcome as it support both ordered and unordered factors 

# We will two seperate instances of dummyVars for creations of the dummy variables for the features and the outcome. We can do it using a single instance but the code is very confusing and difficult to follow.

# Use formula interface to create template for dummy vars. 

dummy_vars_features <- dummyVars(formula = "~.", data = mic_modified_train_set_ann[,c(mic_ordinal_variables, mic_part_ordinal_variables)], sep = "_", levelsOnly = FALSE, fullRank = FALSE)

dummy_vars_outcome <- dummyVars(formula = "~.", data = mic_modified_train_set_ann["LET_IS"], sep = "_", levelsOnly = FALSE, fullRank = FALSE)

# Create dummy vars for Training and Testing datasets. 
mic_modified_train_set_ann_dummy <- as.data.frame(predict(dummy_vars_features, mic_modified_train_set_ann[,c(mic_ordinal_variables, mic_part_ordinal_variables)]))

mic_modified_test_set_ann_dummy <- as.data.frame(predict(dummy_vars_features, mic_modified_test_set_ann[,c(mic_ordinal_variables, mic_part_ordinal_variables)]))

mic_modified_train_set_ann_outcome <-  as.data.frame(predict(dummy_vars_outcome, mic_modified_train_set_ann["LET_IS"]))

mic_modified_test_set_ann_outcome <- as.data.frame(predict(dummy_vars_outcome, mic_modified_test_set_ann["LET_IS"]))

# Collect Training dummy vars names. Same applies for Testing CV. 
dummy_vars_names <- colnames(mic_modified_train_set_ann_dummy)

# Modify Training CV and Testing CV Datasets to remove the existing columns for variables related to  "ID", Ordinal,  Partially Ordinal, Complications and "LET_IS".

mic_modified_train_set_ann %<>% select(-c("ID",mic_ordinal_variables,mic_part_ordinal_variables, mic_complications,"LET_IS"))

mic_modified_test_set_ann %<>% select(-c("ID",mic_ordinal_variables,mic_part_ordinal_variables, mic_complications,"LET_IS"))



# Collect Modified Training Column names. Same applies for Testing dataset
mic_modified_col_names <- colnames(mic_modified_train_set_ann)

# Combine Collected Modified Training column names with Training dummy vars names for predictors

mic_modified_col_names <- c(mic_modified_col_names, dummy_vars_names)

# Create combined Training and Testing datasets by binding together the Modified Training and Testing datasets with the respective dummy vars daatasets
mic_modified_train_set_ann <- cbind(mic_modified_train_set_ann, mic_modified_train_set_ann_dummy)

mic_modified_test_set_ann <- cbind(mic_modified_test_set_ann, mic_modified_test_set_ann_dummy)

# Assign column names to the newly created combined Training and Testing datasets
colnames(mic_modified_train_set_ann) <- mic_modified_col_names

colnames(mic_modified_test_set_ann) <- mic_modified_col_names


# Remove data that is not required anymore

rm(dummy_vars_features, dummy_vars_outcome, mic_modified_train_set_ann_dummy, mic_modified_test_set_ann_dummy, mic_modified_col_names)


## ----Perform Final Analysis for MIC Dataset using Neural Networks with Categories for outcome LET_IS using Sequential API, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE------------------------------

########## Hack to achieve Consistent results between Keras Runs ############
# Unload and Reload Keras
unload(package = "keras3", quiet = TRUE)
library(keras3)
# Set Up Environment
set.seed(1024)
keras3::set_random_seed(1024)
#tensorflow::set_random_seed(1024, disable_gpu = TRUE) #Uncomment if using TensorFlow
reticulate::py_set_seed(0)
#############################################################################


########################## WARNING WARNING WARNING ##########################
############## DOUBLE CHECK INPUT FEATURES TO PREVENT LEAKAGE ###############
feature_names <- c(mic_nominal_variables, mic_continuous_variables, dummy_vars_names)

train_features <- as.matrix(mic_modified_train_set_ann[feature_names])
train_targets <- as.matrix(mic_modified_train_set_ann_outcome)

val_features <- as.matrix(mic_modified_test_set_ann[feature_names])
val_targets <- as.matrix(mic_modified_test_set_ann_outcome)

####################################
# As earlier, We need to ensure that the input variables are scaled & centered so that they we do not have any one factor unnecessarily influencing the outcome and all have adequate representation
# As earlier, we also need to remove Columns from the list that produce NA when scaled

####### Normal Scaling ############
feature_names <- colnames(train_features)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features) == 0)

index_features_scaling <- which(!feature_names  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features[,index_features_scaling])

train_features <- train_features[,c(feature_names_for_scaling)]
val_features <- val_features[,c(feature_names_for_scaling)]

train_features %<>% scale()
val_features %<>% scale(center = attr(train_features, "scaled:center"),
                        scale = attr(train_features, "scaled:scale"))

####### Normal Scaling ############
train_features %<>% scale()
val_features %<>% scale(center = attr(train_features, "scaled:center"),
                        scale = attr(train_features, "scaled:scale"))

# Let us build the ANN

model <-
  keras3::keras_model_sequential(input_shape = ncol(train_features)) |>
  layer_dense(units = 3072) |>
  layer_dropout(rate = 0.3, seed = 1024) |>
  layer_dense(units = 3072) |>
  layer_dropout(rate = 0.3, seed = 1024) |>
  layer_dense(units = 2304) |>
  layer_dropout(rate = 0.3, seed = 1024) |>
  layer_dense(units = 8, activation = 'softmax')


# Collect counts for initial weight generation
counts <- table(mic_train_let_is_final_multi) # Counts for Training Set
# Used for Diagnostic purposes. Commented out for Report creation
# print("counts for training set")
# counts
# print("counts for testing set")
# table(mic_test_let_is_final_multi) # Counts for Validation Set

# Configure Weights. Weights are updated manually with multipliers
weight_for_0 = as.numeric(1 / counts["0"])*2.6
weight_for_1 = as.numeric(1 / counts["1"])*0.225
weight_for_2 = as.numeric(1 / counts["2"])*0.05
weight_for_3 = as.numeric(1 / counts["3"])*0.25
weight_for_4 = as.numeric(1 / counts["4"])*0.05
weight_for_5 = as.numeric(1 / counts["5"])*0.0750
weight_for_6 = as.numeric(1 / counts["6"])*0.0875
weight_for_7 = as.numeric(1 / counts["7"])*0.06125

# Train the Model 

model |> compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy','categorical_accuracy')
)

##############

class_weight <- list("0" = weight_for_0,
                     "1" = weight_for_1,
                     "2" = weight_for_2,
                     "3" = weight_for_3,
                     "4" = weight_for_4,
                     "5" = weight_for_5,
                     "6" = weight_for_6,
                     "7" = weight_for_7)




model |> fit(
  train_features, train_targets,
  validation_data = list(val_features, val_targets),
  batch_size = 2048,
  class_weight = class_weight,
  epochs = 30,
  verbose = 0 # Set verbose=2 during development, tuning and testing
)


# Evaluate Model 
# Commented out for Report creation
# model |> evaluate(val_features, val_targets)

# Prepare Predictions

probs <- model |> predict(val_features)

pred_ann_seq_api_multi <- max.col(probs) - 1L

# Print table of predictions
print("================================",quote=FALSE)
print("The Table of Predicted values is",quote=FALSE)
table(as.factor(pred_ann_seq_api_multi))
print("================================",quote=FALSE)

# Print table of actual values
print("=====================================",quote=FALSE)
print("The table of actual values is",quote=FALSE)
table(as.factor(mic_test_let_is_final_multi))
print("================================",quote=FALSE)

# Print overall accuracy
print("================================",quote=FALSE)
print("The overall accuracy is : ",quote=FALSE)
mean(mic_test_let_is_final_multi == pred_ann_seq_api_multi)

# Record and print confusion matrix
print("================================",quote=FALSE)
print("The Confusion Matrix is",quote=FALSE)
mic_ann_seq_cm_multi_final <- confusionMatrix(data = as.factor(pred_ann_seq_api_multi), reference = as.factor(mic_test_let_is_final_multi))
mic_ann_seq_cm_multi_final
print("================================")

# Extract and print summary of death events
print("================================",quote=FALSE)
print("The Number of Deaths correctly categorised is :",quote=FALSE)
sum(mic_ann_seq_cm_multi_final$table[2,2], mic_ann_seq_cm_multi_final$table[3,3], mic_ann_seq_cm_multi_final$table[4,4], mic_ann_seq_cm_multi_final$table[5,5], mic_ann_seq_cm_multi_final$table[6,6], mic_ann_seq_cm_multi_final$table[7,7], mic_ann_seq_cm_multi_final$table[8,8])
print("================================",quote=FALSE)

print("================================",quote=FALSE)
print("The Number of Deaths correctly detected is :",quote=FALSE)
sum(mic_ann_seq_cm_multi_final$table[2:8,2:8])
print("================================",quote=FALSE)

print("================================",quote=FALSE)
print("The Number of Live cases flagged as Deaths is :",quote=FALSE)
sum(mic_ann_seq_cm_multi_final$table[2:8,1])
print("================================",quote=FALSE)

# Remove data that is no longer required
rm(class_weight, metrics, train_features, train_targets, val_features, val_targets, counts, deaths, feature_names, model, n_deaths_detected, n_deaths_missed, n_live_flagged, pred_correct, val_pred)

rm(index_features_scaling, feature_names_for_scaling, mic_seq_api_class_weight)

rm(probs)

rm(weight_for_0, weight_for_1, weight_for_2, weight_for_3, weight_for_4, weight_for_5, weight_for_6, weight_for_7)



## ----Perform Initial Analysis for MIC Dataset using Neural Networks with Categories for outcome LET_IS with Functional API, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE, , fig.align='center'-------


########## Hack to achieve Consistent results between Keras Runs ############
# Unload and Reload Keras
unload(package = "keras3", quiet = TRUE)
library(keras3)
# Set Up Environment
set.seed(1024)
keras3::set_random_seed(1024)
#tensorflow::set_random_seed(1024, disable_gpu = TRUE) #Uncomment if using TensorFlow
reticulate::py_set_seed(0)
#############################################################################



############# Create the Input Data #########################################


########## Demographic & History Features ###########################

train_features_demographic_history <- as.matrix(mic_modified_train_set_ann[,c(mic_demographic_history_col_indices)])
val_features_demographic_history <- as.matrix(mic_modified_test_set_ann[,c(mic_demographic_history_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_demographic_history <- colnames(train_features_demographic_history)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_demographic_history) == 0)

index_features_scaling <- which(!feature_names_demographic_history  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_demographic_history[,index_features_scaling])

train_features_demographic_history <- train_features_demographic_history[,c(feature_names_for_scaling)]
val_features_demographic_history <- val_features_demographic_history[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_demographic_history %<>% scale()
val_features_demographic_history %<>% 
        scale(center = attr(train_features_demographic_history, "scaled:center"),
        scale = attr(train_features_demographic_history, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )



########## Infarction Features ###########################

train_features_infarction <- as.matrix(mic_modified_train_set_ann[,c(mic_infarction_col_indices)])
val_features_infarction <- as.matrix(mic_modified_test_set_ann[,c(mic_infarction_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_infarction <- colnames(train_features_infarction)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_infarction) == 0)

index_features_scaling <- which(!feature_names_infarction  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_infarction[,index_features_scaling])

train_features_infarction <- train_features_infarction[,c(feature_names_for_scaling)]
val_features_infarction <- val_features_infarction[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_infarction %<>% scale()
val_features_infarction %<>% 
        scale(center = attr(train_features_infarction, "scaled:center"),
        scale = attr(train_features_infarction, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )



########## Emergency ICU Features ###########################

train_features_emergency_icu <- as.matrix(mic_modified_train_set_ann[,c(mic_emergency_icu_col_indices)])
val_features_emergency_icu <- as.matrix(mic_modified_test_set_ann[,c(mic_emergency_icu_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_emergency_icu <- colnames(train_features_emergency_icu)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_emergency_icu) == 0)

index_features_scaling <- which(!feature_names_emergency_icu  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_emergency_icu[,index_features_scaling])

train_features_emergency_icu <- train_features_emergency_icu[,c(feature_names_for_scaling)]
val_features_emergency_icu <- val_features_emergency_icu[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_emergency_icu %<>% scale()
val_features_emergency_icu %<>% 
        scale(center = attr(train_features_emergency_icu, "scaled:center"),
        scale = attr(train_features_emergency_icu, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## ECG Features ###########################

train_features_ecg <- as.matrix(mic_modified_train_set_ann[,c(mic_ecg_col_indices)])
val_features_ecg <- as.matrix(mic_modified_test_set_ann[,c(mic_ecg_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_ecg <- colnames(train_features_ecg)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_ecg) == 0)

index_features_scaling <- which(!feature_names_ecg  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_ecg[,index_features_scaling])

train_features_ecg <- train_features_ecg[,c(feature_names_for_scaling)]
val_features_ecg <- val_features_ecg[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_ecg %<>% scale()
val_features_ecg %<>% 
        scale(center = attr(train_features_ecg, "scaled:center"),
        scale = attr(train_features_ecg, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )

########## FT Features ###########################

train_features_ft <- as.matrix(mic_modified_train_set_ann[,c(mic_ft_col_indices)])
val_features_ft <- as.matrix(mic_modified_test_set_ann[,c(mic_ft_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_ft <- colnames(train_features_ft)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_ft) == 0)

index_features_scaling <- which(!feature_names_ft  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_ft[,index_features_scaling])

train_features_ft <- train_features_ft[,c(feature_names_for_scaling)]
val_features_ft <- val_features_ft[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_ft %<>% scale()
val_features_ft %<>% 
        scale(center = attr(train_features_ft, "scaled:center"),
        scale = attr(train_features_ft, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## Serum Features ###########################

train_features_serum <- as.matrix(mic_modified_train_set_ann[,c(mic_serum_col_indices)])
val_features_serum <- as.matrix(mic_modified_test_set_ann[,c(mic_serum_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_serum <- colnames(train_features_serum)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_serum) == 0)

index_features_scaling <- which(!feature_names_serum  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_serum[,index_features_scaling])

train_features_serum <- train_features_serum[,c(feature_names_for_scaling)]
val_features_serum <- val_features_serum[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_serum %<>% scale()
val_features_serum %<>% 
        scale(center = attr(train_features_serum, "scaled:center"),
        scale = attr(train_features_serum, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## Relapse Features ###########################

train_features_relapse <- as.matrix(mic_modified_train_set_ann[,c(mic_relapse_col_indices)])
val_features_relapse <- as.matrix(mic_modified_test_set_ann[,c(mic_relapse_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_relapse <- colnames(train_features_relapse)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_relapse) == 0)

index_features_scaling <- which(!feature_names_relapse  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_relapse[,index_features_scaling])

train_features_relapse <- train_features_relapse[,c(feature_names_for_scaling)]
val_features_relapse <- val_features_relapse[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_relapse %<>% scale()
val_features_relapse %<>% 
        scale(center = attr(train_features_relapse, "scaled:center"),
        scale = attr(train_features_relapse, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## Medicine Features ###########################

train_features_medicine <- as.matrix(mic_modified_train_set_ann[,c(mic_medicine_col_indices)])
val_features_medicine <- as.matrix(mic_modified_test_set_ann[,c(mic_medicine_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_medicine <- colnames(train_features_medicine)

mic_train_list_col_sd_eq_0 <- which(colSds(train_features_medicine) == 0)

index_features_scaling <- which(!feature_names_medicine  %in% names(mic_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_medicine[,index_features_scaling])

train_features_medicine <- train_features_medicine[,c(feature_names_for_scaling)]
val_features_medicine <- val_features_medicine[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_medicine %<>% scale()
val_features_medicine %<>% 
        scale(center = attr(train_features_medicine, "scaled:center"),
        scale = attr(train_features_medicine, "scaled:scale"))

rm(mic_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )



################# CREATE TRAINING AND VALIDATION TARGETS ########################

train_targets <- as.matrix(mic_modified_train_set_ann_outcome)
val_targets <- as.matrix(mic_modified_test_set_ann_outcome)

###################################################################################

# Let us define the input shapes. 
input_shape_demographic_history  <- ncol(train_features_demographic_history)
input_shape_infarction <- ncol(train_features_infarction)
input_shape_emergency_icu <- ncol(train_features_emergency_icu)
input_shape_ecg <- ncol(train_features_ecg)
input_shape_ft <- ncol(train_features_ft)
input_shape_serum <- ncol(train_features_serum)
input_shape_relapse <- ncol(train_features_relapse)
input_shape_medicine <- ncol(train_features_medicine)



# Let us build the Keras Inputs & Features
input_demographic_history <- keras_input(shape(input_shape_demographic_history), name = "demographic_history")
input_infarction <- keras_input(shape(input_shape_infarction), name = "infarction")
input_emergency_icu <- keras_input(shape(input_shape_emergency_icu), name = "emergency_icu")
input_ecg <- keras_input(shape(input_shape_ecg), name = "ecg")
input_ft <- keras_input(shape(input_shape_ft), name = "ft")
input_serum <- keras_input(shape(input_shape_serum), name = "serum")
input_relapse <- keras_input(shape(input_shape_relapse), name = "relapse")
input_medicine <- keras_input(shape(input_shape_medicine), name = "medicine")

# Let us build the ANN
########################################

demographic_history_features <- 
    layer_dense(object = input_demographic_history, units = 1024) |>
    layer_dense(units = 1024) |>
    layer_dense(units = 1024) |>
    layer_dense(units = 1024) |>
    layer_dropout(rate = 0.3, seed = 1024) 

infarction_features <- 
    layer_dense(object = input_infarction, units = 2048) |>
    layer_dense(units = 2048) |>
    layer_dropout(rate = 0.3, seed = 1024) 

emergency_icu_features <- 
    layer_dense(object = input_emergency_icu, units = 1280) |>
    layer_dense(units = 1280) |>
    layer_dense(units = 1280) |>
    layer_dense(units = 1280) |>
    layer_dropout(rate = 0.3, seed = 1024)

ecg_features <- 
    layer_dense(object = input_ecg, units = 1024) |> 
    layer_dense(units = 1024) |>
    layer_dense(units = 1024) |>
    layer_dense(units = 1024) |>
    layer_dropout(rate = 0.3, seed = 1024)
      
ft_features <- 
    layer_dense(object = input_ft, units = 64) |>  
    layer_dropout(rate = 0.3, seed = 1024)    

serum_features <- 
    layer_dense(object = input_serum, units = 64) |> 
    layer_dropout(rate = 0.3, seed = 1024)

relapse_features <- 
    layer_dense(object = input_relapse, units = 96) |> 
    layer_dropout(rate = 0.3, seed = 1024) 

medicine_features <- 
    layer_dense(object = input_medicine, units = 960) |> 
    layer_dense(units = 960) |>
    layer_dense(units = 960) |>
    layer_dense(units = 960) |>
    layer_dropout(rate = 0.3, seed = 1024)


########################################


# Let us combine the Feature Layers together

combined_features <- layer_concatenate(list(demographic_history_features, infarction_features, emergency_icu_features, ecg_features, ft_features, serum_features, relapse_features, medicine_features))



pred_functional_api <- layer_dense(object = combined_features, units = 8, activation = "softmax")

# Instantiate an end-to-end model 

functional_api_model <- keras_model(
  inputs = list(input_demographic_history, input_infarction, input_emergency_icu, input_ecg, input_ft, input_serum, input_relapse, input_medicine),
  outputs = list(pred_functional_api)
)



# Collect counts for initial weight generation
counts <- table(mic_train_let_is_final_multi) # Counts for Training Set
# Used for Diagnostic purposes. Commented out for Report creation
# print("counts for training set")
# counts
# print("counts for testing set")
# table(mic_test_let_is_final_multi) # Counts for Validation Set

# Configure weights. Weights are updated manually with multipliers
weight_for_0 = as.numeric(1 / counts["0"])*3.5
weight_for_1 = as.numeric(1 / counts["1"])*1.1
weight_for_2 = as.numeric(1 / counts["2"])*0.2
weight_for_3 = as.numeric(1 / counts["3"])*0.7
weight_for_4 = as.numeric(1 / counts["4"])*0.15
weight_for_5 = as.numeric(1 / counts["5"])*0.3
weight_for_6 = as.numeric(1 / counts["6"])*0.3
weight_for_7 = as.numeric(1 / counts["7"])*0.4



# Train the Model 

functional_api_model |> compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy','categorical_accuracy')
)

##############
class_weight <- list("0" = weight_for_0,
                     "1" = weight_for_1,
                     "2" = weight_for_2,
                     "3" = weight_for_3,
                     "4" = weight_for_4,
                     "5" = weight_for_5,
                     "6" = weight_for_6,
                     "7" = weight_for_7)

######  Fit model ##############

functional_api_model |> 
  fit(
  x = list(demographic_history = train_features_demographic_history, infarction = train_features_infarction, emergency_icu = train_features_emergency_icu, ecg = train_features_ecg, ft = train_features_ft, serum = train_features_serum, relapse = train_features_relapse, medicine = train_features_medicine ),
  y = train_targets,
  validation_data = list(list(val_features_demographic_history, val_features_infarction, val_features_emergency_icu, val_features_ecg, val_features_ft, val_features_serum, val_features_relapse, val_features_medicine),val_targets),
  batch_size = 2048,
  epochs = 30,
  class_weight = class_weight, 
  verbose = 0 # Set verbose=2 during development, tuning and testing
)


# Evaluate Model
# Commented out for report creation
# functional_api_model |> evaluate(list(val_features_demographic_history, val_features_infarction, val_features_emergency_icu, val_features_ecg, val_features_ft, val_features_serum, val_features_relapse, val_features_medicine),val_targets)

# Prepare Predictions

probs <- functional_api_model |> predict(list(val_features_demographic_history,  val_features_infarction, val_features_emergency_icu, val_features_ecg, val_features_ft, val_features_serum, val_features_relapse, val_features_medicine))

pred_ann_func_api_multi <- max.col(probs) - 1L

# Print table of predicted values
print("================================",quote=FALSE)
print("The Table of Predicted values is", quote = FALSE)
table(as.factor(pred_ann_func_api_multi))
print("================================",quote=FALSE)

# Print table of actual values
print("================================",quote=FALSE)
print("The Table of actual values is", quote = FALSE)
table(as.factor(mic_test_let_is_final_multi))

# Print overall accuracy
print("================================",quote=FALSE)
print("The overall accuracy is", quote = FALSE)
mean(mic_test_let_is_final_multi  == pred_ann_func_api_multi)

# Record and print confusion matrix
print("================================",quote=FALSE)
print("The Confusion Matrix is", quote = FALSE)
mic_ann_func_cm_multi_final <- confusionMatrix(data = as.factor(pred_ann_func_api_multi), reference = as.factor(mic_test_let_is_final_multi))
mic_ann_func_cm_multi_final
print("================================",quote=FALSE)

# Extract and print summary of death events
print("================================",quote=FALSE)
print("The Number of Deaths correctly categorised is :",quote=FALSE)
sum(mic_ann_func_cm_multi_final$table[2,2], mic_ann_func_cm_multi_final$table[3,3], mic_ann_func_cm_multi_final$table[4,4], mic_ann_func_cm_multi_final$table[5,5], mic_ann_func_cm_multi_final$table[6,6], mic_ann_func_cm_multi_final$table[7,7], mic_ann_func_cm_multi_final$table[8,8])
print("================================",quote=FALSE)

print("================================",quote=FALSE)
print("The Number of Deaths correctly detected is :",quote=FALSE)
sum(mic_ann_func_cm_multi_final$table[2:8,2:8])
print("================================",quote=FALSE)

print("================================",quote=FALSE)
print("The Number of Live cases flagged as Deaths is :",quote=FALSE)
sum(mic_ann_func_cm_multi_final$table[2:8,1])
print("================================",quote=FALSE)

# Remove data that is no longer required

rm(metrics, counts, deaths_functional_api_final, feature_names, model, n_deaths_detected, n_deaths_missed, n_live_flagged, pred_correct_functional_api_final, val_pred_functional_api_final)

rm( demographic_history_features, ecg_features, emergency_icu_features, ft_features,   infarction_features, medicine_features, relapse_features, serum_features, combined_features)

rm( input_demographic_history, input_ecg, input_emergency_icu,  input_ft,  input_infarction, input_medicine, input_relapse, input_serum)

rm( train_features_demographic_history, train_features_ecg, train_features_emergency_icu,  train_features_ft,   train_features_infarction, train_features_medicine, train_features_relapse, train_features_serum)

rm( val_features_demographic_history, val_features_ecg, val_features_emergency_icu, val_features_ft,  val_features_infarction, val_features_medicine, val_features_relapse, val_features_serum)

rm( feature_names_demographic_history, feature_names_ecg, feature_names_emergency_icu,  feature_names_ft, feature_names_infarction, feature_names_medicine, feature_names_relapse, feature_names_serum)

rm( input_shape_demographic_history, input_shape_ecg, input_shape_emergency_icu, input_shape_ft,   input_shape_infarction, input_shape_medicine, input_shape_relapse, input_shape_serum)

rm( mic_demographic_history_col_indices, mic_ecg_col_indices, mic_emergency_icu_col_indices,  mic_ft_col_indices,   mic_infarction_col_indices, mic_medicine_col_indices, mic_relapse_col_indices, mic_serum_col_indices)

rm(functional_api_model, probs)

rm(class_weight)

rm(train_features, train_targets, val_features, val_targets)

rm(mic_functional_api_class_weight)

rm(weight_for_0, weight_for_1, weight_for_2, weight_for_3, weight_for_4, weight_for_5, weight_for_6, weight_for_7)




## ----MIC Results Summary - Binary Outcome, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE--------------------------------------------------------------------------------------------------------------

# Print Survival Rate for reference

print(c("The Survival Rate for patients on this dataset is: ", survival_rate),quote = FALSE)


# Prepare final results table for Binary outcome

mic_summary_bin <- data.frame(c("Naive Bayes without Imputation","Naive Bayes with Imputation", "XGBoost", "ANN Sequential API", "ANN Functional API"), c( mic_nb_cm_bin_2$overall["Accuracy"], mic_nb_wi_cm_bin_1$overall["Accuracy"], mic_xgb_cm_bin$overall["Accuracy"], mic_ann_seq_cm_bin$overall["Accuracy"], mic_ann_func_cm_bin$overall["Accuracy"]), c( mic_nb_cm_bin_2$byClass["Balanced Accuracy"], mic_nb_wi_cm_bin_1$byClass["Balanced Accuracy"], mic_xgb_cm_bin$byClass["Balanced Accuracy"], mic_ann_seq_cm_bin$byClass["Balanced Accuracy"], mic_ann_func_cm_bin$byClass["Balanced Accuracy"]), c( mic_nb_cm_bin_2$table[2,2], mic_nb_wi_cm_bin_1$table[2,2],  mic_xgb_cm_bin$table[2,2], mic_ann_seq_cm_bin$table[2,2], mic_ann_func_cm_bin$table[2,2]), c( mic_nb_cm_bin_2$table[1,2], mic_nb_wi_cm_bin_1$table[1,2], mic_xgb_cm_bin$table[1,2], mic_ann_seq_cm_bin$table[1,2], mic_ann_func_cm_bin$table[1,2]),c( mic_nb_cm_bin_2$table[2,1], mic_nb_wi_cm_bin_1$table[2,1], mic_xgb_cm_bin$table[2,1], mic_ann_seq_cm_bin$table[2,1], mic_ann_func_cm_bin$table[2,1]))

# Print results 
knitr::kable(x = mic_summary_bin, col.names = c("Model", "overall accuracy", "balanced accuracy", "deaths detected", "deaths missed", "live flagged"), caption = "Myocardial Infarction Complications - Results Summary - Binary Outcome ", digits = 4) %>% kable_styling(font_size = 8)



## ----MIC Results Summary - Categorical Outcome, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE---------------------------------------------------------------------------------------------------------

# Prepare final results table for categorical outcome
mic_summary_multi <- data.frame(c("XGBoost", "ANN Sequential API", "ANN Functional API"), c(mic_xgb_cm_multi_final$overall["Accuracy"], mic_ann_seq_cm_multi_final$overall["Accuracy"], mic_ann_func_cm_multi_final$overall["Accuracy"]), c(mic_xgb_cm_multi_final$table[2,2] + mic_xgb_cm_multi_final$table[3,3] + mic_xgb_cm_multi_final$table[4,4] + mic_xgb_cm_multi_final$table[5,5] + mic_xgb_cm_multi_final$table[6,6] + mic_xgb_cm_multi_final$table[7,7] + mic_xgb_cm_multi_final$table[8,8] , mic_ann_seq_cm_multi_final$table[2,2] + mic_ann_seq_cm_multi_final$table[3,3] + mic_ann_seq_cm_multi_final$table[4,4] + mic_ann_seq_cm_multi_final$table[5,5] + mic_ann_seq_cm_multi_final$table[6,6] + mic_ann_seq_cm_multi_final$table[7,7] + mic_ann_seq_cm_multi_final$table[8,8], mic_ann_func_cm_multi_final$table[2,2] + mic_ann_func_cm_multi_final$table[3,3] + mic_ann_func_cm_multi_final$table[4,4] + mic_ann_func_cm_multi_final$table[5,5] + mic_ann_func_cm_multi_final$table[6,6] + mic_ann_func_cm_multi_final$table[7,7] + mic_ann_func_cm_multi_final$table[8,8]) , c(sum(mic_xgb_cm_multi_final$table[2:8,2:8]), sum(mic_ann_seq_cm_multi_final$table[2:8,2:8]), sum(mic_ann_func_cm_multi_final$table[2:8,2:8])), c(sum(mic_xgb_cm_multi_final$table[1,2:8]), sum(mic_ann_seq_cm_multi_final$table[1,2:8]), sum(mic_ann_func_cm_multi_final$table[1,2:8])), c(sum(mic_xgb_cm_multi_final$table[2:8,1]), sum(mic_ann_seq_cm_multi_final$table[2:8,1]), sum(mic_ann_func_cm_multi_final$table[2:8,1])))

# Print results 
knitr::kable(x = mic_summary_multi, col.names = c("Model", "Accuracy", "Deaths Accurately Categorised", "Deaths Detected", "Deaths Missed", "Live Cases Flagged"), caption = "MIC - Results Summary - Categorical Outcome ", digits = 4) %>% kable_styling(font_size = 8)




## ----MIC Results Summary - Categorical Outcome - Distributions of predictions, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE--------------------------------------------------------------------------

# Prepare Tables of Actual and Predicted values for easy visualisation

mic_test_let_is_final_multi_table <- data.frame(table(mic_test_let_is_final_multi))
colnames(mic_test_let_is_final_multi_table) <- c("LET_IS","Actual Values")

pred_xgboost_mic_categorical_final_table <- data.frame(table(pred_xgboost_mic_categorical_final))
colnames(pred_xgboost_mic_categorical_final_table) <- c("LET_IS","XGBoost")

pred_ann_seq_api_multi_table <- data.frame(table(pred_ann_seq_api_multi))
colnames(pred_ann_seq_api_multi_table) <- c("LET_IS","ANN Sequential API")

pred_ann_func_api_multi_table <- data.frame(table(pred_ann_func_api_multi))
colnames(pred_ann_func_api_multi_table) <- c("LET_IS","ANN Functional API")

mic_distributions_multi <- mic_test_let_is_final_multi_table %>% 
    left_join(pred_xgboost_mic_categorical_final_table, by = "LET_IS") %>% 
    left_join(pred_ann_seq_api_multi_table, by = "LET_IS") %>% 
    left_join(pred_ann_func_api_multi_table, by = "LET_IS") %>%
    replace_na(repl = 0)


# Print distributions

knitr::kable(x = mic_distributions_multi, col.names = c("LET_IS","Actual Values", "XGBoost", "ANN Sequential API", "ANN Functional API"), caption = "MIC - Table of distributions") %>% kable_styling(font_size = 8)

# Compile summary of actual values and predictions

mic_multi_accuracy_final <- data.frame(c(mic_test_let_is_final_multi_table[1,1], mic_test_let_is_final_multi_table[2,1], mic_test_let_is_final_multi_table [3,1], mic_test_let_is_final_multi_table [4,1], mic_test_let_is_final_multi_table [5,1], mic_test_let_is_final_multi_table [6,1], mic_test_let_is_final_multi_table [7,1], mic_test_let_is_final_multi_table [8,1]), c(mic_test_let_is_final_multi_table[1,2], mic_test_let_is_final_multi_table[2,2], mic_test_let_is_final_multi_table [3,2], mic_test_let_is_final_multi_table [4,2], mic_test_let_is_final_multi_table [5,2], mic_test_let_is_final_multi_table [6,2], mic_test_let_is_final_multi_table [7,2], mic_test_let_is_final_multi_table [8,2]), c(mic_xgb_cm_multi_final$table[1,1], mic_xgb_cm_multi_final$table[2,2] , mic_xgb_cm_multi_final$table[3,3], mic_xgb_cm_multi_final$table[4,4], mic_xgb_cm_multi_final$table[5,5],  mic_xgb_cm_multi_final$table[6,6], mic_xgb_cm_multi_final$table[7,7], mic_xgb_cm_multi_final$table[8,8]), c(mic_ann_seq_cm_multi_final$table[1,1], mic_ann_seq_cm_multi_final$table[2,2], mic_ann_seq_cm_multi_final$table[3,3], mic_ann_seq_cm_multi_final$table[4,4], mic_ann_seq_cm_multi_final$table[5,5], mic_ann_seq_cm_multi_final$table[6,6],  mic_ann_seq_cm_multi_final$table[7,7], mic_ann_seq_cm_multi_final$table[8,8]), c(mic_ann_func_cm_multi_final$table[1,1], mic_ann_func_cm_multi_final$table[2,2], mic_ann_func_cm_multi_final$table[3,3], mic_ann_func_cm_multi_final$table[4,4], mic_ann_func_cm_multi_final$table[5,5], mic_ann_func_cm_multi_final$table[6,6],  mic_ann_func_cm_multi_final$table[7,7], mic_ann_func_cm_multi_final$table[8,8]))

# Print summary

knitr::kable(x = mic_multi_accuracy_final, col.names = c("LET_IS", "Actual Values", "XGBoost", "ANN Sequential API", "ANN Functional API"),caption = "MIC - Table of Accurate Predictions") %>% kable_styling(font_size = 8)

# Remove data that is no longer required

rm(mic_test_let_is_final_multi_table, pred_xgboost_mic_categorical_final_table,pred_ann_seq_api_multi_table, pred_ann_func_api_multi_table, mic_multi_accuracy_final)


## ----MIC Final Analysis ANN Cleanup - Categorical , include=FALSE, warning=FALSE, echo=FALSE, message = FALSE----------------------------------------------------------------------------------------------------

# Remove data that is no longer required and run garbage collector to free up memory
rm(mic_modified_train_set_ann, mic_modified_test_set_ann)

rm( mic_modified_train_set_ann_outcome, mic_modified_test_set_ann_outcome, dummy_vars_names)

rm(mic_nb_cm_bin_1, mic_nb_cm_bin_2, mic_nb_wi_cm_bin_1, mic_nb_wi_cm_bin_2, mic_xgb_cm_bin, mic_ann_seq_cm_bin, mic_ann_func_cm_bin)

rm ( mic_test_let_is_final_multi, mic_train_let_is_final_multi)

rm(pred_xgboost_mic_categorical_final, pred_ann_seq_api_multi, pred_ann_func_api_multi )

rm(mic_xgb_cm_multi_final, mic_ann_seq_cm_multi_final, mic_ann_func_cm_multi_final)

rm(mic_summary_bin, mic_summary_multi, mic_distributions_multi)

rm(survival_rate)


gc()




## ----MIC Data Clean up Training and Holdout Sets, include=FALSE, warning=FALSE, echo=FALSE, message = FALSE------------------------------------------------------------------------------------------------------

# Perform final cleanup of dataset and run garbage collector

rm(mic_continuous_variables, mic_ordinal_variables, mic_part_ordinal_variables,mic_nominal_variables, mic_complications, survival_rate, dummy_vars_names)

rm( mic_demographic_history_features, mic_ecg_features, mic_emergency_icu_features, mic_ft_features,  mic_infarction_features, mic_medicine_features, mic_relapse_features, mic_serum_features)

rm(mic_orig_train, mic_orig_test,  mic_data_orig, mic_data, mic_orig_imputed_mice, mic_orig_train_imputed_mice, mic_orig_test_imputed_mice)

rm(mic_test_index)

gc()

##########################################################
# End Analysis of MIC Dataset
##########################################################



## ----Download d130 dataset, include=FALSE, warning=FALSE, echo=FALSE, message = FALSE----------------------------------------------------------------------------------------------------------------------------

##########################################################
# Begin Analysis of d130 Dataset
##########################################################

##########################################################
# Download the Raw Data from the respective repositories as the source for truth for the Datasets
##########################################################


options(timeout = 120)


# Diabetes 130-US Hospitals for Years 1999-2008
# https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
# https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip

dl_d130 <- "diabetes+130-us+hospitals+for+years+1999-2008.zip"
if(!file.exists(dl_d130))
  download.file("https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip", dl_d130)

d130_data_file <- "diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv"
  if(!file.exists(d130_data_file))
  unzip(dl_d130, "diabetic_data.csv")

d130_ids_file <- "diabetes+130-us+hospitals+for+years+1999-2008/IDS_mapping.csv"
  if(!file.exists(d130_ids_file))
  unzip(dl_d130, "IDS_mapping.csv")

d130_data <- read.csv(d130_data_file)
d130_ids_mapping <- read.csv(d130_ids_file)


####### 
# Remove Variables used to hold filenames as they are not required anymore
rm(d130_data_file,d130_ids_file,dl_d130)





## ----Prepare d130 data for Analysis, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE--------------------------------------------------------------------------------------------------------------------

# Create a new modified dataset for analysis

# Let us create the list of variables and their types so that it is easier to process them later

################ Classification of Variables ############

# Weight is only documented here for the sake of completeness. It is not used for any other purpose as 97% of the values are missing. 
d130_continuous_variables <- c("weight") 

# We do not include encounter_id as it is unique for each observation and has no value in Prediction

d130_discrete_variables <- c("time_in_hospital","num_lab_procedures","num_procedures","num_medications","number_outpatient","number_emergency","number_inpatient","number_diagnoses")

d130_categorical_variables <- c("patient_nbr","race","age","admission_type_id","discharge_disposition_id","admission_source_id","payer_code","medical_specialty","diag_1","diag_2","diag_3","max_glu_serum","A1Cresult","metformin","repaglinide","nateglinide","chlorpropamide","glimepiride","acetohexamide","glipizide","glyburide","tolbutamide","pioglitazone","rosiglitazone","acarbose","miglitol","troglitazone","tolazamide","examide","citoglipton","insulin","glyburide.metformin","glipizide.metformin","glimepiride.pioglitazone","metformin.rosiglitazone","metformin.pioglitazone")

d130_binary_variables <- c("gender", "change", "diabetesMed")

# Print structure of imported dataset
print("The structure of our d130 as imported is :",quote = FALSE)
print("================================",quote = FALSE)
str(d130_data)
print("================================",quote = FALSE)

# Split of Responses among the different categories

print("================================",quote=FALSE)
print("The Table of Outcomes is",quote=FALSE)
table(d130_data$readmitted)

print(c("Ratio of NO :",round(sum(d130_data$readmitted == "NO")/nrow(d130_data), digits = 4)), quote=FALSE)

print(c("Ratio of >30 :", round(sum(d130_data$readmitted == ">30")/nrow(d130_data), digits = 4)), quote=FALSE)

print(c("Ratio of <30 ", round(sum(d130_data$readmitted == "<30")/nrow(d130_data), digits = 4)), quote=FALSE)
print("================================",quote=FALSE)


## ----Prepare d130 data for Analysis - create indices and partitions, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE------------------------------------------------------------------------------------

# Create a new modified dataset for analysis
# Convert Binary and Categorical Variables to Factors. Replace weight with 0
d130_data_modified <- d130_data %>% 
    mutate_at(c(2:49), ~(ifelse(. == "?", NA, .))) %>%
    mutate_at(c(d130_categorical_variables, d130_binary_variables), ~as.factor(.)) %>% 
    mutate_at(c(d130_discrete_variables), ~as.integer(.)) %>% 
    mutate_at(c("weight"), ~(. = 0)) 


# Split into Training and Testing Sets

set.seed(1024)

d130_test_index <- createDataPartition(y = d130_data_modified$readmitted, times = 1, p = 0.2, list = FALSE)
d130_modified_train <- d130_data_modified[-d130_test_index,]
d130_modified_test <- d130_data_modified[d130_test_index,]


# Create Datasets for Cross Validation 

set.seed(1024)

d130_test_index_cv <- createDataPartition(y = d130_modified_train$readmitted, times = 1, p = 0.2, list = FALSE)
d130_modified_cv_train_set <- d130_modified_train[-d130_test_index_cv,]
d130_modified_cv_test_set <- d130_modified_train[d130_test_index_cv,]






## ----Check Naive Bayes with Modified d130 Dataset, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE------------------------------------------------------------------------------------------------------

# For the sake of convenience and reproducibility, we are reusing variables after deleting them. However, for this to work, the variables need to be sanitised or deleted after use. If variables are not  sanitised or deleted, data could leak inadvertently.

# Set all entries in the weight column to 0 so that we can tune the naive_bayes() function without errors
d130_modified_cv_train_set <- d130_modified_cv_train_set %>% 
          mutate(readmitted = as.factor(readmitted))

d130_modified_cv_test_set <- d130_modified_cv_test_set %>% 
          mutate(readmitted = as.factor(readmitted))

# Fit Model
# Use Column names in the formula to Predict Complications for easy tracking. 
###########
# Fit Naive Bayes
fit_nb_native_readmitted <- naive_bayes(x = d130_modified_cv_train_set[,2:47], y = d130_modified_cv_train_set$readmitted, laplace = 0, usekernel = FALSE, usepoisson = TRUE)

# Print Summary
print("The Naive Bayes Summary is :",quote = FALSE)
print("================================",quote = FALSE)
summary(fit_nb_native_readmitted)
print("================================",quote = FALSE)

# Prepare Predictions
pred_nb_test_native_readmitted <-predict(object = fit_nb_native_readmitted, newdata = d130_modified_cv_test_set[,2:47])

# Print table of predictions
print("================================",quote = FALSE)
print("The Table of predictions is :",quote = FALSE)
table(as.factor(pred_nb_test_native_readmitted))

# Print table of actual values
print("================================",quote = FALSE)
print("The Table of actual values is :",quote = FALSE)
table(d130_modified_cv_test_set$readmitted)

# Print overall accuracy
print("================================",quote = FALSE)
print("The Overall Accuracy is :",quote = FALSE)
mean(pred_nb_test_native_readmitted == d130_modified_cv_test_set$readmitted)
print("================================",quote = FALSE)

# Print confusion matrix
print("The confusion matrix is :",quote = FALSE)
print("================================",quote = FALSE)
confusionMatrix(data = as.factor(pred_nb_test_native_readmitted), reference = as.factor( d130_modified_cv_test_set$readmitted))
print("================================",quote = FALSE)


rm(fit_nb_native_readmitted, pred_nb_test_native_readmitted)




## ----Check Naive Bayes with Modified d130 Dataset with Binary Classification, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE---------------------------------------------------------------------------

# For the sake of convenience and reproducibility, we are reusing variables after deleting them. However, for this to work, the variables need to be sanitised or deleted after use. If variables are not  sanitised or deleted, data could leak inadvertently.


# Create Datasets for Cross Validation 

d130_modified_cv_train_set_binary <- d130_modified_cv_train_set %>% 
        mutate(readmitted = ifelse(readmitted == "NO", "NO", "YES")) %>% 
        mutate(readmitted = as.factor(readmitted))

d130_modified_cv_test_set_binary <- d130_modified_cv_test_set %>% 
        mutate(readmitted = ifelse(readmitted == "NO", "NO", "YES")) %>% 
        mutate(readmitted = as.factor(readmitted))

###########
# Fit Naive Bayes
fit_nb_native_readmitted_binary <- naive_bayes(x = d130_modified_cv_train_set_binary[,2:49], y = d130_modified_cv_train_set_binary$readmitted, laplace = 1, usekernel = FALSE, usepoisson = TRUE)

# Print Summary
print("The Naive Bayes Summary is :",quote = FALSE)
print("================================",quote = FALSE)
summary(fit_nb_native_readmitted_binary)
print("================================",quote = FALSE)

# Prepare Predictions
pred_nb_test_native_readmitted_binary <-predict(object = fit_nb_native_readmitted_binary, newdata = d130_modified_cv_test_set_binary)

# Print table of predictions 
print("================================",quote = FALSE)
print("The Table of predictions is :",quote = FALSE)
table(as.factor(pred_nb_test_native_readmitted_binary))

# Print table of actual values
print("================================",quote = FALSE)
print("The Table of actual values is :",quote = FALSE)
table(d130_modified_cv_test_set_binary$readmitted)

# Print overall accuracy
print("================================",quote = FALSE)
print("The Overall Accuracy is :",quote = FALSE)
mean(pred_nb_test_native_readmitted_binary == d130_modified_cv_test_set_binary$readmitted)

# Print confusion matrix
print("================================",quote = FALSE)
print("The confusion matrix is :",quote = FALSE)
confusionMatrix(data = pred_nb_test_native_readmitted_binary, reference = d130_modified_cv_test_set_binary$readmitted)
print("================================",quote = FALSE)


rm(fit_nb_native_readmitted_binary, pred_nb_test_native_readmitted_binary)



## ----d130 Initial Analysis - Naive Bayes data cleanup , include=FALSE, warning=FALSE, echo=FALSE, message = FALSE------------------------------------------------------------------------------------------------

rm (d130_modified_cv_train_set, d130_modified_cv_test_set)

rm(d130_data_modified, d130_modified_cv_train_set_binary, d130_modified_cv_test_set_binary, d130_modified_train, d130_modified_test)

gc()



## ----Prepare Predictions with Random Forest for d130, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE, out.width= "50%"---------------------------------------------------------------------------------

# For the sake of convenience and reproducibility, we are reusing variables after deleting them. However, for this to work, the variables need to be sanitised or deleted after use. If variables are not  sanitised or deleted, data could leak inadvertently.

# Use Original Dataset as Random Forest struggles with Large Numbers of Factors
# Mutate readmitted field as factors as we want Random Forest to perform classification
# Set weight=0 for all observations

d130_data_orig <- d130_data %>% 
              mutate_at(c("weight"), ~(. = 0)) %>%
              mutate_at(c("readmitted"), ~as.factor(.))

# Use previously created indices for partitioning

d130_orig_train <- d130_data_orig[-d130_test_index,]
d130_orig_test <- d130_data_orig[d130_test_index,]  

d130_modified_cv_train_set <- d130_orig_train[-d130_test_index_cv,]
d130_modified_cv_test_set <- d130_orig_train[d130_test_index_cv,] 


# Print Random Forest mtry and error rates
print("=====================================",quote = FALSE)
print("The Random Forest mtry values and error rates are",quote = FALSE) 

# Use tuneRF to choose best mtry value
set.seed(1024)
best_mtry_rf_tune <- tuneRF(x = d130_modified_cv_train_set[,c(2:49)], y = d130_modified_cv_train_set[,50], stepFactor = 0.5, improve = 0.00001, trace = TRUE, plot = TRUE, doBest = TRUE)

# Extract and print details of Random Forest 
print("=====================================",quote = FALSE)
print( c(" Details for Random Forest for the d130 Dataset are: "),quote = FALSE, justify = "left") 
print(c("Prediction Type :",best_mtry_rf_tune$type),quote = FALSE)
print(c("Number of Trees (ntree) :",best_mtry_rf_tune$ntree),quote = FALSE)
print(c("mtry value :",best_mtry_rf_tune$mtry),quote = FALSE)
print("=====================================",quote = FALSE)

# Prepare Predictions
pred_rf <- predict(best_mtry_rf_tune, newdata = d130_modified_cv_test_set[,c(2:49)], type = "response")

# Print table of predictions
print("================================",quote = FALSE)
print("The Table of predictions is :",quote = FALSE)
table(as.factor(pred_rf))

# Print table of actual values
print("================================",quote = FALSE)
print("The Table of actual values is :",quote = FALSE)
table(d130_modified_cv_test_set$readmitted)

# Print overall accuracy
print("================================",quote = FALSE)
print("The Overall Accuracy is :",quote = FALSE)
mean(pred_rf == d130_modified_cv_test_set$readmitted)

# Print confusion matrix
print("================================",quote = FALSE)
print("The confusion matrix is :",quote = FALSE)
confusionMatrix(data = pred_rf,reference = d130_modified_cv_test_set$readmitted)
print("================================",quote = FALSE)

# Extract and print the 10 most important Features identified by Random Forest
imp <- as.data.frame(randomForest::importance(best_mtry_rf_tune))
imp <- data.frame(Importance = imp$MeanDecreaseGini,
           names   = rownames(imp))
imp <- imp[order(imp$Importance, decreasing = TRUE),]

print("The first 10 Features in order of decreasing importance in prediction are:",quote = FALSE)
print("=============================================",quote = FALSE)
knitr::kable(x = imp[1:10,], col.names = c("Col Id", "Importance", "Names"), caption = "MIC Prediction - Variable Importance (MeanDecreaseGini)")
print("=============================================",quote = FALSE)


## ----d130 Initial Analysis - Random Forest data cleanup , include=FALSE, warning=FALSE, echo=FALSE, message = FALSE----------------------------------------------------------------------------------------------

# Remove Data that is no longer required

rm(d130_data_orig, d130_orig_train, d130_orig_test)
rm(d130_modified_cv_train_set,d130_modified_cv_test_set, best_mtry_rf_tune, pred_rf, imp)

gc()



## ----Perform Initial Consolidation of Unique Values - 1, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE------------------------------------------------------------------------------------------------

############ Extract Summary about Patient Numbers and Diag Codes ############

print(c("============================="),quote = FALSE, justify = "Center")
# Number of Unique "patient_nbr for the entire d130 dataset"
print(c("The Total Number of Unique Patients in the entire d130 dataset is :",length(unique(d130_data$patient_nbr))), justify = "Left",quote = FALSE)

# Number of Unique "diag_1" codes for the entire d130 dataset
print(c("The Total Number of Unique 'diag_1' codes in the entire d130 dataset is :",length(unique(d130_data$diag_1))), justify = "Left",quote = FALSE)

# Number of Unique "diag_2" codes for the entire d130 dataset
print(c("The Total Number of Unique 'diag_2' codes in the entire d130 dataset is :",length(unique(d130_data$diag_2))), justify = "Left",quote = FALSE)

# Number of Unique "diag_3" codes for the entire d130 dataset
print(c("The Total Number of Unique 'diag_3' codes in the entire d130 dataset is :",length(unique(d130_data$diag_3))), justify = "Left",quote = FALSE)

print(c("============================="),quote = FALSE, justify = "Center")


## ----Perform Initial Consolidation of Unique Values - 2, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE------------------------------------------------------------------------------------------------


############ Extract Summary about Diabetes and Diag Codes ############

# Build Regular Expression pattern for CVD and Diabetes based on ICD9 codes
cvd_pattern <- "^39([3-8])|^40(2|4)|(^4(1|2).+)|^785"
diabetes_pattern = "^2(49|50).*"


# Extract and Print diag_1 codes related to CVD
cvd_diag1_unique <- unique(str_extract(d130_data$diag_1, cvd_pattern)) %>% na.omit()

# List of Unique "diag_1" codes referring to CVD
print(c("============================="),quote = FALSE, justify = "Center")
print(c(" The list of Unique 'diag_1' codes referring to CVD is :"))
print(c(cvd_diag1_unique),quote = FALSE, justify = "Left")

#  Number of Unique "diag_1" codes referring to CVD
print(c("The Total Number of Unique 'diag_1' codes referring to CVD is :",length(cvd_diag1_unique)),quote = FALSE, justify = "Left")

# Number of Observations in "diag_1" referring to CVD
print(c("The Total Number of Observations in 'diag_1' referring to CVD is :", sum(str_detect(d130_data$diag_1, cvd_pattern))),quote = FALSE, justify = "Left")
print(c("============================="),quote = FALSE, justify = "Center")

# Extract and Print diag_2 codes related to diabetes
diabetes_diag2_unique <- unique(str_extract(d130_data$diag_2,diabetes_pattern)) %>% na.omit()

# List of Unique "diag_2" codes referring to Diabetes
print(c("============================="),quote = FALSE, justify = "Center")
print(c(" The list of Unique 'diag_2' codes referring to Diabetes is :"))
print(c(diabetes_diag2_unique),quote = FALSE, justify = "Left")

# Number of Unique "diag_2" codes referring to Diabetes
print(c("The Total Number of Unique 'diag_2' codes referring to Diabetes is :",length(diabetes_diag2_unique)),quote = FALSE, justify = "Left")

# Number of Observations in "diag_2" referring to Diabetes
print(c("The Total Number of Observations in 'diag_2' referring to Diabetes is :", sum(str_detect(d130_data$diag_2, diabetes_pattern))),quote = FALSE, justify = "Left")

print(c("============================="),quote = FALSE, justify = "Center")

# Extract and Print diag_2 codes related to diabetes
diabetes_diag3_unique <- unique(str_extract(d130_data$diag_3, diabetes_pattern)) %>% na.omit()

# List of Unique "diag_3" codes referring to Diabetes
print(c("============================="),quote = FALSE, justify = "Center")
print(c(" The list of Unique 'diag_3' codes referring to Diabetes is :"))
print(c(diabetes_diag3_unique),quote = FALSE, justify = "Left")

# Number of Unique "diag_3" codes referring to Diabetes
print(c("The Total Number of Unique 'diag_3' codes referring to Diabetes is :",length(diabetes_diag3_unique)),quote = FALSE, justify = "Left")

# Number of Observations in "diag_3" referring to Diabetes
print(c("The Total Number of Observations in 'diag_3' referring to Diabetes is :", sum(str_detect(d130_data$diag_3, diabetes_pattern))),quote = FALSE, justify = "Left")
print(c("============================="),quote = FALSE, justify = "Center")




## ----Perform Initial Consolidation of Unique Values - 3, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE------------------------------------------------------------------------------------------------

#########

# Clean up Data to remove entries where the Patient Expired or was Moved to Hospice
# Sum used for diagnostic purposes. Commented out for Report creation
# sum(d130_data$discharge_disposition_id == c(11,13,14,19,20,21))

d130_data_cleaned_up <- d130_data %>% filter(discharge_disposition_id != c(11,13,14,19,20,21,23)) 

d130_cvd <- d130_data_cleaned_up %>% filter(diag_1 %in% cvd_diag1_unique)

######### Check Counts for Diagnostic Purposes ##########
# Commented out for Report creation
# Count of observations with each diag_1 code
#d130_cvd %>% dplyr::count(diag_1)

# Count of observations with each diag_2 code 
#d130_cvd %>% dplyr::count(diag_2)

# Count of observations with each diag_3 code 
#d130_cvd %>% dplyr::count(diag_3)

# Retain only cases where Secondary or Additional Secondary Diagnoses indicate Diabetes
d130_cvd <-  d130_cvd %>% 
      filter( diag_2 %in% diabetes_diag2_unique | diag_3 %in% diabetes_diag3_unique)

# Number of Unique "patient_nbr" for the Filtered Dataset
print(c("The Number of Unique Patients for the Dataset filtered for CVD cases with diabetes is :", length(unique(d130_cvd$patient_nbr))),quote = FALSE, justify = "Left")

######### Check Counts for Diagnostic Purposes ##########
# Commented out for Report creation
# Count of observations with each diag_2 code after filtering
#d130_cvd %>% dplyr::count(diag_2,sort = TRUE)

# Count of observations with each diag_3 code after filtering
#d130_cvd %>% dplyr::count(diag_3, sort = TRUE)

# Print Structure of the Filtered Dataset
print(c("============================="),quote = FALSE, justify = "Center")

print(c("The Structure of the Filtered Datset is :"))
str(d130_cvd)

print(c("============================="),quote = FALSE, justify = "Center")



## ----Perform Initial Consolidation of Unique Values - 4, include=FALSE, warning=FALSE, echo=FALSE, message = FALSE-----------------------------------------------------------------------------------------------

# Remove Data that is no longer required
rm(cvd_diag1_unique,cvd_diag2_unique,cvd_diag3_unique,diabetes_diag1_unique, diabetes_diag2_unique, diabetes_diag3_unique, diabetes_pattern, cvd_pattern, d130_data_cleaned_up)

gc()


## ----Modify d130 data for Analysis using right variable types, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE------------------------------------------------------------------------------------------

# Create a new modified dataset for analysis

# Convert Binary and Categorical Variables to Factors. 
# Set weight=0 for all observations

d130_cvd_modified  <- d130_cvd %>%
    mutate_at(c(2:49), ~(ifelse(. == "?", NA, .))) %>%
    mutate_at(c(d130_categorical_variables, d130_binary_variables,"readmitted"), ~as.factor(.)) %>% 
    mutate_at(c(d130_discrete_variables), ~as.integer(.)) %>% 
    mutate_at(c("weight"), ~as.numeric(0))

# Print Structure of the Filtered Dataset with right variable types
print(c("============================="),quote = FALSE, justify = "Center")

print(c("The Structure of the Filtered Datset with the right variable types is :"))
str(d130_cvd_modified)

print(c("============================="),quote = FALSE, justify = "Center")

# Split into Training and Testing Sets

set.seed(1024)

d130_cvd_test_index <- createDataPartition(y = d130_cvd_modified$readmitted, times = 1, p = 0.2, list = FALSE)
d130_cvd_modified_train <- d130_cvd_modified[-d130_cvd_test_index,]
d130_cvd_modified_test <- d130_cvd_modified[d130_cvd_test_index,]

# Create Datasets for Cross Validation 

set.seed(1024)

d130_cvd_test_index_cv <- createDataPartition(y = d130_cvd_modified_train$readmitted, times = 1, p = 0.2, list = FALSE)
d130_cvd_modified_cv_train_set <- d130_cvd_modified_train[-d130_cvd_test_index_cv,]
d130_cvd_modified_cv_test_set <- d130_cvd_modified_train[d130_cvd_test_index_cv,]


# Split of Responses among the different categories

print("================================",quote=FALSE)
print("The Table of Outcomes is",quote=FALSE)
table(d130_cvd$readmitted)

print(c("The Ratio of Patients who were NOT readmitted is :", round(sum(d130_cvd$readmitted == "NO")/nrow(d130_cvd), digits = 4)),quote = FALSE)

print(c("The Ratio of Patients who were readmitted in >30 days is :", round(sum(d130_cvd$readmitted == ">30")/nrow(d130_cvd), digits = 4)),quote = FALSE)

print(c("The Ratio of Patients who were readmitted in <30 days is :", round(sum(d130_cvd$readmitted == "<30")/nrow(d130_cvd), digits = 4)),quote = FALSE)




## ----Check Naive Bayes with Modified d130 Dataset for CVD, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE----------------------------------------------------------------------------------------------

# For the sake of convenience and reproducibility, we are reusing variables after deleting them. However, for this to work, the variables need to be sanitised or deleted after use. If variables are not  sanitised or deleted, data could leak inadvertently.

# Use Column names in the formula to Predict Complications for easy tracking. 

###########
# Fit Naive Bayes
fit_nb_native_readmitted_cvd <- naive_bayes(x = d130_cvd_modified_cv_train_set[,2:49], y = d130_cvd_modified_cv_train_set$readmitted, laplace = 1, usekernel = FALSE, usepoisson = TRUE)

# Print Summary
print("The Naive Bayes Summary is :",quote = FALSE)
summary(fit_nb_native_readmitted_cvd)

# Prepare Predictions
pred_nb_test_native_readmitted_cvd <-predict(object = fit_nb_native_readmitted_cvd, newdata = d130_cvd_modified_cv_test_set)

# Print table of predictions
print("================================",quote = FALSE)
print("The Table of predictions is :",quote = FALSE)
table(as.factor(pred_nb_test_native_readmitted_cvd))

# PrinttTable of actual values
print("================================",quote = FALSE)
print("The Table of actual values is :",quote = FALSE)
table(d130_cvd_modified_cv_test_set$readmitted)

# Print overall accuracy
print("================================",quote = FALSE)
print("The Overall Accuracy is :",quote = FALSE)
mean(pred_nb_test_native_readmitted_cvd == d130_cvd_modified_cv_test_set$readmitted)

# Print confusion matrix
print("================================",quote = FALSE)
print("The confusion matrix is :",quote = FALSE)
confusionMatrix(data = pred_nb_test_native_readmitted_cvd, reference = d130_cvd_modified_cv_test_set$readmitted)
print("================================",quote = FALSE)

rm(fit_nb_native_readmitted_cvd, pred_nb_test_native_readmitted_cvd)




## ----Check Naive Bayes with Modified d130 Dataset with Binary Classification for CVD, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE-------------------------------------------------------------------

# For the sake of convenience and reproducibility, we are reusing variables after deleting them. However, for this to work, the variables need to be sanitised or deleted after use. If variables are not  sanitised or deleted, data could leak inadvertently.

# Consolidate "<30" and ">30" into a simple "YES"

d130_cvd_modified_cv_train_set_binary <- d130_cvd_modified_cv_test_set %>% mutate(readmitted = ifelse(readmitted == "NO", "NO", "YES")) %>% mutate(readmitted = as.factor(readmitted))

d130_cvd_modified_cv_test_set_binary <- d130_cvd_modified_cv_test_set %>% mutate(readmitted = ifelse(readmitted == "NO", "NO", "YES")) %>% mutate(readmitted = as.factor(readmitted))

###########
# Fit Naive Bayes
fit_nb_native_readmitted_binary_cvd <- naive_bayes(x = d130_cvd_modified_cv_train_set_binary[,2:49], y = d130_cvd_modified_cv_train_set_binary$readmitted, laplace = 1, usekernel = FALSE, usepoisson = TRUE)

# Print Summary
print("The Naive Bayes Summary is :",quote = FALSE)
print("================================",quote = FALSE)
summary(fit_nb_native_readmitted_binary_cvd)
print("================================",quote = FALSE)

# Prepare Predictions
pred_nb_test_native_readmitted_binary_cvd <-predict(object = fit_nb_native_readmitted_binary_cvd, newdata = d130_cvd_modified_cv_test_set_binary)

# Print table of predictions
print("================================",quote = FALSE)
print("The Table of predictions is :",quote = FALSE)
table(as.factor(pred_nb_test_native_readmitted_binary_cvd))

# Print table of actual values
print("================================",quote = FALSE)
print("The Table of actual values is :",quote = FALSE)
table(d130_cvd_modified_cv_test_set_binary$readmitted)

# Print overall accuracy
print("================================",quote = FALSE)
print("The Overall Accuracy is :",quote = FALSE)
mean(pred_nb_test_native_readmitted_binary_cvd == d130_cvd_modified_cv_test_set_binary$readmitted)

# Print confusion matrix
print("================================",quote = FALSE)
print("The confusion matrix is :",quote = FALSE)
confusionMatrix(data = pred_nb_test_native_readmitted_binary_cvd, reference = d130_cvd_modified_cv_test_set_binary$readmitted)
print("================================",quote = FALSE)

rm(fit_nb_native_readmitted_binary_cvd, pred_nb_test_native_readmitted_binary_cvd)



## ----d130 CVD Initial Analysis - Naive Bayes data cleanup , include=FALSE, warning=FALSE, echo=FALSE, message = FALSE--------------------------------------------------------------------------------------------

# Clean up data that is no longer required

rm(d130_data_modified)

rm (d130_cvd_modified_cv_train_set, d130_cvd_modified_cv_test_set)

rm(d130_cvd_modified_cv_train_set_binary, d130_cvd_modified_cv_test_set_binary)

gc()



## ----Prepare Predictions with Random Forest for d130 CVD, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE, out.width= "50%"-----------------------------------------------------------------------------


# For the sake of convenience and reproducibility, we are reusing variables after deleting them. However, for this to work, the variables need to be sanitised or deleted after use. If variables are not  sanitised or deleted, data could leak inadvertently.

# Use Original Dataset as Random Forest struggles with Large Numbers of Factors
# Mutate readmitted field as factors as we want Random Forest to perform classification
# Set weight=0 for all observations
d130_cvd_data_orig <- d130_cvd %>%   
              mutate_at(c("readmitted"), ~as.factor(.)) %>%
              mutate_at(c("weight"), ~(. = 0)) 

# Use Indices created earlier for creating the Training and Testing datasets 
d130_cvd_orig_train <- d130_cvd_data_orig[-d130_cvd_test_index,]
d130_cvd_orig_test <- d130_cvd_data_orig[d130_cvd_test_index,]


# Create Datasets for Cross Validation 
d130_cvd_modified_cv_train_set <- d130_cvd_orig_train[-d130_cvd_test_index_cv,]
d130_cvd_modified_cv_test_set <- d130_cvd_orig_train[d130_cvd_test_index_cv,]


print("=====================================",quote = FALSE)
print("The Random Forest mtry values and error rates are",quote = FALSE) 

set.seed(1024)
# Fit Random Forest
best_mtry_rf_tune_d130_cvd <- tuneRF(x = d130_cvd_modified_cv_train_set[,c(2:49)], y = d130_cvd_modified_cv_train_set[,50], stepFactor = 0.5, improve = 0.00001, trace = TRUE, plot = TRUE, doBest = TRUE)

# Extract and print details of Random Forest 
print("=====================================",quote = FALSE)
print( c(" Details for Random Forest  for the d130 Dataset are: "),quote = FALSE, justify = "left") 
print(c("Prediction Type :",best_mtry_rf_tune_d130_cvd$type),quote = FALSE)
print(c("Number of Trees (ntree) :",best_mtry_rf_tune_d130_cvd$ntree),quote = FALSE)
print(c("mtry value :",best_mtry_rf_tune_d130_cvd$mtry),quote = FALSE)
print("=====================================",quote = FALSE)

# Prepare Predictions
pred_rf_d130_cvd <- predict(best_mtry_rf_tune_d130_cvd, newdata = d130_cvd_modified_cv_test_set[,c(2:49)], type = "response")

# Print table of predictions
print("================================",quote = FALSE)
print("The Table of predictions is :",quote = FALSE)
table(as.factor(pred_rf_d130_cvd))

# Print table of actual values
print("================================",quote = FALSE)
print("The Table of actual values is :",quote = FALSE)
table(d130_cvd_modified_cv_test_set$readmitted)

# Print overall accuracy
print("================================",quote = FALSE)
print("The Overall Accuracy is :",quote = FALSE)
mean(pred_rf_d130_cvd == d130_cvd_modified_cv_test_set[,50])

# Print confusion matrix
print("================================",quote = FALSE)
print("The confusion matrix is :",quote = FALSE)
confusionMatrix(data = pred_rf_d130_cvd,reference = d130_cvd_modified_cv_test_set[,50])


# Extract and print the 10 most important Features identified by Random Forest
imp <- as.data.frame(randomForest::importance(best_mtry_rf_tune_d130_cvd))
imp <- data.frame(Importance = imp$MeanDecreaseGini,
           names   = rownames(imp))
imp <- imp[order(imp$Importance, decreasing = TRUE),]

print("The first 10 Features in order of decreasing importance in prediction are:",quote = FALSE)
print("=============================================",quote = FALSE)
knitr::kable(x = imp[1:10,], col.names = c("Col Id", "Importance", "Names"), caption = "MIC Prediction - Variable Importance (MeanDecreaseGini)")
print("=============================================",quote = FALSE)

rm(best_mtry_rf_tune_d130_cvd, pred_rf_d130_cvd, imp, d130_cvd_data_orig, d130_cvd_orig_train, d130_cvd_orig_test)


## ----d130 CVD Initial Analysis - Random Forest data cleanup , include=FALSE, warning=FALSE, echo=FALSE, message = FALSE------------------------------------------------------------------------------------------

# Remove Data that is no longer required

rm(d130_data_orig, d130_orig_train, d130_orig_test)

rm(d130_modified_cv_train_set,d130_modified_cv_test_set)

gc()



## ----Prepare d130 CVD Encounters Dataset for Initial Analysis using XGBoost - One Hot Encoding, include=FALSE, warning=FALSE, echo=FALSE, message = FALSE--------------------------------------------------------

# For the sake of convenience and reproducibility, we are reusing variables after deleting them. However, for this to work, the variables need to be sanitised or deleted after use. If variables are not  sanitised or deleted, data could leak inadvertently.

# Some Cleaning up is required 

# There is a single observation for which the gender is not known. We will remove the observation as the patient_nbr is also unique and we cannot infer the value otherwise.

# Imputation does not work, 
# We will replace missing values with NA
# We will replace all rows in column 6 (weight) with 0

# Do not set outcome "readmitted" as we do not want to one_hot encode it

d130_cvd_boost  <- d130_cvd %>%
        filter( gender != "Unknown/Invalid") %>%
        mutate_at(c(2:49), ~(ifelse(. == "?", NA, .))) %>%
        mutate_at(c("weight"), ~(as.numeric(0))) %>%
        mutate_at(c(d130_categorical_variables, d130_binary_variables), ~as.factor(.)) %>% 
        mutate_at(c(d130_discrete_variables), ~as.integer(.)) 


################# Identify and Remove Columns with a Single Factor Level #################

single_level_factors <- c()

for (i in 1:ncol(d130_cvd_boost)){
  if (nlevels(d130_cvd_boost[,i]) == 1){
    single_level_factors <- c(single_level_factors,colnames(d130_cvd_boost)[i])
    }
}
rm(i)

# Remove above Columns as they cause unwanted issues during processing later

d130_cvd_boost <- d130_cvd_boost %>% select(-c(single_level_factors))


# Create a new vector for Dummy Variables with the names of the Level1 Factors removed


index_level1_factors <- which(!d130_categorical_variables %in% single_level_factors)

d130_categorical_variables_dummy <- d130_categorical_variables[index_level1_factors]

################# Expand Categorical values using Dummy Variables ################# 


# We will use a new data.table to store the Dummy Variables. We will remove Existing Columns in this new data.table to exclude them from further calculations. 

# Ensure Order of operations is as under. Else we could have erroneous results.

# Convert to data.table for one-hot encoding
d130_cvd_boost <- data.table(d130_cvd_boost)


# Create one-hot variables
d130_cvd_boost <- one_hot(dt = d130_cvd_boost, cols = c(d130_categorical_variables_dummy, d130_binary_variables), sparsifyNAs = FALSE, naCols = FALSE, dropCols = TRUE, dropUnusedLevels = TRUE)

d130_cvd_boost <- as.data.frame(d130_cvd_boost)

rm(index_categorical_var, d130_categorical_variables_dummy)

# Create Training and Testing Datasets

d130_cvd_boost_train <- d130_cvd_boost[-d130_cvd_test_index,] # Training & Validation Set
d130_cvd_boost_test <- d130_cvd_boost[d130_cvd_test_index,] # Holdout Set

# Create Datasets for Cross Validation 

d130_cvd_modified_cv_train_set <- d130_cvd_boost_train[-d130_cvd_test_index_cv,] # Training CV
d130_cvd_modified_cv_test_set <- d130_cvd_boost_train[d130_cvd_test_index_cv,] # Testing CV


# Remove Variables and Datasets that are not needed anymore
rm(index_level1_factors, single_level_factors)



## ----Check XGBoost with Modified d130 CVD Dataset, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE------------------------------------------------------------------------------------------------------


# For the sake of convenience and reproducibility, we are reusing variables after deleting them. However, for this to work, the variables need to be sanitised or deleted after use. If variables are not  sanitised or deleted, data could leak inadvertently.

# Use Datasets created earlier

########################## WARNING WARNING WARNING ##########################
############## DOUBLE CHECK INPUT FEATURES TO PREVENT LEAKAGE ###############
####### DOUBLE CHECK PARENTHESIS TO ENSURE THEY CHOOSE RIGHT COLUMNS #########

# Create XGBoost DMatrix

dtrain_d130_cvd <- xgb.DMatrix(data = as.matrix(d130_cvd_modified_cv_train_set[,c(2:(ncol(d130_cvd_modified_cv_train_set)-1))]), label = as.factor(d130_cvd_modified_cv_train_set$readmitted), nthread = 8)


# Fit XGBoost
fit_xgboost_d130_cvd <- xgb.train(
    data = dtrain_d130_cvd,
    max_depth = 10,
    eta = 0.15,
    nthread = 8,
    nrounds = 240,
    objective = "multi:softmax", 
    params = list("num_class" = 8, "booster" = "gbtree"),
    verbose = 0 # set verbose=2 during development, tuning and testing
)

# Print Summary
print("================================",quote = FALSE)
print("The XGBoost Summary is :",quote = FALSE)
fit_xgboost_d130_cvd
print("================================",quote = FALSE)

# Prepare Predictions

pred_xgboost_d130_cvd <-predict(object = fit_xgboost_d130_cvd, newdata = as.matrix(d130_cvd_modified_cv_test_set[,c(2:(ncol(d130_cvd_modified_cv_test_set)-1))]))

# Used for diagnostics during development and testing. commented out for report creation
# table(as.factor(pred_xgboost_d130_cvd))

# Print table of predictions
print("================================",quote = FALSE)
print("The Table of predictions is :",quote = FALSE)
pred_xgboost_d130_cvd_refactored <- ifelse(pred_xgboost_d130_cvd == 1, "<30", (ifelse(pred_xgboost_d130_cvd == 2, ">30", "NO")))
table(as.factor(pred_xgboost_d130_cvd_refactored))

# Print table of actual values
#print("================================",quote = FALSE)
print("The Table of actual values is :",quote = FALSE)
table(d130_cvd_modified_cv_test_set$readmitted)

# Print overall accuracy
print("================================",quote = FALSE)
print("The Overall Accuracy is :",quote = FALSE)
mean(pred_xgboost_d130_cvd_refactored == d130_cvd_modified_cv_test_set$readmitted)

# Print confusion matrix
print("================================",quote = FALSE)
print("The confusion matrix is :",quote = FALSE)
confusionMatrix(data = as.factor(pred_xgboost_d130_cvd_refactored), reference = as.factor(d130_cvd_modified_cv_test_set$readmitted))
print("================================",quote = FALSE)

rm(dtrain_d130_cvd, fit_xgboost_d130_cvd, pred_xgboost_d130_cvd, pred_xgboost_d130_cvd_refactored)



## ----d130 CVD Initial Analysis - XGBoost and ADABoost data cleanup , include=FALSE, warning=FALSE, echo=FALSE, message = FALSE-----------------------------------------------------------------------------------

# Remove Data that is no longer required
  
rm(d130_cvd_modified_cv_train_set, d130_cvd_modified_cv_test_set)

rm(d130_cvd_data_xgboost)


gc()


## ----Prepare d130 CVD Encounters Dataset for Initial Analysis using Neural Networks - One Hot Encoding, include=FALSE, warning=FALSE, echo=FALSE, message = FALSE------------------------------------------------

# Since Neural Networks do not understand Factors, we will use the Original Numeric Values. 

# Some Cleaning up is required 

################ Create Grouping of Variables for Functional API ############

# To prevent too much complexity in our models, we will only maintain 6 groups namely "demographics", "history", "hospitalisation", "medicine" and "diagnoses"

d130_patient_nbr <- c("patient_nbr")

d130_demographics_categorical <- c("race","age")  
d130_demographics_binary <- c("gender")

d130_medicine_categorical <- c("metformin","repaglinide","nateglinide","chlorpropamide","glimepiride","acetohexamide","glipizide","glyburide","tolbutamide","pioglsitazone","rosiglitazone","acarbose","miglitol","troglitazone","tolazamide","examide","citoglipton","insulin","glyburide.metformin","glipizide.metformin","glimepiride.pioglitazone","metformin.rosiglitazone","metformin.pioglitazone")

d130_medicine_binary <- c("change", "diabetesMed")

d130_history_discrete <- c("number_outpatient","number_emergency","number_inpatient")

d130_hospitalisation_discrete <- c("time_in_hospital","num_lab_procedures","num_procedures","num_medications","number_diagnoses")
d130_hospitalisation_categorical  <- c("admission_type_id","discharge_disposition_id","admission_source_id","payer_code","medical_specialty","max_glu_serum", "A1Cresult")

d130_diag_categorical <- c("diag_1", "diag_2","diag_3")

###############

# There is a single observation for which the gender is not known. We will remove the observation as the patient_nbr is also unique and we cannot infer the value otherwise.

# Imputation does not work, Instead we will replace all rows in column 6 (weight) with 0
# We will use columns 11 (payer_code) and 12 (medical_specialty) as-is since ANN cannot handle NA values and we will lose information in these columns if we zero them out.In the existing dataset, NA are already coded using "?" 
 

d130_cvd_modified  <- d130_cvd %>%
        filter( gender != "Unknown/Invalid") %>%
        mutate_at(c("weight"), ~(.=0)) %>%
        mutate_at(c(d130_categorical_variables, d130_binary_variables), ~as.factor(.)) %>% 
        mutate_at(c(d130_discrete_variables), ~as.integer(.)) %>% 
        mutate_at(c("readmitted"), ~(. = as.factor(.)))


# Create Training and Testing Sets

d130_cvd_train <- d130_cvd_modified[-d130_cvd_test_index,] # Training & Validation Set
d130_cvd_test <- d130_cvd_modified[d130_cvd_test_index,] # Holdout Set

##############
# Verify that all NA are removed before creating Datasets for Cross Validation
# Used during development and Testing. Commented out for Report Creation
# sum(is.na(d130_cvd_train))
# str(d130_cvd_train)

################# Identify and Remove Columns with a Single Factor Level #################

single_level_factors <- c()

for (i in 1:ncol(d130_cvd_modified)){
  if (nlevels(d130_cvd_modified[,i]) == 1){
    single_level_factors <- c(single_level_factors,colnames(d130_cvd_modified)[i])
    }
}
rm(i)

# Remove above Columns as they cause unwanted issues during processing later

d130_cvd_train <- d130_cvd_train %>% select(-c(single_level_factors))


# Create a new vector for Dummy Variables with the names of the Level1 Factors removed
# Add readmitted column to the vector for creation of Dummy Variables

index_level1_factors <- which(!d130_categorical_variables %in% single_level_factors)

d130_categorical_variables_dummy <- c(d130_categorical_variables[index_level1_factors], "readmitted")

################# Expand Categorical values using Dummy Variables ################# 


# We will use a new data.table to store the Dummy Variables. We will remove Existing Columns in this new data.table to exclude them from further calculations. 

# Ensure Order of operations is as under. Else we could have erroneous results.

# Store the Original Prediction Column as it will be lost during conversion later.  

d130_cvd_readmitted_train <- d130_cvd_train[,"readmitted"]

# Convert to data.table for one-hot encoding
d130_cvd_train <- data.table(d130_cvd_train)


# Create one-hot variables
d130_cvd_train <- one_hot(dt = d130_cvd_train, cols = c(d130_categorical_variables_dummy, d130_binary_variables), sparsifyNAs = FALSE, naCols = FALSE, dropCols = TRUE, dropUnusedLevels = TRUE)


d130_cvd_train <- as.data.frame(d130_cvd_train)


rm(index_categorical_var, d130_categorical_variables_dummy)

######### Extract relevant Column Names from Created Dummy Vars Set ###############

# Create respective groups of feature/predictor variables
# Not required for "patient_nbr" as it is only a single categorical entry
d130_demographics_all <- c(d130_demographics_binary, d130_demographics_categorical)
d130_medicine_all <- c(d130_medicine_categorical, d130_medicine_binary)
d130_hospitalisation_all <- c(d130_hospitalisation_discrete, d130_hospitalisation_categorical)
d130_diag_all <- c(d130_diag_categorical)

# Initialise Empty Vectors to collect and store respective column indices
d130_patient_nbr_col_indices <- c() 
d130_demographics_col_indices <- c()
d130_medicine_col_indices <- c()
d130_hospitalisation_col_indices <- c()
d130_diag_col_indices <- c()

# Extract the Column indices and store them in respective vectors. 

for (i in 1:(length(d130_patient_nbr))) {
  
 d130_patient_nbr_col <- str_which( string = colnames(d130_cvd_train), pattern = d130_patient_nbr[i])
 d130_patient_nbr_col_indices <- c(d130_patient_nbr_col_indices, d130_patient_nbr_col)
 rm(d130_patient_nbr_col)
}


for (i in 1:(length(d130_demographics_all))) {
  
 d130_demographics_col <- str_which( string = colnames(d130_cvd_train), pattern = d130_demographics_all[i])
 d130_demographics_col_indices <- c(d130_demographics_col_indices, d130_demographics_col)
 rm(d130_demographics_col)
}

for (i in 1:(length(d130_medicine_all))) {
  
 d130_medicine_col <- str_which( string = colnames(d130_cvd_train), pattern = (c(d130_medicine_all[i])))
 d130_medicine_col_indices <- c(d130_medicine_col_indices, d130_medicine_col)
 rm(d130_medicine_col)
}

for (i in 1:(length(d130_hospitalisation_all))) {
  
 d130_hospitalisation_col <- str_which( string = colnames(d130_cvd_train), pattern = d130_hospitalisation_all[i])
 d130_hospitalisation_col_indices <- c(d130_hospitalisation_col_indices, d130_hospitalisation_col)
 rm(d130_hospitalisation_col)
}

# Column Index Extraction for History is not required as it is very simple and consists of only 3 discrete predictors. The vriable "d130_history_discrete" can be used as-is

for (i in 1:(length(d130_diag_all))) {
  
 d130_diag_col <- str_which( string = colnames(d130_cvd_train), pattern = d130_diag_all[i])
 d130_diag_col_indices <- c(d130_diag_col_indices, d130_diag_col)
 rm(d130_diag_col)
}

rm(i)

# Used for Diagnostics during development and testing. Commented out for Report creation
# Check if we have collected them all
# sum(length(d130_demographics_col_indices)+ length(d130_history_discrete) + length(d130_hospitalisation_col_indices)) + +length(d130_medicine_col_indices) + length(d130_diag_col_indices)

# Create Datasets for Cross Validation
# Use existing Indices

d130_cvd_cv_train_set <- d130_cvd_train[-d130_cvd_test_index_cv,]
d130_cvd_cv_test_set <- d130_cvd_train[d130_cvd_test_index_cv,]

d130_cvd_readmitted_cv_train <- d130_cvd_readmitted_train[-d130_cvd_test_index_cv]
d130_cvd_readmitted_cv_test <- d130_cvd_readmitted_train[d130_cvd_test_index_cv]

# Remove Variables and Datasets that are not needed anymore
rm(index_level1_factors, single_level_factors)




## ----Perform Initial Analysis for d130 CVD Dataset using Neural Networks after expansion - Functional API, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE, fig.align='center', out.width="50%"---------

########## Hack to achieve Consistent results between Keras Runs ############
# Unload and Reload Keras
unload(package = "keras3", quiet = TRUE)
library(keras3)
# Set Up Environment
set.seed(1024)
keras3::set_random_seed(1024)
#tensorflow::set_random_seed(1024, disable_gpu = TRUE) #Uncomment if using TensorFlow
reticulate::py_set_seed(0)
#############################################################################


########################## WARNING WARNING WARNING ##########################
############## DOUBLE CHECK INPUT FEATURES TO PREVENT LEAKAGE ###############

############# Create the Input Data#########################################

########## Patient Number ###########################

train_features_patient_nbr <- as.matrix(d130_cvd_cv_train_set[,c(d130_patient_nbr_col_indices)])
val_features_patient_nbr <- as.matrix(d130_cvd_cv_test_set[,c(d130_patient_nbr_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_patient_nbr <- colnames(train_features_patient_nbr)

d130_train_list_col_sd_eq_0 <- which(colSds(train_features_patient_nbr) == 0)

index_features_scaling <- which(!feature_names_patient_nbr  %in% names(d130_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_patient_nbr[,index_features_scaling])

train_features_patient_nbr <- train_features_patient_nbr[,c(feature_names_for_scaling)]
val_features_patient_nbr <- val_features_patient_nbr[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_patient_nbr %<>% scale()
val_features_patient_nbr %<>% 
        scale(center = attr(train_features_patient_nbr, "scaled:center"),
        scale = attr(train_features_patient_nbr, "scaled:scale"))

rm(d130_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## Demographic ###########################

train_features_demographic <- as.matrix(d130_cvd_cv_train_set[,c(d130_demographics_col_indices)])
val_features_demographic <- as.matrix(d130_cvd_cv_test_set[,c(d130_demographics_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_demographic <- colnames(train_features_demographic)

d130_train_list_col_sd_eq_0 <- which(colSds(train_features_demographic) == 0)

index_features_scaling <- which(!feature_names_demographic  %in% names(d130_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_demographic[,index_features_scaling])

train_features_demographic <- train_features_demographic[,c(feature_names_for_scaling)]
val_features_demographic <- val_features_demographic[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_demographic %<>% scale()
val_features_demographic %<>% 
        scale(center = attr(train_features_demographic, "scaled:center"),
        scale = attr(train_features_demographic, "scaled:scale"))

rm(d130_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )

########## Hospitalisation ###########################

train_features_hospitalisation <- as.matrix(d130_cvd_cv_train_set[,c(d130_hospitalisation_col_indices)])
val_features_hospitalisation <- as.matrix(d130_cvd_cv_test_set[,c(d130_hospitalisation_col_indices)])

####### Remove Columns which cause NaN to be generated ########

feature_names_hospitalisation <- colnames(train_features_hospitalisation)


d130_train_list_col_sd_eq_0 <- which(colSds(train_features_hospitalisation) == 0)

index_features_scaling <- which(!feature_names_hospitalisation  %in% names(d130_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_hospitalisation[,index_features_scaling])

train_features_hospitalisation <- train_features_hospitalisation[,c(feature_names_for_scaling)]
val_features_hospitalisation <- val_features_hospitalisation[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_hospitalisation %<>% scale()
val_features_hospitalisation %<>% 
        scale(center = attr(train_features_hospitalisation, "scaled:center"),
        scale = attr(train_features_hospitalisation, "scaled:scale"))

rm(d130_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## History ###########################

train_features_history <- as.matrix(d130_cvd_cv_train_set[,c(d130_history_discrete)])
val_features_history <- as.matrix(d130_cvd_cv_test_set[,c(d130_history_discrete)])

####### Normal Scaling  ############ 

train_features_history %<>% scale()
val_features_history %<>% 
        scale(center = attr(train_features_history, "scaled:center"),
        scale = attr(train_features_history, "scaled:scale"))

########## Medicine ###########################

train_features_medicine <- as.matrix(d130_cvd_cv_train_set[,c(d130_medicine_col_indices)])
val_features_medicine <- as.matrix(d130_cvd_cv_test_set[,c(d130_medicine_col_indices)])

####### Remove Columns which cause NaN to be generated ########

feature_names_medicine <- colnames(train_features_medicine)

d130_train_list_col_sd_eq_0 <- which(colSds(train_features_medicine) == 0)

index_features_scaling <- which(!feature_names_medicine  %in% names(d130_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_medicine[,index_features_scaling])

train_features_medicine <- train_features_medicine[,c(feature_names_for_scaling)]
val_features_medicine <- val_features_medicine[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_medicine %<>% scale()
val_features_medicine %<>% 
        scale(center = attr(train_features_medicine, "scaled:center"),
        scale = attr(train_features_medicine, "scaled:scale"))

rm(d130_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## Diagnoses ###########################

train_features_diagnoses <- as.matrix(d130_cvd_cv_train_set[,c(d130_diag_col_indices)])
val_features_diagnoses <- as.matrix(d130_cvd_cv_test_set[,c(d130_diag_col_indices)])

####### Remove Columns which cause NaN to be generated Before Scaling  ########

feature_names_diagnoses <- colnames(train_features_diagnoses)

d130_train_list_col_sd_eq_0 <- which(colSds(train_features_diagnoses) == 0)

index_features_scaling <- which(!feature_names_diagnoses  %in% names(d130_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_diagnoses[,index_features_scaling])

train_features_diagnoses <- train_features_diagnoses[,c(feature_names_for_scaling)]
val_features_diagnoses <- val_features_diagnoses[,c(feature_names_for_scaling)]


####### Normal Scaling  ############ 

train_features_diagnoses %<>% scale()
val_features_diagnoses %<>% scale(center = attr(train_features_diagnoses, "scaled:center"),
                        scale = attr(train_features_diagnoses, "scaled:scale"))

rm(d130_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )

###################################

####### Create Training & Validation Targets #######################

train_targets <- as.matrix(d130_cvd_cv_train_set[,c("readmitted_<30", "readmitted_>30", "readmitted_NO")])
val_targets <- as.matrix(d130_cvd_cv_test_set[,c("readmitted_<30", "readmitted_>30", "readmitted_NO")])

###################################

# Let us define the input shapes. 
input_shape_patient_nbr<- ncol(train_features_patient_nbr)
input_shape_demographic <- ncol(train_features_demographic)
input_shape_hospitalisation <- ncol(train_features_hospitalisation)
input_shape_history <- ncol(train_features_history)
input_shape_medicine <- ncol(train_features_medicine)
input_shape_diagnoses <- ncol(train_features_diagnoses)

# Let us build the Keras Inputs & Features
input_patient_nbr <- keras_input(shape(input_shape_patient_nbr), name = "patient_nbr")
input_demographic <- keras_input(shape(input_shape_demographic), name = "demographic")
input_hospitalisation <- keras_input(shape(input_shape_hospitalisation), name = "hospitalisation")
input_history <- keras_input(shape(input_shape_history), name = "history")
input_medicine <-  keras_input(shape(input_shape_medicine), name = "medicine")
input_diagnoses <-  keras_input(shape(input_shape_diagnoses), name = "diagnoses")

# Let us build the ANN


patient_nbr_features <-  
    layer_dense(object = input_patient_nbr, units = 992, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024)    

demographic_features <- 
    layer_dense(object = input_demographic, units = 128, activation = "relu") |>  
    layer_dropout(rate = 0.3, seed = 1024) 

hospitalisation_features <- 
    layer_dense(object = input_hospitalisation, units = 736 , activation = "relu") |>  
    layer_dropout(rate = 0.3, seed = 1024) 

history_features <-  
    layer_dense(object = input_history, units = 24, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024) 
  
medicine_features <- 
    layer_dense(object = input_medicine, units = 456, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024) 

diagnoses_features <- 
    layer_dense(object = input_diagnoses, units = 2880, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024) 



# Let us combine the Feature Layers together


combined_features <- layer_concatenate(list(patient_nbr_features, demographic_features, hospitalisation_features, history_features, medicine_features, diagnoses_features))


pred_functional_api <- layer_dense(object = combined_features, units = 3, activation = "softmax")

# Instantiate an end-to-end model 

functional_api_model <- keras_model(
  inputs = list(input_patient_nbr, input_demographic, input_hospitalisation, input_history, input_medicine, input_diagnoses),
  outputs = list(pred_functional_api)
)

# Plot model for Visualisation
# Commented out for Report Creation
# plot(functional_api_model, show_shapes = TRUE)

# summary(functional_api_model)

# Collect counts for generating initial weights if required
counts <- table(d130_cvd_readmitted_cv_train) # Counts for Training Set

# Used for Diagnostic purposes. Commented out for Report creation
# print("counts for training set")
# counts
# print("counts for testing set")
# table(d130_cvd_readmitted_cv_test) # Counts for Validation Set


# Train the Model 

functional_api_model |> compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

######  Fit model ##############

history <- functional_api_model |> 
  fit(
  x = list(patient_nbr = train_features_patient_nbr,demographic = train_features_demographic, hospitalisation = train_features_hospitalisation, history = train_features_history, medicine = train_features_medicine, diagnoses = train_features_diagnoses),
  y = train_targets,
  validation_data = list(list(val_features_patient_nbr,val_features_demographic, val_features_hospitalisation, val_features_history, val_features_medicine, val_features_diagnoses),val_targets),
  batch_size = 8192,
  epochs = 30,
  verbose = 0 # set verbose=2 during development, tuning and testing
)

#### Print History for Reference ########

print("Plot of metrics and trends during training")
plot(history)

######## Evaluate the Model against the Validation Targets ########
# commented out for Report creation
#functional_api_model |> evaluate(list(val_features_patient_nbr,val_features_demographic, val_features_hospitalisation, val_features_history, val_features_medicine, val_features_diagnoses), val_targets)


# Prepare Predictions

d130_probs_functional_api <- functional_api_model |> predict(list(val_features_patient_nbr,val_features_demographic, val_features_hospitalisation, val_features_history, val_features_medicine, val_features_diagnoses))

pred_ann_d130_functional_api <- max.col(d130_probs_functional_api) - 1L

#Used for development and diagnostics. Commented out for Report creation
#table(pred_ann_d130_functional_api)

pred_ann_d130_functional_api_refactored <- ifelse(pred_ann_d130_functional_api == 0, "<30", (ifelse(pred_ann_d130_functional_api == 1, ">30", "NO")))

print("================================",quote = FALSE)
print("The Table of predictions is :",quote = FALSE)
table(as.factor(pred_ann_d130_functional_api_refactored))

print("================================",quote = FALSE)
print("The Table of actual values is :",quote = FALSE)
table(as.factor(d130_cvd_readmitted_cv_test))

print("================================",quote = FALSE)
print("The Overall Accuracy is :",quote = FALSE)
mean(pred_ann_d130_functional_api_refactored == d130_cvd_readmitted_cv_test)

print("================================",quote = FALSE)
print("The Confusion Matrix is :",quote = FALSE)
confusionMatrix(data = as.factor(pred_ann_d130_functional_api_refactored), reference = as.factor(d130_cvd_readmitted_cv_test))
print("================================",quote = FALSE)

#### Remove variables that are not required ##########

rm(metrics, counts, feature_names, model, pred_ann_d130_functional_api, pred_ann_d130_functional_api_refactored)

rm( patient_nbr_features, demographic_features,history_features, hospitalisation_features,medicine_features, diagnoses_features,combined_features)

rm( input_patient_nbr, input_demographic,input_history, input_hospitalisation, input_medicine, input_diagnoses)

rm( train_features_patient_nbr, train_features_demographic, train_features_history, train_features_hospitalisation, train_features_medicine, train_features_diagnoses)

rm( val_features_patient_nbr, val_features_demographic, val_features_history, val_features_hospitalisation, val_features_medicine, val_features_diagnoses)

rm( feature_names_patient_nbr, feature_names_demographic, feature_names_hospitalisation, feature_names_medicine, feature_names_diagnoses)

rm( input_shape_patient_nbr, input_shape_demographic, input_shape_history, input_shape_hospitalisation, input_shape_medicine, input_shape_diagnoses)

rm(d130_patient_nbr_col_indices, d130_demographics_col_indices, d130_hospitalisation_col_indices, d130_medicine_col_indices, d130_diag_col_indices)

rm(functional_api_model, pred_functional_api, d130_probs_functional_api)

rm(class_weight)

rm(train_features, train_targets, val_features, val_targets, history)





## ----Perform Initial Analysis for d130 CVD Dataset using Neural Networks after expansion - All Variables, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE, out.width="50%"------------------------------

########## Hack to achieve Consistent results between Keras Runs ############
# Unload and Reload Keras
unload(package = "keras3", quiet = TRUE)
library(keras3)
# Set Up Environment
set.seed(1024)
keras3::set_random_seed(1024)
#tensorflow::set_random_seed(1024, disable_gpu = TRUE) #Uncomment if using TensorFlow
reticulate::py_set_seed(0)
#############################################################################

#All Variables. We will use a Sequential API to build a Model and check its prediction abilities

########################## WARNING WARNING WARNING ##########################
############## DOUBLE CHECK INPUT FEATURES TO PREVENT LEAKAGE ###############
####### DOUBLE CHECK PARENTHESIS TO ENSURE THEY CHOOSE RIGHT COLUMNS #########

feature_names <- colnames(d130_cvd_cv_train_set[2:(ncol(d130_cvd_cv_train_set)-3)])

train_features <- as.matrix(d130_cvd_cv_train_set[feature_names])
train_targets <- as.matrix(d130_cvd_cv_train_set[,c("readmitted_<30", "readmitted_>30", "readmitted_NO")])
val_features <- as.matrix(d130_cvd_cv_test_set[feature_names])
val_targets <- as.matrix(d130_cvd_cv_test_set[,c("readmitted_<30", "readmitted_>30", "readmitted_NO")])

# We also need to ensure that the input variables are scaled & centered so that they we do not have any one factor unnecessarily influencing the outcome and all have adequate representation
####### Remove Columns which cause NaN to be generated Before Scaling  ########

feature_names <- colnames(train_features)

d130_train_list_col_sd_eq_0 <- which(colSds(train_features) == 0)

index_features_scaling <- which(!feature_names  %in% names(d130_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features[,index_features_scaling])

train_features <- train_features[,c(feature_names_for_scaling)]
val_features <- val_features[,c(feature_names_for_scaling)]


####### Normal Scaling  ############ 


train_features %<>% scale()
val_features %<>% scale(center = attr(train_features, "scaled:center"),
                        scale = attr(train_features, "scaled:scale"))

sum(is.nan(train_features))
sum(is.nan(val_features))

# Let us build the Nueral Network 


model <-
  keras3::keras_model_sequential(input_shape = ncol(train_features)) |>
  layer_dense(units = 768, activation = 'relu') |> #768 
  layer_dropout(rate = 0.3) |>
  layer_dense(units = 512, activation = 'relu') |> #512 
  layer_dropout(rate = 0.3) |>
  layer_dense(units = 192) |> #192
  layer_dense(units = 192, activation = 'relu') |> #192
  layer_dropout(rate = 0.3) |>
  layer_dense(3, activation = "softmax")


# Train the Model 

model |> compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)


##############

history <- model |> fit(
  train_features, train_targets,
  validation_data = list(val_features, val_targets),
  batch_size = 8192,
  epochs = 30,
  verbose = 0 # Set verbose=2 during development, tuning and testing
)

#### Print History for Reference ########

print("Plot of metrics and trends during training")
plot(history)

# Evaluate Model
# Connented out for Report creation
# model |> evaluate(val_features, val_targets)

# Prepare Predictions

d130_probs_discrete <- model |> predict(val_features)

pred_ann_d130_discrete <- max.col(d130_probs_discrete) - 1L

# Used duing development and diagnosis. Commented out for Report creation
# table(pred_ann_d130_discrete)

# Convert predictions to values that are comparable with actual values
pred_ann_d130_discrete_refactored <- ifelse(pred_ann_d130_discrete == 0, "<30", (ifelse(pred_ann_d130_discrete == 1, ">30", "NO")))

# Print table of predictions
print("================================",quote = FALSE)
print("The Table of predictions is :",quote = FALSE)
table(as.factor(pred_ann_d130_discrete_refactored))

# Print table of actual values
print("================================",quote = FALSE)
print("The Table of actual values is :",quote = FALSE)
table(as.factor(d130_cvd_readmitted_cv_test))

# Print overall accuracy
print("================================",quote = FALSE)
print("The Overall Accuracy is :",quote = FALSE)
mean(pred_ann_d130_discrete_refactored == d130_cvd_readmitted_cv_test)

# Print confusion matrix
print("================================",quote = FALSE)
print("The Confusion Matrix is :",quote = FALSE)
confusionMatrix(data = as.factor(pred_ann_d130_discrete_refactored), reference = as.factor(d130_cvd_readmitted_cv_test))
print("================================",quote = FALSE)

# Remove Data that is no longer required

rm(d130_train_list_cols_sum_eq_0, d130_train_list_cols_sd_eq_0,d130_val_list_cols_sum_eq_0, feature_names_for_scaling, index_features_scaling, pred_ann_d130_discrete, pred_ann_d130_discrete_refactored)

rm(d130_probs_discrete, history, model, train_features, train_targets, val_features, val_targets)





## ----d130 CVD Initail Analysis -ANN data cleanup , include=FALSE, warning=FALSE, echo=FALSE, message = FALSE-----------------------------------------------------------------------------------------------------

rm(d130_cvd_readmitted_train, d130_cvd_readmitted_cv_train, d130_cvd_readmitted_cv_test)

rm(d130_cvd_cv_train_set, d130_cvd_cv_test_set)

rm(d130_test_index_cv)

gc()



## ----Final Analysis with Naive Bayes for Modified d130 CVD Holdout Dataset, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE-----------------------------------------------------------------------------


# Use Column names in the formula to Predict outcomes for easy tracking.

###########
# Fit Naive Bayes
fit_nb_native_readmitted_cvd_final <- naive_bayes(x = d130_cvd_modified_train[,2:49], y = d130_cvd_modified_train$readmitted, laplace = 1, usekernel = FALSE, usepoisson = TRUE)

# Prepare Predictions
pred_nb_test_native_readmitted_cvd_final <-predict(object = fit_nb_native_readmitted_cvd_final, newdata = d130_cvd_modified_test[,2:49])

# Print Table of predictions
print("================================",quote = FALSE)
print("The Table of predictions is :",quote = FALSE)
table(as.factor(pred_nb_test_native_readmitted_cvd_final))

# Print table of actual values
print("================================",quote = FALSE)
print("The Table of actual values is :",quote = FALSE)
table(d130_cvd_modified_test$readmitted)

# Print overall accuracy
print("================================",quote = FALSE)
print("The Overall Accuracy is :",quote = FALSE)
mean(pred_nb_test_native_readmitted_cvd_final == d130_cvd_modified_test$readmitted)

# Print confusion matrix
print("================================",quote = FALSE)
print("The confusion matrix is :",quote = FALSE)
d130_cvd_nb_cm_final <- confusionMatrix(data = pred_nb_test_native_readmitted_cvd_final, reference = d130_cvd_modified_test$readmitted)
d130_cvd_nb_cm_final
print("================================",quote = FALSE)

rm(fit_nb_native_readmitted_cvd_final)





## ----d130 CVD Final Analysis - Naive Bayes data cleanup , include=FALSE, warning=FALSE, echo=FALSE, message = FALSE----------------------------------------------------------------------------------------------

# Remove Data that is no longer required and run the garbage collector
rm(d130_data_modified)

rm (d130_cvd_modified_train, d130_cvd_modified_test)


gc()


## ----Final Analysis with XGBoost for Modified d130 CVD Holdout Dataset, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE---------------------------------------------------------------------------------

# For the sake of convenience, we are reusing variables after deleting them. However, this is not recommended as data could leak inadvertently  


# Use Datasets created earlier

d130_cvd_modified_train_set <- d130_cvd_boost_train  

d130_cvd_modified_test_set <- d130_cvd_boost_test 

########################## WARNING WARNING WARNING ##########################
############## DOUBLE CHECK INPUT FEATURES TO PREVENT LEAKAGE ###############
####### DOUBLE CHECK PRENTHESIS TO ENSURE THEY CHOOSE RIGHT COLUMNS #########

# Create XGBoost DMatrix

dtrain_d130_cvd_final <- xgb.DMatrix(data = as.matrix(d130_cvd_modified_train_set[,c(2:(ncol(d130_cvd_modified_train_set)-1))]), label = as.factor(d130_cvd_modified_train_set$readmitted), nthread = 8)


# Fit XGBoost
fit_xgboost_d130_cvd_final <- xgb.train(
    data = dtrain_d130_cvd_final,
    max_depth = 10,
    eta = 0.15,
    nthread = 8,
    nrounds = 240,
    objective = "multi:softmax", 
    params = list("num_class" = 8, "booster" = "gbtree"),
    verbose = 0 # set verbose=2 during development, tuning and testing
)


# Prepare Predictions

pred_xgboost_d130_cvd_final <-predict(object = fit_xgboost_d130_cvd_final, newdata = as.matrix(d130_cvd_modified_test_set[,c(2:(ncol(d130_cvd_modified_test_set)-1))]))

# Used for diagnostics during development and testing. commented out for report creation
# table(as.factor(pred_xgboost_d130_cvd_final))

# Print table of predictions
print("================================",quote = FALSE)
print("The Table of predictions is :",quote = FALSE)
pred_xgboost_d130_cvd_final_refactored <- ifelse(pred_xgboost_d130_cvd_final == 1, "<30", (ifelse(pred_xgboost_d130_cvd_final == 2, ">30", "NO")))
table(as.factor(pred_xgboost_d130_cvd_final_refactored))

# Print table of actual values
print("================================",quote = FALSE)
print("The Table of actual values is :",quote = FALSE)
table(d130_cvd_modified_test_set$readmitted)

# Print overall accuracy
print("================================",quote = FALSE)
print("The Overall Accuracy is :",quote = FALSE)
mean(pred_xgboost_d130_cvd_final_refactored == d130_cvd_modified_test_set$readmitted)

# Print confusion matrix
print("================================",quote = FALSE)
print("The confusion matrix is :",quote = FALSE)
d130_cvd_xgb_cm_final <- confusionMatrix(data = as.factor(pred_xgboost_d130_cvd_final_refactored), reference = as.factor(d130_cvd_modified_test_set$readmitted))
d130_cvd_xgb_cm_final
print("================================",quote = FALSE)

rm(dtrain_d130_cvd_final ,fit_xgboost_d130_cvd_final, pred_xgboost_d130_cvd_final,  d130_cvd_modified_cv_test_set_readmitted_refact_final)



## ----d130 CVD Final Analysis - XGBoost data cleanup , include=FALSE, warning=FALSE, echo=FALSE, message = FALSE--------------------------------------------------------------------------------------------------

# Remove Data that is no longer required and run the garbage collector
rm(d130_cvd_boost, d130_cvd_boost_train, d130_cvd_boost_test)

gc()



## ----Prepare d130 CVD Encounters Dataset for Final Analysis using Neural Networks - One Hot Encoding, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE---------------------------------------------------
  
# Used during development and Testing. Commented out for Report creation
# Verify that all NA are removed before creating Datasets for Training and Testing
#sum(is.na(d130_cvd_modified))


##############
# Used during development and Testing. Commented out for Report creation
#str(d130_cvd_modified)

##### Identify and Remove Columns with a Single Factor Level ######

single_level_factors <- c()

for (i in 1:ncol(d130_cvd_modified)){
  if (nlevels(d130_cvd_modified[,i]) == 1){
    single_level_factors <- c(single_level_factors,colnames(d130_cvd_modified)[i])
    }
}
rm(i)

# Remove above Columns as they cause unwanted issues during processing later

d130_cvd_modified <- d130_cvd_modified %>% select(-c(single_level_factors))


# Create a new vector for Dummy Variables with the names of the Level1 Factors removed
# Add readmitted column to the vector for creation of Dummy Vars

index_level1_factors <- which(!d130_categorical_variables %in% single_level_factors)

d130_categorical_variables_dummy <- c(d130_categorical_variables[index_level1_factors], "readmitted")

# Expand Categorical values using Dummy Variables


# We will use a new data.table to store the Dummy Vars. We will remove Existing Columns in this new data.table to exclude them from further calculations. We will bind them to the Original Dataframe later to ensure that we recreate the original variables and the Dummy Vars in one Table. 

# Ensure Order of operations is as under. Else we could have erroneous results.

# Store the Original Prediction Column as it will be lost during conversion later.  

d130_cvd_readmitted_modified <- d130_cvd_modified[,"readmitted"]

# Convert to data.table for one_hot 
# Since this is the last set of algorithms for which we are performing the predictions, we are recycling the same variable. Can be changed to a new variable if required.

d130_cvd_modified <- data.table(d130_cvd_modified)



d130_cvd_modified <- one_hot(dt = d130_cvd_modified, cols = c(d130_categorical_variables_dummy, d130_binary_variables), sparsifyNAs = FALSE, naCols = FALSE, dropCols = TRUE, dropUnusedLevels = TRUE)


d130_cvd_modified <- as.data.frame(d130_cvd_modified)


rm(d130_categorical_variables_dummy)

######### Extract relevant Column Names from Created Dummy Vars Set ###############

# Create respective groups of feature/predictor variables
# Not required for "patient_nbr" as it is only a single categorical entry
d130_demographics_all <- c(d130_demographics_binary, d130_demographics_categorical)
d130_medicine_all <- c(d130_medicine_categorical, d130_medicine_binary)
d130_hospitalisation_all <- c(d130_hospitalisation_discrete, d130_hospitalisation_categorical)
d130_diag_all <- c(d130_diag_categorical)

# Initialise Empty Vectors to collect and store respective column indices
d130_patient_nbr_col_indices <- c() 
d130_demographics_col_indices <- c()
d130_medicine_col_indices <- c()
d130_hospitalisation_col_indices <- c()
d130_diag_col_indices <- c()

# Extract the Column indices and store them in respective vectors. 

for (i in 1:(length(d130_patient_nbr))) {
  
 d130_patient_nbr_col <- str_which( string = colnames(d130_cvd_modified), pattern = d130_patient_nbr[i])
 d130_patient_nbr_col_indices <- c(d130_patient_nbr_col_indices, d130_patient_nbr_col)
 rm(d130_patient_nbr_col)
}


for (i in 1:(length(d130_demographics_all))) {
  
 d130_demographics_col <- str_which( string = colnames(d130_cvd_modified), pattern = d130_demographics_all[i])
 d130_demographics_col_indices <- c(d130_demographics_col_indices, d130_demographics_col)
 rm(d130_demographics_col)
}

for (i in 1:(length(d130_medicine_all))) {
  
 d130_medicine_col <- str_which( string = colnames(d130_cvd_modified), pattern = (c(d130_medicine_all[i])))
 d130_medicine_col_indices <- c(d130_medicine_col_indices, d130_medicine_col)
 rm(d130_medicine_col)
}

for (i in 1:(length(d130_hospitalisation_all))) {
  
 d130_hospitalisation_col <- str_which( string = colnames(d130_cvd_modified), pattern = d130_hospitalisation_all[i])
 d130_hospitalisation_col_indices <- c(d130_hospitalisation_col_indices, d130_hospitalisation_col)
 rm(d130_hospitalisation_col)
}

# Column Index Extraction for History is not required as it is very simple and consists of only 3 discrete predictors. The vriable "d130_history_discrete" can be used as-is

for (i in 1:(length(d130_diag_all))) {
  
 d130_diag_col <- str_which( string = colnames(d130_cvd_modified), pattern = d130_diag_all[i])
 d130_diag_col_indices <- c(d130_diag_col_indices, d130_diag_col)
 rm(d130_diag_col)
}

rm(i)

# Used during development and Testing. Commented out for Report creation
# Check if we have collected them all
# sum(length(d130_demographics_col_indices)+ length(d130_history_discrete) + length(d130_hospitalisation_col_indices)) + +length(d130_medicine_col_indices) + length(d130_diag_col_indices)

# Create Datasets for Cross Validation 
# Use Indices created earlier

d130_cvd_train <- d130_cvd_modified[-d130_cvd_test_index,] # Training & Validation Set
d130_cvd_test <- d130_cvd_modified[d130_cvd_test_index,] # Holdout Set

d130_readmitted_train <- d130_cvd_readmitted_modified[-d130_cvd_test_index]
d130_readmitted_test <- d130_cvd_readmitted_modified[d130_cvd_test_index]

# Remove Variables and Datasets that are not needed anymore
rm(d130_cvd_test_index, d130_cvd_test_index_cv, single_level_factors, index_level1_factors)




## ----Perform Final Analysis for d130 CVD Dataset using Neural Networks after expansion - Functional API, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE, fig.align='center'----------------------------

########## Hack to achieve Consistent results between Keras Runs ############
# Unload and Reload Keras
unload(package = "keras3", quiet = TRUE)
library(keras3)
# Set Up Environment
set.seed(1024)
keras3::set_random_seed(1024)
#tensorflow::set_random_seed(1024, disable_gpu = TRUE) #Uncomment if using TensorFlow
reticulate::py_set_seed(0)
#############################################################################


########################## WARNING WARNING WARNING ##########################
############## DOUBLE CHECK INPUT FEATURES TO PREVENT LEAKAGE ###############

############# Create the Input Data#########################################


########## Patient Number ###########################

train_features_patient_nbr <- as.matrix(d130_cvd_train[,c(d130_patient_nbr_col_indices)])
val_features_patient_nbr <- as.matrix(d130_cvd_test[,c(d130_patient_nbr_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_patient_nbr <- colnames(train_features_patient_nbr)

d130_train_list_col_sd_eq_0 <- which(colSds(train_features_patient_nbr) == 0)

index_features_scaling <- which(!feature_names_patient_nbr  %in% names(d130_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_patient_nbr[,index_features_scaling])

train_features_patient_nbr <- train_features_patient_nbr[,c(feature_names_for_scaling)]
val_features_patient_nbr <- val_features_patient_nbr[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_patient_nbr %<>% scale()
val_features_patient_nbr %<>% 
        scale(center = attr(train_features_patient_nbr, "scaled:center"),
        scale = attr(train_features_patient_nbr, "scaled:scale"))

rm(d130_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## Demographic ###########################

train_features_demographic <- as.matrix(d130_cvd_train[,c(d130_demographics_col_indices)])
val_features_demographic <- as.matrix(d130_cvd_test[,c(d130_demographics_col_indices)])

####### Remove Columns which cause NaN ########

feature_names_demographic <- colnames(train_features_demographic)

d130_train_list_col_sd_eq_0 <- which(colSds(train_features_demographic) == 0)

index_features_scaling <- which(!feature_names_demographic  %in% names(d130_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_demographic[,index_features_scaling])

train_features_demographic <- train_features_demographic[,c(feature_names_for_scaling)]
val_features_demographic <- val_features_demographic[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_demographic %<>% scale()
val_features_demographic %<>% 
        scale(center = attr(train_features_demographic, "scaled:center"),
        scale = attr(train_features_demographic, "scaled:scale"))

rm(d130_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )

########## Hospitalisation ###########################

train_features_hospitalisation <- as.matrix(d130_cvd_train[,c(d130_hospitalisation_col_indices)])
val_features_hospitalisation <- as.matrix(d130_cvd_test[,c(d130_hospitalisation_col_indices)])

####### Remove Columns which cause NaN to be generated ########

feature_names_hospitalisation <- colnames(train_features_hospitalisation)


d130_train_list_col_sd_eq_0 <- which(colSds(train_features_hospitalisation) == 0)

index_features_scaling <- which(!feature_names_hospitalisation  %in% names(d130_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_hospitalisation[,index_features_scaling])

train_features_hospitalisation <- train_features_hospitalisation[,c(feature_names_for_scaling)]
val_features_hospitalisation <- val_features_hospitalisation[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_hospitalisation %<>% scale()
val_features_hospitalisation %<>% 
        scale(center = attr(train_features_hospitalisation, "scaled:center"),
        scale = attr(train_features_hospitalisation, "scaled:scale"))

rm(d130_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## History ###########################

train_features_history <- as.matrix(d130_cvd_train[,c(d130_history_discrete)])
val_features_history <- as.matrix(d130_cvd_test[,c(d130_history_discrete)])

####### Normal Scaling  ############ 

train_features_history %<>% scale()
val_features_history %<>% 
        scale(center = attr(train_features_history, "scaled:center"),
        scale = attr(train_features_history, "scaled:scale"))

########## Medicine ###########################

train_features_medicine <- as.matrix(d130_cvd_train[,c(d130_medicine_col_indices)])
val_features_medicine <- as.matrix(d130_cvd_test[,c(d130_medicine_col_indices)])

####### Remove Columns which cause NaN to be generated ########

feature_names_medicine <- colnames(train_features_medicine)

d130_train_list_col_sd_eq_0 <- which(colSds(train_features_medicine) == 0)

index_features_scaling <- which(!feature_names_medicine  %in% names(d130_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_medicine[,index_features_scaling])

train_features_medicine <- train_features_medicine[,c(feature_names_for_scaling)]
val_features_medicine <- val_features_medicine[,c(feature_names_for_scaling)]

####### Normal Scaling  ############ 

train_features_medicine %<>% scale()
val_features_medicine %<>% 
        scale(center = attr(train_features_medicine, "scaled:center"),
        scale = attr(train_features_medicine, "scaled:scale"))

rm(d130_train_list_col_sd_eq_0, index_features_scaling, feature_names_for_scaling )


########## Diagnoses ###########################

train_features_diagnoses <- as.matrix(d130_cvd_train[,c(d130_diag_col_indices)])
val_features_diagnoses <- as.matrix(d130_cvd_test[,c(d130_diag_col_indices)])

####### Remove Columns which cause NaN to be generated Before Scaling  ########

feature_names_diagnoses <- colnames(train_features_diagnoses)

d130_train_list_col_sd_eq_0 <- which(colSds(train_features_diagnoses) == 0)

index_features_scaling <- which(!feature_names_diagnoses  %in% names(d130_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features_diagnoses[,index_features_scaling])

train_features_diagnoses <- train_features_diagnoses[,c(feature_names_for_scaling)]
val_features_diagnoses <- val_features_diagnoses[,c(feature_names_for_scaling)]


####### Normal Scaling  ############ 

train_features_diagnoses %<>% scale()
val_features_diagnoses %<>% scale(center = attr(train_features_diagnoses, "scaled:center"),
                        scale = attr(train_features_diagnoses, "scaled:scale"))

rm(d130_train_list_cols_sum_eq_0, index_features_scaling, feature_names_for_scaling )

###################################

####### Create Training & Validation Targets #######################

train_targets <- as.matrix(d130_cvd_train[,c("readmitted_<30", "readmitted_>30", "readmitted_NO")])
val_targets <- as.matrix(d130_cvd_test[,c("readmitted_<30", "readmitted_>30", "readmitted_NO")])

###################################

# Let us define the input shapes. 
input_shape_patient_nbr<- ncol(train_features_patient_nbr)
input_shape_demographic <- ncol(train_features_demographic)
input_shape_hospitalisation <- ncol(train_features_hospitalisation)
input_shape_history <- ncol(train_features_history)
input_shape_medicine <- ncol(train_features_medicine)
input_shape_diagnoses <- ncol(train_features_diagnoses)

# Let us build the Keras Inputs & Features
input_patient_nbr <- keras_input(shape(input_shape_patient_nbr), name = "patient_nbr")
input_demographic <- keras_input(shape(input_shape_demographic), name = "demographic")
input_hospitalisation <- keras_input(shape(input_shape_hospitalisation), name = "hospitalisation")
input_history <- keras_input(shape(input_shape_history), name = "history")
input_medicine <-  keras_input(shape(input_shape_medicine), name = "medicine")
input_diagnoses <-  keras_input(shape(input_shape_diagnoses), name = "diagnoses")

# Let us build the ANN

patient_nbr_features <-  
    layer_dense(object = input_patient_nbr, units = 992, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024)    

demographic_features <- 
    layer_dense(object = input_demographic, units = 128, activation = "relu") |>  
    layer_dropout(rate = 0.3, seed = 1024) 

hospitalisation_features <- 
    layer_dense(object = input_hospitalisation, units = 736 , activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024) 

history_features <-  
    layer_dense(object = input_history, units = 24, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024) 
  
medicine_features <- 
    layer_dense(object = input_medicine, units = 456, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024) 

diagnoses_features <- 
    layer_dense(object = input_diagnoses, units = 2880, activation = "relu") |> 
    layer_dropout(rate = 0.3, seed = 1024) 


# Let us combine the feature layers together

combined_features <- layer_concatenate(list(patient_nbr_features, demographic_features, hospitalisation_features, history_features, medicine_features, diagnoses_features))


pred_functional_api <- layer_dense(object = combined_features, units = 3, activation = "softmax")

# Instantiate an end-to-end model 

functional_api_model <- keras_model(
  inputs = list(input_patient_nbr, input_demographic, input_hospitalisation, input_history, input_medicine, input_diagnoses),
  outputs = list(pred_functional_api)
)

# Collect counts to generate initial weights if required
counts <- table(d130_readmitted_train) # Counts for Training Set

# Used for Diagnostic purposes. Commented out for Report creation
# print("counts for training set")
# counts
# print("counts for testing set")
# table(d130_readmitted_test) # counts for Validation Set


# Train the Model 

functional_api_model |> compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy','categorical_accuracy')
)

######  Fit model ##############

functional_api_model |> 
  fit(
  x = list(patient_nbr = train_features_patient_nbr,demographic = train_features_demographic, hospitalisation = train_features_hospitalisation, history = train_features_history, medicine = train_features_medicine, diagnoses = train_features_diagnoses),
  y = train_targets,
  validation_data = list(list(val_features_patient_nbr,val_features_demographic, val_features_hospitalisation, val_features_history, val_features_medicine, val_features_diagnoses),val_targets),
  batch_size = 8192,
  epochs = 30,
  verbose = 0 # Set verbose=2 during development, tuning and testing
)


######## Evaluate the Model against the Validation Targets ########
# Commented out for report creation
#functional_api_model |> evaluate(list(val_features_patient_nbr,val_features_demographic, val_features_hospitalisation, val_features_history, val_features_medicine, val_features_diagnoses), val_targets)


# Prepare Predictions

d130_probs_functional_api <- functional_api_model |> predict(list(val_features_patient_nbr,val_features_demographic, val_features_hospitalisation, val_features_history, val_features_medicine, val_features_diagnoses))

pred_ann_d130_functional_api <- max.col(d130_probs_functional_api) - 1L

# Used for development and diagnostics. commented out for Report creation
#table(pred_ann_d130_functional_api)

# Convert predictions so that they can be compared with actual values
pred_ann_d130_functional_api_refactored <- ifelse(pred_ann_d130_functional_api == 0, "<30", (ifelse(pred_ann_d130_functional_api == 1, ">30", "NO")))

# Print table of predictions
print("================================",quote = FALSE)
print("The Table of predictions is :",quote = FALSE)
table(as.factor(pred_ann_d130_functional_api_refactored))

# Print table of actual values
print("================================",quote = FALSE)
print("The Table of actual values is :",quote = FALSE)
table(as.factor(d130_readmitted_test))

# Print overall accuracy
print("================================",quote = FALSE)
print("The Overall Accuracy is :",quote = FALSE)
mean(pred_ann_d130_functional_api_refactored == d130_readmitted_test)

# Print confusion matrix
print("================================",quote = FALSE)
print("The confusion matrix is :",quote = FALSE)
d130_cvd_ann_func_cm_final <- confusionMatrix(data = as.factor(pred_ann_d130_functional_api_refactored), reference = as.factor(d130_readmitted_test))
d130_cvd_ann_func_cm_final
print("================================",quote = FALSE)

#### Remove variables that are not required ##########

rm(metrics, counts, feature_names, model, pred_ann_d130_functional_api)

rm( patient_nbr_features, demographic_features,history_features, hospitalisation_features,medicine_features, diagnoses_features,combined_features)

rm( input_patient_nbr, input_demographic,input_history, input_hospitalisation, input_medicine, input_diagnoses)

rm( train_features_patient_nbr, train_features_demographic, train_features_history, train_features_hospitalisation, train_features_medicine, train_features_diagnoses)

rm( val_features_patient_nbr, val_features_demographic, val_features_history, val_features_hospitalisation, val_features_medicine, val_features_diagnoses)

rm( feature_names_patient_nbr, feature_names_demographic, feature_names_hospitalisation, feature_names_medicine, feature_names_diagnoses)

rm(input_shape_patient_nbr, input_shape_demographic, input_shape_history, input_shape_hospitalisation, input_shape_medicine, input_shape_diagnoses)

rm(d130_patient_nbr_col_indices, d130_demographics_col_indices, d130_hospitalisation_col_indices, d130_medicine_col_indices, d130_diag_col_indices)

rm(functional_api_model, pred_functional_api, d130_probs_functional_api)

rm(class_weight)

rm(train_features, train_targets, val_features, val_targets, history)





## ----Perform Final Analysis for d130 CVD Dataset using Neural Networks after expansion - All Variables, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE-------------------------------------------------

########## Hack to achieve Consistent results between Keras Runs ############
# Unload and Reload Keras
unload(package = "keras3", quiet = TRUE)
library(keras3)
# Set Up Environment
set.seed(1024)
keras3::set_random_seed(1024)
#tensorflow::set_random_seed(1024, disable_gpu = TRUE) #Uncomment if using TensorFlow
reticulate::py_set_seed(0)
#############################################################################


########################## WARNING WARNING WARNING ##########################
############## DOUBLE CHECK INPUT FEATURES TO PREVENT LEAKAGE ###############
####### DOUBLE CHECK PARENTHESIS TO ENSURE THEY CHOOSE RIGHT COLUMNS #########

feature_names <- colnames(d130_cvd_train[2:(ncol(d130_cvd_train)-3)])

train_features <- as.matrix(d130_cvd_train[feature_names])
train_targets <- as.matrix(d130_cvd_train[,c("readmitted_<30", "readmitted_>30", "readmitted_NO")])

val_features <- as.matrix(d130_cvd_test[feature_names])
val_targets <- as.matrix(d130_cvd_test[,c("readmitted_<30", "readmitted_>30", "readmitted_NO")])

# We also need to ensure that the input variables are scaled & centered so that they we do not have any one factor unnecessarily influencing the outcome and all have adequate representation
####### Remove Columns which cause NaN to be generated Before Scaling  ########

feature_names <- colnames(train_features)

d130_train_list_col_sd_eq_0 <- which(colSds(train_features) == 0)

index_features_scaling <- which(!feature_names  %in% names(d130_train_list_col_sd_eq_0))

feature_names_for_scaling <- colnames(train_features[,index_features_scaling])

train_features <- train_features[,c(feature_names_for_scaling)]
val_features <- val_features[,c(feature_names_for_scaling)]


####### Normal Scaling  ############ 


train_features %<>% scale()
val_features %<>% scale(center = attr(train_features, "scaled:center"),
                        scale = attr(train_features, "scaled:scale"))

sum(is.nan(train_features))
sum(is.nan(val_features))

# Let us build the Nueral Network 

model <-
  keras3::keras_model_sequential(input_shape = ncol(train_features)) |>
  layer_dense(units = 768, activation = 'relu') |> #768 
  layer_dropout(rate = 0.3) |>
  layer_dense(units = 512, activation = 'relu') |> #512 
  layer_dropout(rate = 0.3) |>
  layer_dense(units = 192) |> #192
  layer_dense(units = 192, activation = 'relu') |> #192
  layer_dropout(rate = 0.3) |>
  layer_dense(3, activation = "softmax")


# Train the Model 

model |> compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)



model |> fit(
  train_features, train_targets,
  validation_data = list(val_features, val_targets),
  batch_size = 8192,
  epochs = 30,
  verbose = 0 # Set verbose=2 during development, tuning and testing
)



# Evaluate Model
# Commented out for Report creation
# model |> evaluate(val_features, val_targets)

# Prepare Predictions

d130_probs_seq_api_final <- model |> predict(val_features)

pred_ann_d130_seq_api_final <- max.col(d130_probs_seq_api_final) - 1L

# Used for development and diagnostics. Commented out for Report creation 
#table(d130_probs_seq_api_final)

# Convert predictions so that the can be compared to actual values
pred_ann_d130_seq_api_refactored <- ifelse(pred_ann_d130_seq_api_final == 0, "<30", (ifelse(pred_ann_d130_seq_api_final == 1, ">30", "NO")))

# Print table of predictions
print("================================",quote = FALSE)
print("The Table of predictions is :",quote = FALSE)
table(as.factor(pred_ann_d130_seq_api_refactored))

# print table of actual values
print("================================",quote = FALSE)
print("The Table of actual values is :",quote = FALSE)
table(as.factor(d130_readmitted_test))

# print overall accuracy
print("================================",quote = FALSE)
print("The Overall Accuracy is :",quote = FALSE)
mean(pred_ann_d130_seq_api_refactored == d130_readmitted_test)

# print confusion matrix
print("================================",quote = FALSE)
print("The Confusion Matrix is :",quote = FALSE)
d130_cvd_ann_seq_cm_final <- confusionMatrix(data = as.factor(pred_ann_d130_seq_api_refactored), reference = as.factor(d130_readmitted_test))
d130_cvd_ann_seq_cm_final

##### Remove Variables that are not required anymore ######

rm(d130_train_list_col_sd_eq_0,d130_val_list_cols_sum_eq_0, feature_names_for_scaling, feature_names, index_features_scaling, pred_ann_d130_seq_api_final)

rm(d130_probs_seq_api_final, history, model, train_features, train_targets, val_features, val_targets)



## ----d130 CVD Results Summary - Categorical Outcome, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE----------------------------------------------------------------------------------------------------

# Print Split of Responses among the different categories for reference
print("================================",quote=FALSE)
print("The Ratios of Outcomes for reference is",quote=FALSE)

print(c("Ratio of NO :",round(sum(d130_cvd$readmitted == "NO")/nrow(d130_data), digits = 4)), quote=FALSE)

print(c("Ratio of >30 :", round(sum(d130_cvd$readmitted == ">30")/nrow(d130_data), digits = 4)), quote=FALSE)

print(c("Ratio of <30 ", round(sum(d130_cvd$readmitted == "<30")/nrow(d130_data), digits = 4)), quote=FALSE)
print("================================",quote=FALSE)

# Prepare final results table for categorical outcome
d130_cvd_summary <- data.frame(c(d130_cvd_nb_cm_final$overall["Accuracy"]),c( d130_cvd_xgb_cm_final$overall["Accuracy"]), c(d130_cvd_ann_seq_cm_final$overall["Accuracy"]),c( d130_cvd_ann_func_cm_final$overall["Accuracy"]))
rownames(d130_cvd_summary) <- "Accuracy"
colnames(d130_cvd_summary) <- c("Naive Bayes", "XGBoost", "ANN Sequential API", "ANN Functional API")

# Print results 
knitr::kable(x = d130_cvd_summary, caption = "d130 CVD - Results Summary", digits = 4) %>% kable_styling(font_size = 8)



## ----d130 CVD Results Summary - Distributions of predictions, include=TRUE, warning=FALSE, echo=FALSE, message = FALSE-------------------------------------------------------------------------------------------

# Prepare Tables of Actual and Predicted values for easy visualisation

d130_cvd_readmitted_actuals_final <- data.frame(table(d130_readmitted_test))
colnames(d130_cvd_readmitted_actuals_final) <- c("readmitted","Actual Values")

pred_nb_test_native_readmitted_cvd_final_table <- data.frame(table(pred_nb_test_native_readmitted_cvd_final))
colnames(pred_nb_test_native_readmitted_cvd_final_table) <- c("readmitted","Naive Bayes")

pred_xgboost_d130_cvd_final_refactored_table <- data.frame(table(pred_xgboost_d130_cvd_final_refactored))
colnames(pred_xgboost_d130_cvd_final_refactored_table) <- c("readmitted","XGBoost")

pred_ann_d130_seq_api_refactored_table <- data.frame(table(pred_ann_d130_seq_api_refactored))
colnames(pred_ann_d130_seq_api_refactored_table) <- c("readmitted","ANN Sequential API")

pred_ann_d130_functional_api_refactored_table <- data.frame(table(pred_ann_d130_functional_api_refactored))
colnames(pred_ann_d130_functional_api_refactored_table) <- c("readmitted","ANN Functional API")

d130_cvd_distributions_final <- d130_cvd_readmitted_actuals_final %>% 
    left_join(pred_nb_test_native_readmitted_cvd_final_table, by = "readmitted") %>% 
    left_join(pred_xgboost_d130_cvd_final_refactored_table, by = "readmitted") %>% 
    left_join(pred_ann_d130_seq_api_refactored_table, by = "readmitted") %>% 
    left_join(pred_ann_d130_functional_api_refactored_table, by = "readmitted") %>% 
    replace_na(repl = 0)


# Print distributions

knitr::kable(x = d130_cvd_distributions_final, col.names = c("readmitted","Actual Values", "Naive Bayes","XGBoost", "ANN Sequential API", "ANN Functional API"), caption = "d130 CVD - Table of distributions") %>% kable_styling(font_size = 8)



# Print Accurate Values

d130_cvd_accuracy_final <- data.frame(c(d130_cvd_readmitted_actuals_final[1,1], d130_cvd_readmitted_actuals_final[2,1], d130_cvd_readmitted_actuals_final [3,1]) , c(d130_cvd_readmitted_actuals_final[1,2], d130_cvd_readmitted_actuals_final[2,2], d130_cvd_readmitted_actuals_final [3,2]) , c(d130_cvd_nb_cm_final$table[1,1], d130_cvd_nb_cm_final$table[2,2], d130_cvd_nb_cm_final$table[3,3]), c(d130_cvd_xgb_cm_final$table[1,1], d130_cvd_xgb_cm_final$table[2,2], d130_cvd_xgb_cm_final$table[3,3]), c(d130_cvd_ann_seq_cm_final$table[1,1], d130_cvd_ann_seq_cm_final$table[2,2], d130_cvd_ann_seq_cm_final$table[3,3]), c( d130_cvd_ann_func_cm_final$table[1,1],  d130_cvd_ann_func_cm_final$table[2,2],  d130_cvd_ann_func_cm_final$table[3,3]))


knitr::kable(x = d130_cvd_accuracy_final, col.names = c("readmitted","Actual Values", "Naive Bayes","XGBoost", "ANN Sequential API", "ANN Functional API"),caption = "d130 CVD - Table of Accurate Predictions") %>% kable_styling(font_size = 8)


rm(d130_cvd_readmitted_actuals_final, pred_nb_test_native_readmitted_cvd_final_table, pred_xgboost_d130_cvd_final_refactored_table, pred_ann_d130_seq_api_refactored_table, pred_ann_d130_functional_api_refactored_table, d130_cvd_distributions_final, d130_cvd_accuracy_final)


## ----d130 CVD Final Analysis -ANN data cleanup , include=FALSE, warning=FALSE, echo=FALSE, message = FALSE-------------------------------------------------------------------------------------------------------

rm(d130_binary_variables, d130_categorical_variables, d130_continuous_variables, d130_discrete_variables)

rm(d130_demographics_all, d130_demographics_binary, d130_demographics_categorical, d130_diag_all, d130_diag_categorical, d130_history_discrete, d130_hospitalisation_all, d130_hospitalisation_categorical, d130_hospitalisation_discrete, d130_medicine_all, d130_medicine_binary, d130_medicine_categorical, d130_patient_nbr )

rm(pred_nb_test_native_readmitted_cvd_final, pred_xgboost_d130_cvd_final_refactored, pred_ann_d130_seq_api_refactored, pred_ann_d130_functional_api_refactored)

rm(d130_cvd_nb_cm_final, d130_cvd_xgb_cm_final, d130_cvd_ann_seq_cm_final, d130_cvd_ann_func_cm_final, d130_cvd_summary)

rm(d130_test_index)

rm(d130_cvd, d130_cvd_modified, d130_cvd_train, d130_cvd_test, d130_cvd_modified_train_set, d130_cvd_modified_test_set, d130_data, d130_ids_mapping)

rm(d130_cvd_readmitted_modified, d130_readmitted_train, d130_readmitted_test)

##########################################################
# End Analysis of d130 Dataset
##########################################################

rm(nCores)

gc()

##########################################################
# End Project
##########################################################

