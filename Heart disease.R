#Load the packages
library(tidyverse) #data cleaning
library(skimr)  #descriptive statistics
library(ggvis) #scatterplots
library(corrgram)
library(corrplot)
library(GGally)
library(ggplot2)
library(dplyr) 
library(purrr)
library(caTools)
library(reshape2)
library(caret)
library(randomForest)
library(e1071)
library(xgboost)
library(pROC)
library(nnet)
library(DMwR2)
library(xgboost)
library(rpart)
library(ipred)
library(rpart.plot)
library(stats)
library(lattice)
library(knitr)



# Load dataset 
heart_data <- read.csv("C:/Users/LENOVO/Desktop/RStudio/Dataset/heart.csv")

# Convert heart_data to a data frame
heart_df <- as.data.frame(heart_data)

# view the head
head(heart_df)

# Count the occurrences of unique values in the "target" column
table(heart_df$target)

# Count the occurrences of unique values in the "cp" column
table(heart_df$cp)

# Using 'mutate' and 'case_when' to rename values
heart_df <- heart_df %>%
  mutate(cp = case_when(
    cp == 0 ~ 1,
    cp == 1 ~ 2,
    cp == 2 ~ 3,
    cp == 3 ~ 4,
    TRUE ~ cp  # If no match, keep the original value
  ))

# Checking the count the occurrences of unique values in the "cp" column
table(heart_df$cp)

# view the head
head(heart_df)

#dimensions of the data
dim(heart_df)

#data structure
str(heart_df)

# list types for each attribute
sapply(heart_df, class)

#column names
colnames(heart_df)

#data summary
summary(heart_df)

#checking for missing values
sum(is.na(heart_df))

# Check missing values for each of the columns
heart_df %>%
  purrr::map_df(~ sum(is.na(.)))

# View summary statistics with numeric hist
skimmed_data <- skim(heart_df)
View(skimmed_data)



# Exploratory Data Analysis

# Convert "target" to a factor
heart_df$target <- factor(heart_df$target)


age.plot <- ggplot(heart_df, mapping = aes(x = age, fill = target)) +
  geom_histogram() +
  facet_wrap(vars(age)) +
  labs(title = "Prevelance of Heart Disease Across Age", x = "Age (years)", y = "Count", fill = "Heart Disease")

age.plot

# Negative(0) and  Positive(1) Heart disease(target)
ggplot(heart_df, aes(x = target, fill = target)) +
  geom_bar() +
  scale_fill_manual(values = c("#FA8072", "skyblue")) +
  labs(title = "Negative(0) and Positive(1) Heart Disease")


# Convert "target" to a factor
heart_df$cp <- factor(heart_df$cp)

# Chest pain count
ggplot(heart_df, aes(x = cp, fill = cp)) +
  geom_bar() +
  scale_fill_manual(values = c("#997950","#F9A602", "#FA8072", "skyblue")) +
  labs(title = "Chest Pain Types")

# Age distribution
ggplot(heart_df, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "skyblue", color = "#FA8072") +
  labs(title = "Distribution of Age", x = "Age", y = "Count")

# Convert "sex" to a factor
heart_df$sex <- factor(heart_df$sex)

# Female(0) and Male(1)
ggplot(heart_df, aes(x = sex, fill = sex)) +
  geom_bar() +
  scale_fill_manual(values = c("#FA8072", "skyblue")) +
  labs(title = "Female(0) and Male(1)")


# Box plot for resting blood pressure 
ggplot(heart_df, aes(x = "", y = trestbps, fill = "Resting Blood Pressure")) +
  geom_boxplot() +
  labs(title = "Box Plot of Resting Blood Pressure",
       y = "Resting Blood Pressure",
       x = "") +
  theme(legend.position = "none")

# Box plot for cholesterol
ggplot(heart_df, aes(x = "", y = chol, fill = "Cholesterol")) +
  geom_boxplot() +
  labs(title = "Box Plot of Cholesterol",
       y = "Cholesterol",
       x = "") +
  theme(legend.position = "none")


# Calculate the correlation matrix
numeric_heart_df <- heart_df[, sapply(heart_df, is.numeric)]
corr_matrix <- cor(numeric_heart_df)

# Heatmap 
ggplot(data = melt(corr_matrix), aes(Var1, Var2, fill = value)) +
  geom_tile() +
  geom_text(aes(label = round(value, 2)), vjust = 1) +  # Add labels with 2 decimal places
  scale_fill_gradient(low = "skyblue", high = "#FA8072") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Correlation Heatmap")

# Specify the variables and hue (color) for the pair plot
vars <- c("age", "trestbps", "chol", "thalach", "oldpeak")
hue_var <- "target"

# Create the pair plot with red and blue colors
ggpairs(heart_df, columns = vars, title="Correlogram", aes(color = factor(target))) +
  scale_color_manual(values = c("#FA8072", "skyblue"))


# Convert "slope" to a factor
heart_df$slope <- factor(heart_df$slope)

# ST Depression vs Heart Disease(target)
ggplot(heart_df, aes(x = target, y = oldpeak, fill = slope)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(
    title = "ST Depression vs Heart Disease(target)",
    x = "Heart Disease",
    y = "ST Depression"
  ) +
  scale_fill_manual(values = c("#F9A602", "#FA8072", "skyblue"))


# ST Depression vs Heart Disease(target) Against Chest pain
ggplot(heart_df, aes(x = target, y = oldpeak, fill = cp)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(
    title = "ST Depression vs Heart Disease(target) Against Chest pain",
    x = "Heart Disease",
    y = "ST Depression"
  ) +
  scale_fill_manual(values = c("#997950","#F9A602", "#FA8072", "skyblue"))


# Convert 'target' to a factor
heart_df$target <- as.factor(heart_df$target)

# Scatter Plot of Age vs. Resting Blood Pressure
ggplot(data = heart_df, aes(x = age, y = trestbps, color = target)) +
  geom_point() +
  scale_color_manual(values = c("#FA8072", "skyblue")) +  # Set custom colors
  labs(
    title = "Scatter Plot of Age vs. Resting Blood Pressure",
    x = "Age",
    y = "Resting Blood Pressure",
    color = "Target"
  )

# Normalization, Encoding and Splitting of Data

# Extracting the columns to be standardized
columns_to_standardize <- c("age", "trestbps", "chol", "thalach", "oldpeak", "ca")

# Standardizing the selected columns in the dataframe
heart_df[, columns_to_standardize] <- scale(heart_df[, columns_to_standardize])

# encode categorical columns
categorical_col = c("sex", "cp", "slope")
heart_df[,categorical_col] = lapply(heart_df[categorical_col], factor)

# Seed set for reproducibility
set.seed(123)

# Splitting the dataset into training and testing sets (70% train, 30% test)
trainIndex <- createDataPartition(heart_df$target, p = 0.7, list = FALSE)
train_df <- heart_df[trainIndex, ]
test_df <- heart_df[-trainIndex, ]

#Model Info
names(getModelInfo())

# Model Training

# Convert "target" to a factor
train_df$target <- factor(train_df$target)
test_df$target <- factor(test_df$target)

# Function to calculate metrics and plot confusion matrix
metrics_score <- function(actual, predicted) {
  print(confusionMatrix(actual, predicted))
  cm <- confusionMatrix(actual, predicted)$table
  ggplot(data = as.data.frame(cm), aes(x = Prediction, y = Reference)) +
    geom_tile(aes(fill = Freq), color = "white") +
    geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
    scale_fill_gradient(low = "skyblue", high = "#FA8072") +
    labs(x = "Predicted", y = "Actual") +
    theme_minimal()
}

# Create a list to store trained models
trained_models <- list()

# Define control parameters for modeling
ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3)  # Control parameters for cross-validation

# Train Logistic Regression model
log_reg_model <- train(target ~ ., data = train_df, method = "glm", trControl = ctrl)
 
# Add to the list 
trained_models[["Logistic Regression"]] <- log_reg_model

# Checking performance on the test dataset
y_pred_test <- predict(log_reg_model, newdata = test_df)
metrics_score(test_df$target, y_pred_test)

# Summary of the model
summary(log_reg_model)


# Train Naive Bayes
nvb_model <- train(target ~ ., data = train_df, 
                   method = "nb",
                   trControl = ctrl)

# Add to the list 
trained_models[["Naive Bayes"]] <- nvb_model


# Checking performance on the test dataset
y_pred_test <- predict(nvb_model, newdata = test_df)
metrics_score(test_df$target, y_pred_test)


# Train Support Vector Machine (SVM)
svm_model <- train(target ~ ., data = train_df, 
                   method = "svmRadial", 
                   trControl = ctrl, 
                   ranges = expand.grid(C = c(0.1, 1, 10), gamma = c(0.1, 1, 10)) )

# Add to the list 
trained_models[["Support Vector Machine"]] <- svm_model

# Checking performance on the test dataset
y_pred_test <- predict(svm_model, newdata = test_df)
metrics_score(test_df$target, y_pred_test)

# Train Decision Tree
decision_tree_model <- train(target ~ ., data = train_df, 
                             method = "rpart", 
                             trControl = ctrl, 
                             control = rpart.control(minsplit = 20, cp = 0.01))

# Add to the list 
trained_models[["Decision Tree"]] <- decision_tree_model

# Checking performance on the test dataset
y_pred_test <- predict(decision_tree_model, newdata = test_df)
metrics_score(test_df$target, y_pred_test)

# Plotting the decision tree
rpart.plot(decision_tree_model$finalModel)

# Train Gradient Boosting
xgb_model <- train(
  target ~ ., data = train_df, 
  method = "xgbTree", 
  trControl = ctrl,
  tuneGrid = expand.grid(
    nrounds = c(50, 100),
    max_depth = c(3, 6),
    eta = c(0.01, 0.1),  
    gamma = c(0, 0.5),    
    colsample_bytree = c(0.6, 0.8), 
    min_child_weight = c(1, 3),      
    subsample = c(0.6, 0.8)         
  )
)

# Add to the list
trained_models[["Gradient Boosting"]] <- xgb_model

# Checking performance on the test dataset
y_pred_test <- predict(xgb_model, newdata = test_df)
metrics_score(test_df$target, y_pred_test)


# Evaluate the models
results <- resamples(trained_models)

# Summarize the model performance
summary(results)

# Comparing the AUC and Accuracy of the Models

# make a list of AUC and Accuracy
AUC = list()
Accuracy = list()

# getting the predictions and confusion of the models
Pred_lr <- predict(log_reg_model, test_df)
lrConfMat <- confusionMatrix(Pred_lr, test_df[,"target"])

Pred_nvb <- predict(nvb_model, test_df)
nvbConfMat <- confusionMatrix(Pred_nvb, test_df[,"target"])

Pred_svm <- predict(svm_model, test_df)
svmConfMat <- confusionMatrix(Pred_svm, test_df[,"target"])

Pred_dt <- predict(decision_tree_model, test_df)
dtConfMat <- confusionMatrix(Pred_dt, test_df[,"target"])

Pred_xgb <- predict(xgb_model, test_df)
xgbConfMat <- confusionMatrix(Pred_xgb, test_df[,"target"])

# Extracting the AUC and Accuracy scores for the models
AUC$lr <- roc(as.numeric(test_df$target),as.numeric(as.matrix((Pred_lr))))$auc
Accuracy$RF <- lrConfMat$overall['Accuracy']

AUC$nvb <- roc(as.numeric(test_df$target),as.numeric(as.matrix((Pred_nvb))))$auc
Accuracy$nvb <- nvbConfMat$overall['Accuracy']

AUC$svm <- roc(as.numeric(test_df$target),as.numeric(as.matrix((Pred_svm))))$auc
Accuracy$svm <- svmConfMat$overall['Accuracy']

AUC$dt <- roc(as.numeric(test_df$target),as.numeric(as.matrix((Pred_dt))))$auc
Accuracy$dt <- dtConfMat$overall['Accuracy']

AUC$xgb <- roc(as.numeric(test_df$target),as.numeric(as.matrix((Pred_xgb))))$auc
Accuracy$xgb <- xgbConfMat$overall['Accuracy']

# combine AUC and Accuracy as dataframe
do.call(rbind, Map(data.frame, AUC=AUC, Accuracy=Accuracy))



# plot the ROC curve for all the models
model_names <- c('Logistic Regression', 'Naive Bayes', 'SVM', 'Decision Tree', 'Gradient Boosting')

for (i in seq_along(trained_models)) {
  model <- trained_models[[i]]
  name <- model_names[i]
  
  # ROC Curve
  y_scores <- predict(model, newdata = test_df, type = "raw")
  y_scores <- as.numeric(y_scores)
  roc_curve <- roc(test_df$target, y_scores)
  auc <- auc(roc_curve)
  
  plot(roc_curve, col = "skyblue", main = "Receiver Operating Characteristic (ROC) Curve",
       xlab = "False Positive Rate", ylab = "True Positive Rate")
  legend("bottomright", legend = c(paste(name, " (AUC =", round(auc, 2), ")")), col = "skyblue", lty = 1)
  abline(a = 0, b = 1, lty = 2, col = "gray")  # Add diagonal reference line
}


# plotting the performance metrics
model_names <- c('Logistic Regression', 'Naive Bayes', 'SVM', 'Decision Tree','Gradient Boosting')
auc <- c(0.8414910, 0.8338178, 0.9216860, 0.7742701, 0.9904459)
accuracy <- c(0.8431373, 0.8333333, 0.9215686, 0.7745098, 0.9901961)
kappa <- c(0.6851, 0.6668, 0.8431, 0.5486, 0.9804)
sensitivity <- c(0.8855, 0.8141, 0.9139, 0.7703, 0.9803)
specificity <- c(0.8114, 0.8533, 0.9290, 0.7785, 1.0000)

# Combine metrics for comparison
metrics <- data.frame(Model = model_names, AUC = auc, Accuracy = accuracy,
                      Kappa = kappa, Sensitivity = sensitivity, Specificity = specificity)

# Reshaping the data
metrics_melted <- melt(metrics, id.vars = "Model", variable.name = "Metric", value.name = "Value")

# Plotting side-by-side bar chart
ggplot(metrics_melted, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  labs(title = "Model Performance Metrics", x = "Model", y = "Value", fill = "Metric") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Get the feature importance from the best model
feature_importance <- varImp(xgb_model, scale = FALSE)
plot(feature_importance)
