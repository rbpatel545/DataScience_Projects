
getwd()
setwd('C:/Users/Admin/Desktop/ML_Project/Flight_Fare')
install.packages('ggplot2')
install.packages('tidyr')
install.packages('dplyr')
install.packages('caret')
install.packages('e1071')
install.packages('Epi')
install.packages('lubridate')
install.packages("readr")
install.packages("plotly")
install.packages("gridExtra")
install.packages("stats")
install.packages("glmnet")
install.packages("xgboost")

library(ggplot2)
library(e1071)
library(tidyr)
library(dplyr)
library(caret)
library(lubridate)
library(Epi)
library(readr)
library(plotly)
library(gridExtra)
library(stats)
library(glmnet)
library(rpart)
library(randomForest)
library(xgboost)
library(FNN)

ff_df = read.csv("Clean_Dataset.csv",header = TRUE, stringsAsFactors = FALSE, na.strings = c(""," ",NA))
ff_df =  subset(ff_df, select= -c(X))
sapply(ff_df, class)

#Checking for Null Values & Complete Cases
sum(is.na(ff_df))
sum(complete.cases(ff_df))

#Calculation correlation matrix for numeric and integer type dataset

ff_df_Num <- subset(ff_df, select= c("duration", "days_left","price"))
cor_mat<- cor(ff_df_Num)
print(cor_mat)
# as per the output there is no significant correlation between Price, duration and day_left

######################Exploratory Analysis####################################
#Checking Distribution of Target Variable - as seen below Price is not normal but - bimodal Distribution

ggplot(ff_df, aes(x = price)) +
  geom_histogram(binwidth = 10, fill = "skyblue", color = "black") +
  labs(x = "Price", y = "Probability", title = "Price Distribution Histogram")

# 1 : Grouping by Airline Carrier Type and count of Flights taken by them
df1 <- ff_df %>% group_by(flight, airline) %>% summarise(count = n()) %>% ungroup()
airline_counts <- count(df1, airline) %>% arrange(desc(n))
print(airline_counts)

colors <- c("#FF0000", "#00FF00", "#0000FF", "#FF00FF", "#FFFF00", "#00FFFF")

ggplot(df1, aes(airline)) +
  geom_bar(fill = colors) +
  ggtitle("Flights Count of Different Airlines") + scale_fill_manual(values = colors) +
  xlab("Airline") +
  ylab("Count") +
  theme_minimal()
#As can be seen Indigo has the highest value counts and appears to popular choice

#2 : Checking the Share between Class - Economy & Business
df2 <- ff_df %>% group_by(airline,flight, class) %>% summarise(count=n())%>% ungroup()
class_counts <- as.data.frame(table(df2$class))


#Plotting Pie Chart which shows Economy Class has highest share
pie <- plot_ly(class_counts, labels = class_counts$Var1, values = class_counts$Freq, type = "pie", textinfo = "label+percent",
               marker = list(colors = c("steelblue", "orange"))) %>%
  layout(title = "Classes of Different Airlines", showlegend = TRUE,
         legend = list(title = "Class", labels = c("Business", "Economy")))

print(pie)


# 3 : Price statistics for Different Airlines using Box Plot across Airlines, Class & No of Stops
# Median Prices for Vistara is Highest
# Price Range for Air India & Vistara is highest
# Relatively same prices for other Airlines
#As expected business class prices are way high ~10x the Economy
#One Stop Flights are more expensive

colors2 <- c("orange", "steelblue", "green", "red", "purple", "yellow")
ggplot( ff_df, aes(x = airline, y= price))+ geom_boxplot( fill = colors2, color ="Black")+
  ggtitle("Airlines vs Prices")+ xlab("Airline") + ylab("Price")+ theme_minimal()

  
ggplot(ff_df, aes(x = class, y = price)) + geom_boxplot(fill = "steelblue", color = "black") +
ggtitle("Class Vs Price") +  xlab("Airline") + ylab("Price") + theme_minimal()
  
 
colnames(ff_df) 
ggplot(ff_df, aes(x = stops, y = price)) + geom_boxplot(fill = "steelblue", color = "black") +
  ggtitle("Stops Vs Price") +  xlab("Stops") + ylab("Price") + theme_minimal()

# 4 : Prices changes with respect to Time of Departure and arrival
# Price range for Morning and Night Departure &  Arrival is high
# Late night Arrivals & Departures are less costly


p1 <- ggplot(ff_df, aes(x = departure_time, y = price)) + geom_boxplot(fill = "orange", color = "black") +
  ggtitle("Departure Vs Price") +  xlab("Departure") + ylab("Price") + theme_minimal() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

p2 <- ggplot(ff_df, aes(x = arrival_time, y = price)) + geom_boxplot(fill = "orange", color = "black") +
  ggtitle("Arrival Vs Price") +  xlab("Arrival") + ylab("Price") + theme_minimal()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

grid.arrange(p1, p2, ncol = 2)


# 5 : Prices changes with respect to Source and Destination
#Departure and Arrival Prices are high for both Source and Destination City
colnames(ff_df)
p3 <- ggplot( ff_df, aes(x= source_city, y=price))+ geom_boxplot(fill=colors2, color ="Black")+
  xlab("Source City")+ ylab("Prices")+ggtitle("Prices vs Source City")+ theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

p4<-ggplot( ff_df, aes(x= destination_city, y= price))+ geom_boxplot(fill=colors2, color ="Black")+
  xlab("Destination")+ ylab("Prices")+ggtitle("Prices vs Destination City")+ theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

grid.arrange(p3, p4, ncol = 2)

dates <- seq(as.Date("2022-01-01"), as.Date("2022-12-31"), by = "day")
values <- sin(seq(0, 2 * pi, length.out = length(dates)))

################################# Trendline to be created####
###Price vs Days left##################### ##

# Create a data frame
df <- data.frame(date = dates, value = values)

# Plot the trend graph
ggplot(df, aes(x = date, y = value)) +
  geom_line() +
  ggtitle("Trend Graph over Time Series") +
  xlab("Date") +
  ylab("Value") +
  theme_minimal() +
  theme(plot.title = element_text(size = 15, face = "bold"),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12))
################################################

##### Label Encoding Character dataset to
#### Creating a backup dataframe for working ###
dim(ff_df)
ff_en <- as.data.frame(nrow(300153))
ff_en <- ff_df[1:300153,]
sapply(ff_en, class)

#Code for Ordinal Encoding of - departure_time, arrival_time, class
#departure_time & arrival_time - basis levels - Early_Morning","Morning","Afternoon","Evening","Night","Late_Night
unique(ff_en$departure_time)
unique(ff_en$arrival_time)
encode_ordinal <- function(x, order = unique(x)) {
  x <- as.numeric(factor(x, levels = c("Early_Morning","Morning","Afternoon","Evening","Night","Late_Night"), exclude = NULL))
  x
}
#Tabulating encoded values
table(ff_en[["departure_time"]], encode_ordinal(ff_en[["departure_time"]]), useNA = "ifany")
table(ff_en[["arrival_time"]], encode_ordinal(ff_en[["arrival_time"]]), useNA = "ifany")
#Encoding Oridinal values
ff_en[["departure_time"]] <- encode_ordinal(ff_en[["departure_time"]])
ff_en[["arrival_time"]] <- encode_ordinal(ff_en[["arrival_time"]])
head(ff_en,4)

#Class basis - Economy & Business
unique(ff_en$class)
encode_ordinal_class <- function(x, order = unique(x)) {
  x <- as.numeric(factor(x, levels = c("Economy","Business"), exclude = NULL))
  x
}
table(ff_en[["class"]], encode_ordinal_class(ff_en[["class"]]), useNA = "ifany")
ff_en[["class"]] <- encode_ordinal_class(ff_en[["class"]])

#Stops basis - zero, one, two_or_more
unique(ff_en$stops)
encode_ordinal_stops <- function(x, order = unique(x)) {
  x <- as.numeric(factor(x, levels = c("zero","one","two_or_more"), exclude = NULL))
  x
}
ff_en[["stops"]] <- encode_ordinal_stops(ff_en[["stops"]])


#For remaining Labels we are Encoding using label encoder - airline, flight, source
for (col in names(ff_en)) {
  if (class(ff_en[[col]]) == "character") {
    ff_en[[col]] <- as.integer(factor(ff_en[[col]]))
  }
}

head(ff_en, n=5)


X =  subset(ff_en, select= -c(price))
Y =  subset(ff_en, select= c(price))

# Set the random seed for reproducibility
set.seed(42)

# Split the data into training and testing sets
train_indices <- createDataPartition(ff_en$price, p = 0.6, list = FALSE)
x_train <- X[train_indices, ]
x_test <- X[-train_indices, ]
y_train <- Y[train_indices]
y_test <- Y[-train_indices]

# Print the shape of the training and testing sets
cat("x_train shape:", dim(x_train), "\n")
cat("x_test shape:", dim(x_test), "\n")
cat("y_train shape:", length(y_train), "\n")
cat("y_test shape:", length(y_test), "\n")

##### Using Caret library applying Normalization #######
feature_range <- c(0, 1)
preprocess_obj <- preProcess(x_train, method = c("range"), range = feature_range)
x_train <- predict(preprocess_obj, newdata = x_train)
x_test <- predict(preprocess_obj, newdata = x_test)

# Convert the scaled data to data frames
x_train <- as.data.frame(x_train)
x_test <- as.data.frame(x_test)

############ Applying Model on Training Data ######

train_data <- cbind(y_train, x_train)
head(train_data, n=1)
  
# Create objects of regression models with default hyper parameters
#### Linear Regression Model #############
train_data_lm <- cbind(y_train, x_train)
modelmlg <- lm(formula = y_train ~ ., data = train_data_lm) #Linear Regression
summary(modelmlg)
x_test_lm <- x_test
y_test_lm <- y_test
## Now predicting new prices (y_pred) using above model on x_test dataset and comparing with y_test (actual)
y_pred_lm <- predict(modelmlg, newdata = x_test_lm)


rmse_lm <- sqrt(mean((y_test_lm - y_pred_lm)^2))
r_squared_lm <- cor(y_test_lm, y_pred_lm)^2
n_lm <- length(y_test_lm)
p_lm <- length(coef(modelmlg)) - 1
adj_r_squared_lm <- 1 - (1 - r_squared_lm) * ((n_lm - 1) / (n_lm - p_lm - 1))
mape_lm <- mean(abs((y_test_lm - y_pred_lm) / y_test_lm)) * 100

# Print the evaluation metrics
cat("RMSE:", rmse_lm, "\n") #6932.509 
cat("R-squared:", r_squared_lm , "\n")  #0.9068228 
cat("Adjusted R-squared:", adj_r_squared_lm, "\n") #0.9068125
cat("MAPE:", mape_lm, "%\n") #43.199 %

#########################################################
# 2nd Algorithm - XGboost -Ensembling Technique

train_matrix_XG <- as.matrix(train_data[, -1]) # Train independent variables
label_matrix_XG <- as.matrix(train_data$y_train) # Train target variables

# Define parameters for the XGBoost model
params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse"
)

modelXGR <- xgb.train(
  params = params,
  data = xgb.DMatrix(train_matrix_XG, label = label_matrix_XG),
  nrounds = 10
)
# Obtain feature importance
importance <- xgb.importance(model = modelXGR)
# Print feature importance
print(importance)

#Testing model on y_test dataset
x_test_df <- as.data.frame(x_test)
y_test_df <- y_test
y_pred_xg <- predict(modelXGR, newdata = as.matrix(x_test_df))

rmse_xg <- sqrt(mean((y_test_df- y_pred_xg)^2))
r_squared_xg <- cor(y_test_df, y_pred_xg)^2
n_xg <- length(y_test_df)
p_xg <- length(coef(modelXGR)) - 1
adj_r_squared <- 1 - (1 - r_squared_xg) * ((n_xg - 1) / (n_xg - p_xg - 1))
mape_xg <- mean(abs((y_test_df - y_pred_xg) / y_test_df)) * 100

# Print the evaluation metrics
cat("RMSE:", rmse_xg, "\n") # RMSE: 4368.85
cat("R-squared:", r_squared_xg, "\n") #0.9648945 
cat("Adjusted R-squared:", adj_r_squared, "\n") #0.9648906
cat("MAPE:", mape_xg, "%\n") #17.12296 %

######################## Additional Models ########################
# you can use x_train and y_train for training dataset
#Once model is generated it can be tested on x_test
# y_pred from above can be compared to calculate RMSE, R-Sq, etc.

####### #3 Model : Decision Tree ############################
x_test_dcr <- x_test
y_test_dcr <- y_test
train_data_dcr <- cbind(y_train, x_train)

modeldcr <- rpart(formula = y_train ~ ., data = train_data_dcr) # Decision Tree
summary(modeldcr)
y_pred_dcr <- predict(modeldcr, newdata=x_test_dcr)

rmse_dcr <- sqrt(mean((y_test_dcr - y_pred_dcr)^2))
r_squared_dcr <- cor(y_test_dcr, y_pred_dcr)^2
n_dcr <- length(y_test_dcr)
p_dcr <- length(coef(modeldcr)) - 1
adj_r_squared_dcr <- 1 - (1 - r_squared_dcr) * ((n_dcr - 1) / (n_dcr - p_dcr - 1))
mape_dcr <- mean(abs((y_test_dcr - y_pred_dcr) / y_test_dcr)) * 100

# Print the evaluation metrics
cat("RMSE:", rmse_dcr, "\n") # RMSE: 6556.101 
cat("R-squared:", r_squared_dcr, "\n") #0.9166663  
cat("Adjusted R-squared:", adj_r_squared_dcr, "\n") #0.9166571
cat("MAPE:", mape_dcr, "%\n") #41.81517 %

################## Model 4 :KNN with no of neighbours as 5 #################################
## There is no universal thumb rule to select value of 5 but sqrt of n is followed by many
#In our case we were getting better results with K =5 , also computation time was very high hence we used K =5

x_test_knn <- x_test
y_test_knn <- y_test
x_train_knn <- x_train
y_train_knn <- y_train
train_data_knn <- cbind(y_train_knn, x_train_knn)

knnmodel = knnreg(x_train_knn, y_train_knn,k=5)
y_pred_knn = predict(knnmodel,  newdata=x_test_knn)


rmse_knn <- sqrt(mean((y_test_knn - y_pred_knn)^2))
r_squared_knn <- cor(y_test_knn, y_pred_knn)^2
n_knn <- length(y_test_knn)
p_knn <- length(coef(knnmodel)) - 1
adj_r_squared_knn <- 1 - (1 - r_squared_knn) * ((n_knn - 1) / (n_knn - p_knn - 1))
mape_knn <- mean(abs((y_test_knn - y_pred_knn) / y_test_knn)) * 100

# Print the evaluation metrics
cat("RMSE:", rmse_knn, "\n") # RMSE:  
cat("R-squared:", r_squared_knn, "\n") #  
cat("Adjusted R-squared:", adj_r_squared_knn, "\n") #
cat("MAPE:", mape_knn, "%\n") # 

####################################

### We tried using support vector algorithm and decision tree but due to large dataset and computational
#limitations we were not able to execute the algorithms
####### Support Vector Regressor ########
### Reducing Size of Original Dataset and executing
#ff_svm <- ff_en[1:30000,]

#X_svm =  subset(ff_svm, select= -c(price))
#Y_svm =  subset(ff_svm, select= c(price))

# Set the random seed for reproducibility
#set.seed(42)

# Split the data into training and testing sets
#train_indices_svm <- createDataPartition(ff_svm$price, p = 0.7, list = FALSE)
#x_train <- X_svm[train_indices_svm, ]
#x_test <- X_svm[-train_indices_svm, ]
#y_train <- Y_svm[train_indices_svm]
#y_test <- Y_svm[-train_indices_svm]

##### Using Caret library applying Normalization #######
#feature_range <- c(0, 1)
#preprocess_obj <- preProcess(x_train, method = c("range"), range = feature_range)
#x_train <- predict(preprocess_obj, newdata = x_train)
#x_test <- predict(preprocess_obj, newdata = x_test)

# Convert the scaled data to data frames
#x_train <- as.data.frame(x_train)
#x_test <- as.data.frame(x_test)
#train_data <- cbind(y_train, x_train)
#head(train_data, n=1)

#svmfit = svm(y_train ~ ., data = train_data , kernel = "linear", cost = 10, scale = FALSE)

#y_pred_svm = predict(svmfit,  newdata=x_test)


#rmse_svm <- sqrt(mean((y_test - y_pred_svm)^2))
#r_squared_svm <- cor(y_test, y_pred_svm)^2
#n_svm <- length(y_test)
#p_svm <- length(coef(svmfit)) - 1
#adj_r_squared_svm <- 1 - (1 - r_squared_svm) * ((n_svm - 1) / (n_svm - p_svm - 1))
#mape_svm <- mean(abs((y_test - y_pred_svm) / y_test)) * 100

# Print the evaluation metrics
#cat("RMSE:", rmse_svm, "\n") # RMSE: 2831.485  
#cat("R-squared:", r_squared_svm, "\n") # R-squared: 0.4310435 
#cat("Adjusted R-squared:", adj_r_squared_svm, "\n") #Adjusted R-squared: 0.4308538
#cat("MAPE:", mape_svm, "%\n") # MAPE: 33.82037 %


####################################################


