
#here total count is the sum of registered and casual users
# we have to predict casual and registered users and sum them and
#compare to count(find RMSE on total count)




#loading data

train=read.csv("day.csv")
summary(train)

#encoding factor varibles
train$season  = factor(train$season, labels = c("Spring", "Summer", "Fall", "Winter"))
train$weathersit = factor(train$weathersit, labels = c("Good", "Normal", "Bad"))
train$yr= factor(train$yr, labels = c(2011,2012))
train$mnth=factor(train$mnth,labels=c('jan','feb','march',
                                      'apr','may','jun','jul','aug','sep',
                                      'oct','nov','dec'))
train$holiday = factor(train$holiday, labels = c('Yes','no'))
train$weekday=factor(train$weekday,labels=c('sun','mom','tue','wed',
                                            'thu','fri','sat'))
train$workingday=factor(train$workingday,labels = c('No','Yes'))

table(is.na(train)) # There is no missing data in the data set
dat=train

#########################
#
#Exploratory data analysis
#
#########################

## Understanding the distribution of numerical variables and generating a frequency table for numeric variables
q<- par(mfrow = c(4,2))
p <- par(mar = rep(2,4))
hist(train$registered)
hist(train$casual)
hist(train$cnt)
hist(train$temp)
hist(train$atemp)
hist(train$hum)
hist(train$windspeed)
#plot a histogram for each numerical variables and analyze the distribution.


library(ggplot2)
ggplot(train, aes(x = season, fill = season)) + geom_bar()
ggplot(train, aes(x = weathersit, fill = weathersit)) + geom_bar()
ggplot(train, aes(x = mnth, fill = mnth)) + geom_bar()
ggplot(train, aes(x = yr, fill = yr)) + geom_bar()
ggplot(train, aes(x = holiday, fill = holiday)) + geom_bar()
ggplot(train, aes(x = workingday, fill = workingday)) + geom_bar()
ggplot(train, aes(x = weekday, fill = weekday)) + geom_bar()


#continuous varibles


ggplot(data=dat,aes(dat$hum))+geom_histogram(aes(fill=..count..),bins=70)+scale_fill_gradient("Count", low="Steelblue",high = "Red")+labs(title="Histogram for humidity") +labs(x="humidity", y="Count")


ggplot(data=dat,aes(dat$temp))+geom_histogram(aes(fill=..count..),bins=10)+scale_fill_gradient("Count", low="Steelblue",high = "Red")+labs(title="Histogram for temp") +labs(x="temp", y="Count")


ggplot(data=dat,aes(dat$atemp))+geom_histogram(aes(fill=..count..),bins=10)+scale_fill_gradient("Count", low="Steelblue",high = "Red")+labs(title="Histogram for atemp") +labs(x="atemp", y="Count")

ggplot(data=dat,aes(dat$registered))+geom_histogram(aes(fill=..count..),bins = 70)+scale_fill_gradient("Count", low="Steelblue",high = "Red")+labs(title="Histogram for registered users") +labs(x="registered users", y="Count")

ggplot(data=dat,aes(dat$casual))+geom_histogram(aes(fill=..count..),binwidth = 100)+scale_fill_gradient("Count", low="Steelblue",high = "Red")+labs(title="Histogram for casual users") +labs(x="casual users", y="Count")

ggplot(data=dat,aes(dat$windspeed))+geom_histogram(aes(fill=..count..),bins = 60)+scale_fill_gradient("Count", low="Steelblue",high = "Red")+labs(title="Histogram of windspeed") +labs(x="windspeed", y="Count")
ggplot(data=dat,aes(dat$cnt))+geom_histogram(aes(fill=..count..),bins = 60)+scale_fill_gradient("Count", low="Steelblue",high = "Red")+labs(title="Histogram of total count") +labs(x="total count(cas+reg)", y="Count")

#checking for outliers in the data
boxplot(train$registered)
boxplot(train$casual)#outliers detected
boxplot(train$cnt)
boxplot(train$temp)
boxplot(train$atemp)
boxplot(train$hum)
boxplot(train$windspeed)
library(ggplot2)

#season vs count



ggplot(dat, aes(x = season, y = cnt, colour = season)) +
  geom_point( aes(group = season)) +
  #geom_line( aes(group = season)) +
  scale_x_discrete("season") +
  scale_y_continuous("Count") +
  theme_minimal() +
  ggtitle("count vs season") + 
  theme(plot.title=element_text(size=18))



#year vs count
ggplot(dat, aes(x = yr, y = cnt, colour = yr)) +
  geom_point( aes(group = yr)) +
  #geom_line( aes(group = season)) +
  scale_x_discrete("year") +
  scale_y_continuous("Count") +
  theme_minimal() +
  ggtitle("People rent bikes more in 2012, and much less in 2011.\n") + 
  theme(plot.title=element_text(size=18))

#month vs count
ggplot(dat, aes(x = mnth, y = cnt, colour = mnth)) +
  geom_point( aes(group = mnth)) +
  #geom_line( aes(group = season)) +
  scale_x_discrete("month") +
  scale_y_continuous("Count") +
  theme_minimal() +
  ggtitle("total users across different months.\n") + 
  theme(plot.title=element_text(size=18))

#weather vs count
ggplot(dat, aes(x = weathersit, y = cnt, colour = weathersit)) +
  geom_point( aes(group = weathersit)) +
  #geom_line( aes(group = season)) +
  scale_x_discrete("weathers") +
  scale_y_continuous("Count") +
  theme_minimal() +
  ggtitle("total users across different weathers.\n") + 
  theme(plot.title=element_text(size=18))

#weekday vs count
ggplot(dat, aes(x = weekday, y = cnt, colour =weekday)) +
  geom_point( aes(group = weekday)) +
  #geom_line( aes(group = season)) +
  scale_x_discrete("days") +
  scale_y_continuous("Count") +
  theme_minimal() +
  ggtitle("total users across different weekdays.\n") + 
  theme(plot.title=element_text(size=18))

#working day vs count

ggplot(dat, aes(x = workingday, y = cnt, colour = workingday)) +
  geom_point( aes(group = workingday)) +
  #geom_line( aes(group = season)) +
  scale_x_discrete("working day") +
  scale_y_continuous("Count") +
  theme_minimal() +
  ggtitle("total users across workingdays.\n") + 
  theme(plot.title=element_text(size=18))

#holiday vs count
ggplot(dat, aes(x = holiday, y = cnt,colour=holiday)) +
  geom_point( aes(group = holiday)) +
  #geom_line( aes(group = season)) +
  scale_x_discrete("holiday") +
  scale_y_continuous("Count") +
  theme_minimal() +
  ggtitle("total users across holidays.\n") + 
  theme(plot.title=element_text(size=18))


##################################
#
#
#
# plotting registered users and casual users across 
#different feature
#
#
#
#################################

#season vs registered
sea_reg=ggplot(dat, aes(x = season, y = registered,colour=season)) +
  geom_point( aes(group = season)) +
  #geom_line( aes(group = season)) +
  scale_x_discrete("season") +
  scale_y_continuous("registered") +
  theme_minimal() +
  ggtitle("registered users across seasons") + 
  theme(plot.title=element_text(size=18))
#season vs casual users
sea_cas=ggplot(dat, aes(x = season, y = casual,colour=season)) +
  geom_point( aes(group = season)) +
  #geom_line( aes(group = season)) +
  scale_x_discrete("season") +
  scale_y_continuous("registered") +
  theme_minimal() +
  ggtitle(" casual users across seasons") + 
  theme(plot.title=element_text(size=18))

grid.arrange(sea_reg,sea_cas)




#yr vs registered users
year_reg=ggplot(dat, aes(x = yr, y = registered,colour=yr)) +
  geom_point( aes(group = yr)) +
  #geom_line( aes(group = season)) +
  scale_x_discrete("yr") +
  scale_y_continuous("registered") +
  theme_minimal() +
  ggtitle(" registered users across years") + 
  theme(plot.title=element_text(size=18))
#yr vs casual users
year_cas=ggplot(dat, aes(x = yr, y = casual,colour=yr)) +
  geom_point( aes(group = yr)) +
  #geom_line( aes(group = season)) +
  scale_x_discrete("yr") +
  scale_y_continuous("casual") +
  theme_minimal() +
  ggtitle("casual users across years") + 
  theme(plot.title=element_text(size=18))

grid.arrange(year_reg,year_cas)

#month vs registered users
month_reg=ggplot(dat, aes(x = mnth, y = registered,colour=mnth)) +
  geom_point( aes(group = mnth)) +
  #geom_line( aes(group = season)) +
  scale_x_discrete("month") +
  scale_y_continuous("registered") +
  theme_minimal() +
  ggtitle(" registered users across months") + 
  theme(plot.title=element_text(size=18))
#month vs casual users
month_cas=ggplot(dat, aes(x = mnth, y = casual,colour=mnth)) +
  geom_point( aes(group = mnth)) +
  #geom_line( aes(group = season)) +
  scale_x_discrete("month") +
  scale_y_continuous("casual") +
  theme_minimal() +
  ggtitle("casual users months") + 
  theme(plot.title=element_text(size=18))

grid.arrange(month_reg,month_cas)

#weekday vs registered users
weekday_reg=ggplot(dat, aes(x = weekday, y = registered,colour=weekday)) +
  geom_point( aes(group = weekday)) +
  #geom_line( aes(group = season)) +
  scale_x_discrete("weekday") +
  scale_y_continuous("registered") +
  theme_minimal() +
  ggtitle(" registered users across weekdays") + 
  theme(plot.title=element_text(size=18))
#weekday vs casual users
weekday_cas=ggplot(dat, aes(x = weekday, y = casual,colour=weekday)) +
  geom_point( aes(group = weekday)) +
  #geom_line( aes(group = season)) +
  scale_x_discrete("weekday") +
  scale_y_continuous("casual") +
  theme_minimal() +
  ggtitle("casual users across weekdays") + 
  theme(plot.title=element_text(size=18))

grid.arrange(weekday_reg,weekday_cas)


#working day vs registered users
workday_reg=ggplot(dat, aes(x = workingday, y = registered,colour=workingday)) +
  geom_point( aes(group = workingday)) +
  #geom_line( aes(group = season)) +
  scale_x_discrete("workingday") +
  scale_y_continuous("registered") +
  theme_minimal() +
  ggtitle(" registered users across working days") + 
  theme(plot.title=element_text(size=18))
#workingday vs casual users
workday_cas=ggplot(dat, aes(x = workingday, y = casual,colour=workingday)) +
  geom_point( aes(group = workingday)) +
  #geom_line( aes(group = season)) +
  scale_x_discrete("workinng day") +
  scale_y_continuous("casual") +
  theme_minimal() +
  ggtitle("casual users across working day") + 
  theme(plot.title=element_text(size=18))

grid.arrange(workday_reg,workday_cas)

#holidayday vs registered users
holiday_reg=ggplot(dat, aes(x = holiday, y = registered,colour=holiday)) +
  geom_point( aes(group = holiday)) +
  #geom_line( aes(group = season)) +
  scale_x_discrete("hoiday") +
  scale_y_continuous("registered") +
  theme_minimal() +
  ggtitle(" registered users across holidays") + 
  theme(plot.title=element_text(size=18))
#holiday vs casual users
holiday_cas=ggplot(dat, aes(x = holiday, y = casual,colour=holiday)) +
  geom_point( aes(group = holiday)) +
  #geom_line( aes(group = season)) +
  scale_x_discrete("holiday") +
  scale_y_continuous("casual") +
  theme_minimal() +
  ggtitle("casual users across holiday") + 
  theme(plot.title=element_text(size=18))

grid.arrange(holiday_reg,holiday_cas)

#weather  vs registered users
weather_reg=ggplot(dat, aes(x = weathersit, y = registered,colour=weathersit)) +
  geom_point( aes(group = weathersit)) +
  #geom_line( aes(group = season)) +
  scale_x_discrete("weather") +
  scale_y_continuous("registered") +
  theme_minimal() +
  ggtitle(" registered users across different weathers") + 
  theme(plot.title=element_text(size=18))
#weather vs casual users
weather_cas=ggplot(dat, aes(x = weathersit, y = casual,colour=weathersit)) +
  geom_point( aes(group = weathersit)) +
  #geom_line( aes(group = season)) +
  scale_x_discrete("weather") +
  scale_y_continuous("casual") +
  theme_minimal() +
  ggtitle("casual users across different weathers") + 
  theme(plot.title=element_text(size=18))

grid.arrange(weather_reg,weather_cas)


###########################
#
#
#Feature engineering
#
#
##########################

#correlation plot
library(corrplot)
num=c('windspeed','hum','temp','atemp','casual','registered','cnt')
corr=cor(dat[num])
corrplot(corr,method="number")


#"atemp" is variable is not taken into since "atemp" and "temp" has 
#got strong correlation with each other. 
#During model building any one of the variable has to be dropped since 
#they will exhibit multicollinearity in the data.
cols=c("atemp")
train[cols]=NULL
#as it has many outliers i am replacing values that lie
qn = quantile(train$casual, c( 0.95), na.rm = TRUE)
print(qn)
train$casual[train$casual>qn[1]]=qn[1]
# as casual variable is updated we should update total count varible
# is sum of casual and registered varible
train$cnt=train$casual+train$registered





################
#
# creating new feature
#Creating bins for casual count variable based on its relation with
# month column
#
#
###############

library(rpart)
install.packages("rpart.plot")
library(rpart.plot)
d=rpart(casual ~ mnth, data = dat)
#plotting tree
rpart.plot(d)
#creating new variable according to the graph
train$newcas=0
a=train$mnth=='jan' | train$mnth=='feb' | 
  train$mnth=='march' |
  train$mnth=='nov' | train$mnth =='dec'
for (i in(1:731)){
  
if(a[i]==TRUE){
  train$newcas[train$mnth=='jan' | train$mnth=='feb' | 
                 train$mnth=='march' |
                 train$mnth=='nov' | train$mnth =='dec']=1
}else{
  train$newcas=2
}
}

b=train$mnth=='jan' | train$mnth=='feb' | 
  train$mnth=='dec'
for (i in(1:731)){
  if(a[i]==TRUE ){
    
  if( b[i]==TRUE){
  
    train$newcas[i]=3
  }else{
    train$newcas[i]=4
  }
  }
  else{}
}

c=train$mnth=='apr' | train$mnth=='oct'
for (i in(1:731)){
  if(a[i]==FALSE ){
    
    if( c[i]==TRUE){
      
      train$newcas[i]=5
    }else{
      train$newcas[i]=6
    }
  }
  else{}
}

d1= rpart(registered ~ mnth, data = dat)
#plotting the tree
rpart.plot(d1)
#creating new varible according to decision tree
train$reg_mnth=0
d=train$mnth=='jan' | train$mnth=='feb' | 
  train$mnth=='march'| train$mnth =='dec'
for (i in(1:731)){
  
  if(d[i]==TRUE){
    train$reg_mnth[train$mnth=='jan' | train$mnth=='feb' | 
                   train$mnth=='march' |
                   train$mnth=='nov' | train$mnth =='dec']=1
  }else{
    train$reg_mnth=2
  }
}

e=train$mnth=='jan' | train$mnth=='feb' 
  
for (i in(1:731)){
  if(d[i]==TRUE ){
    
    if( e[i]==TRUE){
      
      train$reg_mnth[i]=3
    }else{
      train$reg_mnth[i]=4
    }
  }
  else{}
}

f=train$mnth=='apr' | train$mnth=='nov'
for (i in(1:731)){
  if(d[i]==FALSE ){
    
    if( f[i]==TRUE){
      
      train$reg_mnth[i]=5
    }else{
      train$reg_mnth[i]=6
    }
  }
  else{}
}

library(dummies)
dummy_weather=data.frame(dummy(train$weathersit))
dummy_season=data.frame(dummy(train$season))
dummy_weekday=data.frame(dummy(train$weekday))
dummy_holiday=data.frame(dummy(train$holiday))
dummy_month=data.frame(dummy(train$mnth))
dummy_yr=data.frame(dummy(train$yr))
dummy_workingday=data.frame(dummy(train$workingday))
#removing the factor column
train = subset(train, select = -c(weathersit,season,
                                  weekday,holiday,mnth,yr,
                                  workingday))
#concatenation dummy variable

train=cbind(train,dummy_holiday,dummy_weather,dummy_month,
            dummy_season,dummy_weekday,dummy_yr,dummy_workingday)

#converting newly created varibles to category  type
train$newcas=as.factor(train$newcas)
train$reg_mnth=as.factor(train$reg_mnth)


#removing instant and dteday columns
train$instant=NULL
train$dteday=NULL

#############################
#
#
# Modelling
#
#
##############################

library(DAAG)
#feature selection using boruta package
library(Boruta)


train=train[sample(nrow(train)),]
library(caret)
#feature selection using boruta
finail.boruta_cas=Boruta(casual~., data = train[,c(1:4,7:40)], doTrace = 2)
selected_features_cas=getSelectedAttributes(finail.boruta_cas, withTentative = F)
formula_cas=as.formula(paste("casual~",paste(selected_features_cas,collapse = "+")))
#feature selection using boruta

finail.boruta_reg=Boruta(registered~., data = train[,c(1:3,5,7:40)], doTrace = 2)
selected_features_reg=getSelectedAttributes(finail.boruta_reg, withTentative = F)
formula_reg=as.formula(paste("registered~",paste(selected_features_reg,collapse = "+")))

#creating model  using selected features
myfunction_model=function(model){
  print(model)
  #summary of the model
  
  
  train_control=trainControl(method = "repeatedcv", 
                             number = 10, 
                             repeats = 6)
  model_casual= train(casual~.,data=train[1:500,-(5:6)],
                          metric="RMSE", method=model,trControl=train_control)
  print(' model on casual count')
  print(model_casual)
  prediction_cas = predict(model_casual, train[501:731,])
  print('test RMSE of casual count prediction')
  
  print(RMSE(prediction_cas,train[501:731,4]))
  print('model on registered model')
  
  
  model_registered= train(registered~.,data=train[1:500,c(-4,-6)],
                              metric="RMSE", method=model,trControl=train_control)
  print('registered count')
  print(model_registered)
  prediction.registered = predict(model_registered, train[501:731,])
  print('test RMSE of registered prediction')
  print(RMSE(prediction.registered,train[501:731,5]))
  total_count=prediction_cas+prediction.registered
  print('Test RMSE on Total count(casual +registered)')
  print(RMSE(total_count,train[551:731,6]))
}

#ridge regression
myfunction_model('ridge')

#linear regression

myfunction_model('lm')

#elastic net regression
myfunction_model('lasso')

#generalized linear model
myfunction_model('glm')


#elastic regression

myfunction_model('enet')

#decision tree
myfunction_model('rpart')

#random forest
myfunction_model('rf')

############################
# from the above models even after removing statistically unsignificant variables
#the RMSE is high and R-squared value is  low
############################



####################
#
#
#Principal component analysis on  registered users
#
#
####################



#divide the new data



#removing casual,total count,new_cas,new_reg
pca.train = train[1:550,c(-4,-6,-7,-8)]
pca.test =train[551:731,c(-4,-6,-7,-8)]
#principal component analysis
prin_comp <- prcomp(pca.train)
#outputs the mean of variables
prin_comp$center

#outputs the standard deviation of variables
prin_comp$scale
dim(prin_comp$x)
biplot(prin_comp, scale = 0)

#compute standard deviation of each principal component
std_dev = prin_comp$sdev

#compute variance
pr_var = std_dev^2
#proportion of variance explained
prop_varex =pr_var/sum(pr_var)
#scree plot
plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

#cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")
#add a training set with principal components
train.data = data.frame(registered = pca.train$registered, prin_comp$x)

#we are interested in first 40 PCAs as we have seen from the graph
# and the target variable ,so in total 41(including target variable)
train.data =train.data[,1:26]

#transform test into PCA
test.data=predict(prin_comp, newdata = pca.test)
test.data= as.data.frame(test.data)

#select the first 40 components
test.data=test.data[,1:25]
#linear regression



####################
#
#
#Principal component analysis on casual users
#
#
####################


#removing registered,total count,new_cas,new_reg

pca.train.cas = train[1:550,c(-5,-6,-7,-8)]
pca.test.cas =train[551:731,c(-5,-6,-7,-8)]
#principal component analysis
prin_comp.cas <- prcomp(pca.train.cas)
#outputs the mean of variables
prin_comp.cas$center

#outputs the standard deviation of variables
prin_comp.cas$scale
dim(prin_comp.cas$x)
biplot(prin_comp, scale = 0)

#compute standard deviation of each principal component
std_dev.cas = prin_comp.cas$sdev

#compute variance
pr_var.cas = std_dev.cas^2
#proportion of variance explained
prop_varex.cas =pr_var.cas/sum(pr_var.cas)
#scree plot
plot(prop_varex.cas, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

#cumulative scree plot
plot(cumsum(prop_varex.cas), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")
#add a training set with principal components
train.data.cas = data.frame(casual = pca.train.cas$casual, prin_comp.cas$x)

#we are interested in first 40 PCAs as we have seen from the graph
# and the target variable ,so in total 41(including target variable)
train.data.cas =train.data.cas[,1:26]

#transform test into PCA
test.data.cas=predict(prin_comp.cas, newdata = pca.test.cas)
test.data.cas= as.data.frame(test.data.cas)

#select the first 40 components
test.data.cas=test.data.cas[,1:25]
#linear regression

###########################
#
#
# function that will predict casual and registered users and 
#sum them and compare to total count(finding RMSE)
#
#
#############################

#creating model with PCA components

myfunction_pca=function(model){
  print('Princiapl component analysis')
  print(model)
  #summary of the model
  
  train_control=trainControl(method = "repeatedcv", 
                             number = 10, 
                             repeats = 6)
  pca_model_casual= train(casual ~.,data=train.data.cas,
                   metric="RMSE", method=model,trControl=train_control)
  print(' model on casual count')
  print(pca_model_casual)
  pca.prediction_cas = predict(pca_model_casual, test.data.cas)
  print('test RMSE of casual count prediction')
  
  print(RMSE(pca.prediction_cas,train[551:731,4]))
  print('model on registered model')
  pca_model_registered= train(registered ~.,data=train.data,
                   metric="RMSE", method=model,trControl=train_control)
  print('registered count')
  print(pca_model_registered)
  pca.prediction.registered = predict(pca_model_registered, test.data)
  print('test RMSE of registered prediction')
  print(RMSE(pca.prediction.registered,train[551:731,5]))
  total_count=pca.prediction_cas+pca.prediction.registered
  print('Test RMSE on Total count(casual +registered)')
  RMSE(total_count,train[551:731,6])
}



#############################
#
#
# Modelling
#
#
##############################

#ridge regression
myfunction_pca('ridge')

#linear regression

myfunction_pca('lm')

#generalized linear model

myfunction_pca('glm')

#elastic regression
myfunction_pca('enet')

#elastic net regression
myfunction_pca('lasso')


#decision tree
myfunction_pca('rpart')

#random forest
myfunction_pca('rf')


