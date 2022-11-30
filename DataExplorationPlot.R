## Load libraries
library(ggplot2)

## Manually enter count of images
AsianFemale = 1385
AsianMale = 1056
BlackFemale = 2052
BlackMale = 2196
IndianFemale = 1389
IndianMale = 1997
OtherFemale = 1389
OtherMale = 1997
WhiteFemale = 3666
WhiteMale = 4558

## Create the data frame based on counts of images
race = c("Asian", "Asian", "Black", "Black", "Indian", "Indian", "White", "White", "Other", "Other")
gender = c("Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male")
count = c(AsianFemale, AsianMale, BlackFemale, BlackMale, IndianFemale, IndianMale, WhiteFemale, WhiteMale, OtherFemale, OtherMale)
dataFrame1 = data.frame(race, gender, count)

## Plot the data
ggplot(data=table1, aes(x=race, y=count, fill=gender)) + geom_bar(stat="identity") + ggtitle("Number of Images by Race & Gender") + xlab("Race") + ylab("Count") + theme(axis.text=element_text(size=18), axis.title=element_text(size=20,face="bold"), plot.title = element_text(hjust = 0.5, size=20,face="bold"))

