setwd("/Users/hmishfaq/Documents/fall17/cs325b/final_paper_visualization")
library(ggmap)
library(ggplot2)

#train
df.train <- read.table("./data_split/bangladesh_2015_train.csv", sep = ",", header = TRUE)
df.train.has.na <- apply(df.train, 1, function(x){any(is.na(x))})
sum(df.train.has.na)
df.train <- df.train[!df.train.has.na,]

#val
df.val <- read.table("./data_split/bangladesh_2015_valid.csv", sep = ",", header = TRUE)
df.val.has.na <- apply(df.val, 1, function(x){any(is.na(x))})
sum(df.val.has.na)
df.val <- df.val[!df.val.has.na,]

sbbox <- make_bbox(lon = df.train$long1, lat = df.train$lat1, f = .2)
sbbox
sq_map <- get_map(location = "Bangladesh", zoom = 7,maptype = "terrain")

p <- ggmap(sq_map) + geom_point(data = df.train, mapping = aes(x = long1, y = lat1, color = "train"),size=0.8) + 
  geom_point(data = df.val, mapping = aes(x = long1, y = lat1,color="validation"),size=0.8)
p <- p + ggtitle("Training/Validation Scatterplot") + xlab("Longitude") + ylab("Latitude")
p <- p + scale_color_manual(values = c("train"='red', "validation"='blue'),breaks = c('train', 'validation'))
p <- p+theme(plot.title = element_text(hjust = 0.5))
p




