# scatter-plot of training-validation data for India
library(ggmap)
library(ggplot2)

#train
df.train <- read.table("./data_split/india_train.csv", sep = ",", header = TRUE)
df.train.has.na <- apply(df.train, 1, function(x){any(is.na(x))})
sum(df.train.has.na)
df.train <- df.train[!df.train.has.na,]

#val
df.val <- read.table("./data_split/india_valid.csv", sep = ",", header = TRUE)
df.val.has.na <- apply(df.val, 1, function(x){any(is.na(x))})
sum(df.val.has.na)
df.val <- df.val[!df.val.has.na,]

sbbox <- make_bbox(lon = df.train$longitude, lat = df.train$latitude, f = .2)
sbbox
# sq_map <- get_map(location = sbbox, maptype = "satellite", source = "google")
sq_map <- get_map(location = "India", zoom = 7,maptype = "terrain")

p <- ggmap(sq_map) + geom_point(data = df.train, mapping = aes(x = longitude, y = latitude, color = "train"),size=0.8) + 
  geom_point(data = df.val, mapping = aes(x = longitude, y = latitude,color="validation"),size=0.8)
p <- p + ggtitle("Training/Validation Scatterplot") + xlab("Longitude") + ylab("Latitude")
p <- p + scale_color_manual(values = c("train"='red', "validation"='blue'),breaks = c('train', 'validation'))
p <- p+theme(plot.title = element_text(hjust = 0.5))
p

