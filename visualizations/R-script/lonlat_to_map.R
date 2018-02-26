setwd("/Users/hmishfaq/Documents/fall17/cs325b/visualization")
library(ggmap)
library(ggplot2)
sisquoc <- read.table("../data/Bangladesh_CE_2011.csv", sep = ",", header = TRUE)
# sisquoc <- read.table("../data/Bangladesh_CE_2015.csv", sep = ",", header = TRUE)

sisquoc.has.na <- apply(sisquoc, 1, function(x){any(is.na(x))})
sum(sisquoc.has.na)
sisquoc <- sisquoc[!sisquoc.has.na,]

sbbox <- make_bbox(lon = sisquoc$long, lat = sisquoc$lat, f = .2)
sbbox
sq_map <- get_map(location = sbbox, maptype = "satellite", source = "google")
#ggmap(sq_map) + geom_point(data = sisquoc, mapping = aes(x = long, y = lat), color = "red")
ggmap(sq_map) + geom_point(data = sisquoc, mapping = aes(x = long, y = lat),size=.1, color = "red")
# ggmap(sq_map) + geom_point(data = sisquoc, mapping = aes(x = long, y = lat), color = "red")



##############INDIA ##################
setwd("/Users/hmishfaq/Documents/fall17/cs325b/visualization")
library(ggmap)
library(ggplot2)
sisquoc <- read.table("../data/India_pov_pop.csv", sep = ",", header = TRUE)

sisquoc.has.na <- apply(sisquoc, 1, function(x){any(is.na(x))})
sum(sisquoc.has.na)
sisquoc <- sisquoc[!sisquoc.has.na,]

sbbox <- make_bbox(lon = sisquoc$longitude, lat = sisquoc$latitude, f = .2)
sbbox
sq_map <- get_map(location = sbbox, maptype = "satellite", source = "google")
#ggmap(sq_map) + geom_point(data = sisquoc, mapping = aes(x = long, y = lat), color = "red")
#for bd size = .1
ggmap(sq_map) + geom_point(data = sisquoc, mapping = aes(x = longitude, y = latitude),size=.0001, color = "red")
# ggmap(sq_map) + geom_point(data = sisquoc, mapping = aes(x = long, y = lat), color = "red")


