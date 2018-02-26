##############INDIA ##################
# This is the script for plotting pred expenditure (z-score) heatmap of India validation dataset
setwd("/Users/hmishfaq/Documents/fall17/cs325b/visualization")
library(ggmap)
library(ggplot2)
library(akima)

setwd("/Users/hmishfaq/Documents/fall17/cs325b/final_paper_visualization")
df <- read.table("./data/india_valid_preds.csv", sep = ",", header = TRUE)
head(df)
df.has.na <- apply(df, 1, function(x){any(is.na(x))})
sum(df.has.na)
df <- df[!df.has.na,]
df$true[which(df$true >3)] <-3 #changing to account for outlier
hist(df$true)
hist(df$pred)
df <- df[, !(colnames(df) %in% c("true"))]

dim(df)
head(df)

df <- df[is.finite(rowSums(df)),]

fldcons <- with(df, interp(x = longitude, y = latitude, z = pred,duplicate = "strip"))

library(akima)
# rgb.palette <- colorRampPalette(c("red", "orange", "blue"),space = "rgb")
filled.contour(x = fldcons$x,
               y = fldcons$y,
               z = fldcons$z,
               color.palette =colorRampPalette(c("red", "green")),asp=1, 
               xlab = "Longitude",
               ylab = "Latitude",
               main = "India Expenditure (pred)",
               key.title = title(main = "Z-score", cex.main = 1))

