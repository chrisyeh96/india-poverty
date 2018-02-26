library(akima)
setwd("/Users/hmishfaq/Documents/fall17/cs325b/final_paper_visualization/")

df <- read.table("./data/bangladesh_with_nightlights.csv", sep = ",", header = TRUE)
df.has.na <- apply(df, 1, function(x){any(is.na(x))})
sum(df.has.na)
df <- df[!df.has.na,]
df <- df[, !(colnames(df) %in% c("Unnamed..0","totexp_m","X","Village","a01","aeu","Upazila","hh_type","div","District","Union"))]
df <- df[!(df$long < 85),]
df <- df[!(df$lat< 19),]
df <- df[!(df$lat> 29),]
df$totexp_m_pc[which(df$totexp_m_pc >2)] <-2 #changing to account for outlier
df$totexp_m_pc[which(df$totexp_m_pc < -1)] <- -1 #changing to account for outlier

hist(df$totexp_m_pc)


hist(df$long)
hist(df$lat)
head(df)
str(df)
colnames(df)
fld <- with(df, interp(x = long, y = lat, z = totexp_m_pc,duplicate = "mean"))

filled.contour(x = fld$x,
               y = fld$y,
               z = fld$z,
               color.palette =
                 colorRampPalette(c("green", "red")),
               xlab = "Longitude",
               ylab = "Latitude",
               main = "Bangladesh expenditure",
               key.title = title(main = "totexp_m_pc (mm)", cex.main = 1))


library(ggplot2)
library(reshape2)

# prepare data in long format
df <- melt(fld$z, na.rm = TRUE)
names(df) <- c("x", "y", "totexp_m_pc")
df$long <- fld$x[df$x]
df$lat <- fld$y[df$y]

ggplot(data = df, aes(x = long, y = lat, z = totexp_m_pc)) +
  geom_tile(aes(fill = totexp_m_pc)) +
  stat_contour() +
  ggtitle("Bangladesh expenditure") +
  xlab("Longitude") +
  ylab("Latitude") +
  scale_fill_continuous(name = "totexp_m_pc (mm)",
                        low = "green", high = "red") +
  theme(plot.title = element_text(size = 25, face = "bold"),
        legend.title = element_text(size = 15),
        axis.text = element_text(size = 15),
        axis.title.x = element_text(size = 20, vjust = -0.5),
        axis.title.y = element_text(size = 20, vjust = 0.2),
        legend.text = element_text(size = 10))


# grab a map. get_map creates a raster object
library(ggmap)

sbbox <- make_bbox(lon = df$lon, lat = df$lat, f = .2)
sbbox
map <- get_map(location = sbbox,maptype = "terrain",source = "google",zoom=7)


g1 <- ggmap(map)

g1 +
  stat_contour(data = df, aes(x = long, y = lat, z = totexp_m_pc)) +
  geom_tile(data = df, aes(x = long, y = lat, z = totexp_m_pc, fill = totexp_m_pc), alpha = 0.7) +
  ggtitle("Bangladesh Expenditure (Z-score)") +
  xlab("Longitude") +
  ylab("Latitude") +
  scale_fill_continuous(name = "Z-score",
                        low = "gold", high = "blue",limits=c(-1, 2), breaks=seq(-1,2,by=0.5)) +
  theme(plot.title = element_text(face = "bold",hjust = 0.5),
        legend.title = element_text(size = 15),
        axis.text = element_text(size = 15),
        axis.title.x = element_text(vjust = -0.5),
        axis.title.y = element_text(vjust = 0.2)) +
  coord_map()
 

