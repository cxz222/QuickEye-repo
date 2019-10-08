library(geomorph)
library(rgl)
library(stringr)

setwd('C:/Users/zcyhi/Documents/Insight Project')

rgl.bg(color = c('white'))

# Generate 8 images from different angles for each object
# For each image, use randomly chosen field of view (FOV) to introduce some randomness
save_png <- function(th, ph, num){
  rgl.viewpoint(theta = th, phi = ph, fov = sample(30:90, 1))
  obj <- read.ply(paste('models_cad/obj_', num, '.ply', sep=''))
  rgl.snapshot(
    paste('rotated images/obj', num, as.character(th), as.character(ph), '.png', sep='_'))
}

th_val <- c(-135, -45, 45, 135)
ph_val <- c(-45, 45)

num_val <- str_pad(1:30, 2, pad = "0")

for (th in th_val){
  for (ph in ph_val){
    for (num in num_val){
      save_png(th, ph, num)
    }
  }
}
