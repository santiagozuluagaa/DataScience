# Source: https://www.youtube.com/watch?v=IN2XmBhILt4
# Build to illustrate the example given by StatQuest with Josh Starmer in
# Neural Networks Pt. 2: Backpropagation Main Ideas

# Estimating ONLY one parameter 
# b3 (one of the bias) is the parameter to estimate

## Libraries ----
library(ggplot2)
library(ggpubr)
library(stringr)
library(magick)
options(scipen = 999)

## Parameters estimated ----
# They were already estimated
w1 <- 3.34
w2 <- -3.53
w3 <- -1.22
w4 <- -2.30
b1 <- -1.43
b2 <- 0.57


## Activation function ----
# Softplus
softplus <- function(x){
  log(1+exp(x))
}

## Neural Network Output ----
# Dosage is the Input
# b3 is the parameter to estimate
NN <- function(Dosage, b3){
  
  # f1
  a1 <- Dosage * w1 + b1
  f1 <- w3 * softplus(a1)
  
  # f2
  a2 <- Dosage * w2 + b2
  f2 <- w4 * softplus(a2)
  
  # Output
  f1 + f2 + b3
}

## Backpropagation ----

### d Loss Functions / d b3 ----
# Derivate of the loss function with respect to b3
dSSR_db3 <- function(Observed, Predicted){
  sum(-2 * (Observed - Predicted))
}


### stepSize function ----
stepSize <- function(dSSR_db3, learning_rate){
  dSSR_db3 * learning_rate
}


### update b3 ----
update_b3 <- function(b3){
  b3 - stepSize
}

### SSR function
SSR <- function(Observed, Predicted){
  sum((Observed - Predicted)^2)
}

### Put together ----
# Initial values 
Dosage <- c(0, 0.5, 1)
Observed <- c(0, 1, 0)

### Testing Functions ----
Predicted <- NN(Dosage = Dosage, b3 = 0)
dSSR_db3(Observed, Predicted)


backPropagation <- function(Dosage, 
                            Observed, 
                            b3 = 0, 
                            learning_rate = 0.1, 
                            max_iter = 10000,
                            stepSize_tol = 0.01){
  
  # Initial value for b3
  b3 <- 0 
  # Initial value for the iterarion
  iter <- 1
  # Initial value for start the loop (just a big number)
  stepSize <- 999 
  
  # Dataframe where it'll saved the info for each iteration
  # datos BP (BackPropagation)
  datosBP <- data.frame(
                   iter = rep(0, 10000),
                   b3 = rep(0, 10000),
                   stepSize = rep(0, 10000),
                   SSR = rep(0, 10000),
                   dSSR_db3 = rep(0, 10000)
  )
  
  while(abs(stepSize) > stepSize_tol & iter < max_iter){
    
    # Save Iter
    datosBP$iter[iter] <- iter
    
    # Save b3
    datosBP$b3[iter] <- b3   
    
    # Sabe stepSize
    datosBP$stepSize[iter] <- stepSize  
    
    # Prediction from the NN
    Predicted <- NN(Dosage, b3)
    
    # SSR
    SSR <- SSR(Observed, Predicted)
    # Save SSR
    datosBP$SSR[iter] <- SSR 
    
    # dSSR_db3
    dSSR_db3 <- dSSR_db3(Observed, Predicted)
    # Save dSSR_db3
    datosBP$dSSR_db3[iter] <- dSSR_db3 
    
    # Step Size
    stepSize <- stepSize(dSSR_db3, learning_rate)
    
    # New b3
    b3 <- b3 - stepSize(dSSR_db3, learning_rate)
  
    # Iteration
    iter <- iter + 1
    

  }
    # Save final values
  
    # iter sabed
    datosBP$iter[iter] <- iter
    
    # b3 saved
    datosBP$b3[iter] <- b3   
    
    # stepSize saved
    datosBP$stepSize[iter] <- stepSize
    
    # Save SSR
    datosBP$SSR[iter] <- SSR 
    
    # Save dSSR_db3
    datosBP$dSSR_db3[iter] <- dSSR_db3 
    
  return(datosBP[1:iter,])
}

### Call backPropagation
datosBP <- backPropagation(Dosage = Dosage, Observed = Observed, stepSize_tol = 0.01)

## Animation ----
# Dosage and values Observed
Dosage <- c(0, 0.5, 1)
Observed <- c(0, 1, 0)
datos <- data.frame(Dosage = Dosage,
                    Observed = Observed)

### Function to plot

plot_datosBP <- function(x, 
                        y, 
                        xlim0, 
                        xlim1, 
                        ylim0,
                        ylim1,
                        iter){
  
  p <- ggplot(data = datosBP[1:iter,]) +
    geom_point(aes(x = {{x}}, y = {{y}}), color = "red", size = 3) +
    geom_line(aes(x = {{x}}, y = {{y}}), color = "blue", size = 1) +
    xlim(c(xlim0, xlim1)) +
    ylim(c(ylim0, ylim1))
  
  return(p)
}

### Plot SSR ----
for (iter in datosBP$iter){
  
  iteracion <- iter
  ### Plot data ----
  # Dataframe
  Dosage_Points <- seq(0, 1, 0.02)
  Predicted_Points <- NN(Dosage = Dosage_Points, b3 = datosBP$b3[iter])
  datos_curva <- data.frame(Dosage_Points = Dosage_Points,
                            Predicted_Points = Predicted_Points)
  
  p <- ggplot() +
    geom_point(data = datos, 
               mapping = aes(x = Dosage, y = Observed), color = "red", size = 3) +
    geom_line(data = datos_curva, 
              mapping = aes(x = Dosage_Points, y = Predicted_Points)) +
    xlim(c(0, 1)) +
    ylim(c(-3, 2)) +
    annotate("text", x = 0.5, y = 1.5, size = 5, 
             label = "To be fitted")
  
  
  p2 <- plot_datosBP(b3, SSR, 0, 3, 0, 25, iter = iter)
  p3 <- plot_datosBP(b3, dSSR_db3, 0, 3, -18, 2, iter = iter)
  p4 <- plot_datosBP(iter, SSR, 0, 8, 0, 25, iter = iter)
  p5 <- plot_datosBP(iter, dSSR_db3, 0, 8, -18, 2, iter = iter)
  
  
  text <- ggplot() + 
    annotate("text", x = 1, y = 8, size = 5, 
             label = "Optimizing a parameter (b3) \n in a Neural Network") +
    annotate("text", x = 1, y = 6, size = 5, 
             label = paste0("Iteration: ", round(datosBP$iter[iter],0))) +
    annotate("text", x = 1, y = 5, size = 5, 
             label = paste0("b3: ", round(datosBP$b3[iter],4))) + 
    annotate("text", x = 1, y = 4, size = 5, 
             label = paste0("Step size: ", round(datosBP$stepSize[iter],4))) +
    annotate("text", x = 1, y = 1.5, size = 3, 
             label = "Created By: Santiago Zuluaga | 2022 \n Example from Josh Starmer (StatQuest)") +
    ylim(c(1, 9)) +
    theme_void() +
    theme(panel.background = element_rect(fill = "white"))
  
  final_p <- ggarrange(p, p2, p3, text, p4, p5,
                       ncol = 3, nrow = 2)
    
  
  fp <- paste0("plot-animation", datosBP$iter[iter],".png")
  
  ggsave(plot = final_p, 
         path = "images-animation/",
         filename = fp, 
         device = "png",
         width = 7.5,
         height = 5)
}


# GIF CREATION
# For more info about gif creation: https://www.nagraj.net/notes/gifs-in-r/
# Use a RProject o make the changes
dir_imags <- "images-animation/"
imgs <- paste0(dir_imags, dir(dir_imags)[str_detect(dir(dir_imags), "png")])
plot_list <- lapply(imgs, image_read)

## join the images together
plot_joined <- image_join(plot_list)

## animate at 2 frames per second
plot_animated <- image_animate(plot_joined, fps = 2)

## view animated image
plot_animated

## save to disk
image_write(image = plot_animated,
            path = "BackPropagation.gif")


