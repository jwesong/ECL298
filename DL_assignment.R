## Deep Learning Assignment 
## 1. Work through 10.9 lab (10.9.1- 10.9.4)
## 2. Question 7,8 for practice question
# See all plots as noted in git

# install.packages(c("keras3"))
## Not working, thus
# keras3::install_keras()
# install.packages("glmnet")
library(keras3)
library(ISLR2)
library(glmnet)
library(terra)

# ====================================================
## 10.10.7 
# ====================================================

'Fit a neural network to the Default data. Use a single hidden layer
with 10 units, and dropout regularization. Have a look at Labs 10.9.1– 10.9.2 
for guidance. Compare the classification performance of your model with that of 
linear logistic regression.'

Default <- Default
# first convert yes/no to 1/0
Default$default <- ifelse(Default$default == "Yes", 1, 0)

x <- model.matrix(default ~ . - 1, data = Default)
y <- Default$default

set.seed(13)
n <- nrow(x)
ntest <- trunc(n/3)
testid <- sample(1:n, ntest)
x <- scale(x)

glm_fit <- glm(default ~ ., data = Default[-testid, ],
               family = binomial)

glm_prob <- predict(glm_fit,
                    Default[testid, ],
                    type = "response")

glm_pred <- ifelse(glm_prob > 0.5, 1, 0)

mean(glm_pred != y[testid]) ## [1] 0.02910291

modnn <- keras_model_sequential(input_shape = ncol(x)) |>
   layer_dense(units = 10, activation = "relu") |>
   layer_dropout(rate = 0.4) |>
   layer_dense(units = 1, activation = "sigmoid")  
#the output layer is a single sigmoid for the binary classification task. see ISLR p455
compile(modnn,
        loss = "binary_crossentropy",
        optimizer = optimizer_rmsprop(),
        metrics = "accuracy")
history <- fit(modnn,
               x[-testid, ], y[-testid],
               epochs = 50,
               batch_size = 32,
               validation_data =
                  list(x[testid, ], y[testid]))
plot(history)

## DL_HW_plot7 saved, see git

evaluate(modnn, x[testid, ], y[testid])
'$accuracy
[1] 0.9636964

$loss
[1] 0.0883306
'

# since binomial 1/0 
n_prob <- predict(modnn, x[testid, ])
n_pred <- ifelse(n_prob > 0.5, 1, 0)
mean(n_pred != y[testid])  ## [1] 0.03630363


# ====================================================
## 10.10.8
# ====================================================
## all 10 photos in upload in photos folder within the git project




# ====================================================
## 10.9.1 A Single Layer Network on the Hitters Data
# ====================================================

Gitters <- na.omit(Hitters)
n <- nrow(Gitters)
set.seed(13)
ntest <- trunc(n / 3)
testid <- sample(1:n, ntest)

# linear
lfit <- lm(Salary ~ ., data = Gitters[-testid, ])
lpred <- predict(lfit, Gitters[testid, ])
with(Gitters[testid, ], mean(abs(lpred - Salary)))  ## [1] 254.6687

# lasso
x <- scale(model.matrix(Salary ~ . - 1, data = Gitters))
y <- Gitters$Salary

library(glmnet)
cvfit <- cv.glmnet(x[-testid, ], y[-testid], type.measure = "mae")
cpred <- predict(cvfit, x[testid, ], s = "lambda.min")
mean(abs(y[testid] - cpred)) ## [1] 252.2994

library(keras3)  # new version of keras, from RH
modnn <- keras_model_sequential(input_shape = ncol(x)) |>
   layer_dense(units = 50, activation = "relu") |>
   layer_dropout(rate = 0.4) |> 
   layer_dense(units = 1)

compile(modnn, loss = "mse", optimizer = optimizer_rmsprop(),
        metrics = list("mean_absolute_error"))

history <- fit(modnn, x[-testid, ], y[-testid], epochs = 1500, 
               batch_size=32, validation_data=list(x[testid, ], y[testid]))

evaluate(modnn, x[testid, ], y[testid]) ## $loss [1] 113770.6 , $mean_absolute_error [1] 247.0294

# From textbook
# plot DL_HW_plot1 saved
plot(history)  
npred <- predict(modnn, x[testid, ])
mean(abs(y[testid] - npred))


# ====================================================
## 10.9.2  A Multilayer Network on the MNIST Digit Data
# ====================================================

mnist <- dataset_mnist()

# illustrate with terra
library(terra)
show_digit <- \(i) {
   plot(rast(mnist$train$x[i,,]), legend=FALSE,
        axes=FALSE, col=gray(255:0/255), mar=c(0,0,0,0))
   text(2, 26, mnist$train$y[i], cex=2)
}
par(mfrow=c(5,5), mar=c(0,0,0,0))
for (i in 1:25) show_digit(i)

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
g_test <- mnist$test$y
dim(x_train)
dim(x_test)

###
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(g_test, 10)

x_train <- x_train / 255
x_test <- x_test / 255

###
modelnn <- keras_model_sequential(input_shape = 784) |>
   layer_dense(units = 256, activation = "relu") |>
   layer_dropout(rate = 0.4) |>
   layer_dense(units = 128, activation = "relu") |>
   layer_dropout(rate = 0.3) |>
   layer_dense(units = 10, activation = "softmax")

summary(modelnn)

compile(modelnn, loss = "categorical_crossentropy",
        optimizer = optimizer_rmsprop(), metrics = "accuracy")

system.time(
   history <- fit(modelnn, x_train, y_train, epochs = 30, 
                  batch_size = 128, validation_split = 0.2)
)
# plot DL_HW_plot2 saved

plot(history, smooth = FALSE)

evaluate(modelnn, x_test, y_test) 

# in the book:
accuracy <- function(pred, truth) {
  mean(drop(as.numeric(pred)) == drop(truth))
}  

predict(modelnn, x_test) |> 
  op_argmax(-1, zero_indexed = T) |> accuracy(g_test)

###
modellr <- keras_model_sequential(input_shape = 784)  |>
   layer_dense(units = 10, activation = "softmax")

summary(modellr)

compile(modellr, loss = "categorical_crossentropy",
        optimizer = optimizer_rmsprop(), metrics = c("accuracy"))

fit(modellr, x_train, y_train, epochs = 30,
    batch_size = 128, validation_split = 0.2)

evaluate(modellr, x_test, y_test) 
'$accuracy
[1] 0.9273

$loss
[1] 0.2674221'

# ====================================================
## 10.9.3 Convolutional Neural Networks
# ====================================================

cifar100 <- dataset_cifar100()
names(cifar100)
x_train <- cifar100$train$x

# first images
par(mar = c(0, 0, 0, 0), mfrow = c(5, 5))
for (i in 1:25) {
   terra::plotRGB(terra::rast(x_train[i,,, ]))
   terra::halo(3,30, cifar100$train$y[i], cex=1.5)
}

# elephants
index <- which(cifar100$train$y == 31)[1:25]
for (i in index) {
   terra::plotRGB(terra::rast(x_train[i,,, ]))
}

y_train <- cifar100$train$y
x_test <- cifar100$test$x
y_test <- cifar100$test$y
dim(x_train)
range(x_train[1,,, 1])
x_train <- x_train / 255
x_test <- x_test / 255
y_train <- to_categorical(y_train, 100)
y_test <- to_categorical(y_test, 100)
dim(y_train)

model <- keras_model_sequential(input_shape = c(32, 32, 3)) |>
   layer_conv_2d(filters = 32, kernel_size = c(3, 3),
                 padding = "same", activation = "relu") |>
   layer_max_pooling_2d(pool_size = c(2, 2)) |>
   layer_conv_2d(filters = 64, kernel_size = c(3, 3),
                 padding = "same", activation = "relu") |>
   layer_max_pooling_2d(pool_size = c(2, 2)) |>
   layer_conv_2d(filters = 128, kernel_size = c(3, 3),
                 padding = "same", activation = "relu") |>
   layer_max_pooling_2d(pool_size = c(2, 2)) |>
   layer_conv_2d(filters = 256, kernel_size = c(3, 3),
                 padding = "same", activation = "relu") |>
   layer_max_pooling_2d(pool_size = c(2, 2)) |>
   layer_flatten() |>
   layer_dropout(rate = 0.5) |>
   layer_dense(units = 512, activation = "relu") |>
   layer_dense(units = 100, activation = "softmax")

summary(model)

compile(model, loss = "categorical_crossentropy",
        optimizer = optimizer_rmsprop(), 
        metrics = "accuracy")

history <- fit(model, x_train, y_train, epochs = 30,
               batch_size = 128, validation_split = 0.2)
# DL_HW_plot3 saved

evaluate(model, x_test, y_test)
'$accuracy
[1] 0.4422

$loss
[1] 2.711993'


# ====================================================
## 10.9.4 Using Pretrained CNN Models
# ====================================================

burl <- "https://www.statlearning.com/s/book_images.zip"
if (!file.exists("book_images")) {
   download.file(burl, basename(burl), mode="wb")
   unzip("book_images.zip")
}
image_files <- list.files("book_images", pattern=".jpg$", full.names = TRUE)
num_images <- length(image_files)

x <- array(dim = c(num_images, 224, 224, 3))
for (i in 1:num_images) {
   img <- image_load(image_files[i], target_size = c(224, 224))
   x[i,,, ] <- image_to_array(img)
}

x <- imagenet_preprocess_input(x)
model <- application_resnet50(weights = "imagenet")
summary(model)

pred6 <- predict(model, x) |>
   imagenet_decode_predictions(top = 3)
names(pred6) <- basename(image_files)
print(pred6)

