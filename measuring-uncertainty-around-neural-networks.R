### The argument "proba specifies the dropout probability.
### There is no dropout at the input layer.
T.replicas <- 30
proba <- 0.005
pred.data1 <- matrix(0, ncol = T.replicas, nrow = nrow(data.test))

### In order to impose dropout also at test phase, the layer dropout
### is specified outside the sequential neural network (this allows
### t0 set the Boolean condition while training)
dropout_1 <- layer_dropout(rate = proba)
dropout_2 <- layer_dropout(rate = proba)

input <-layer_input(shape = dim(data.train)[[2]])

output <- input %>%
  layer_dense(units = 3, activation = "relu") %>%
  dropout_1(training = TRUE) %>%
  layer_dense(units = 2, activation = "relu") %>%
  dropout_2(training = TRUE) %>%
  layer_dense(units = 1)

model <- keras_model(input, output)

model %>% compile (loss = "mean_squared_error",
                   optimizer_adam(lr = 0.01, beta_1 = 0.9,
                   beta_2 = 0.999, epsilon = NULL, decay = 0,
                   amsgrad = FALSE, clipnorm = NULL,
                   clipvalue = NULL), metrics =
                   list("mean_absolute_error", "mean_squared_error"))

history <- model %>% fit(data.train, labels.train, epochs = 80,
                 verbose = 0, validation_data = list(data.test,
                 labels.test), callbacks =
                 callback_model_checkpoint("weightdropsim.best.hdf5",
                 monitor = "val_loss", versbose = 0,
                 save_best_only = TRUE, save_weights_only = FALSE,
                 mode = c("auto","min","max"), period = 1))

### the MC-dropout allows to approximate the predictive distribution
### by performing T.replicas stochastic forward passes:
for(i in 1:T.replicas) {
  pred.data1[,i] <- model %>% predict(model = model, x = data.test) %>%
    as.vector()
}

### Computing the first and second moments of the distribution in
### order to estimate the uncertainty of the fitted neural network.
forecast_avg <- apply(pred.data1, 1, mean)
forecast_sd <- apply(pred.data1, 1, sd)
