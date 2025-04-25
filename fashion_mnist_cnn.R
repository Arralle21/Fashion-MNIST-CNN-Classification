# =============================================================================
# Fashion MNIST Classification with CNN in R
# Submitted by: Abdullahi Mohamed Jibril
# Date 04-25-2025
# =============================================================================
# This R script demonstrates how to build a CNN model similar to the Python version
# for classifying Fashion MNIST images using the keras and tensorflow packages.
# =============================================================================

# =============================================================================
# 1. REQUIRED LIBRARIES
# =============================================================================
# If installing the packages, uncomment these lines
# install.packages("keras")
# install.packages("tensorflow")
# install.packages("tidyverse")
# install.packages("reticulate")
# tensorflow::install_tensorflow()
# keras::install_keras()

library(keras)
library(tensorflow)
library(tidyverse)
library(reticulate)

# =============================================================================
# 2. SETUP AND CONFIGURATION
# =============================================================================
# Set random seed for reproducibility
set.seed(42)
tf$random$set_seed(42)

# Define the class names for Fashion MNIST
class_names <- c(
  'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
)

# =============================================================================
# 3. DATA LOADING AND PREPARATION
# =============================================================================
# Path to your data - update this path to where your data is stored
data_path <- "C:/Users/Abdullah/OneDrive/Documents/NXU/Programming using Python and R/Module 6"

# Read CSV files
train_data <- read.csv(file.path(data_path, "fashion-mnist_train.csv"))
test_data <- read.csv(file.path(data_path, "fashion-mnist_test.csv"))

# Separate features and labels
X_train <- train_data %>% select(-label) %>% as.matrix()
y_train <- train_data$label

X_test <- test_data %>% select(-label) %>% as.matrix()
y_test <- test_data$label

# Reshape and normalize
X_train <- array_reshape(X_train, c(nrow(X_train), 28, 28, 1)) / 255.0
X_test <- array_reshape(X_test, c(nrow(X_test), 28, 28, 1)) / 255.0

# One-hot encode labels
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

print(paste("Training data shape:", paste(dim(X_train), collapse = " x ")))
print(paste("Test data shape:", paste(dim(X_test), collapse = " x ")))

# =============================================================================
# 4. DATA VISUALIZATION (optional)
# =============================================================================
# Example to visualize some training images
visualize_samples <- function(n_samples = 10) {
  par(mfrow=c(2, 5), mar=c(0, 0, 1.5, 0))
  for (i in 1:n_samples) {
    index <- sample(1:nrow(X_train), 1)
    img <- X_train[index,,,]
    label <- which.max(y_train[index,]) - 1
    
    image(t(apply(img[,,1], 2, rev)), 
          axes = FALSE, 
          col = gray.colors(100, start = 1, end = 0),
          main = class_names[label + 1])
  }
}

# Uncomment to run visualization
# visualize_samples()

# =============================================================================
# 5. MODEL ARCHITECTURE
# =============================================================================
# Build the 6-layer CNN model
model <- keras_model_sequential() %>%
  # Layer 1: Conv + Pooling
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', 
                padding = 'same', input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Layer 2: Conv + Pooling
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu', 
                padding = 'same') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Layer 3: Conv + Pooling
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu', 
                padding = 'same') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Layer 4: Flatten
  layer_flatten() %>%
  
  # Layer 5: Dense with Dropout
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  
  # Layer 6: Output
  layer_dense(units = 10, activation = 'softmax')

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

# Print model summary
summary(model)

# =============================================================================
# 6. MODEL TRAINING
# =============================================================================
# Define early stopping callback to prevent overfitting
early_stop <- callback_early_stopping(
  monitor = 'val_loss',
  patience = 3,
  restore_best_weights = TRUE
)

# Train the model
history <- model %>% fit(
  X_train, y_train,
  epochs = 10,  # Reduced for quicker execution
  batch_size = 64,
  validation_split = 0.1,
  callbacks = list(early_stop),
  verbose = 1
)

# =============================================================================
# 7. MODEL EVALUATION
# =============================================================================
# Evaluate the model on test data
evaluation <- model %>% evaluate(X_test, y_test, verbose = 1)
print(paste("Test Loss:", round(evaluation[1], 4)))
print(paste("Test Accuracy:", round(evaluation[2], 4)))

# =============================================================================
# 8. VISUALIZE TRAINING HISTORY
# =============================================================================
plot_training_history <- function(history) {
  par(mfrow = c(1, 2), mar = c(4, 4, 2, 1))
  
  # Plot accuracy
  plot(history$metrics$accuracy, type = "l", col = "blue",
       xlab = "Epoch", ylab = "Accuracy", main = "Model Accuracy", ylim = c(0, 1))
  lines(history$metrics$val_accuracy, col = "red")
  legend("bottomright", legend = c("Train", "Validation"), 
         col = c("blue", "red"), lty = 1)
  
  # Plot loss
  plot(history$metrics$loss, type = "l", col = "blue",
       xlab = "Epoch", ylab = "Loss", main = "Model Loss")
  lines(history$metrics$val_loss, col = "red")
  legend("topright", legend = c("Train", "Validation"), 
         col = c("blue", "red"), lty = 1)
  
  # Optional: Save the plot
  # dev.copy(png, 'training_history.png', width = 1200, height = 500)
  # dev.off()
}

# Uncomment to plot training history
# plot_training_history(history)

# =============================================================================
# 9. MAKE PREDICTIONS
# =============================================================================
# Function to visualize predictions
visualize_predictions <- function(num_samples = 5) {
  # Select random samples from test set
  set.seed(123)  # For reproducible random selection
  sample_indices <- sample(1:nrow(X_test), num_samples)
  
  # Create plots
  par(mfrow = c(num_samples, 2), mar = c(3, 3, 3, 1))
  
  for (i in 1:num_samples) {
    idx <- sample_indices[i]
    img <- X_test[idx,,,]
    true_label <- which.max(y_test[idx,]) - 1  # Subtract 1 to get 0-based index
    
    # Make prediction
    pred <- predict(model, array_reshape(img, c(1, 28, 28, 1)))
    pred_label <- which.max(pred) - 1  # Subtract 1 to get 0-based index
    confidence <- max(pred)
    
    # Plot image
    image(t(apply(img[,,1], 2, rev)), 
          col = gray.colors(100, start = 1, end = 0),
          axes = FALSE,
          main = paste0("True: ", class_names[true_label + 1], 
                        "\nPredicted: ", class_names[pred_label + 1]))
    
    # Plot prediction probabilities
    barplot(pred[1,], 
            names.arg = class_names, 
            horiz = TRUE, 
            las = 1,
            cex.names = 0.7,
            main = paste0("Confidence: ", round(confidence, 2)),
            xlab = "Probability")
  }
  
  # Optional: Save the visualization
  # dev.copy(png, 'predictions.png', width = 1500, height = num_samples * 300, res = 100)
  # dev.off()
}

# Uncomment to visualize predictions
# visualize_predictions(5)

# =============================================================================
# 10. CONFUSION MATRIX
# =============================================================================
# Function to create and visualize confusion matrix
create_confusion_matrix <- function() {
  # Get predictions for all test data
  predictions <- model %>% predict(X_test) %>% k_argmax() %>% as.integer()
  true_labels <- max.col(y_test) - 1  # Convert from one-hot back to single labels
  
  # Create confusion matrix
  conf_matrix <- table(Predicted = class_names[predictions + 1], 
                       Actual = class_names[true_labels + 1])
  
  # Print confusion matrix
  print(conf_matrix)
  
  # Calculate per-class accuracy
  per_class_accuracy <- diag(conf_matrix) / colSums(conf_matrix)
  print("Per-class accuracy:")
  print(round(per_class_accuracy, 3))
  
  return(conf_matrix)
}

# Uncomment to create confusion matrix
# conf_matrix <- create_confusion_matrix()

# =============================================================================
# 11. SAVE THE MODEL
# =============================================================================
# Save model and class names
save_model <- function() {
  model_path <- 'fashion_mnist_cnn.h5'
  save_model_hdf5(model, model_path)
  print(paste("Model saved to:", model_path))
  
  class_names_file <- 'fashion_mnist_class_names.rds'
  saveRDS(class_names, class_names_file)
  print(paste("Class names saved to:", class_names_file))
}

# Uncomment to save the model
# save_model()

# =============================================================================
# 12. MODEL LOADING AND INFERENCE (for future use)
# =============================================================================
# Function to load saved model and make predictions
load_and_predict <- function(image_path) {
  # Load the saved model
  loaded_model <- load_model_hdf5('fashion_mnist_cnn.h5')
  loaded_class_names <- readRDS('fashion_mnist_class_names.rds')
  
  # Load and preprocess the image
  # This is pseudocode - in reality you'd need image processing functions
  # img <- load_and_preprocess_image(image_path)
  
  # Make prediction
  # pred <- predict(loaded_model, img)
  # pred_label <- which.max(pred) - 1
  # print(paste("Predicted class:", loaded_class_names[pred_label + 1]))
}

# =============================================================================
# CONCLUSION
# =============================================================================
print("Fashion MNIST CNN model script completed.")