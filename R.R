original_dataset_dir <- "~/PinguOrNot/pingu"
base_dir <- "~/PinguOrNot/pingu"
dir.create(base_dir)
train_dir <- file.path(base_dir, "train")
dir.create(train_dir)
validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)
test_dir <- file.path(base_dir, "test")
dir.create(test_dir)
train_pingu_dir <- file.path(train_dir, "pingu")
dir.create(train_pingu_dir)
train_other_dir <- file.path(train_dir, "other")
dir.create(train_other_dir)
validation_pingu_dir <- file.path(validation_dir, "pingu")
dir.create(validation_pingu_dir)
validation_other_dir <- file.path(validation_dir, "other")
dir.create(validation_other_dir)
test_pingu_dir <- file.path(test_dir, "pingu")
dir.create(test_pingu_dir)
test_other_dir <- file.path(test_dir, "other")
dir.create(test_other_dir)

 
cat("total training pingu images:", length(list.files(train_pingu_dir)), "\n")
cat("total training other images:", length(list.files(train_other_dir)), "\n")
cat("total validation pingu images:",
      length(list.files(validation_pingu_dir)), "\n")
cat("total validation other images:",
      length(list.files(validation_other_dir)), "\n")

library(keras)
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)

train_datagen <- image_data_generator(rescale = 1/255)             
validation_datagen <- image_data_generator(rescale = 1/255)        

train_generator <- flow_images_from_directory(
  train_dir,                                                       
  train_datagen,                                                   
  target_size = c(150, 150),                                       
  batch_size = 20,                                                 
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)

save_model_hdf5("./PinguOrNot/pingu.hd5")

pred <- function(img_path) {
  img <- image_load(img_path, target_size = c(150, 150)) %>%           
    image_to_array() %>% 
    array_reshape(dim = c(1, 150, 150, 3))# %>%                         
  #imagenet_preprocess_input()      
  preds <- model %>% predict_proba(img)
  
  img <- image_load(img_path, target_size = c(150, 150)) %>%
    image_to_array()
  plot(as.raster(img,max=255))
  preds
}

img_path <- "~/PinguOrNot/pingu/validation/other/00000001.jpg"
print(pred(img_path))

####### DISPLAY ACTIVATION MAPS ################
img <- image_load(img_path, target_size = c(150, 150))                 
img_tensor <- image_to_array(img)
img_tensor <- array_reshape(img_tensor, c(1, 150, 150, 3))
img_tensor <- img_tensor / 255                                         
dim(img_tensor)  
plot(as.raster(img_tensor[1,,,]))

layer_outputs <- lapply(model$layers[1:8], function(layer) layer$output)      
activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)
activations <- activation_model %>% predict(img_tensor)

plot_channel <- function(channel) {
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(channel), axes = FALSE, asp = 1,
        col = terrain.colors(12))
}

image_size <- 58
images_per_row <- 16

for (i in 1:8) {
  
  layer_activation <- activations[[i]]
  layer_name <- model$layers[[i]]$name
  
  n_features <- dim(layer_activation)[[4]]
  n_cols <- n_features %/% images_per_row
  
  png(paste0("pingu_activations_", i, "_", layer_name, ".png"),
      width = image_size * images_per_row,
      height = image_size * n_cols)
  op <- par(mfrow = c(n_cols, images_per_row), mai = rep_len(0.02, 4))
  
  for (col in 0:(n_cols-1)) {
    for (row in 0:(images_per_row-1)) {
      channel_image <- layer_activation[1,,,(col*images_per_row) + row + 1]
      plot_channel(channel_image)
    }
  }
  
  par(op)
  dev.off()
}