library(shiny)
library(keras)

model <- load_model_hdf5("~/PinguOrNot/pingu.hd5")

pred <- function(img_path) {
  img <- image_load(img_path, target_size = c(150, 150)) %>%           
    image_to_array() %>% 
    array_reshape(dim = c(1, 150, 150, 3)) #%>%                         
  #imagenet_preprocess_input()      
  preds <- model %>% predict_proba(img)
  
  img <- image_load(img_path, target_size = c(150, 150)) %>%
    image_to_array()
  plot(as.raster(img,max=255))
  preds
}



server <- shinyServer(function(input, output) {
  output$files <- renderTable(input$files)
  
  files <- reactive({
    files <- input$files
    files$datapath <- gsub("\\\\", "/", files$datapath)
    files
  })
  
  
  output$images <- renderUI({
    if(is.null(input$files)) return(NULL)
    image_output_list <- 
      lapply(1:nrow(files()),
             function(i)
             {
               imagename = paste0("image", i)
               #list(imageOutput(imagename),renderText(pred(files()$datapath[i])))
               if(pred(files()$datapath[i]) == 1)
                 list(renderText("Pingu"),imageOutput(imagename,inline=T))
               else
                 list(renderText("Not Pingu"),imageOutput(imagename,inline=T))
             })
    
    do.call(tagList, image_output_list)
  })
  
  observe({
    if(is.null(input$files)) return(NULL)
    for (i in 1:nrow(files()))
    {
      print(i)
      local({
        my_i <- i
        imagename = paste0("image", my_i)
        print(imagename)
        #pred(files()$datapath[my_i])
        print(paste0("prediction: ",pred(files()$datapath[my_i])))
        output[[imagename]] <- 
          renderImage({
            list(src = files()$datapath[my_i],
                 alt = "Image failed to render")
          }, deleteFile = FALSE)
      })
    }
  })
  
})

ui <- shinyUI(fluidPage(
  titlePanel("Pingu Or Not"),
  sidebarLayout(
    sidebarPanel(
      fileInput(inputId = 'files', 
                label = 'Select an Image',
                multiple = TRUE,
                accept=c('image/png', 'image/jpeg'))
    ),
    mainPanel(
      tableOutput('files'),
      uiOutput('images')
    )
  )
))

shinyApp(ui=ui,server=server)