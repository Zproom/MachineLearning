#' analysis.R
#' This file contains analysis for the report. In particular, it creates
#' convergence plots showing the relationship between the number of nodes
#' in the tree and the validation error during the pruning process.

library(tidyverse)
library(this.path)

home_dir <- dirname(this.path())
setwd(dirname(this.path()))

data_dir <- "../data/"
chart_dir <- "../charts/"

#' This function creates a convergence plot for a given dataset.
#' @param data_file The full file path to the validation error data, as a
#' string.
#' @param chart_file The full file path to the output chart file, as a string.
#' @param dataset The name of the dataset to be displayed in the chart, as a
#' string.
#' @return Saves a .png in chart_file.
create_convergence_plot <- function(data_file, chart_file, dataset) {
  prune_data <- read_csv(data_file, col_names = c("num_nodes", "error"))
  prune_data_final <- prune_data %>% group_by(num_nodes) %>%
    summarize(average_error = mean(error))
  breaks <- min(prune_data_final$num_nodes):max(prune_data_final$num_nodes)
  if (max(breaks) >= 10) breaks <- seq(min(breaks), max(breaks), by = 2)
  plot <- ggplot(data = prune_data_final, aes(x = num_nodes, 
                                              y = average_error)) +
    geom_line() +
    scale_x_continuous(breaks = breaks, labels = breaks) +
    labs(x = "Total Number of Nodes in Tree",
         y = "Validation Error",
         title = dataset,
         subtitle = "Convergence Plot")
  ggsave(chart_file,
         plot,
         dpi = 800,
         height = 4,
         width = 3.4,
         units = "in")
}

###############################################################################
# Cancer data
###############################################################################

data_file <- paste0(data_dir, "breast-cancer-wisconsin_prune.csv")
chart_file <- paste0(chart_dir, "breast-cancer-wisconsin_prune.png")
dataset <- "Breast Cancer Wisconsin"
create_convergence_plot(data_file, chart_file, dataset)

###############################################################################
# Car data
###############################################################################

data_file <- paste0(data_dir, "car_prune.csv")
chart_file <- paste0(chart_dir, "car_prune.png")
dataset <- "Car Evaluation"
create_convergence_plot(data_file, chart_file, dataset)

###############################################################################
# Voting data
###############################################################################

data_file <- paste0(data_dir, "house-votes-84_prune.csv")
chart_file <- paste0(chart_dir, "house-votes-84_prune.png")
dataset <- "Congressional Voting Records"
create_convergence_plot(data_file, chart_file, dataset)

###############################################################################
# Abalone data
###############################################################################

data_file <- paste0(data_dir, "abalone_prune.csv")
chart_file <- paste0(chart_dir, "abalone_prune.png")
dataset <- "Abalone"
create_convergence_plot(data_file, chart_file, dataset)

###############################################################################
# Computer data
###############################################################################

data_file <- paste0(data_dir, "machine_prune.csv")
chart_file <- paste0(chart_dir, "machine_prune.png")
dataset <- "Computer Hardware"
create_convergence_plot(data_file, chart_file, dataset)

###############################################################################
# Forest Fire data
###############################################################################

data_file <- paste0(data_dir, "forestfires_prune.csv")
chart_file <- paste0(chart_dir, "forestfires_prune.png")
dataset <- "Forest Fires"
create_convergence_plot(data_file, chart_file, dataset)