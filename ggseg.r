library(readxl)
library(ggseg)
library(ggsegSchaefer)
library(tidyverse)
library(ggplot2)
library(wesanderson)
library(dplyr)

setwd('/example/working/directory/')

# Labels data to load
# First column is an index not used
# Second column should contain the region name matching what ggseg expects for the atlas
# Third column is the hemisphere ('left' or 'right')
labels <- read_excel("./ST218_Labels.xlsx")

labels = data.frame(labels)
colnames(labels)[2]  <- "region"
colnames(labels)[3]  <- "hemi"

ggseg_figure <- function(data_file,output_file,data_thresh,int_fig_thresh=TRUE){
  ###############################
  # Parameters
  # ----------
  # data_file         :   string path to csv file with BSR data to be plotted
  #                   should have a value in each row in same order as labels
  # output_file       :   string path to output png
  # data_thresh       :   float. Absolute value at which to threshold data
  # int_fig_thresh    :   bool. Whether to create figure legend threshold at integer
  #                   value (ceiling) rather than exact float
  # Returns
  # -------
  # None
  ###############################

  # Load data
  PlotData <- read_csv(data_file,col_names=c('bsr'))
  if (int_fig_thresh){
    figure_thresh = ceiling(max(c(abs(min(PlotData)),max(PlotData))))
    limits = c(-1*figure_thresh,figure_thresh)
  }
  else{
    limits = c(min(PlotData),max(PlotData))
  }
  plot.df<-cbind(labels, PlotData[,])
  plot.df$bsr[plot.df$bsr > -1*data_thresh & plot.df$bsr < data_thresh] <- NA
  plot.df[ plot.df == "NaN" ] <- NA #setting the NA values from excel to actual NAs
  pal <- wes_palette("Zissou1", 50, type = "continuous")
  
  newdata <- subset(plot.df, bsr!= "NA")
  someData <- tibble(
    region = c(newdata$region), 
    p = c(as.double(newdata$bsr)),
    groups = c(newdata$hemi)
  )
  
  # plotting
  sp<-someData%>%
    ggseg(atlas = schaefer17_200,
          mapping=aes(fill=as.double(p)),
          position="stacked", colour="black")+
    scale_color_manual(values = pal)+
    theme(legend.title=element_blank(), text=element_text(family="Arial"), axis.text=element_text(family="Arial"))
  sp+scale_fill_gradientn(colours = pal, limits=limits)
  ggsave(filename=output_file, width=1800, height=1200, device="png", units="px", bg='white')
}

results_dir = "./results/"
figures_dir = "./figures/"
data_names = c("SC_le_age_CattellTotal_1000_its_244_subs_0_to_50_age_range",
               "SC_le_age_CattellTotal_1000_its_350_subs_50_to_150_age_range",
               "SC_le_age_CattellTotal_1000_its_594_subs_0_to_150_age_range",
               "SC_ne_updated_age_CattellTotal_1000_its_244_subs_0_to_50_age_range",
               "SC_ne_updated_age_CattellTotal_1000_its_350_subs_50_to_150_age_range",
               "SC_ne_updated_age_CattellTotal_1000_its_594_subs_0_to_150_age_range"
               )

for (i in 1:length(data_names)){
  ggseg_figure( paste(results_dir,data_names[i],"/",data_names[i],"_lv1_bsr.csv",sep=""),
                paste(figures_dir,data_names[i],"_lv1_bsr_brain_image.png",sep=""),
                data_thresh=2.0,
                )
}

figures_data_dir = "/example/data/directory/"

ggseg_figure(paste(figures_data_dir,"TVBSchaeferTian220_SC_nodal_efficiency_OA_nodal_means.csv",sep=""),
             paste(figures_data_dir,"TVBSchaeferTian220_SC_nodal_efficiency_OA_nodal_means.png",sep=""),
             data_thresh=0, int_fig_thresh = FALSE)

ggseg_figure(paste(figures_data_dir,"TVBSchaeferTian220_SC_nodal_efficiency_YA_nodal_means.csv",sep=""),
             paste(figures_data_dir,"TVBSchaeferTian220_SC_nodal_efficiency_YA_nodal_means.png",sep=""),
             data_thresh=0, int_fig_thresh = FALSE)

ggseg_figure(paste(figures_data_dir,"TVBSchaeferTian220_SC_local_efficiency_OA_nodal_means.csv",sep=""),
             paste(figures_data_dir,"TVBSchaeferTian220_SC_local_efficiency_OA_nodal_means.png",sep=""),
             data_thresh=0, int_fig_thresh = FALSE)

ggseg_figure(paste(figures_data_dir,"TVBSchaeferTian220_SC_local_efficiency_YA_nodal_means.csv",sep=""),
             paste(figures_data_dir,"TVBSchaeferTian220_SC_local_efficiency_YA_nodal_means.png",sep=""),
             data_thresh=0, int_fig_thresh = FALSE)