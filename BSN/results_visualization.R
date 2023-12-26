# Results ALAS

df <- read.csv('C:/Users/Milton/Documents/Doc/Empatica-Project-ALAS-main-v2/Paper_Data/no_reference/results_10-fold.csv')
df <- df[, -1]
df[, 'CSV'] <- paste0(substr(df$CSV, 5, 5), substr(df$CSV, 8, 8))
df[, 'N_Fold'] <- rep(c(rep(5, 180/2), rep(10, 180/2)), 4)
print(df)

#df <- df[(df$CSV == 'EEG_Q39M69N100') & (df$), ]

library(lattice)


xyplot(Acc_Test ~ N_Feat | Model + N_Class + Validation, data = df[df$CSV == 'EEG_Q39M69N100', ])

features_l <- c(list(c(2, 4)), list(c(8, 10)))
title_label <- c('Low-features (2, 4)', 'High-features (8, 10)')


for (idx_feat in 1:length(features_l)){
     png(paste0('results_', idx_feat, '.png'), 1080, 480)
     
     par(mfrow=c(1, 2))
     a <- barchart(Acc_Test ~ CSV | Model + Validation, ylim = c(0.2, 0.9),
                   df[df$N_Feat == features_l[[idx_feat]][1],], groups = N_Class, ylab = 'Accuracy (%)')
     b <- barchart(Acc_Test ~ CSV | Model + Validation, ylim = c(0.2, 0.9),
                   df[df$N_Feat == features_l[[idx_feat]][2],], groups = N_Class, ylab = '')
     print(a, position = c(0, 0, 0.5, 1), more = TRUE)
     print(b, position = c(0.5, 0, 1, 1))
     
     #mtext(title_label[idx_feat], side = 3, outer = TRUE, line = 1)
     
     dev.off()
     
     #print(barchart(Acc_Test ~ CSV | Model + Validation,
     #               df[df$N_Feat == n_feat,], groups = N_Class, ylab = 'Accuracy (%)'))
}

custom_axis <- function(side, at, labels, ...){
     if (side == 'left'){
          labels <- rep(" ", length(labels))
     }
     axis.default(side, at, labels, ...)
}



for (n_fold in unique(df$N_Fold)){
     png(paste0('results_', n_fold, '-fold.png'), 480*2, 480*2)
     par(mfrow = c(2, 2))
     a <- barchart(Acc_Test ~ CSV | Model + Validation, ylim = c(0.2, 0.9), main = '2 features',
                    df[(df$N_Feat == 2) & (df$N_Fold == n_fold),], groups = N_Class, ylab = 'Accuracy (%)')
     b <- barchart(Acc_Test ~ CSV | Model + Validation, ylim = c(0.2, 0.9), main = '4 features',
                   df[df$N_Feat == 4  & (df$N_Fold == n_fold),], groups = N_Class, ylab = '', scales = list(y = list(at=NULL)))
     c <- barchart(Acc_Test ~ CSV | Model + Validation, ylim = c(0.2, 0.9), main = '8 features',
                   df[df$N_Feat == 8  & (df$N_Fold == n_fold),], groups = N_Class, ylab = 'Accuracy (%)')
     d <- barchart(Acc_Test ~ CSV | Model + Validation, ylim = c(0.2, 0.9), main = '10 features',
                   df[df$N_Feat == 10  & (df$N_Fold == n_fold),], groups = N_Class, ylab = '', scales = list(y = list(at=NULL)))
     
     print(a, position = c(0, 0.5, 0.5, 1), more = TRUE)
     print(b, position = c(0.5, 0.5, 1, 1), more = TRUE)
     print(c, position = c(0, 0, 0.5, 0.5), more = TRUE)
     print(d, position = c(0.5, 0, 1, 0.5))
     dev.off()
}

# Fragile accuracy
library(lattice)

df <- read.csv('C:/Users/Milton/Documents/Doc/Empatica-Project-ALAS-main-v2/Paper_Data/fragile_acc.csv')
df <- df[, -c(1:4)]
colnames(df) <- c('Validation', 'Model', 'Class', 'Acc', 'SD')

df['N'] <- ifelse(df$Validation == 'LOO', 16, 10)
df['SE'] <- df$SD / sqrt(df$N)
df['ulim'] <- df$Acc + df$SE
df['llim'] <- df$Acc
df['err'] <- ifelse(df$Class == 2, -0.175, 0.175)

#png('FragAcc.png', width=480, height=240)
pdf('FragAcc.pdf', width=7, height = 3.5)
b <- barchart(Acc ~ Model | Validation, df, groups = Class, ylim =c(0.2, 1),
         auto.key = list(space = "top", rectangles = TRUE, points = FALSE,
                         title='Class', cex.title= 1, columns=2),
         ylab = 'Mode accuracy', xlab = 'Model',
         scales = list( y=list(at=seq(0.2,1,0.2)), alternating = FALSE, tck = c(1,0)),
         panel=function(x,y,..., subscripts)
         {panel.barchart(x, y, subscripts = subscripts, ...)
               ll = df$llim[subscripts]
               ul = df$ulim[subscripts]
               
               # vertical error bars
               panel.segments(as.numeric(x) + df$err[subscripts], ll,
                              as.numeric(x) + df$err[subscripts], ul, col=1, lwd=1)
               
               # lower horizontal cap
               # panel.segments(as.numeric(x) + df$err[subscripts] - 0.1, ll,
               #                as.numeric(x) + df$err[subscripts] + 0.1, ll, col=1, lwd=1)
               
               # upper horizontal cap
               panel.segments(as.numeric(x) + df$err[subscripts] - 0.1, ul, 
                              as.numeric(x) + df$err[subscripts] + 0.1, ul, col=1, lwd=1)
})
print(b)
dev.off()

# Robust accuracy
library(lattice)
library(tidyr)

df <- read.csv('C:/Users/Milton/Documents/Doc/Empatica-Project-ALAS-main-v2/Paper_Data/robust_acc.csv')
df <- df[, -c(7)]
df <- data.frame(pivot_longer(df,
                              cols=c('Accuracy', 'F1.Score', 'Precision', 'Recall'),
                              names_to='Metric', values_to = 'Score'))
df['N_Fold'] <- as.factor(df$N_Fold)

df_mean <- df[df$Meaning == 'Mean',]
df_sd <- df[df$Meaning == 'SD',]


df_sd['SE'] <- df_sd$Score / sqrt(5*100)
df['ulim'] <- df_mean$Score + df_sd$SE
df['llim'] <- df_mean$Score

df['err'] <- ifelse(df$Metric == 'Accuracy', -0.25,
                    ifelse(df$Metric=='F1.Score', -0.0875,
                           ifelse(df$Metric=='Precision', 0.0875, 0.25)))

# png('robustacc.png', width = 480+120, height = 180)
pdf('robustacc.pdf', width=7, height=2.1)
b <- barchart(Score ~ N_Fold, df[df$Meaning == 'Mean',], groups = Metric, ylim =c(0.80, 1),
         auto.key = list(space = "right", rectangles = TRUE, points = FALSE, columns=1),
         ylab = 'Performance metric', xlab = 'Number of stratified folds',
         scales = list( y=list(at=seq(0.8,1,0.04)), alternating = FALSE, tck = c(1,0)),
         panel=function(x,y,..., subscripts)
         {panel.barchart(x, y, subscripts = subscripts, ...)
              ll = df$llim[subscripts]
              ul = df$ulim[subscripts]
              
              # vertical error bars
              panel.segments(as.numeric(x) + df$err[subscripts], ll,
                             as.numeric(x) + df$err[subscripts], ul, col=1, lwd=1)
              
              # lower horizontal cap
              # panel.segments(as.numeric(x) + df$err[subscripts] - 0.05, ll,
              #              as.numeric(x) + df$err[subscripts] + 0.05, ll, col=1, lwd=1)
              
              # upper horizontal cap
              panel.segments(as.numeric(x) + df$err[subscripts] - 0.05, ul, 
                             as.numeric(x) + df$err[subscripts] + 0.05, ul, col=1, lwd=1)
         })
print(b)
dev.off()
