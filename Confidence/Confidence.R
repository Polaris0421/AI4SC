sigmoid_weight <- function(R){
  return(exp(abs(R)-max(R))/(1+exp(abs(R)-max(R))))
}

weighted_quantile <- function(w,R,prob){
  w_sort = sort(w)
  R_sort = sort(R) # R need to be absolute value
  n = length(R_sort)
  q = 0
  R_t = 0
  for (i in 1:n){
    if (q<prob){
      q = q + w_sort[i]
    }
    else{
      R_t = R_sort[i]
      break
    } 
    
  }
  return(R_t)
}

## 容忍度设定
delta = 1

## 有序置信度计算
val = read.csv('311_val_outputs.csv')
test = read.csv('311_test_outputs.csv')

val_res = abs(val$target-val$prediction)
test_predict = test$prediction
recover = rep(0,length(test$prediction))
for (i in 1:length(test$prediction)){
  candidate = abs(test_predict[i]-val$target)
  index = which.min(candidate)
  recover[i] = test_predict[i] + val_res[index]
}
conf_order = rep(0,length(test$prediction))
for (j in 1:length(conf_order)){
  conf_order[j] = mean((recover[j]+delta) > val$target)
}

## 无序置信度计算

val = read.csv('316_val_outputs.csv')
test = read.csv('316_test_outputs.csv')

val_res = abs(val$target-val$prediction)
test_predict = test$prediction
recover = rep(0,length(test$prediction))
for (i in 1:length(test$prediction)){
  candidate = abs(test_predict[i]-val$target)
  index = which.min(candidate)
  recover[i] = test_predict[i] + val_res[index]
}
w = abs(val_res)/sum(abs(val_res))
conf_disorder = rep(0,length(test$prediction))
for (j in 1:length(conf_disorder)){
  conf_disorder[j] = sum(w*((recover[j]+delta) > val$target))
}

data = data.frame('分位数' = c(c(1:length(conf_order))/length(conf_order),c(1:length(conf_disorder)/length(conf_disorder))),
                    '置信度' = c(sort(conf_order),sort(conf_disorder)),
                  '晶体结构' = c(rep('有序',length(conf_order)),rep('无序',length(conf_disorder))))

library(ggplot2)

## 绘制图像
pp<- ggplot(data,aes(x=分位数,y=置信度,color=晶体结构,shape = 晶体结构))+
  geom_point(size = 2, alpha=0.5)+
  geom_line(alpha=0.5)+
  geom_jitter(aes(color=晶体结构))+
  geom_hline(yintercept = 0.3, linetype = 'dashed')+
  scale_color_manual(values = c('#BFBFFE','#6D6DFF'))+
  theme_bw()+
  theme(text = element_text(family = 'Kai'),
    axis.title = element_text(size = 20),
        panel.grid = element_blank(),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        axis.text = element_text(size = 16))
pp

