

setwd("/home/luojiawei/mimic3_R_work")

library(RPostgreSQL)
library(stringr)
library(magrittr)
library(lubridate)
library(rlist)
library(data.table)

source("/home/luojiawei/RL_AP/工具函数1.R")

# 访问数据库
drv<-dbDriver("PostgreSQL")
con<-dbConnect(drv,host="localhost",port="5432",dbname="mimic3",user="ljw",password="123456")

# 处理 admission 数据集
rs<-dbSendQuery(con,statement = "select * from admissions;")
admissions<-fetch(rs,n=-1)
admissions[1:2,]



## 读取 patients 数据集
rs<-dbSendQuery(con,statement = "select * from patients;")
patients<-fetch(rs,n=-1)
patients[1:2,]

admissions1<-merge(admissions, patients, by=c('subject_id'), all.x=T)
admissions1[1:5,]

admissions1$age<-round(interval(admissions1$dob,admissions1$admittime)/years(1),0)+1
admissions1$age[1:100]

admissions1$time_to_death<-round(interval(admissions1$admittime,admissions1$dod)/hours(1),2)
admissions1$time_to_out<-round(interval(admissions1$admittime,admissions1$dischtime)/hours(1),2)
summary(admissions1$time_to_death)


admissions2<-admissions1[,c(1,3,4,5,6,10,11,12,13,14,16,21:ncol(admissions1))]
names(admissions2)
for(i in c(6,7,8,9,10)){
	admissions2[[i]][is.na(admissions2[[i]])]<-"NA"
}
admissions2[1:5,]

fun<-function(x, n){
	
	x <- ifelse(x<=n, 1, 0)
	x[is.na(x)]<-0
	x
	
}
admissions2$death_in_24h<-fun(admissions2$time_to_death, 24)
admissions2$death_in_48h<-fun(admissions2$time_to_death, 48)
admissions2$death_in_90d<-fun(admissions2$time_to_death, 24*90)

names(admissions2)

d_admissions<-list()
for(i in c(6,7,8,9,10,12)){
	x<-as.character(unique(admissions2[[i]]))
	df<-data.frame(index = 1:length(x),value=x)
	d_admissions[[names(admissions2)[i]]]<-df
}

save("d_admissions", file="./admission的字典.RData")

for(i in c(6,7,8,9,10,12)){
	cur_d<-d_admissions[[names(admissions2)[i]]]
	admissions2[[i]]<-unlist(lapply(admissions2[[i]], function(x){cur_d$index[cur_d$value==x]}))
}

save(list=c("admissions","admissions2"), file="./处理后的admission.RData")


# 实验室数据的处理 labevents

rs<-dbSendQuery(con,statement = "select * from labevents;")
labevents<-fetch(rs,n=-1)

sum(is.na(labevents$hadm_id))
labevents1<-labevents[which(!is.na(labevents$hadm_id)),]
dim(labevents1)

## 为每个实验室观测匹配入院时间
names(labevents1)
names(admissions)
labevents1 <- merge(labevents1[,c(2:ncol(labevents1))], admissions[,c(2,3,4,5,6)], 
			        by=c("subject_id", "hadm_id"), all.x=T)
names(labevents1)

labevents1$time<-interval(labevents1$admittime, labevents1$charttime)/hours(1)

save("labevents1", file="./labevents1.RData")
load(file="./labevents1.RData")


## 统计各指标的患者覆盖数量


# length(unique(labevents1$itemid))
# tab_lab <- table(labevents1$itemid)
# tab_lab <-sort(tab_lab, decreasing=T)
# length(tab_lab)

# tab_lab1<-lapply(1:100, function(i){

		# n<-length(unique(labevents1$hadm_id[which(labevents1$itemid==names(tab_lab)[i])]))
		# return(n)

# }) %>% unlist

# save(list=c("tab_lab","tab_lab1"), file="./各lab指标的admission覆盖数量统计.RData")

load("./各lab指标的admission覆盖数量统计.RData")

N<-length(unique(labevents1$hadm_id))
tab_lab2<-data.frame(name=names(tab_lab)[1:100], ratio = round(tab_lab1/N,2))
selected_lab_item <- as.character(tab_lab2$name[tab_lab2$ratio>=0.2])
length(selected_lab_item)
labevents2<-labevents1[which(labevents1$itemid %in% selected_lab_item), ]


##  定量数据
labevents2.n<-labevents2[which(!is.na(labevents2$valuenum)),c(1,2,3,12,6)]
names(labevents2.n)
labevents2.n$time<-round(labevents2.n$time,0)
lab_vars_num<-unique(labevents2.n$itemid)

### 处理极端值
labevents2.n1<-lapply(lab_vars_num,

							function(id){
								
								ds_tmp<-labevents2.n[which(labevents2.n$itemid==id),]
								ran<-quantile(ds_tmp$valuenum, probs=c(0.01,0.99),na.rm=T)
								ds_tmp$valuenum[ds_tmp$valuenum<ran[1] |  ds_tmp$valuenum>ran[2]]<-NA
								return(ds_tmp)
								}
)
labevents2.n<-list.rbind(labevents2.n1)


stat_lab_value_num<-lapply(lab_vars_num,

							function(id){
								
								x<-labevents2.n$valuenum[labevents2.n$itemid==id]
								m<-mean(x,na.rm=T)
								s<-sd(x,na.rm=T)
								if(is.na(s) || s<0.001) s<-0.001
								m1<-median(x,na.rm=T)
								c(m,s,m1)
								}
)
names(stat_lab_value_num)<-lab_vars_num


## 保存处理好的数据
save(list=c("labevents2.n"),
		file="./处理后的labevents数据.RData")
## 保存字典和统计信息
save(list=c("stat_lab_value_num"),
		file="./labevents数据 字典和统计信息.RData")
		


# 处理 chartevents
rs<-dbSendQuery(con,statement = "select * from chartevents;")
chartevents<-fetch(rs,n=-1)
dim(chartevents)
chartevents<-chartevents[which(!is.na(chartevents$hadm_id)),]
chartevents <- merge(chartevents[,c(2:ncol(chartevents))], admissions[,c(2,3,4,5,6)], 
			        by=c("subject_id", "hadm_id"), all.x=T)
chartevents$time<-interval(chartevents$admittime, chartevents$charttime)/hours(1)
save("chartevents", file="./chartevents.RData")
load(file="./chartevents.RData")


## 统计各指标的患者覆盖数量

# tab_chart <- table(chartevents$itemid)
# tab_chart <-sort(tab_chart, decreasing=T)
# length(tab_chart)

# tab_chart1<-lapply(1:100, function(i){

		# n<-length(unique(chartevents$hadm_id[which(chartevents$itemid==names(tab_chart)[i])]))
		# return(n)

# }) %>% unlist

# save(list=c("tab_chart","tab_chart1"), file="./各chart指标的admission覆盖数量统计.RData")
load("./各chart指标的admission覆盖数量统计.RData")


N<-length(unique(chartevents$hadm_id))
tab_chart2<-data.frame(name=names(tab_chart)[1:100], ratio = round(tab_chart1/N,2))
selected_chart_item <- as.character(tab_chart2$name[tab_chart2$ratio>=0.2])

chartevents1<-chartevents[which(chartevents$itemid %in% selected_chart_item), ]


##  定量数据

chartevents1.n<-chartevents1[which(!is.na(chartevents1$valuenum)),c(1,2,4,18,9)]
chartevents1.n$time<-round(chartevents1.n$time,0)
chart_vars_num<-unique(chartevents1.n$itemid)

names(chartevents1.n)

### 处理极端值
chartevents1.n1<-lapply(chart_vars_num,

							function(id){
								
								ds_tmp<-chartevents1.n[which(chartevents1.n$itemid==id),]
								ran<-quantile(ds_tmp$valuenum, probs=c(0.01,0.99),na.rm=T)
								ds_tmp$valuenum[ds_tmp$valuenum<ran[1] |  ds_tmp$valuenum>ran[2]]<-NA
								return(ds_tmp)
								}
)
chartevents1.n<-list.rbind(chartevents1.n1)

stat_chart_value_num<-lapply(chart_vars_num,

							function(id){
								
								x<-chartevents1.n$valuenum[chartevents1.n$itemid==id]
								m<-mean(x,na.rm=T)
								s<-sd(x,na.rm=T)
								if(is.na(s) || s<0.001) s<-0.001
								m1<-median(x,na.rm=T)
								c(m,s,m1)
								}
)
names(stat_chart_value_num)<-chart_vars_num

## 保存处理好的数据
save(list=c("chartevents1.n"),
		file="./处理后的chartevents数据.RData")
## 保存字典和统计信息
save(list=c("stat_chart_value_num"),
		file="./chartevents数据 字典和统计信息.RData")


# 处理文本信息
# noteevents <- fread("/home/luojiawei/mimic3_R_work/noteevents_已经分词2.csv",
					# header=T, select = c("subject_id", "hadm_id", "chartdate", "text1"))
# length(unique(noteevents$hadm_id))
# names(admissions)
# noteevents$hadm_id<-as.character(noteevents$hadm_id)
# str(noteevents)
# str(admissions)
# noteevents<-merge(noteevents, admissions[,c(1,2,3)], 
				   # by=c("subject_id","hadm_id"),
			       # all.x=T)
# noteevents$time<-round(interval(noteevents$admittime,noteevents$chartdate)/hours(1),0)
# noteevents$time<-round(noteevents$time,0)

# save("noteevents", file="./noteevents.RData")

load(file="./noteevents.RData")

# 处理 prescriptions

prescriptions<-fread("/home/luojiawei/mimic3/mimic3_data/PRESCRIPTIONS.csv",
					header=T, fill=T)
names(prescriptions)<-tolower(names(prescriptions))
prescriptions<-prescriptions[which(!is.na(prescriptions$hadm_id)),]
prescriptions<-prescriptions[which(prescriptions$formulary_drug_cd != ""), ]

prescriptions1<-merge(prescriptions, admissions[,c(2,3,4,5,6)], 
		by=c("subject_id", "hadm_id"),
		all.x=T)
		
prescriptions1$time<-round(interval(prescriptions1$admittime,prescriptions1$startdate)/hours(1),2)
prescriptions1<-prescriptions1[which(!is.na(prescriptions1$time)),]
prescriptions1$time<-round(prescriptions1$time, 0)


## 统计各药物的患者覆盖数量

# tab_pres<-table(prescriptions[["formulary_drug_cd"]])
# tab_pres<-sort(tab_pres, decreasing=T)
# length(tab_pres)

# tab_pres1<-lapply(1:1000, function(i){
	# n<-length(unique(prescriptions1$hadm_id[prescriptions1[["formulary_drug_cd"]]==names(tab_pres)[i]]))
	# return(n)
# }) %>% unlist


N<-length(unique(prescriptions1$hadm_id))
tab_pres2<-data.frame(name=names(tab_pres)[1:1000], ratio = round(tab_pres1/N,2))

# save(list=c("tab_pres","tab_pres1"), file="./各prescription药物的admission覆盖数量统计.RData")
load(file="./各prescription药物的admission覆盖数量统计.RData")

selected_drug <- as.character(tab_pres2$name[tab_pres2$ratio>=0.2])

prescriptions1<-prescriptions1[which(prescriptions1$formulary_drug_cd %in% selected_drug),]

uniq_drug<-unique(prescriptions1$formulary_drug_cd)
d_drug<-data.frame(index=1:length(uniq_drug),
						   itemid = uniq_drug)
						   
save("d_drug", file="./医嘱的字典.RData")

prescriptions1<-merge(prescriptions1, admissions[,c(2:4)], 
		by=c("subject_id", "hadm_id"),
		all.x=T)
prescriptions1$time<-round(interval(prescriptions1$admittime,prescriptions1$startdate)/hours(1),2)
prescriptions1<-prescriptions1[which(!is.na(prescriptions1$time)),]
save("prescriptions1", file="./处理后的prescriptions.RData")


#########################################################

#            开始为每个患者生成 患者为中心的 数据

load(file="./admission的字典.RData")
load(file="./处理后的admission.RData")
load(file="./处理后的labevents数据.RData")
load(file="./labevents数据 字典和统计信息.RData")
load(file="./处理后的chartevents数据.RData")
load(file="./chartevents数据 字典和统计信息.RData")
load(file="./noteevents.RData")
load(file="./医嘱的字典.RData")
load(file="./处理后的prescriptions.RData")


admissions2<-admissions2[admissions2$age>=18,]
all_hadm_id <- unique(admissions2$hadm_id)

all_hadm_id<-intersect(all_hadm_id, chartevents1.n$hadm_id)
all_hadm_id<-intersect(all_hadm_id, labevents2.n$hadm_id)
all_hadm_id<-intersect(all_hadm_id, prescriptions1$hadm_id)
all_hadm_id<-intersect(all_hadm_id, noteevents$hadm_id)

ids<-c()

## 24小时死亡
pos_ids<-admissions2$hadm_id[admissions2$death_in_24h==1]
set.seed(2022)
pos_ids<-sample(pos_ids, size=length(pos_ids), replace=F)
if(length(pos_ids)>=6000){
	pos_ids<-pos_ids[1:6000]
}
neg_ids<-admissions2$hadm_id[admissions2$death_in_24h==0]
set.seed(2022)
neg_ids<-sample(neg_ids, size=length(neg_ids), replace=F)

train_ids<-c()
train_ids<-c(train_ids, pos_ids[1:floor(length(pos_ids)*0.7)])
train_ids<-c(train_ids, if(length(neg_ids)>=(1*length(pos_ids))) neg_ids[1:(floor(1*length(pos_ids)*0.7))] else neg_ids[1:floor(length(neg_ids)*0.7)])

train_ids<-train_ids[train_ids%in%all_hadm_id]

write.csv(data.frame(id = as.character(train_ids)),
		  file="./id_files/train_id_death24h.csv", row.names=F)

test_ids<-c()
test_ids<-c(test_ids, pos_ids[(floor(length(pos_ids)*0.7)+1):length(pos_ids)])
test_ids<-c(test_ids, if(length(neg_ids)>=(1*length(pos_ids))) neg_ids[(floor(1*length(pos_ids)*0.7)+1):(1*length(pos_ids))] else neg_ids[(floor(length(neg_ids)*0.7)+1):length(neg_ids)])
test_ids<-test_ids[test_ids%in%all_hadm_id]
write.csv(data.frame(id = as.character(test_ids)),
		  file="./id_files/test_id_death24h.csv", row.names=F)
ids<-union(ids, c(train_ids, test_ids))


for(i in 1:length(ids)){
	cur_id <- ids[i]
	folder_path<-paste0("./all_admissions/", cur_id)
	# if(!dir.exists(folder_path)){
		# dir.create(folder_path)
	# }
	if(!dir.exists(folder_path)){
		dir.create(folder_path)
	} else{
		next
	}
	
	#################### labevents ####################
	
	ds_lab_num <- labevents2.n[labevents2.n$hadm_id == cur_id,]
	
	t_range<-sort(unique(ds_lab_num$time), decreasing = F)
	ds_lab_num1<-rbind()
	for(cur_t in t_range){
		ds_lab_num_tmp<-ds_lab_num[ds_lab_num$time==cur_t,]
		cur_x_lab<-lapply(names(stat_lab_value_num), function(item){
				
				x<-ds_lab_num_tmp$value[ds_lab_num_tmp$itemid==item]
				m<-median(x,na.rm=T)
				if(is.na(m)){
					return(NA)
				} else{
					return(m)
				}
		}) %>% unlist
		ds_lab_num1<-rbind(ds_lab_num1, cur_x_lab)
	}
	
	row.names(ds_lab_num1)<-NULL
	
	index<-apply(ds_lab_num1, 2, function(x){
		all(is.na(x))
	 }) & is.na(ds_lab_num1[1,])
	
	ds_lab_num1[1, index]<-unlist(lapply(stat_lab_value_num, 
														function(x)x[3]))[index]
														
	ds_lab_num1<-rbind(ds_lab_num1)
	ds_lab_num1<-apply(ds_lab_num1, 2, function(x){fill_NA(x, t_range)})
	
	ds_lab_num1<-as.data.frame(rbind(ds_lab_num1))
	names(ds_lab_num1)<-names(stat_lab_value_num)
	
	ds_lab_num1<-sapply(names(stat_lab_value_num), function(item){
			x<-ds_lab_num1[[as.character(item)]]
			stat<-stat_lab_value_num[[as.character(item)]]
			m<-stat[1]
			s<-stat[2]
			return((x-m)/s)
	}) %>% rbind(.) %>% as.data.frame
	
	ds_lab_num1<-cbind("date"=t_range, ds_lab_num1)
	
	fwrite(x = ds_lab_num1,file=paste0(folder_path,"/labevents2_n.csv"), row.names=F)
	
	#################### chartevents ####################
	
	ds_chart_num <- chartevents1.n[chartevents1.n$hadm_id == cur_id,]

	t_range<-sort(unique(ds_chart_num$time), decreasing = F)

	ds_chart_num1<-rbind()
	for(cur_t in t_range){
		ds_chart_num_tmp<-ds_chart_num[ds_chart_num$time==cur_t,]
		cur_x_chart<-lapply(names(stat_chart_value_num), function(item){
				
				x<-ds_chart_num_tmp$value[ds_chart_num_tmp$itemid==item]
				m<-median(x,na.rm=T)
				if(is.na(m)){
					return(NA)
				} else{
					return(m)
				}
		}) %>% unlist
		ds_chart_num1<-rbind(ds_chart_num1, cur_x_chart)
	}
	
	row.names(ds_chart_num1)<-NULL
	
	index<-apply(ds_chart_num1, 2, function(x){
		all(is.na(x))
	 }) & is.na(ds_chart_num1[1,])
	
	ds_chart_num1[1, index]<-unlist(lapply(stat_chart_value_num, 
														function(x)x[3]))[index]
														
	ds_chart_num1<-rbind(ds_chart_num1)
	ds_chart_num1<-apply(ds_chart_num1, 2, function(x){fill_NA(x, t_range)})
	
	ds_chart_num1<-as.data.frame(rbind(ds_chart_num1))
	names(ds_chart_num1)<-names(stat_chart_value_num)
	
	ds_chart_num1<-sapply(names(stat_chart_value_num), function(item){
			x<-ds_chart_num1[[as.character(item)]]
			stat<-stat_chart_value_num[[as.character(item)]]
			m<-stat[1]
			s<-stat[2]
			return((x-m)/s)
	}) %>% rbind(.) %>% as.data.frame
	
	ds_chart_num1<-cbind("date"=t_range, ds_chart_num1)
	
	fwrite(x = ds_chart_num1,file=paste0(folder_path,"/chartevents1_n.csv"), row.names=F)
			   
	############  prescriptions  ############
	
	ds_pres <- prescriptions1[prescriptions1$hadm_id == cur_id,]

	t_range<-sort(unique(ds_pres$time), decreasing = F)

	ds_pres1<-rbind()
	for (cur_t in t_range) {
	  # cut_t<-t_range[1]
	  ds_tmp<-ds_pres[which(ds_pres$time==cur_t),]
	  ds_pres1<-rbind(ds_pres1,
					 lapply(d_drug$itemid, function(x){
						if(any(x %in% ds_tmp$formulary_drug_cd)) return(1) else return(0)
					}) %>% unlist)
	}

	ds_pres1<- as.data.frame(ds_pres1)
	ds_pres1<-cbind("date"=t_range, ds_pres1)
	
	fwrite(x = ds_pres1,file=paste0(folder_path,"/prescriptions.csv"), row.names=F)
			   
	############  noteevents ############
	
	ds_note <- unique(na.omit(noteevents[which(noteevents$hadm_id==cur_id),]))
	
	t_range<-sort(unique(ds_note$time), decreasing = F)
	
	ds_note1 <- rbind(NULL)
	for (cur_t in t_range) {
		
		# cut_t<-t_range[1]
		ds_tmp<-rbind(ds_note[which(ds_note$time==cur_t),])
		if(nrow(ds_tmp)==1){
		
			ds_note1<-rbind(ds_note1, ds_tmp)
			
		} else{
		
			ds_tmp$text1[1]<-paste0(ds_tmp$text1, collapse=" ")
			ds_tmp$text1[1]<-gsub("  "," ",ds_tmp$text1[1])
			ds_note1<-rbind(ds_note1, ds_tmp[1,])
			
		}
	}
	
	ds_note1<-as.data.frame(rbind(ds_note1))
	
	fwrite(x = ds_note1,
			   file=paste0(folder_path,"/noteevents.csv"), row.names=F)
			   
	fwrite(x = admissions2[which(admissions2$hadm_id==cur_id),],
			   file=paste0(folder_path,"/admissions.csv"), row.names=F)
			   
	if(i %% 10 == 0) print(paste0(i ,' | ', length(ids)))
	
}



