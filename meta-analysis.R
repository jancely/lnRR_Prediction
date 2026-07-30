################Fig1#######
install.packages("ggplot2")
library(ggplot2)
library(grid)
library(sysfonts)
library(showtextdb)
library(showtext)
library(Matrix)
library(lme4)
library(lmerTest)
packageVersion("lme4")
packageVersion("boot")
packageVersion("randomForst")
packageVersion("akima")
packageVersion("piecewiseSEM")
packageVersion("ggplot2")
datalm1=read.csv("ALL-3.csv",sep=",",header=TRUE)
datalm1$index=as.factor(datalm1$index)
datalm1$lnRR=as.numeric(as.character(datalm1$lnRR))
p1<-ggplot(data = datalm1,aes(x=index,y=lnRR,fill=index, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=3, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"All Intercropping"=24,"All Rotation"=25))+
geom_text(aes(x = index, y = up+1, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3 ,check_overlap = FALSE)+
scale_y_continuous(limits=c(-108,61), breaks = c(-100,-80,-60,-40,-20,0,20,40,60))+
scale_x_discrete(breaks=c("NGHGB","CH4 emission","N2O emission","CO2 emission","indirect N2O","Nitrate leaching","SOC stock"),labels=c("NGHGB",expression(paste("C",H[4]," ","emission")),expression(paste(N[2],"O"," ","emission")),expression(paste("C",O[2]," ","emission")),expression(paste("indirect"," ",N[2],"O"," ","emission")),"Nitrate leaching","SOC stock"))+
labs(x = "", y = "Change ( % )",colour = 'black')+
theme(legend.title = element_blank(),
    legend.position=c(0.8,0.7),
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    #axis.text.y = element_blank(),
    axis.text.y = element_text(colour = 'black', size = 12),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p1
##################
datalm1=read.csv("ALL-4.csv",sep=",",header=TRUE)
datalm1$index=as.factor(datalm1$index)
datalm1$lnRR=as.numeric(as.character(datalm1$lnRR))
p1<-ggplot(data = datalm1,aes(x=index,y=lnRR,fill=index, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=3, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = index, y = up+1, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3 ,check_overlap = FALSE)+
scale_y_continuous(limits=c(-85,65), breaks = c(-80,-60,-40,-20,0,20,40,60))+
scale_x_discrete(breaks=c("CH4 emission","CO2 emission","indirect N2O","N2O emission","NGHGB","Nitrate leaching","SOC stock"),labels=c(expression(paste("C",H[4]," ","emission")),expression(paste("C",O[2]," ","emission")),expression(paste("indirect"," ",N[2],"O"," ","emission")),expression(paste(N[2],"O"," ","emission")),"NGHGB","Nitrate leaching","SOC stock"))+
labs(x = "", y = "Change ( % )",colour = 'black')+
theme(legend.title = element_blank(),
    legend.position=c(0.8,0.7),
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    #axis.text.y = element_blank(),
    axis.text.y = element_text(colour = 'black', size = 12),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p1
ggsave("Fig1.pdf", plot = p1, device = cairo_pdf, dpi = 600,width = 6, height = 6)
###################################################################
#################Fig2##########
datalm1=read.csv("SOC-MAP.csv",sep=",",header=TRUE)
datalm1$MAP=as.factor(datalm1$MAP)
datalm1$lnRR=as.numeric(as.character(datalm1$lnRR))
p1<-ggplot(data = datalm1,aes(x=MAP,y=lnRR,fill=MAP, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = MAP, y = up+1, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3 ,check_overlap = FALSE)+
scale_y_continuous(limits=c(-8,40), breaks = c(-5,0,10,20,30,40))+
scale_x_discrete(limits=c(">1600","","800~1600","","400~800","","<400","","All"))+
labs(title=" ",x = "Mean annual precipitation", y = "SOC stock ( % change )",colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    #axis.text.y = element_blank(),
    axis.text.y = element_text(colour = 'black', size = 12),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p1
#ggsave("Fa2-1.pdf", plot = p1, device = cairo_pdf, dpi = 600)
#######
###NL#####
datalm2=read.csv("NL-MAP.csv",sep=",",header=TRUE)
datalm2$MAP=as.factor(datalm2$MAP)
datalm2$lnRR=as.numeric(as.character(datalm2$lnRR))
p2<-ggplot(data = datalm2,aes(x=MAP,y=lnRR,fill=MAP, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = MAP, y = up+1, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3, check_overlap = FALSE)+
scale_y_continuous(limits=c(-43,83), breaks = c(-40,0,40,60,80))+
scale_x_discrete(limits=c(">1600","","800~1600","","400~800","","<400","","All"))+
labs(title="",x = "", y = "Nitrate leaching ( % change )",colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="right",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    #axis.text.y = element_text(colour = 'black', size = 16),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p2
#ggsave("Fa2-2.pdf", plot = p2, device = cairo_pdf, dpi = 600)
##############
####CO2#####
datalm3=read.csv("CO2-MAP.csv",sep=",",header=TRUE)
datalm3$MAP=as.factor(datalm3$MAP)
datalm3$lnRR=as.numeric(as.character(datalm3$lnRR))
p3<-ggplot(data = datalm3,aes(x=MAP,y=lnRR,fill=MAP, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = MAP, y = up+1, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3, check_overlap = FALSE)+
scale_y_continuous(limits=c(-45,33), breaks = c(-40,-30,-20,-10,0,10,20,30))+
scale_x_discrete(limits=c(">1600","","800~1600","","400~800","","<400","","All"))+
labs(title="",x = "Mean annual precipitation (mm)", y = expression(paste(CO[2]," ","emission"," ","(","%"," ","change",")")),colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    #axis.text.y = element_blank(),
    axis.text.y = element_text(colour = 'black', size = 12),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p3
#ggsave("Fa2-3.pdf", plot = p3, device = cairo_pdf, dpi = 600)
##########
###N2O######
datalm4=read.csv("N2O-MAP.csv",sep=",",header=TRUE)
datalm4$MAP=as.factor(datalm4$MAP)
datalm4$lnRR=as.numeric(as.character(datalm4$lnRR))
p4<-ggplot(data = datalm4,aes(x=MAP,y=lnRR,fill=MAP, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = MAP, y = up+2, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3, check_overlap = FALSE)+
scale_y_continuous(limits=c(-17,30), breaks = c(-20,-10,0,10,20,30))+
scale_x_discrete(limits=c(">1600","","800~1600","","400~800","","<400","","All"))+
labs(title="",x = "", y = expression(paste(N[2],"O"," ","emission"," ","(","%"," ","change",")")),colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    #axis.text.y = element_text(colour = 'black', size = 16),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p4
#ggsave("Fa2-4.pdf", plot = p4, device = cairo_pdf, dpi = 600)
###########
#####CH4#####
datalm5=read.csv("CH4-MAP.csv",sep=",",header=TRUE)
datalm5$MAP=as.factor(datalm5$MAP)
datalm5$lnRR=as.numeric(as.character(datalm5$lnRR))
p5<-ggplot(data = datalm5,aes(x=MAP,y=lnRR,fill=MAP, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = MAP, y = up+2, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3, check_overlap = FALSE)+
scale_y_continuous(limits=c(-45,123), breaks = c(-40,0,40,80,120))+
scale_x_discrete(limits=c(">1600","","800~1600","","400~800","","<400","","All"))+
labs(title="",x = "", y = expression(paste("C",H[4]," ","emission"," ","(","%"," ","change",")")),colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    #axis.text.y = element_text(colour = 'black', size = 16),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p5
#ggsave("Fa2-5.pdf", plot = p5, device = cairo_pdf, dpi = 600)
#
ggsave("F2a-1.pdf", plot = p1, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F2a-2.pdf", plot = p2, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F2a-3.pdf", plot = p3, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F2a-4.pdf", plot = p4, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F2a-5.pdf", plot = p5, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
grid.newpage()
layout_1<-grid.layout(nrow = 2,ncol = 4,widths = c(1,1),heights=c(5,5))
pushViewport(viewport(layout = layout_1))
print(p1,vp=viewport(layout.pos.row = 1,layout.pos.col = c(1,2)))
print(p2,vp=viewport(layout.pos.row = 1,layout.pos.col = c(3,4)))
print(p3,vp=viewport(layout.pos.row = 2,layout.pos.col = c(1,2)))
print(p4,vp=viewport(layout.pos.row = 2,layout.pos.col = 3))
print(p5,vp=viewport(layout.pos.row = 2,layout.pos.col = 4))
#################################################
#######################################################
###NL#####
datalm2=read.csv("NL-MAT.csv",sep=",",header=TRUE)
datalm2$MAT=as.factor(datalm2$MAT)
datalm2$lnRR=as.numeric(as.character(datalm2$lnRR))
p2<-ggplot(data = datalm2,aes(x=MAT,y=lnRR,fill=MAT, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = MAT, y = up+2, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3, check_overlap = FALSE)+
scale_y_continuous(limits=c(-40,168), breaks = c(-40,0,40,80,120,160))+
scale_x_discrete(limits=c(">15","","10~15","","5~10","","<5","","All"))+
labs(title="",x = "", y = "Nitrate leaching ( % change )",colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="right",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    #axis.text.y = element_text(colour = 'black', size = 16),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p2
#ggsave("F2b-2.pdf", plot = p2, device = cairo_pdf, dpi = 600)
##############
####CO2#####
datalm3=read.csv("CO2-MAT.csv",sep=",",header=TRUE)
datalm3$MAT=as.factor(datalm3$MAT)
datalm3$lnRR=as.numeric(as.character(datalm3$lnRR))
p3<-ggplot(data = datalm3,aes(x=MAT,y=lnRR,fill=MAT, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = MAT, y = up+2, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3, check_overlap = FALSE)+
scale_y_continuous(limits=c(-32,46), breaks = c(-30,-20,-10,0,10,20,30,40))+
scale_x_discrete(limits=c(">15","","10~15","","5~10","","<5","","All"))+
labs(title="",x = "Mean annual temperature(°C)", y = expression(paste(CO[2]," ","emission"," ","(","%"," ","change",")")),colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    #axis.text.y = element_blank(),
    axis.text.y = element_text(colour = 'black', size = 12),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p3
#ggsave("F2b-3.pdf", plot = p3, device = cairo_pdf, dpi = 600)
##########
###N2O######
datalm4=read.csv("N2O-MAT.csv",sep=",",header=TRUE)
datalm4$MAT=as.factor(datalm4$MAT)
datalm4$lnRR=as.numeric(as.character(datalm4$lnRR))
p4<-ggplot(data = datalm4,aes(x=MAT,y=lnRR,fill=MAT, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = MAT, y = up+2, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3, check_overlap = FALSE)+
scale_y_continuous(limits=c(-25,46), breaks = c(-20,-10,0,10,20,30,40,50))+
scale_x_discrete(limits=c(">15","","10~15","","5~10","","<5","","All"))+
labs(title="",x = "", y = expression(paste(N[2],"O"," ","emission"," ","(","%"," ","change",")")),colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    #axis.text.y = element_text(colour = 'black', size = 16),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p4
#ggsave("F2b-4.pdf", plot = p4, device = cairo_pdf, dpi = 600)
###########
#####CH4#####
datalm5=read.csv("CH4-MAT.csv",sep=",",header=TRUE)
datalm5$MAT=as.factor(datalm5$MAT)
datalm5$lnRR=as.numeric(as.character(datalm5$lnRR))
p5<-ggplot(data = datalm5,aes(x=MAT,y=lnRR,fill=MAT, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = MAT, y = up+2, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3, check_overlap = FALSE)+
scale_y_continuous(limits=c(-63,90), breaks = c(-60,-30,0,30,60,90))+
scale_x_discrete(limits=c(">15","","10~15","","5~10","","<5","","All"))+
labs(title="",x = "", y = expression(paste("C",H[4]," ","emission"," ","(","%"," ","change",")")),colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    #axis.text.y = element_text(colour = 'black', size = 16),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p5
#ggsave("F2b-5.pdf", plot = p5, device = cairo_pdf, dpi = 600)
#
ggsave("F2b-1.pdf", plot = p1, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F2b-2.pdf", plot = p2, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F2b-3.pdf", plot = p3, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F2b-4.pdf", plot = p4, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F2b-5.pdf", plot = p5, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
grid.newpage()
layout_1<-grid.layout(nrow = 2,ncol = 4,widths = c(1,1),heights=c(5,5))
pushViewport(viewport(layout = layout_1))
print(p1,vp=viewport(layout.pos.row = 1,layout.pos.col = c(1,2)))
print(p2,vp=viewport(layout.pos.row = 1,layout.pos.col = c(3,4)))
print(p3,vp=viewport(layout.pos.row = 2,layout.pos.col = c(1,2)))
print(p4,vp=viewport(layout.pos.row = 2,layout.pos.col = 3))
print(p5,vp=viewport(layout.pos.row = 2,layout.pos.col = 4))
############
datalm1=read.csv("SOCALL2-lm3Texture3.csv",sep=",",header=TRUE)
datalm1$Texture=as.factor(datalm1$Texture)
datalm1$lnRR=as.numeric(as.character(datalm1$lnRR))
p1<-ggplot(data = datalm1,aes(x=Texture,y=lnRR,fill=Texture, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = Texture, y = up+1, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3 ,check_overlap = FALSE)+
scale_y_continuous(limits=c(-1,15), breaks = c(0,5,10,15))+
scale_x_discrete(limits=c("Sand","","Loam","","Clay","","All"))+
labs(title="",x = "Soil texture", y = "SOC stock ( % change )",colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    #axis.text.y = element_blank(),
    axis.text.y = element_text(colour = 'black', size = 12),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p1
#ggsave("F2c-1.pdf", plot = p1, device = cairo_pdf, dpi = 600)
####NL######
datalm2=read.csv("NLALL3-lm3Texture3.csv",sep=",",header=TRUE)
datalm2$Texture=as.factor(datalm2$Texture)
datalm2$lnRR=as.numeric(as.character(datalm2$lnRR))
p2<-ggplot(data = datalm2,aes(x=Texture,y=lnRR,fill=Texture, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = Texture, y = up+1, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3, check_overlap = FALSE)+
scale_y_continuous(limits=c(-32,20), breaks = c(-30,-20,-10,0,10,20))+
scale_x_discrete(limits=c("Sand","","Loam","","Clay","","All"))+
labs(title="",x = "", y = "Nitrate leaching ( % change )",colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="right",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    #axis.text.y = element_text(colour = 'black', size = 16),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p2
#ggsave("F2c-2.pdf", plot = p2, device = cairo_pdf, dpi = 600)
#####CO2#########
datalm3=read.csv("CO2ALL2-lm3Texture3.csv",sep=",",header=TRUE)
datalm3$Texture=as.factor(datalm3$Texture)
datalm3$lnRR=as.numeric(as.character(datalm3$lnRR))
p3<-ggplot(data = datalm3,aes(x=Texture,y=lnRR,fill=Texture, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = Texture, y = up+1, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3, check_overlap = FALSE)+
scale_y_continuous(limits=c(-20,15), breaks = c(-20,-15,-10,-5,0,5,10,15))+
scale_x_discrete(limits=c("Sand","","Loam","","Clay","","All"))+
labs(title="",x = "Soil texture", y = expression(paste(CO[2]," ","emission"," ","(","%"," ","change",")")),colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    #axis.text.y = element_blank(),
    axis.text.y = element_text(colour = 'black', size = 12),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p3
#ggsave("F2c-3.pdf", plot = p3, device = cairo_pdf, dpi = 600)
#####N2O###
datalm4=read.csv("N2OALL2-lm3Texture3.csv",sep=",",header=TRUE)
datalm4$Texture=as.factor(datalm4$Texture)
datalm4$lnRR=as.numeric(as.character(datalm4$lnRR))
p4<-ggplot(data = datalm4,aes(x=Texture,y=lnRR,fill=Texture, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = Texture, y = up+1, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3, check_overlap = FALSE)+
scale_y_continuous(limits=c(-15,8), breaks = c(-15,-10,-5,0,5))+
scale_x_discrete(limits=c("Sand","","Loam","","Clay","","All"))+
labs(title="",x = "", y = expression(paste(N[2],"O"," ","emission"," ","(","%"," ","change",")")),colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    #axis.text.y = element_text(colour = 'black', size = 16),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p4
#ggsave("F2c-4.pdf", plot = p4, device = cairo_pdf, dpi = 600)
#####CH4######
datalm5=read.csv("CH4ALL2-lm3Texture3.csv",sep=",",header=TRUE)
datalm5$Texture=as.factor(datalm5$Texture)
datalm5$lnRR=as.numeric(as.character(datalm5$lnRR))
p5<-ggplot(data = datalm5,aes(x=Texture,y=lnRR,fill=Texture, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = Texture, y = up+1, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3, check_overlap = FALSE)+
scale_y_continuous(limits=c(-25,53), breaks = c(-20,0,20,40,60))+
scale_x_discrete(limits=c("Sand","","Loam","","Clay","","All"))+
labs(title="",x = "", y = expression(paste("C",H[4]," ","emission"," ","(","%"," ","change",")")),colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    #axis.text.y = element_text(colour = 'black', size = 16),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p5
#ggsave("F2c-5.pdf", plot = p5, device = cairo_pdf, dpi = 600)
ggsave("F2c-1.pdf", plot = p1, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F2c-2.pdf", plot = p2, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F2c-3.pdf", plot = p3, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F2c-4.pdf", plot = p4, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F2c-5.pdf", plot = p5, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
grid.newpage()
layout_1<-grid.layout(nrow = 2,ncol = 4,widths = c(1,1),heights=c(5,5))
pushViewport(viewport(layout = layout_1))
print(p1,vp=viewport(layout.pos.row = 1,layout.pos.col = c(1,2)))
print(p2,vp=viewport(layout.pos.row = 1,layout.pos.col = c(3,4)))
print(p3,vp=viewport(layout.pos.row = 2,layout.pos.col = c(1,2)))
print(p4,vp=viewport(layout.pos.row = 2,layout.pos.col = 3))
print(p5,vp=viewport(layout.pos.row = 2,layout.pos.col = 4))
###################################################################################
###############Fig3##############
datalm1=read.csv("SOCALL2-lm1family1-3.csv",sep=",",header=TRUE)
datalm1$crop=as.factor(datalm1$crop)
datalm1$lnRR=as.numeric(as.character(datalm1$lnRR))
p1<-ggplot(data = datalm1,aes(x=crop,y=lnRR,fill=crop, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = crop, y = up+1, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3 ,check_overlap = FALSE)+
scale_y_continuous(limits=c(-5,45), breaks = c(-5,0,10,20,30,40))+
#scale_x_discrete(limits=c("Poaceae+Fallow","","excluded Poaceae","","excluded Leguminosae","","Poaceae+Leguminosae","","only included Poaceae","","All"))+
scale_x_discrete(limits=c("excluded Poaceae","","excluded Leguminosae","","Poaceae+Leguminosae","","only included Poaceae","","All"))+
labs(title="",x = "Crop combination", y = "SOC stock ( % change )",colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    #axis.text.y = element_blank(),
    axis.text.y = element_text(colour = 'black', size = 12),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p1
#ggsave("F3a-1.pdf", plot = p1, device = cairo_pdf, dpi = 600,width = 5, height = 2.5)
####NL#########
datalm2=read.csv("NLALL3-lm1family1-3.csv",sep=",",header=TRUE)
datalm2$crop=as.factor(datalm2$crop)
datalm2$lnRR=as.numeric(as.character(datalm2$lnRR))
p2<-ggplot(data = datalm2,aes(x=crop,y=lnRR,fill=crop, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = crop, y = up+1, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3, check_overlap = FALSE)+
scale_y_continuous(limits=c(-55,10), breaks = c(-50,-40,-30,-20,-10,0,10))+
scale_x_discrete(limits=c("Poaceae+Fallow","","excluded Poaceae","","excluded Leguminosae","","Poaceae+Leguminosae","","only included Poaceae","","All"))+
labs(title="",x = "", y = "Nitrate leaching ( % change )",colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="right",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    #axis.text.y = element_text(colour = 'black', size = 16),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p2
#ggsave("F3a-2.pdf", plot = p2, device = cairo_pdf, dpi = 600,width = 5, height = 2.5)
#######CO2######
datalm3=read.csv("CO2ALL2-lm1family1-3.csv",sep=",",header=TRUE)
datalm3$crop=as.factor(datalm3$crop)
datalm3$lnRR=as.numeric(as.character(datalm3$lnRR))
p3<-ggplot(data = datalm3,aes(x=crop,y=lnRR,fill=crop, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = crop, y = up+1, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3, check_overlap = FALSE)+
scale_y_continuous(limits=c(-17,110), breaks = c(-20,0,20,40,60,80,100))+
scale_x_discrete(limits=c("excluded Poaceae","","excluded Leguminosae","","Poaceae+Leguminosae","","only included Poaceae","","All"))+
labs(title="",x = "Crop combination", y = expression(paste(CO[2]," ","emission"," ","(","%"," ","change",")")),colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    #axis.text.y = element_blank(),
    axis.text.y = element_text(colour = 'black', size = 12),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p3
#ggsave("F3a-3.pdf", plot = p3, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
####N2O######
datalm4=read.csv("N2OALL2-lm1family1-3.csv",sep=",",header=TRUE)
datalm4$crop=as.factor(datalm4$crop)
datalm4$lnRR=as.numeric(as.character(datalm4$lnRR))
p4<-ggplot(data = datalm4,aes(x=crop,y=lnRR,fill=crop, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = crop, y = up+1, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3, check_overlap = FALSE)+
scale_y_continuous(limits=c(-17,10), breaks = c(-15,-10,-5,0,5,10))+
scale_x_discrete(limits=c("excluded Poaceae","","excluded Leguminosae","","Poaceae+Leguminosae","","only included Poaceae","","All"))+
labs(title="",x = "", y = expression(paste(N[2],"O"," ","emission"," ","(","%"," ","change",")")),colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    #axis.text.y = element_text(colour = 'black', size = 16),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p4
#ggsave("F3a-4.pdf", plot = p4, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
#####CH4####
datalm5=read.csv("CH4ALL2-lm1family1-3.csv",sep=",",header=TRUE)
datalm5$crop=as.factor(datalm5$crop)
datalm5$lnRR=as.numeric(as.character(datalm5$lnRR))
p5<-ggplot(data = datalm5,aes(x=crop,y=lnRR,fill=crop, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = crop, y = up+1, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3, check_overlap = FALSE)+
scale_y_continuous(limits=c(-38,60), breaks = c(-40,-20,0,20,40,60))+
scale_x_discrete(limits=c("excluded Poaceae","","excluded Leguminosae","","Poaceae+Leguminosae","","only included Poaceae","","All"))+
labs(title="",x = "", y = expression(paste("C",H[4]," ","emission"," ","(","%"," ","change",")")),colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    #axis.text.y = element_text(colour = 'black', size = 16),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p5
#ggsave("F3a-5.pdf", plot = p5, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F3a-1.pdf", plot = p1, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F3a-2.pdf", plot = p2, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F3a-3.pdf", plot = p3, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F3a-4.pdf", plot = p4, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F3a-5.pdf", plot = p5, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
grid.newpage()
layout_1<-grid.layout(nrow = 2,ncol = 4,widths = c(1,1),heights=c(5,5))
pushViewport(viewport(layout = layout_1))
print(p1,vp=viewport(layout.pos.row = 1,layout.pos.col = c(1,2)))
print(p2,vp=viewport(layout.pos.row = 1,layout.pos.col = c(3,4)))
print(p3,vp=viewport(layout.pos.row = 2,layout.pos.col = c(1,2)))
print(p4,vp=viewport(layout.pos.row = 2,layout.pos.col = 3))
print(p5,vp=viewport(layout.pos.row = 2,layout.pos.col = 4))
###################################
###图3-13######
datalm1=read.csv("SOCALL2-lm6FNF3.csv",sep=",",header=TRUE)
datalm1$FNF=as.factor(datalm1$FNF)
datalm1$lnRR=as.numeric(as.character(datalm1$lnRR))
p1<-ggplot(data = datalm1,aes(x=FNF,y=lnRR,fill=FNF, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = FNF, y = up+1, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3 ,check_overlap = FALSE)+
scale_y_continuous(limits=c(-7,32), breaks = c(-5,0,10,20,30))+
scale_x_discrete(limits=c(">200","","101~200","","0~100","","All"))+
labs(title="",x ="Nitrogen", y = "SOC stock ( % change )",colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_text(size = 10),
    #axis.title.y = element_blank(),
    #axis.text.y = element_blank(),
    axis.text.y = element_text(colour = 'black', size = 12),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p1
#####NL####
datalm2=read.csv("NLALL3-FNF3.csv",sep=",",header=TRUE)
datalm2$FNF=as.factor(datalm2$FNF)
datalm2$lnRR=as.numeric(as.character(datalm2$lnRR))
p2<-ggplot(data = datalm2,aes(x=FNF,y=lnRR,fill=FNF, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = FNF, y = up+1, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3 ,check_overlap = FALSE)+
scale_y_continuous(limits=c(-42,68), breaks = c(-40,-20,0,20,40,60))+
scale_x_discrete(limits=c(">200","","101~200","","0~100","","All"))+
labs(title="",x ="", y = "Nitrate leaching ( % change )",colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="right",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    #axis.text.y = element_text(colour = 'black', size = 12),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p2
#####CO2#####
datalm3=read.csv("CO2ALL2-lm6FNF3.csv",sep=",",header=TRUE)
datalm3$FNF=as.factor(datalm3$FNF)
datalm3$lnRR=as.numeric(as.character(datalm3$lnRR))
p3<-ggplot(data = datalm3,aes(x=FNF,y=lnRR,fill=FNF, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"All Intercropping"=24,"All Rotation"=25,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = FNF, y = up+1, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3, check_overlap = FALSE)+
scale_y_continuous(limits=c(-32,5), breaks = c(-30,-20,-10,0,5))+
scale_x_discrete(limits=c(">200","","101~200","","0~100","","All"))+
labs(title="",x = expression(paste("Nitrogen application rate"," ","(","kg"," ","N"," ",ha^{-1},")")), y = expression(paste(CO[2]," ","emission"," ","(","%"," ","change",")")),colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    # axis.title.y = element_text(size = 10),
    #axis.title.y = element_blank(),
    #axis.text.y = element_blank(),
    axis.text.y = element_text(colour = 'black', size = 12),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p3
#####N2O####
datalm4=read.csv("N2OALL2-lm6FNF3.csv",sep=",",header=TRUE)
datalm4$FNF=as.factor(datalm4$FNF)
datalm4$lnRR=as.numeric(as.character(datalm4$lnRR))
p4<-ggplot(data = datalm4,aes(x=FNF,y=lnRR,fill=FNF, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"All Intercropping"=24,"All Rotation"=25,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = FNF, y = up+1, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3, check_overlap = FALSE)+
scale_y_continuous(limits=c(-32,20), breaks = c(-30,-20,-10,0,10,20))+
scale_x_discrete(limits=c(">200","","101~200","","0~100","","All"))+
labs(title="",x ="", y = expression(paste(N[2],"O"," ","emission"," ","(","%"," ","change",")")),colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    #axis.text.y = element_text(colour = 'black', size = 16),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p4
###CH4###
datalm5=read.csv("CH4ALL2-lm6FNF3.csv",sep=",",header=TRUE)
datalm5$FNF=as.factor(datalm5$FNF)
datalm5$lnRR=as.numeric(as.character(datalm5$lnRR))
p5<-ggplot(data = datalm5,aes(x=FNF,y=lnRR,fill=FNF, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"All Intercropping"=24,"All Rotation"=25,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = FNF, y = up+1, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3, check_overlap = FALSE)+
scale_y_continuous(limits=c(-62,120), breaks = c(-40,0,40,80,120))+
scale_x_discrete(limits=c(">200","","101~200","","0~100","","All"))+
labs(title="",x = "", y = expression(paste("C",H[4]," ","emission"," ","(","%"," ","change",")")),colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    #axis.text.y = element_text(colour = 'black', size = 16),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p5
ggsave("F3b-1.pdf", plot = p1, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F3b-2.pdf", plot = p2, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F3b-3.pdf", plot = p3, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F3b-4.pdf", plot = p4, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F3b-5.pdf", plot = p5, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
grid.newpage()
layout_1<-grid.layout(nrow = 2,ncol = 4,widths = c(1,1),heights=c(5,5))
pushViewport(viewport(layout = layout_1))
print(p1,vp=viewport(layout.pos.row = 1,layout.pos.col = c(1,2)))
print(p2,vp=viewport(layout.pos.row = 1,layout.pos.col = c(3,4)))
print(p3,vp=viewport(layout.pos.row = 2,layout.pos.col = c(1,2)))
print(p4,vp=viewport(layout.pos.row = 2,layout.pos.col = 3))
print(p5,vp=viewport(layout.pos.row = 2,layout.pos.col = 4))

############图3-14###########################
datalm1=read.csv("SOCALL2-LCR6.csv",sep=",",header=TRUE)
datalm1$LCR7=as.factor(datalm1$LCR7)
datalm1$lnRR7=as.numeric(as.character(datalm1$lnRR7))
p1<-ggplot(data = datalm1,aes(x=LCR7,y=lnRR7,fill=LCR7, shape=Planting7)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low7, ymax = up7), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Rotation"=21))+
geom_text(aes(x = LCR7, y = up7+0.5, label = samplesize7),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3 ,check_overlap = FALSE)+
scale_y_continuous(limits=c(-2,10), breaks = c(-2,0,2,4,6,8,10))+
scale_x_discrete(limits=c(">30","","20~30","","10~20","","3~10","","All"))+
labs(title="",x =expression(paste("Duration (year)")), y = "SOC stock ( % change )",colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    #axis.text.y = element_blank(),
    axis.text.y = element_text(colour = 'black', size = 12),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p1
#####间作####
datalm1=read.csv("SOCALL2-LCR6.csv",sep=",",header=TRUE)
datalm1$LCR8=as.factor(datalm1$LCR8)
datalm1$lnRR8=as.numeric(as.character(datalm1$lnRR8))
p2<-ggplot(data = datalm1,aes(x=LCR8,y=lnRR8,fill=LCR8, shape=Planting8)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low8, ymax = up8), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22))+
geom_text(aes(x = LCR8, y = up8+1, label = samplesize8),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3 ,check_overlap = FALSE)+
scale_y_continuous(limits=c(-1,52), breaks = c(0,10,20,30,40,50))+
scale_x_discrete(limits=c(">9","6~9","3~6","","All"))+
labs(title="",x =expression(paste("Duration (year)")), y = "SOC stock ( % change )",colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    #axis.text.y = element_blank(),
    axis.text.y = element_text(colour = 'black', size = 12),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p2
ggsave("F3c-1.pdf", plot = p1, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
ggsave("F3c-2.pdf", plot = p2, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
##############
####water#####
datalm3=read.csv("NLALL3-lm5water-2.csv",sep=",",header=TRUE)
datalm3$water=as.factor(datalm3$water)
datalm3$lnRR=as.numeric(as.character(datalm3$lnRR))
p3<-ggplot(data = datalm3,aes(x=water,y=lnRR,fill=water, shape=Planting)) + 
geom_hline(yintercept=0,linetype = "dashed",size=0.3)+
geom_errorbar(position=position_dodge(-0.8),aes(ymin = low, ymax = up), width=0.3,size=0.3)+
geom_point(position=position_dodge(-0.8), size=4, stroke = 0.3) + 
scale_shape_manual(values=c("All"=23,"Intercropping"=22,"Rotation"=21))+
geom_text(aes(x = water, y = up+1, label = samplesize),
        position = position_dodge(width = -0.9),vjust = 0.4, hjust=0, size = 3, check_overlap = FALSE)+
scale_y_continuous(limits=c(-45,27), breaks = c(-40,-30,-20,-10,0,10,20))+
scale_x_discrete(limits=c("Irrigation","","Rainfed","","All"))+
labs(title="",x = expression(paste("Water management")), y = "Nitrate leaching ( % change )",colour = 'black')+
theme(legend.title = element_blank(),
    legend.position="none",
    legend.key = element_rect(fill = "white",size = 4),
    legend.background = element_blank(),
    legend.text=element_text(size=12),
    panel.background = element_rect(fill = 'white', colour = 'white'),
    axis.title=element_text(size=13),
    #axis.title.y = element_blank(),
    #axis.text.y = element_blank(),
    axis.text.y = element_text(colour = 'black', size = 12),
    axis.text.x = element_text(colour = 'black', size = 12),
    axis.line = element_line(colour = 'black',size=0.6),
    axis.line.y = element_blank(),
    axis.ticks = element_line(colour = 'black',size=0.6),
    axis.ticks.y = element_blank())+
#geom_segment(x = 4.6, y = -0.3, xend = 4.6, yend = 0.5, colour = "black",size=0.8)+
guides(fill = "none")+
coord_flip()
p3
ggsave("F3d.pdf", plot = p3, device = cairo_pdf, dpi = 600,width = 5.0, height = 2.5)
grid.newpage()
layout_1<-grid.layout(nrow = 2,ncol = 5,widths = c(1,1),heights=c(5,5))
pushViewport(viewport(layout = layout_1))
print(p1,vp=viewport(layout.pos.row = c(1,2),layout.pos.col = c(1,2,3)))
print(p2,vp=viewport(layout.pos.row = 2,layout.pos.col = c(4,5)))
######################################################################################
##############Fig4#######################
library(base)
library(graphics)
library(stringr)
library(data.table)
library(stats)
library(rJava)
library(xlsx)
library(xlsxjars)
library(readxl)
library(openxlsx)
library(stringi)
library(akima)
library(grDevices)
library(fields)
library(Hmisc)
library(sp)
library(utils)
library(pointr)
library(fBasics)
library(ggplot2)
library(RColorBrewer)
data1=read.csv("SOClnRR-akima-ID1.csv",sep=",",header=TRUE)###
head(data1)
dim(data1)
Xrange<-max(data1$ISOC)-min(data1$ISOC)
Yrange<-max(data1$Duration)-min(data1$Duration)
Zrange<-max(data1$lnRR)-min(data1$lnRR)
Xrange
Yrange
Zrange
Xmax<-ifelse(max(data1$ISOC)==0,0.1*abs(Xrange),round(max(data1$ISOC)+0.1*abs(Xrange),1))
Xmin<-ifelse(min(data1$ISOC)==0,-0.1*abs(Xrange),round(min(data1$ISOC)-0.1*abs(Xrange),1))
Xmax
Xmin
Ymax<-ifelse(max(data1$Duration)==0,0.1*abs(Yrange),round(max(data1$Duration)+0.1*abs(Yrange),1))
Ymin<-ifelse(min(data1$Duration)==0,-0.1*abs(Yrange),round(min(data1$Duration)-0.1*abs(Yrange),1))
Ymax
Ymin

interplinearS<-interp(data1$ISOC,data1$Duration,data1$lnRR,nx=500,ny=500,linear=TRUE,duplicate = 'mean')
interplinearS$x
interplinearS$y
interplinearS$z

lgnd=1
lgndmar=5.1
lgnd
lgndmar

library(Cairo)
cairo_pdf("F4a.pdf",width = 4.6,height = 3.6)
p1<-image.plot(interplinearS,main=expression(paste("(a)"," ","lnRR"," ","(","SOC stock",")")),col.main='black',xlim=c(Xmin,Xmax),ylim=c(Ymin,Ymax),xlab=expression(paste("Initial SOC"," ","(","g"," "," ",kg^{-1},")")),ylab='Duration (year)',
           legend.shrink=1,legend.width=1,legend.mar=5.1,horizontal=FALSE)
p1+scale_color_distiller(palette="Spectral")
p1+scale_color_distiller(palette="Greens")
p1
dev.off()

####SOC:ISOC+Duration#####
data1=read.csv("SOClnRR-akima-ID2.csv",sep=",",header=TRUE)###
head(data1)
dim(data1)
Xrange<-max(data1$ISOC)-min(data1$ISOC)
Yrange<-max(data1$Duration)-min(data1$Duration)
Zrange<-max(data1$lnRR)-min(data1$lnRR)
Xrange
Yrange
Zrange
Xmax<-ifelse(max(data1$ISOC)==0,0.1*abs(Xrange),round(max(data1$ISOC)+0.1*abs(Xrange),1))
Xmin<-ifelse(min(data1$ISOC)==0,-0.1*abs(Xrange),round(min(data1$ISOC)-0.1*abs(Xrange),1))
Xmax
Xmin
Ymax<-ifelse(max(data1$Duration)==0,0.1*abs(Yrange),round(max(data1$Duration)+0.1*abs(Yrange),1))
Ymin<-ifelse(min(data1$Duration)==0,-0.1*abs(Yrange),round(min(data1$Duration)-0.1*abs(Yrange),1))
Ymax
Ymin

interplinearS<-interp(data1$ISOC,data1$Duration,data1$lnRR,nx=500,ny=500,linear=TRUE,duplicate = 'mean')
interplinearS$x
interplinearS$y
interplinearS$z

lgnd=1
lgndmar=5.1
lgnd
lgndmar

#cairo_pdf("F4b.pdf",width = 4.6,height = 3.6)
p1<-image.plot(interplinearS,main=expression(paste("(b)"," ","lnRR"," ","(","SOC stock",")")),col.main='black',xlim=c(Xmin,Xmax),ylim=c(Ymin,Ymax),xlab=expression(paste("Initial SOC"," ","(","g"," "," ",kg^{-1},")")),ylab='Duration (year)',
           legend.shrink=1,legend.width=1,legend.mar=5.1,horizontal=FALSE)
p1+scale_color_distiller(palette="Spectral")
p1+scale_color_distiller(palette="Greens")
p1
#dev.off()
#####################
####NL:ISOC+CLAY#####
data1=read.csv("NLlnRR-akima-IC1.csv",sep=",",header=TRUE)###
head(data1)
dim(data1)
Xrange<-max(data1$ISOC)-min(data1$ISOC)
Yrange<-max(data1$CLAY)-min(data1$CLAY)
Zrange<-max(data1$lnRR)-min(data1$lnRR)
Xrange
Yrange
Zrange
Xmax<-ifelse(max(data1$ISOC)==0,0.1*abs(Xrange),round(max(data1$ISOC)+0.1*abs(Xrange),1))
Xmin<-ifelse(min(data1$ISOC)==0,-0.1*abs(Xrange),round(min(data1$ISOC)-0.1*abs(Xrange),1))
Xmax
Xmin
Ymax<-ifelse(max(data1$CLAY)==0,0.1*abs(Yrange),round(max(data1$CLAY)+0.1*abs(Yrange),1))
Ymin<-ifelse(min(data1$CLAY)==0,-0.1*abs(Yrange),round(min(data1$CLAY)-0.1*abs(Yrange),1))
Ymax
Ymin

interplinearS<-interp(data1$ISOC,data1$CLAY,data1$lnRR,nx=500,ny=500,linear=TRUE,duplicate = 'mean')
interplinearS$x
interplinearS$y
interplinearS$z

lgnd=1
lgndmar=5.1
lgnd
lgndmar

cairo_pdf("F4c.pdf",width = 4.6,height = 3.6)
p1<-image.plot(interplinearS,main=expression(paste("(c)"," ","lnRR"," ","(","Nitrate leaching",")")),col.main='black',xlim=c(Xmin,Xmax),ylim=c(Ymin,Ymax),xlab=expression(paste("Initial SOC"," ","(","g"," "," ",kg^{-1},")")),ylab='Clay content (%)',
           legend.shrink=1,legend.width=1,legend.mar=5.1,horizontal=FALSE)
p1+scale_color_distiller(palette="Spectral")
p1+scale_color_distiller(palette="Greens")
p1
dev.off()
####NL:ISOC+CLAY#####
data1=read.csv("NLlnRR-akima-IC2.csv",sep=",",header=TRUE)###
head(data1)
dim(data1)
Xrange<-max(data1$ISOC)-min(data1$ISOC)
Yrange<-max(data1$CLAY)-min(data1$CLAY)
Zrange<-max(data1$lnRR)-min(data1$lnRR)
Xrange
Yrange
Zrange
Xmax<-ifelse(max(data1$ISOC)==0,0.1*abs(Xrange),round(max(data1$ISOC)+0.1*abs(Xrange),1))
Xmin<-ifelse(min(data1$ISOC)==0,-0.1*abs(Xrange),round(min(data1$ISOC)-0.1*abs(Xrange),1))
Xmax
Xmin
Ymax<-ifelse(max(data1$CLAY)==0,0.1*abs(Yrange),round(max(data1$CLAY)+0.1*abs(Yrange),1))
Ymin<-ifelse(min(data1$CLAY)==0,-0.1*abs(Yrange),round(min(data1$CLAY)-0.1*abs(Yrange),1))
Ymax
Ymin

interplinearS<-interp(data1$ISOC,data1$CLAY,data1$lnRR,nx=500,ny=500,linear=TRUE,duplicate = 'mean')
interplinearS$x
interplinearS$y
interplinearS$z

lgnd=1
lgndmar=5.1
lgnd
lgndmar

cairo_pdf("F4d.pdf",width = 4.6,height = 3.6)
p1<-image.plot(interplinearS,main=expression(paste("(d)"," ","lnRR"," ","(","Nitrate leaching",")")),col.main='black',xlim=c(Xmin,Xmax),ylim=c(Ymin,Ymax),xlab=expression(paste("Initial SOC"," ","(","g"," "," ",kg^{-1},")")),ylab='Clay content (%)',
           legend.shrink=1,legend.width=1,legend.mar=5.1,horizontal=FALSE)
p1+scale_color_distiller(palette="Spectral")
p1+scale_color_distiller(palette="Greens")
p1
dev.off()
###################################
data1=read.csv("NLALL3-Q333akimaRB-GP-IN.csv",sep=",",header=TRUE)
head(data1)
dim(data1)
Xrange<-max(data1$GP)-min(data1$GP)
Yrange<-max(data1$RB)-min(data1$RB)
Zrange<-max(data1$lnRR)-min(data1$lnRR)
Xrange
Yrange
Zrange
Xmax<-ifelse(max(data1$GP)==0,0.1*abs(Xrange),round(max(data1$GP)+0.1*abs(Xrange),1))
Xmin<-ifelse(min(data1$GP)==0,-0.1*abs(Xrange),round(min(data1$GP)-0.1*abs(Xrange),1))
Xmax
Xmin
Ymax<-ifelse(max(data1$RB)==0,0.1*abs(Yrange),round(max(data1$RB)+0.1*abs(Yrange),1))
Ymin<-ifelse(min(data1$RB)==0,-0.1*abs(Yrange),round(min(data1$RB)-0.1*abs(Yrange),1))
Ymax
Ymin

interplinearS<-interp(data1$GP,data1$RB,data1$lnRR,nx=500,ny=500,linear=TRUE,duplicate = 'mean')
interplinearS$x
interplinearS$y
interplinearS$z

lgnd=1
lgndmar=5.1
lgnd
lgndmar
cairo_pdf("F4e.pdf",width = 4.6,height = 3.6)
p1<-image.plot(interplinearS,main=expression(paste("(e)"," ","lnRR"," ","(","Nitrate leaching",")")),col.main='black',xlim=c(Xmin,Xmax),ylim=c(Ymin,Ymax),xlab=expression(paste("Ratio of Growth period")),ylab=expression(paste("Ratio of Root biomass")),
           legend.shrink=1,legend.width=1,legend.mar=5.1,horizontal=FALSE)
p1+scale_color_distiller(palette="Spectral")
p1+scale_color_distiller(palette="Greens")
p1
dev.off()

data1=read.csv("NLALL3-Q333akimaRB-GP-RO.csv",sep=",",header=TRUE)
head(data1)
dim(data1)
Xrange<-max(data1$GP)-min(data1$GP)
Yrange<-max(data1$RB)-min(data1$RB)
Zrange<-max(data1$lnRR)-min(data1$lnRR)
Xrange
Yrange
Zrange
Xmax<-ifelse(max(data1$GP)==0,0.1*abs(Xrange),round(max(data1$GP)+0.1*abs(Xrange),1))
Xmin<-ifelse(min(data1$GP)==0,-0.1*abs(Xrange),round(min(data1$GP)-0.1*abs(Xrange),1))
Xmax
Xmin
Ymax<-ifelse(max(data1$RB)==0,0.1*abs(Yrange),round(max(data1$RB)+0.1*abs(Yrange),1))
Ymin<-ifelse(min(data1$RB)==0,-0.1*abs(Yrange),round(min(data1$RB)-0.1*abs(Yrange),1))
Ymax
Ymin

interplinearS<-interp(data1$GP,data1$RB,data1$lnRR,nx=500,ny=500,linear=TRUE,duplicate = 'mean')
interplinearS$x
interplinearS$y
interplinearS$z

lgnd=1
lgndmar=5.1
lgnd
lgndmar

cairo_pdf("F4f.pdf",width = 4.6,height = 3.6)
p1<-image.plot(interplinearS,main=expression(paste("(f)"," ","lnRR"," ","(","Nitrate leaching",")")),col.main='black',xlim=c(Xmin,Xmax),ylim=c(Ymin,Ymax),xlab=expression(paste("Ratio of Growth period")),ylab=expression(paste("Ratio of Root biomass")),
           legend.shrink=1,legend.width=1,legend.mar=5.1,horizontal=FALSE)
p1+scale_color_distiller(palette="Spectral")
p1+scale_color_distiller(palette="Greens")
p1
dev.off()
#######################
####N2O:Nrate+Clay#####
data1=read.csv("N2OlnRR-akima-NC1.csv",sep=",",header=TRUE)###
head(data1)
dim(data1)
Xrange<-max(data1$Nrate)-min(data1$Nrate)
Yrange<-max(data1$CLAY)-min(data1$CLAY)
Zrange<-max(data1$lnRR)-min(data1$lnRR)
Xrange
Yrange
Zrange
Xmax<-ifelse(max(data1$Nrate)==0,0.1*abs(Xrange),round(max(data1$Nrate)+0.1*abs(Xrange),1))
Xmin<-ifelse(min(data1$Nrate)==0,-0.1*abs(Xrange),round(min(data1$Nrate)-0.1*abs(Xrange),1))
Xmax
Xmin
Ymax<-ifelse(max(data1$CLAY)==0,0.1*abs(Yrange),round(max(data1$CLAY)+0.1*abs(Yrange),1))
Ymin<-ifelse(min(data1$CLAY)==0,-0.1*abs(Yrange),round(min(data1$CLAY)-0.1*abs(Yrange),1))
Ymax
Ymin

interplinearS<-interp(data1$Nrate,data1$CLAY,data1$lnRR,nx=500,ny=500,linear=TRUE,duplicate = 'mean')
interplinearS$x
interplinearS$y
interplinearS$z

lgnd=1
lgndmar=5.1
lgnd
lgndmar

cairo_pdf("F4g.pdf",width = 4.6,height = 3.6)
p1<-image.plot(interplinearS,main=expression(paste("(g)"," ","lnRR"," ","(",N[2],"O"," ","emission",")")),col.main='black',xlim=c(Xmin,Xmax),ylim=c(Ymin,Ymax),xlab=expression(paste("Nitrogen application rate"," ","(","kg"," ","N"," ",ha^{-1},")")),ylab='Clay content (%)',
           legend.shrink=1,legend.width=1,legend.mar=5.1,horizontal=FALSE)
p1+scale_color_distiller(palette="Spectral")
p1+scale_color_distiller(palette="Greens")
p1
dev.off()

data1=read.csv("N2OlnRR-akima-NC2.csv",sep=",",header=TRUE)###
head(data1)
dim(data1)
Xrange<-max(data1$Nrate)-min(data1$Nrate)
Yrange<-max(data1$CLAY)-min(data1$CLAY)
Zrange<-max(data1$lnRR)-min(data1$lnRR)
Xrange
Yrange
Zrange
Xmax<-ifelse(max(data1$Nrate)==0,0.1*abs(Xrange),round(max(data1$Nrate)+0.1*abs(Xrange),1))
Xmin<-ifelse(min(data1$Nrate)==0,-0.1*abs(Xrange),round(min(data1$Nrate)-0.1*abs(Xrange),1))
Xmax
Xmin
Ymax<-ifelse(max(data1$CLAY)==0,0.1*abs(Yrange),round(max(data1$CLAY)+0.1*abs(Yrange),1))
Ymin<-ifelse(min(data1$CLAY)==0,-0.1*abs(Yrange),round(min(data1$CLAY)-0.1*abs(Yrange),1))
Ymax
Ymin

interplinearS<-interp(data1$Nrate,data1$CLAY,data1$lnRR,nx=500,ny=500,linear=TRUE,duplicate = 'mean')
interplinearS$x
interplinearS$y
interplinearS$z

lgnd=1
lgndmar=5.1
lgnd
lgndmar

cairo_pdf("F4h.pdf",width = 4.6,height = 3.6)
p1<-image.plot(interplinearS,main=expression(paste("(h)"," ","lnRR"," ","(",N[2],"O"," ","emission",")")),col.main='black',xlim=c(Xmin,Xmax),ylim=c(Ymin,Ymax),xlab=expression(paste("Nitrogen application rate"," ","(","kg"," ","N"," ",ha^{-1},")")),ylab='Clay content (%)',
           legend.shrink=1,legend.width=1,legend.mar=5.1,horizontal=FALSE)
p1+scale_color_distiller(palette="Spectral")
p1+scale_color_distiller(palette="Greens")
p1
dev.off()
#################################################################
#######Fig5######
library(carData)
library(car)
library(piecewiseSEM)
library(lavaan)

DF <-read.csv("SOCALL2-Q3SEM3.csv",header=T)

model="SOC ~ MAT + CLAY + PH + Duration + ISOC + GP3 +BD;
   Duration ~ CLAY + PH + GP3"

fit1=cfa(model = model,data=DF)
fitmeasures(fit1,c("cfi","rmsea","bic","rmsea.ci.upper","gfi","chisq","pvalue"))


SOC.piecewise <- psem(
lm(SOC ~ MAT + CLAY + PH + Duration + ISOC + RB, data = DF),
lm(Duration ~ CLAY + PH, data = DF))

#summary(SOC.piecewise)
coefs(SOC.piecewise)
plot(SOC.piecewise)
######################

DF2 <-read.csv("NLALL3-Q333SEM3.csv",header=T)

model2="NL~ MAP + Nrate + CLAY + Water + TB;
    Nrate ~ MAP + MAT + ISOC + Water + CLAY;
    CLAY ~ MAT + MAP"

fit2=cfa(model = model2,data=DF2)
fitmeasures(fit2,c("cfi","rmsea","bic","rmsea.ci.upper","gfi","chisq","pvalue"))

NL.piecewise <- psem(
lm(NL~ MAP + Nrate + CLAY + Water + TB,data=DF2),
lm(Nrate ~ MAP + MAT + ISOC + Water + CLAY,data=DF2),
lm(CLAY ~ MAT + MAP,data=DF2)
)
AIC(NL.piecewise)#####
#summary(NL.piecewise)
coefs(NL.piecewise)
plot(NL.piecewise)


##########CO2###########
DF3 <-read.csv("CO2ALL2-Q3SEM3.csv",header=T)

model3="CO2 ~ TB + Nrate + MAP + CLAY + MAT;
    TB ~ CLAY + BD + MAT + ISOC + MAP"
fit3=cfa(model = model3,data=DF3)
fitmeasures(fit3,c("cfi","rmsea","bic","rmsea.ci.upper","gfi","chisq","pvalue"))


CO2.piecewise <- psem(
lm(CO2 ~ TB + Nrate + MAP + CLAY + MAT, data = DF3),
lm(TB ~ CLAY + BD + MAT + ISOC + MAP, data = DF3)
)
AIC(CO2.piecewise)#####AIC=34###
#summary(CO2.piecewise)
coefs(CO2.piecewise)
plot(CO2.piecewise)

###N2O#####

DF4 <-read.csv("N2OALL2-Q3SEM3.csv",header=T)
model4="N2O ~ MAP + MAT + ISOC + Nrate + TB + BD;
    Nrate ~ MAT + ISOC + PH"
fit4=cfa(model = model4,data=DF4)
fitmeasures(fit4,c("cfi","rmsea","bic","rmsea.ci.upper","gfi","chisq","pvalue"))


N2O.piecewise <- psem(
lm(N2O ~ MAP + MAT + ISOC + Nrate + TB + BD, data = DF4),
lm(Nrate ~ MAT + ISOC + PH, data = DF4)
)
AIC(N2O.piecewise)#####AIC=35###
#summary(N2O.piecewise)
coefs(N2O.piecewise)
plot(N2O.piecewise)
#############################################################################
############Fig.S2#########
library(maps)
library(ggplot2)
library(ggpubr)
library(sp)
library(sf)
library(maptools)
library(tidyverse)

DT1<-read.csv('SOCALL2-DT.csv')
str(DT1)
head(DT1)
mapworld<-borders("world",regions=".",
              colour="black",fill="white",size=0.01)
mp<-ggplot(data=DT1)+mapworld
print(mp)
mp1<-mp+
geom_point(aes(x=Lo,y=La),size=3)+
aes(color=Planting)+
scale_color_manual(values = c("lightseagreen","salmon"))+
theme(legend.background = element_blank(),
    legend.position="none",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill='#DCDCDC'),
    axis.text.y= element_text(size=16, color="black", face= "bold"),
    axis.text.x= element_text(size=16, color="black", face= "bold"),
    axis.title.x=element_text(size=18),
    axis.title.y=element_text(size=18),
    title=element_text(size=16))+
labs(x="Longitude",y="Latitude",title="SOC stock")
mp1
###
DT2<-read.csv('NLALL3-DT.csv')
str(DT2)
head(DT2)
mapworld<-borders("world",regions=".",
              colour="black",fill="white",size=0.01)
mp2<-ggplot(data=DT2)+mapworld
print(mp2)
mp3<-mp2+
geom_point(aes(x=Lo,y=La),size=3)+
aes(color=Planting)+
scale_color_manual(values = c("lightseagreen","salmon"))+
theme(legend.background = element_blank(),
    legend.position="none",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill='#DCDCDC'),
    axis.text.y= element_text(size=16, color="black", face= "bold"),
    axis.text.x= element_text(size=16, color="black", face= "bold"),
    axis.title.x=element_text(size=18),
    axis.title.y=element_text(size=18),
    title=element_text(size=16))+
labs(x="Longitude",y="Latitude",title="Nitrate leaching")
mp3
###
DT3<-read.csv('CO2ALL2-DT.csv')
str(DT3)
head(DT3)
mapworld<-borders("world",regions=".",
              colour="black",fill="white",size=0.01)
mp4<-ggplot(data=DT3)+mapworld
print(mp4)
mp5<-mp4+
geom_point(aes(x=Lo,y=La),size=3)+
aes(color=Planting)+
scale_color_manual(values = c("lightseagreen","salmon"))+
theme(legend.background = element_blank(),
    legend.position="none",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill='#DCDCDC'),
    axis.text.y= element_text(size=16, color="black", face= "bold"),
    axis.text.x= element_text(size=16, color="black", face= "bold"),
    axis.title.x=element_text(size=18),
    axis.title.y=element_text(size=18),
    title=element_text(size=16))+
labs(x="Longitude",y="Latitude",title=expression(paste("C",O[2]," ","emission")))
mp5


#####图1.2 N2O研究地点分布######
DT4<-read.csv('N2OALL2-DT.csv')
str(DT4)
head(DT4)
mapworld<-borders("world",regions=".", colour="black",fill="white",size=0.01)
mp6<-ggplot(data=DT4)+mapworld
print(mp6)
mp7<-mp6+
geom_point(aes(x=Lo,y=La),size=3)+
aes(color=Planting)+
scale_color_manual(values = c("lightseagreen","salmon"))+
theme(legend.background = element_blank(),
         legend.position="none",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect(fill='#DCDCDC'),
         axis.text.y= element_text(size=16, color="black", face= "bold"),
        axis.text.x= element_text(size=16, color="black", face= "bold"),
        axis.title.x=element_text(size=18),
        axis.title.y=element_text(size=18),
        title=element_text(size=16))+
labs(x="Longitude",y="Latitude",title=expression(paste(N[2],"O"," ","emission")))
mp7


#####图1 CH4研究地点分布######
DT5<-read.csv('CH4ALL2-DT.csv')
str(DT5)
head(DT5)
mapworld<-borders("world",regions=".",
              colour="black",fill="white",size=0.01)
mp8<-ggplot(data=DT5)+mapworld
print(mp8)
mp9<-mp8+
geom_point(aes(x=Lo,y=La),size=3)+
#guides(color=guide_legend(override.aes = list(size=14)))+

#guides(fill=guide_legend(title="Planting种植模式"))+
#scale_color_brewer(breaks=c("Intercropping","Rotation"),labels=c("Intercropping间作","Rotation轮作"))+
aes(color=Planting)+
scale_color_manual(values = c("lightseagreen","salmon"))+
#scale_fill_discrete(labels="Intercropping间作","Rotation轮作")+
theme(legend.background = element_blank(),
    legend.position="none",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill='#DCDCDC'),
    axis.text.y= element_text(size=16, color="black", face= "bold"),
    axis.text.x= element_text(size=16, color="black", face= "bold"),
    axis.title.x=element_text(size=18),
    axis.title.y=element_text(size=18),
    title=element_text(size=16))+
    #legend.text = element_text(size = 40),
    #legend.title=element_text(size=40),
    #title=element_text(size=24))+

labs(x="Longitude",y="Latitude",title=expression(paste("C",H[4]," ","emission")))

mp9

grid.newpage()
layout_1<-grid.layout(nrow = 2,ncol = 3,widths = c(1,1),heights=c(5,5))
pushViewport(viewport(layout = layout_1))
print(mp1,vp=viewport(layout.pos.row = 1,layout.pos.col = 1))
print(mp3,vp=viewport(layout.pos.row = 1,layout.pos.col = 2))
print(mp5,vp=viewport(layout.pos.row = 2,layout.pos.col = 1))
print(mp7,vp=viewport(layout.pos.row = 2,layout.pos.col = 2))
print(mp9,vp=viewport(layout.pos.row = 2,layout.pos.col = 3))
################################
#######################################
###图2，数据的总体分布###
library(ggpubr)
library(ggplot2)
library(ggsci)
library(gcookbook)
###########################################
#setwd("D:/R/RALL3/Fig.2 ZT3")
######实心的图#####shape=20为实心###=1为空心
data1<-read.csv('SOCALL2-ZT2.csv')
data1$Planting<- factor(data1$Planting,levels = c('In','Mo','Ro','Co'),ordered = TRUE)
p1=ggboxplot(data1,x="Planting",y="SOC")+
stat_boxplot(geom = "errorbar",size=1,width=0.5,linetype="solid",
           col=c("lightseagreen","lightseagreen","salmon","salmon"))+
geom_boxplot(size=1,fill="white",linetype="solid",
           col=c("lightseagreen","lightseagreen","salmon","salmon"))+
geom_jitter(width = 0.2,shape=20,size=4)+
aes(color=Planting)+
scale_color_manual(values = c("lightseagreen","lightseagreen","salmon","salmon"))+

labs(title="",x="",y=expression(paste("SOC stock"," ","(","Mg"," ",ha^-1,")")))+
theme(legend.position="none",
    axis.title= element_text(size=16, color="black", face= "bold"),
    axis.text.y= element_text(size=16, color="black", face= "bold"),
    axis.text.x= element_text(size=16, color="black", face= "bold"),
    axis.title.x=element_text(size=18),
    axis.title.y=element_text(size=18),
    title=element_text(size=16))
p1
########
data2<-read.csv('NLALL2-ZT2.csv')
data2$Planting<- factor(data2$Planting,levels = c('In','Mo','Ro','Co'),ordered = TRUE)
p2=ggboxplot(data2,x="Planting",y="NL")+
stat_boxplot(geom = "errorbar",size=1,width=0.5,linetype="solid",
           col=c("lightseagreen","lightseagreen","salmon","salmon"))+
geom_boxplot(size=1,fill="white",linetype="solid",
           col=c("lightseagreen","lightseagreen","salmon","salmon"))+
geom_jitter(width = 0.2,shape=20,size=4)+
aes(color=Planting)+
scale_color_manual(values = c("lightseagreen","lightseagreen","salmon","salmon"))+

labs(title="",x="",y=expression(paste("Nitrate leaching"," ","(","kg"," ",ha^-1,")")))+
theme(legend.position="none",
    axis.title= element_text(size=16, color="black", face= "bold"),
    axis.text.y= element_text(size=16, color="black", face= "bold"),
    axis.text.x= element_text(size=16, color="black", face= "bold"),
    axis.title.x=element_text(size=18),
    axis.title.y=element_text(size=18),
    title=element_text(size=16))
p2
#####
data3<-read.csv('CO2ALL2-ZT2.csv')
data3$Planting<- factor(data3$Planting,levels = c('In','Mo','Ro','Co'),ordered = TRUE)
p3=ggboxplot(data3,x="Planting",y="CO2")+
stat_boxplot(geom = "errorbar",size=1,width=0.5,linetype="solid",
           col=c("lightseagreen","lightseagreen","salmon","salmon"))+
geom_boxplot(size=1,fill="white",linetype="solid",
           col=c("lightseagreen","lightseagreen","salmon","salmon"))+
geom_jitter(width = 0.2,shape=20,size=4)+
aes(color=Planting)+
scale_color_manual(values = c("lightseagreen","lightseagreen","salmon","salmon"))+

labs(title="",x="",y=expression(paste(CO[2],"emisson","(","Mg",CO[2],"eq",ha^{-1},year^{-1},")")))+
theme(legend.position="none",
    axis.title= element_text(size=16, color="black", face= "bold"),
    axis.text.y= element_text(size=16, color="black", face= "bold"),
    axis.text.x= element_text(size=16, color="black", face= "bold"),
    axis.title.x=element_text(size=18),
    axis.title.y=element_text(size=18),
    title=element_text(size=16))
p3

#####2.2 N2O总体分布###
data4<-read.csv('N2OALL2-ZT2.csv')
data4$Planting<- factor(data4$Planting,levels = c('In','Mo','Ro','Co'),ordered = TRUE)
p4=ggboxplot(data4,x="Planting",y="N2O")+
stat_boxplot(geom = "errorbar",size=1,width=0.5,linetype="solid",
           col=c("lightseagreen","lightseagreen","salmon","salmon"))+
geom_boxplot(size=1,fill="white",linetype="solid",
           col=c("lightseagreen","lightseagreen","salmon","salmon"))+
geom_jitter(width = 0.2,shape=20,size=4)+
aes(color=Planting)+
scale_color_manual(values = c("lightseagreen","lightseagreen","salmon","salmon"))+

labs(title="",x="",y=expression(paste(N[2],"O","emission","(","Mg",CO[2],"eq",ha^{-1},year^{-1},")")))+
theme(legend.position="none",
    axis.title= element_text(size=16, color="black", face= "bold"),
    axis.text.y= element_text(size=16, color="black", face= "bold"),
    axis.text.x= element_text(size=16, color="black", face= "bold"),
    axis.title.x=element_text(size=18),
    axis.title.y=element_text(size=18),
    title=element_text(size=16))
p4
#####2.3 CH4总体分布###
data5<-read.csv('CH4ALL2-ZT2.csv')
#shapiro.test(data5$CH4)###P值小于0.05，不符合正态分布##
data5$Planting<- factor(data5$Planting,levels = c('In','Mo','Ro','Co'),ordered = TRUE)
p5<-ggboxplot(data5,x="Planting",y="CH4")+
stat_boxplot(geom = "errorbar",size=1,width=0.5,linetype="solid",
           col=c("lightseagreen","lightseagreen","salmon","salmon"))+
geom_boxplot(size=1,fill="white",linetype="solid",
           col=c("lightseagreen","lightseagreen","salmon","salmon"))+
geom_jitter(width = 0.2,shape=20,size=4)+
aes(color=Planting)+
scale_color_manual(values = c("lightseagreen","lightseagreen","salmon","salmon"))+

labs(title="",x="",y=expression(paste("C",H[4]," ","emission"," ","(","Mg",CO[2],"eq",ha^{-1}," ",year^{-1},")")))+
theme(legend.position="none",
    axis.title= element_text(size=16, color="black", face= "bold"),
    axis.text.y= element_text(size=16, color="black", face= "bold"),
    axis.text.x= element_text(size=16, color="black", face= "bold"),
    axis.title.x=element_text(size=18),
    axis.title.y=element_text(size=18),
    title=element_text(size=16))
p5

####空心图####
data1<-read.csv('SOCALL2-ZT2.csv')
data1$Planting<- factor(data1$Planting,levels = c('In','Mo','Ro','Co'),ordered = TRUE)
p1=ggboxplot(data1,x="Planting",y="SOC")+
stat_boxplot(geom = "errorbar",size=1,width=0.5,linetype="solid",
           col=c("lightseagreen","lightseagreen","salmon","salmon"))+
geom_boxplot(size=1,fill="white",linetype="solid",
           col=c("lightseagreen","lightseagreen","salmon","salmon"))+
geom_jitter(width = 0.2,shape=1,size=4)+
aes(color=Planting)+
scale_color_manual(values = c("lightseagreen","lightseagreen","salmon","salmon"))+

labs(title="",x="",y=expression(paste("SOC stock"," ","(","Mg"," ",ha^-1,")")))+
theme(legend.position="none",
    axis.title= element_text(size=16, color="black", face= "bold"),
    axis.text.y= element_text(size=16, color="black", face= "bold"),
    axis.text.x= element_text(size=16, color="black", face= "bold"),
    axis.title.x=element_text(size=18),
    axis.title.y=element_text(size=18),
    title=element_text(size=16))
p1
########
data2<-read.csv('NLALL2-ZT2.csv')
data2$Planting<- factor(data2$Planting,levels = c('In','Mo','Ro','Co'),ordered = TRUE)
p2=ggboxplot(data2,x="Planting",y="NL")+
stat_boxplot(geom = "errorbar",size=1,width=0.5,linetype="solid",
           col=c("lightseagreen","lightseagreen","salmon","salmon"))+
geom_boxplot(size=1,fill="white",linetype="solid",
           col=c("lightseagreen","lightseagreen","salmon","salmon"))+
geom_jitter(width = 0.2,shape=1,size=4)+
aes(color=Planting)+
scale_color_manual(values = c("lightseagreen","lightseagreen","salmon","salmon"))+

labs(title="",x="",y=expression(paste("Nitrate leaching"," ","(","kg"," ",ha^-1,")")))+
theme(legend.position="none",
    axis.title= element_text(size=16, color="black", face= "bold"),
    axis.text.y= element_text(size=16, color="black", face= "bold"),
    axis.text.x= element_text(size=16, color="black", face= "bold"),
    axis.title.x=element_text(size=18),
    axis.title.y=element_text(size=18),
    title=element_text(size=16))
p2
#####
data3<-read.csv('CO2ALL2-ZT2.csv')
data3$Planting<- factor(data3$Planting,levels = c('In','Mo','Ro','Co'),ordered = TRUE)
p3=ggboxplot(data3,x="Planting",y="CO2")+
stat_boxplot(geom = "errorbar",size=1,width=0.5,linetype="solid",
           col=c("lightseagreen","lightseagreen","salmon","salmon"))+
geom_boxplot(size=1,fill="white",linetype="solid",
           col=c("lightseagreen","lightseagreen","salmon","salmon"))+
geom_jitter(width = 0.2,shape=1,size=4)+
aes(color=Planting)+
scale_color_manual(values = c("lightseagreen","lightseagreen","salmon","salmon"))+

labs(title="",x="",y=expression(paste(CO[2],"emisson","(","Mg",CO[2],"eq",ha^{-1},year^{-1},")")))+
theme(legend.position="none",
    axis.title= element_text(size=16, color="black", face= "bold"),
    axis.text.y= element_text(size=16, color="black", face= "bold"),
    axis.text.x= element_text(size=16, color="black", face= "bold"),
    axis.title.x=element_text(size=18),
    axis.title.y=element_text(size=18),
    title=element_text(size=16))
p3

#####2.2 N2O总体分布###
data4<-read.csv('N2OALL2-ZT2.csv')
data4$Planting<- factor(data4$Planting,levels = c('In','Mo','Ro','Co'),ordered = TRUE)
p4=ggboxplot(data4,x="Planting",y="N2O")+
stat_boxplot(geom = "errorbar",size=1,width=0.5,linetype="solid",
           col=c("lightseagreen","lightseagreen","salmon","salmon"))+
geom_boxplot(size=1,fill="white",linetype="solid",
           col=c("lightseagreen","lightseagreen","salmon","salmon"))+
geom_jitter(width = 0.2,shape=1,size=4)+
aes(color=Planting)+
scale_color_manual(values = c("lightseagreen","lightseagreen","salmon","salmon"))+

labs(title="",x="",y=expression(paste(N[2],"O","emission","(","Mg",CO[2],"eq",ha^{-1},year^{-1},")")))+
theme(legend.position="none",
    axis.title= element_text(size=16, color="black", face= "bold"),
    axis.text.y= element_text(size=16, color="black", face= "bold"),
    axis.text.x= element_text(size=16, color="black", face= "bold"),
    axis.title.x=element_text(size=18),
    axis.title.y=element_text(size=18),
    title=element_text(size=16))
p4
#####2.3 CH4总体分布###
data5<-read.csv('CH4ALL2-ZT2.csv')
#shapiro.test(data5$CH4)###P值小于0.05，不符合正态分布##
data5$Planting<- factor(data5$Planting,levels = c('In','Mo','Ro','Co'),ordered = TRUE)
p5<-ggboxplot(data5,x="Planting",y="CH4")+
stat_boxplot(geom = "errorbar",size=1,width=0.5,linetype="solid",
           col=c("lightseagreen","lightseagreen","salmon","salmon"))+
geom_boxplot(size=1,fill="white",linetype="solid",
           col=c("lightseagreen","lightseagreen","salmon","salmon"))+
geom_jitter(width = 0.2,shape=1,size=4)+
aes(color=Planting)+
scale_color_manual(values = c("lightseagreen","lightseagreen","salmon","salmon"))+

labs(title="",x="",y=expression(paste("C",H[4]," ","emission"," ","(","Mg",CO[2],"eq",ha^{-1}," ",year^{-1},")")))+
theme(legend.position="none",
    axis.title= element_text(size=16, color="black", face= "bold"),
    axis.text.y= element_text(size=16, color="black", face= "bold"),
    axis.text.x= element_text(size=16, color="black", face= "bold"),
    axis.title.x=element_text(size=18),
    axis.title.y=element_text(size=18),
    title=element_text(size=16))
p5
grid.newpage()
layout_1<-grid.layout(nrow = 2,ncol = 3,widths = c(1,1),heights=c(5,5))
pushViewport(viewport(layout = layout_1))
print(p1,vp=viewport(layout.pos.row = 1,layout.pos.col = 1))
print(p2,vp=viewport(layout.pos.row = 1,layout.pos.col = 2))
print(p3,vp=viewport(layout.pos.row = 2,layout.pos.col = 1))
print(p4,vp=viewport(layout.pos.row = 2,layout.pos.col = 2))
print(p5,vp=viewport(layout.pos.row = c(1,2),layout.pos.col = 3))
###################################################################################
#############################################################################
############################Fig.S3######################
packages <- c("ggplot2", "cowplot", "dplyr", "scales")
for (pkg in packages) {
if (!require(pkg, character.only = TRUE)) install.packages(pkg)
library(pkg, character.only = TRUE)}
file_path <- "FNF-All.csv"
raw <- read.csv(file_path, header = FALSE, stringsAsFactors = FALSE, 
fileEncoding = "GBK", skip = 1)
colnames(raw) <- c("indicator", "planting", "group", "n", "change", "lower", "upper")
df <- raw[!is.na(raw$indicator) & raw$indicator != "", ]
num_cols <- c("n", "change", "lower", "upper")
for (col in num_cols) df[[col]] <- as.numeric(gsub("[^0-9.-]", "", df[[col]]))
df <- df[complete.cases(df[, num_cols]), ]
group_levels <- c("All", ">0", "0", "<0")
existing_levels <- intersect(group_levels, unique(df$group))
df$group <- factor(df$group, levels = existing_levels)
colors_fnf <- scales::hue_pal()(length(existing_levels))
names(colors_fnf) <- existing_levels
common_theme <- theme(
legend.title = element_blank(),
legend.position = "none",
legend.key = element_rect(fill = "white", size = 4),
legend.background = element_blank(),
legend.text = element_text(colour = "black", size = 12),
panel.background = element_rect(fill = "white", colour = "white"),
axis.title = element_text(colour = "black", size = 13),
axis.title.y = element_blank(),
axis.text.x = element_text(colour = "black", size = 12),
axis.text.y = element_text(colour = "black", size = 12, 
margin = margin(t = -6, b = -6), lineheight = 0.6),
axis.line = element_line(colour = "black", size = 0.6),
axis.line.y = element_blank(),
axis.ticks = element_line(colour = "black", size = 0.6),
axis.ticks.y = element_blank(),
plot.margin = margin(-2, 2, -2, 2),
panel.spacing = unit(0, "lines"))
plot_metric <- function(df_sub, y_title, text_offset = 3, show_y_ticks = TRUE, 
category_levels = NULL, color_map = NULL,
y_limits = NULL, y_breaks = NULL) {
if (nrow(df_sub) == 0) {return(ggplot() + theme_void() + 
geom_text(aes(x = 0.5, y = 0.5, label = "No data"), size = 5))}
if (is.null(category_levels)) category_levels <- unique(df_sub$group)
df_sub$group <- factor(df_sub$group, levels = category_levels)
df_sub <- df_sub %>% arrange(group, planting)
if (is.null(y_limits)) {
all_vals <- c(df_sub$change, df_sub$lower, df_sub$upper)
all_vals <- all_vals[is.finite(all_vals)]
if (length(all_vals) > 0) {
y_min <- min(all_vals, na.rm = TRUE); y_max <- max(all_vals, na.rm = TRUE)
pad <- (y_max - y_min) * 0.15
y_limits <- c(y_min - pad, y_max + pad)
y_breaks <- pretty(y_limits, n = 5)} else {
y_limits <- c(-10, 10); y_breaks <- c(-10, -5, 0, 5, 10)}}
y_axis_theme <- if (show_y_ticks) {
theme(axis.text.y = element_text(colour = "black", size = 12,
 margin = margin(t = -6, b = -6), lineheight = 0.6))} else {
theme(axis.text.y = element_text(colour = "white", size = 12,
margin = margin(t = -6, b = -6), lineheight = 0.6),
axis.ticks.y = element_blank())}
fill_scale <- if (!is.null(color_map)) scale_fill_manual(values = color_map, drop = FALSE) else scale_fill_discrete(drop = FALSE)
p <- ggplot(df_sub, aes(x = group, y = change, fill = group, shape = planting)) +
geom_hline(yintercept = 0, linetype = "dashed", size = 0.3, colour = "black") +
geom_errorbar(aes(ymin = lower, ymax = upper), 
position = position_dodge(-0.8), width = 0.2, size = 0.3, colour = "black") +
geom_point(position = position_dodge(-0.8), size = 4, stroke = 0.3, colour = "black") +
scale_shape_manual(values = c("All" = 23, "Intercropping" = 22, "Rotation" = 21, "Monoculture" = 24)) +
geom_text(aes(x = group, y = upper + text_offset, label = n),
position = position_dodge(width = -0.9), vjust = 0.4, hjust = 0, 
size = 3, colour = "black", check_overlap = FALSE) +
scale_y_continuous(limits = y_limits, breaks = y_breaks) +
scale_x_discrete(limits = rev(category_levels), expand = c(0, 0)) +
labs(x = NULL, y = y_title) +
common_theme + y_axis_theme + guides(fill = "none") + coord_flip() +
theme(axis.ticks.length.y = unit(0, "cm")) + fill_scalereturn(p)}
create_fnf_plot <- function(data, category_levels, color_map) {
df_CH4 <- data %>% filter(indicator == "CH4")
df_CO2 <- data %>% filter(indicator == "CO2")
df_N2O <- data %>% filter(indicator == "N2O")
df_NL  <- data %>% filter(indicator == "NL")
df_SOC <- data %>% filter(indicator == "SOC")
p_SOC <- plot_metric(df_SOC, "Relative change in SOC stock (%)", 
     text_offset = 3, show_y_ticks = TRUE, 
     category_levels = category_levels, color_map = color_map)
p_NL  <- plot_metric(df_NL,  "Relative change in Nitrate leaching (%)", 
     text_offset = 3, show_y_ticks = FALSE, 
     category_levels = category_levels, color_map = color_map)
p_CO2 <- plot_metric(df_CO2, expression(paste("Relative change in ", CO[2], " emission (%)")), 
     text_offset = 3, show_y_ticks = TRUE, 
     category_levels = category_levels, color_map = color_map)
p_N2O <- plot_metric(df_N2O, expression(paste("Relative change in ", N[2], "O emission (%)")), 
     text_offset = 3, show_y_ticks = FALSE, 
     category_levels = category_levels, color_map = color_map)
p_CH4 <- plot_metric(df_CH4, expression(paste("Relative change in ", CH[4], " emission (%)")), 
text_offset = 3, show_y_ticks = FALSE, 
category_levels = category_levels, color_map = color_map)
p_SOC <- p_SOC + theme(axis.text.y = element_text(face = "bold"))
p_NL  <- p_NL  + theme(axis.text.y = element_text(face = "bold"))
p_CO2 <- p_CO2 + theme(axis.text.y = element_text(face = "bold"))
p_N2O <- p_N2O + theme(axis.text.y = element_text(face = "bold"))
p_CH4 <- p_CH4 + theme(axis.text.y = element_text(face = "bold"))
planting_levels <- intersect(c("All", "Intercropping", "Rotation", "Monoculture"), unique(data$planting))
legend_df <- data.frame(x = seq_along(planting_levels), y = seq_along(planting_levels),
 Planting = factor(planting_levels, levels = planting_levels))
shape_values <- c("All" = 23, "Intercropping" = 22, "Rotation" = 21, "Monoculture" = 24)
shape_values <- shape_values[names(shape_values) %in% planting_levels]
legend_plot <- ggplot(legend_df, aes(x = x, y = y, shape = Planting)) +
geom_point(size = 4, stroke = 0.3, colour = "black", fill = "white") +
scale_shape_manual(values = shape_values, labels = planting_levels) +
theme(legend.title = element_blank(), legend.position = "bottom",
legend.key = element_rect(fill = "white", size = 4),
legend.background = element_blank(),
legend.text = element_text(colour = "black", size = 12),
legend.margin = margin(0, 0, 0, 0)) +
guides(shape = guide_legend(nrow = length(planting_levels), byrow = FALSE))
legend_grob <- cowplot::get_legend(legend_plot)
total_width <- 18; total_height <- 5.5; left_margin <- 0.06; gap_h <- 0.03; gap_v <- 0.04
sub_width <- 0.25
x1 <- left_margin; x2 <- x1 + sub_width + gap_h; x3 <- x2 + sub_width + gap_h
y_top <- 0.5 + gap_v / 2; y_bottom <- 0; height_row <- 0.5 - gap_v / 2
final_plot <- ggdraw() +
draw_plot(p_SOC, x = x1, y = y_top, width = sub_width, height = height_row) +
draw_plot(p_NL,  x = x2, y = y_top, width = sub_width, height = height_row) +
draw_plot(p_CO2, x = x1, y = y_bottom, width = sub_width, height = height_row) +
draw_plot(p_N2O, x = x2, y = y_bottom, width = sub_width, height = height_row) +
draw_plot(p_CH4, x = x3, y = y_bottom, width = sub_width, height = height_row) +
draw_grob(legend_grob, x = 0.55, y = 0.72, width = 0.26, height = 0.25) +
draw_label("Difference in total fertilizer N input (ΔN, kg N ha⁻¹)", 
angle = 90, x = 0.02, y = 0.5, 
vjust = 0.5, hjust = 0.5, size = 18, fontface = "plain")
return(final_plot)}
plot_fnf <- create_fnf_plot(df, category_levels = existing_levels, color_map = colors_fnf)
output_dir <- dirname(file_path)
base_name <- "FNF_Combined_Plot"
ggsave(file.path(output_dir, paste0(base_name, ".png")), plot = plot_fnf, 
width = 18.5, height = 5.5, dpi = 600, bg = "white")
ggsave(file.path(output_dir, paste0(base_name, ".pdf")), plot = plot_fnf, 
width = 18.5, height = 5.5, device = "pdf")
ggsave(file.path(output_dir, paste0(base_name, ".tiff")), plot = plot_fnf, 
width = 18.5, height = 5.5, dpi = 600, bg = "white", compression = "lzw")
#############################################################################
###################################Fig.S4############################
library(ggplot2)
library(readxl)
data_path <- "单位施氮量-All.xlsx"
datalm1 <- read_excel(data_path)
datalm1$index <- as.factor(datalm1$index)
datalm1$lnRR <- as.numeric(as.character(datalm1$lnRR))
p1 <- ggplot(data = datalm1, aes(x = index, y = lnRR, fill = index, shape = Planting)) + 
geom_hline(yintercept = 0, linetype = "dashed", size = 0.3) +
geom_errorbar(position = position_dodge(-0.8), aes(ymin = low, ymax = up), width = 0.3, size = 0.3) +
geom_point(position = position_dodge(-0.8), size = 3, stroke = 0.3) + 
scale_shape_manual(values = c("All" = 23, "Intercropping" = 22, "Rotation" = 21)) +
geom_text(aes(x = index, y = up + 1, label = samplesize),
position = position_dodge(width = -0.9), vjust = 0.4, hjust = 0, size = 3, check_overlap = FALSE) +
scale_y_continuous(limits = c(-50, 80), breaks = seq(-50, 80, by = 10)) +
scale_x_discrete(
breaks = c("CH4 emission", "CO2 emission", "N2O emission",, "Nitrate leaching", "SOC stock"),
labels = c(expression(paste("C", H[4], " ", "emission")),
expression(paste("C", O[2], " ", "emission")),
expression(paste(N[2], "O", " ", "emission")),"Nitrate leaching","SOC stock")) +
labs(x = "", y = "Relative change (%)", colour = 'black') +
theme(legend.title = element_blank(),
legend.position = c(0.88, 0.7),
legend.key = element_rect(fill = "white", size = 4),
legend.background = element_blank(),
legend.text = element_text(size = 12),
panel.background = element_rect(fill = 'white', colour = 'white'),
axis.title = element_text(size = 13),
axis.text.y = element_text(colour = 'black', size = 12),
axis.text.x = element_text(colour = 'black', size = 12),
axis.line = element_line(colour = 'black', size = 0.6),
axis.line.y = element_blank(),
axis.ticks = element_line(colour = 'black', size = 0.6),
axis.ticks.y = element_blank(),
plot.margin = margin(5, 20, 5, 5, "mm")) +guides(fill = "none") +coord_flip()
print(p1)
output_dir <- dirname(data_path)
ggsave(filename = paste0(output_dir, "/Figure1.png"), 
plot = p1, width = 8, height = 6, dpi = 300, units = "in")
ggsave(filename = paste0(output_dir, "/Figure1.pdf"), 
plot = p1, width = 8, height = 6, units = "in", device = "pdf")
ggsave(filename = paste0(output_dir, "/Figure1.tiff"), 
plot = p1, width = 8, height = 6, dpi = 300, units = "in", compression = "lzw")
#############################################################################
#############################################################################
####################Fig.S5########################################
packages <- c("ggplot2", "cowplot", "dplyr", "scales")
for (pkg in packages) {if (!require(pkg, character.only = TRUE)) install.packages(pkg)
library(pkg, character.only = TRUE)}
file_path <- "Climate-All.csv"
raw <- read.csv(file_path, header = FALSE, stringsAsFactors = FALSE, 
fileEncoding = "GBK", skip = 1)
colnames(raw) <- c("indicator", "planting", "group", "n", "change", "lower", "upper")
df <- raw[!is.na(raw$indicator) & raw$indicator != "", ]
num_cols <- c("n", "change", "lower", "upper")
for (col in num_cols) {
df[[col]] <- as.numeric(gsub("[^0-9.-]", "", df[[col]]))}
df <- df[complete.cases(df[, num_cols]), ]
group_levels <- c("All", "Warm–Wet", "Warm–Dry", "Cool–Wet", "Cool–Dry")
existing_levels <- intersect(group_levels, unique(df$group))
df$group <- factor(df$group, levels = existing_levels)
colors_climate <- scales::hue_pal()(length(existing_levels))
names(colors_climate) <- existing_levels
common_theme <- theme(
legend.title = element_blank(),
legend.position = "none",
legend.key = element_rect(fill = "white", size = 4),
legend.background = element_blank(),
legend.text = element_text(colour = "black", size = 12),
panel.background = element_rect(fill = "white", colour = "white"),
axis.title = element_text(colour = "black", size = 13),
axis.title.y = element_blank(),
axis.text.x = element_text(colour = "black", size = 12),
axis.text.y = element_text(colour = "black", size = 12, 
margin = margin(t = -6, b = -6),
lineheight = 0.6),
axis.line = element_line(colour = "black", size = 0.6),
axis.line.y = element_blank(),
axis.ticks = element_line(colour = "black", size = 0.6),
axis.ticks.y = element_blank(),
plot.margin = margin(-2, 2, -2, 2),
panel.spacing = unit(0, "lines"))
plot_metric <- function(df_sub, y_title, text_offset = 3, show_y_ticks = TRUE, 
category_levels = NULL, color_map = NULL,
y_limits = NULL, y_breaks = NULL) {
if (nrow(df_sub) == 0) {
return(ggplot() + theme_void() + 
geom_text(aes(x = 0.5, y = 0.5, label = "No data"), size = 5))}
if (is.null(category_levels)) {category_levels <- unique(df_sub$group)}
df_sub$group <- factor(df_sub$group, levels = category_levels)
df_sub <- df_sub %>% arrange(group, planting)
if (is.null(y_limits)) {
all_vals <- c(df_sub$change, df_sub$lower, df_sub$upper)
all_vals <- all_vals[is.finite(all_vals)]
if (length(all_vals) > 0) {
y_min <- min(all_vals, na.rm = TRUE)
y_max <- max(all_vals, na.rm = TRUE)
pad <- (y_max - y_min) * 0.15
y_limits <- c(y_min - pad, y_max + pad)
y_breaks <- pretty(y_limits, n = 5)} else {y_limits <- c(-10, 10)y_breaks <- c(-10, -5, 0, 5, 10)}}
if (show_y_ticks) {
y_axis_theme <- theme(axis.text.y = element_text(colour = "black", size = 12,
margin = margin(t = -6, b = -6),
lineheight = 0.6))} else {
y_axis_theme <- theme(axis.text.y = element_text(colour = "white", size = 12,
                                                margin = margin(t = -6, b = -6),lineheight = 0.6),axis.ticks.y = element_blank())}
if (!is.null(color_map)) {
fill_scale <- scale_fill_manual(values = color_map, drop = FALSE)
} else {fill_scale <- scale_fill_discrete(drop = FALSE)}
p <- ggplot(df_sub, aes(x = group, y = change, 
fill = group, shape = planting)) +
geom_hline(yintercept = 0, linetype = "dashed", size = 0.3, colour = "black") +
geom_errorbar(aes(ymin = lower, ymax = upper), 
position = position_dodge(-0.8), width = 0.2, size = 0.3, colour = "black") +
geom_point(position = position_dodge(-0.8), size = 4, stroke = 0.3, colour = "black") +
scale_shape_manual(values = c("All" = 23, "Intercropping" = 22, "Rotation" = 21, "Monoculture" = 24)) +
geom_text(aes(x = group, y = upper + text_offset, label = n),
position = position_dodge(width = -0.9), vjust = 0.4, hjust = 0, 
size = 3, colour = "black", check_overlap = FALSE) +
scale_y_continuous(limits = y_limits, breaks = y_breaks) +
scale_x_discrete(limits = rev(category_levels), expand = c(0, 0)) +
labs(x = NULL, y = y_title) +
common_theme +
y_axis_theme +
guides(fill = "none") +
coord_flip() +
theme(axis.ticks.length.y = unit(0, "cm")) +
fill_scalereturn(p)
create_climate_plot <- function(data, category_levels, color_map) {
df_CH4 <- data %>% filter(indicator == "CH4")
df_CO2 <- data %>% filter(indicator == "CO2")
df_N2O <- data %>% filter(indicator == "N2O")
df_NL  <- data %>% filter(indicator == "NL")
df_SOC <- data %>% filter(indicator == "SOC")
p_SOC <- plot_metric(df_SOC, "Relative change in SOC stock (%)", 
text_offset = 3, show_y_ticks = TRUE, 
category_levels = category_levels, color_map = color_map)
p_NL  <- plot_metric(df_NL,  "Relative change in Nitrate leaching (%)", 
text_offset = 3, show_y_ticks = FALSE, 
category_levels = category_levels, color_map = color_map)
p_CO2 <- plot_metric(df_CO2, expression(paste("Relative change in ", CO[2], " emission (%)")), 
text_offset = 3, show_y_ticks = TRUE, 
category_levels = category_levels, color_map = color_map)
p_N2O <- plot_metric(df_N2O, expression(paste("Relative change in ", N[2], "O emission (%)")), 
text_offset = 3, show_y_ticks = FALSE, 
category_levels = category_levels, color_map = color_map)
p_CH4 <- plot_metric(df_CH4, expression(paste("Relative change in ", CH[4], " emission (%)")), 
text_offset = 3, show_y_ticks = FALSE, 
category_levels = category_levels, color_map = color_map)
p_SOC <- p_SOC + theme(axis.text.y = element_text(face = "bold"))
p_NL  <- p_NL  + theme(axis.text.y = element_text(face = "bold"))
p_CO2 <- p_CO2 + theme(axis.text.y = element_text(face = "bold"))
p_N2O <- p_N2O + theme(axis.text.y = element_text(face = "bold"))
p_CH4 <- p_CH4 + theme(axis.text.y = element_text(face = "bold"))
planting_types <- unique(data$planting)
planting_levels <- intersect(c("All", "Intercropping", "Rotation", "Monoculture"), planting_types)
legend_df <- data.frame(
x = seq_along(planting_levels),
y = seq_along(planting_levels),
Planting = factor(planting_levels, levels = planting_levels))
shape_values <- c("All" = 23, "Intercropping" = 22, "Rotation" = 21, "Monoculture" = 24)
shape_values <- shape_values[names(shape_values) %in% planting_levels]
legend_plot <- ggplot(legend_df, aes(x = x, y = y, shape = Planting)) +
geom_point(size = 4, stroke = 0.3, colour = "black", fill = "white") +
scale_shape_manual(values = shape_values,
labels = planting_levels) +
theme(legend.title = element_blank(), legend.position = "bottom",
legend.key = element_rect(fill = "white", size = 4),
legend.background = element_blank(),
legend.text = element_text(colour = "black", size = 12),
legend.margin = margin(0, 0, 0, 0)) +
guides(shape = guide_legend(nrow = length(planting_levels), byrow = FALSE))
legend_grob <- cowplot::get_legend(legend_plot)
total_width <- 18
total_height <- 5.5
left_margin <- 0.06
gap_h <- 0.03
gap_v <- 0.04
sub_width <- 0.25
x1 <- left_margin
x2 <- x1 + sub_width + gap_h
x3 <- x2 + sub_width + gap_h
y_top <- 0.5 + gap_v / 2
y_bottom <- 0
height_row <- 0.5 - gap_v / 2
final_plot <- ggdraw() +
draw_plot(p_SOC, x = x1, y = y_top, width = sub_width, height = height_row) +
draw_plot(p_NL,  x = x2, y = y_top, width = sub_width, height = height_row) +
draw_plot(p_CO2, x = x1, y = y_bottom, width = sub_width, height = height_row) +
draw_plot(p_N2O, x = x2, y = y_bottom, width = sub_width, height = height_row) +
draw_plot(p_CH4, x = x3, y = y_bottom, width = sub_width, height = height_row) +
draw_grob(legend_grob, x = 0.55, y = 0.72, width = 0.26, height = 0.25) +
draw_label("Climate type", angle = 90, x = 0.02, y = 0.5, 
vjust = 0.5, hjust = 0.5, size = 16, fontface = "plain")
return(final_plot)
plot_climate <- create_climate_plot(df, category_levels = existing_levels, color_map = colors_climate)
output_dir <- dirname(file_path)
base_name <- "Climate_Combined_Plot"
ggsave(file.path(output_dir, paste0(base_name, ".png")), plot = plot_climate, 
width = 18.5, height = 5.5, dpi = 600, bg = "white")
ggsave(file.path(output_dir, paste0(base_name, ".pdf")), plot = plot_climate, 
width = 18.5, height = 5.5, device = "pdf")
ggsave(file.path(output_dir, paste0(base_name, ".tiff")), plot = plot_climate, 
width = 18.5, height = 5.5, dpi = 600, bg = "white", compression = "lzw")
##############################
library(ggplot2)
library(readxl)
file_path <- "水旱轮作-All.xlsx"
data <- read_excel(file_path)
desired_order <- c("Overall", "Including rice cultivation", "Excluding rice cultivation")
data$index <- factor(data$index, levels = desired_order)
data$Planting <- factor(data$Planting, levels = c("All", "Intercropping", "Rotation"))
dodge_width <- -0.8
p <- ggplot(data, aes(x = index, y = change, fill = index, shape = Planting)) +
geom_hline(yintercept = 0, linetype = "dashed", size = 0.3) +
geom_errorbar(data = subset(data, index == "Overall"),
aes(ymin = low, ymax = up),
position = position_dodge(dodge_width),
width = 0.3, size = 0.3) +
geom_errorbar(data = subset(data, index == "Including rice cultivation"),
aes(ymin = low, ymax = up),
position = position_dodge(dodge_width),
width = 0.15, size = 0.3) +
geom_errorbar(data = subset(data, index == "Excluding rice cultivation"),
aes(ymin = low, ymax = up),
position = position_dodge(dodge_width),
width = 0.3, size = 0.3) +
geom_point(position = position_dodge(dodge_width),
size = 4, stroke = 0.3) +
scale_shape_manual(values = c("All" = 23, "Intercropping" = 22, "Rotation" = 21)) +
geom_text(aes(x = index, y = up + 1, label = samplesize),
position = position_dodge(dodge_width),
vjust = 0.4, hjust = 0, size = 3, check_overlap = FALSE) +
scale_y_continuous(limits = c(-100, 50), breaks = seq(-100, 50, 25)) +
scale_x_discrete(limits = rev(desired_order)) +
labs(title = " ", x = " ", y = "Relative change (%)") +theme(
legend.title = element_blank(),
legend.position = c(0.98, 0.7),
legend.key = element_rect(fill = "white", size = 4),
legend.background = element_blank(),
legend.text = element_text(size = 12),
panel.background = element_rect(fill = 'white', colour = 'white'),
axis.title = element_text(size = 13),
axis.text.y = element_text(colour = 'black', size = 12),
axis.text.x = element_text(colour = 'black', size = 12),
axis.line = element_line(colour = 'black', size = 0.6),
axis.line.y = element_blank(),
axis.ticks = element_line(colour = 'black', size = 0.6),
axis.ticks.y = element_blank(),
plot.margin = margin(r = 2, unit = "cm")) +guides(fill = "none") +coord_flip()
print(p)
#############################################################################
#############################Fig.S6######################
library(ggplot2)
library(car)
library(carData)
library(palmerpenguins)
library(ggpubr)
library(gcookbook)
######
data=read.csv("FS-MAP-MAT-FN-XG.csv",sep=",",header=TRUE)
####1.SOC-MAP###
data$Planting1<- as.factor(data$Planting1)
data$Planting1<- factor(data$Planting1,levels = c('Intercropping','Rotation'),ordered = TRUE)
p1<-ggplot(data=data,aes(x=MAP1,y=lnRRSOC,color=Planting1,group=Planting1))+
labs(y=expression(paste("lnRR"," ","(","SOC"," ","stock",")")),x="Mean annual precipitation (mm)",title="(a)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting1,fill=Planting1),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(MAP1, lnRRSOC, color = Planting1), method = "spearman",size=7)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,2000),breaks=c(0, 500, 1000, 1500, 2000)) +
scale_y_continuous(limits=c(-0.5,1.5),breaks=c(-0.5, 0, 0.5, 1, 1.5))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p1
ggsave("FS2a.pdf", plot = p1, device = cairo_pdf, dpi = 600,width = 5.5, height = 4)
###########
####2.NL-MAP###
data$Planting2<- as.factor(data$Planting2)
data$Planting2<- factor(data$Planting2,levels = c('Intercropping','Rotation'),ordered = TRUE)
p4<-ggplot(data=data,aes(x=MAP2,y=lnRRNL,color=Planting2,group=Planting2))+
labs(y=expression(paste("lnRR"," ","(","Nitrate leaching",")")),x="Mean annual precipitation (mm)",title="(b)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting2,fill=Planting2),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(MAP2, lnRRNL, color = Planting2), method = "spearman",size=7,label.x.npc = 0.4,label.y.npc = 0.9)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,2000),breaks=c(0, 500, 1000, 1500, 2000)) +
scale_y_continuous(limits=c(-2,2),breaks=c(-2, -1, 0, 1, 2))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p4
ggsave("FS2b.pdf", plot = p4, device = cairo_pdf, dpi = 600,width = 5.5, height = 4)
############
####3.CO2-MAP###
data$Planting3<- factor(data$Planting3,levels = c('Intercropping','Rotation'),ordered = TRUE)
p7<-ggplot(data=data,aes(x=MAP3,y=lnRRCO2,group=Planting3,color=Planting3))+
labs(y=expression(paste("lnRR"," ","(",CO[2]," ","emission",")")),x="Mean annual precipitation (mm)",title="(c)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting3,fill=Planting3),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(MAP3, lnRRCO2, colour = Planting3), method = "spearman",size=7,label.x.npc = 0.1,label.y.npc = 0.2)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,2000),breaks=c(0, 500, 1000, 1500, 2000)) +
scale_y_continuous(limits=c(-3,2),breaks=c(-3,-2, -1, 0, 1, 2))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p7
ggsave("FS2c.pdf", plot = p7, device = cairo_pdf, dpi = 600,width = 5.5, height = 4)
#######################
####4.N2O-MAP###
data$Planting4<- factor(data$Planting4,levels = c('Intercropping','Rotation'),ordered = TRUE)
data$Planting4<- factor(data$Planting4,levels = c('Intercropping','Rotation'),ordered = TRUE)
p10<-ggplot(data=data,aes(x=MAP4,y=lnRRN2O,group=Planting4,color=Planting4))+
labs(y=expression(paste("lnRR"," ","(",N[2],"O"," ","emission",")")),x="Mean annual precipitation (mm)",title="(d)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting4,fill=Planting4),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(MAP4, lnRRN2O, colour = Planting4), method = "spearman",size=7,label.x.npc = 0.5,label.y.npc = 0.9)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,2000),breaks=c(0, 500, 1000, 1500, 2000)) +
scale_y_continuous(limits=c(-3,3),breaks=c(-3,-1.5, 0, 1.5, 3))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p10
ggsave("FS2d.pdf", plot = p10, device = cairo_pdf, dpi = 600,width = 5.5, height = 4)
##################
##5.CH4-MAP###
data=read.csv("FS-MAP-MAT-FN-XG.csv",sep=",",header=TRUE)
data$Planting5<- factor(data$Planting5,levels = c('Intercropping','Rotation'),ordered = TRUE)
data$Planting5<- factor(data$Planting5,levels = c('Intercropping','Rotation'),ordered = TRUE)
p13<-ggplot(data=data,aes(x=MAP5,y=lnRRCH4,group=Planting5,color=Planting5))+
labs(y=expression(paste("lnRR"," ","(",CH[4]," ","emission",")")),x="Mean annual precipitation (mm)",title="(e)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting5,fill=Planting5),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(MAP5, lnRRCH4, colour = Planting5), method = "spearman",size=7,label.x.npc = 0.5,label.y.npc = 0.3)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,2000),breaks=c(0, 500, 1000, 1500, 2000)) +
scale_y_continuous(limits=c(-6,2),breaks=c(-6,-4,-2, 0, 2))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p13
ggsave("FS2e.pdf", plot = p13, device = cairo_pdf, dpi = 600,width = 5.5, height = 4)
################
####6.SOC-MAT###
data=read.csv("FS-MAP-MAT-FN-XG.csv",sep=",",header=TRUE)
data$Planting1<- factor(data$Planting1,levels = c('Intercropping','Rotation'),ordered = TRUE)
p2<-ggplot(data=data,aes(x=MAT1,y=lnRRSOC,group=Planting1,color=Planting1))+
labs(y=expression(paste("lnRR"," ","(","SOC"," ","stock",")")),x="Mean annual temperature (°C)",title="(f)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting1,fill=Planting1),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(MAT1, lnRRSOC, colour = Planting1), method = "spearman",size=7)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,30),breaks=c(0, 10, 20, 30)) +
scale_y_continuous(limits=c(-0.5,1.5),breaks=c(-0.5, 0, 0.5, 1, 1.5))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p2
ggsave("FS2f.pdf", plot = p2, device = cairo_pdf, dpi = 600,width = 5.5, height = 4)
#############
#######7.NL-MAT###
data$Planting2<- as.factor(data$Planting2)
data$Planting2<- factor(data$Planting2,levels = c('Intercropping','Rotation'),ordered = TRUE)
p5<-ggplot(data=data,aes(x=MAT2,y=lnRRNL,color=Planting2,group=Planting2))+
labs(y=expression(paste("lnRR"," ","(","Nitrate leaching",")")),x="Mean annual temperature (°C)",title="(g)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting2,fill=Planting2),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(MAT2, lnRRNL, color = Planting2), method = "spearman",size=7,label.x.npc = 0.45,label.y.npc = 0.9)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,30),breaks=c(0, 10, 20, 30)) +
scale_y_continuous(limits=c(-2,2),breaks=c(-2, -1, 0, 1, 2))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p5
ggsave("FS2g.pdf", plot = p5, device = cairo_pdf, dpi = 600,width = 5.5, height = 4)
###################
####8.CO2-MAT###
data=read.csv("FS-MAP-MAT-FN-XG.csv",sep=",",header=TRUE)
data$Planting3<- factor(data$Planting3,levels = c('Intercropping','Rotation'),ordered = TRUE)
p8<-ggplot(data=data,aes(x=MAT3,y=lnRRCO2,group=Planting3,color=Planting3))+
labs(y=expression(paste("lnRR"," ","(",CO[2]," ","emission",")")),x="Mean annual temperature (°C)",title="(h)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting3,fill=Planting3),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(MAT3, lnRRCO2, colour = Planting3), method = "spearman",size=7,label.x.npc = 0.05,label.y.npc = 0.2)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,30),breaks=c(0, 10, 20, 30)) +
scale_y_continuous(limits=c(-3,2),breaks=c(-3,-2, -1, 0, 1, 2))+

theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
    axis.text.y= element_text(size=18, color="black", face= "bold"),
    axis.text.x= element_text(size=18, color="black", face= "bold"),
    axis.title.x=element_text(size=20),
    axis.title.y=element_text(size=20),
    plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p8
ggsave("FS2h.pdf", plot = p8, device = cairo_pdf, dpi = 600,width = 5.5, height = 4)
#####################
####9.N2O-MAT###
data=read.csv("FS-MAP-MAT-FN-XG.csv",sep=",",header=TRUE)
data$Planting4<- factor(data$Planting4,levels = c('Intercropping','Rotation'),ordered = TRUE)
p11<-ggplot(data=data,aes(x=MAT4,y=lnRRN2O,group=Planting4,color=Planting4))+
labs(y=expression(paste("lnRR"," ","(",N[2],"O"," ","emission",")")),x="Mean annual temperature (°C)",title="(i)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting4,fill=Planting4),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(MAT4, lnRRN2O, colour = Planting4), method = "spearman",size=7,label.x.npc = 0.45,label.y.npc = 0.15)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,30),breaks=c(0, 10, 20, 30)) +
scale_y_continuous(limits=c(-3,3),breaks=c(-3,-1.5, 0, 1.5, 3))+

theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
    axis.text.y= element_text(size=18, color="black", face= "bold"),
    axis.text.x= element_text(size=18, color="black", face= "bold"),
    axis.title.x=element_text(size=20),
    axis.title.y=element_text(size=20),
    plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p11
ggsave("FS2i.pdf", plot = p11, device = cairo_pdf, dpi = 600,width = 5.5, height = 4)
###################
#######10.CH4-MAT###
data$Planting5<- factor(data$Planting5,levels = c('Intercropping','Rotation'),ordered = TRUE)
p14<-ggplot(data=data,aes(x=MAT5,y=lnRRCH4,group=Planting5,color=Planting5))+
labs(y=expression(paste("lnRR"," ","(",CH[4]," ","emission",")")),x="Mean annual temperature (°C)",title="(j)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting5,fill=Planting5),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(MAT5, lnRRCH4, colour = Planting5), method = "spearman",size=7,label.x.npc = 0.45,label.y.npc = 0.3)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,30),breaks=c(0, 10, 20, 30)) +
scale_y_continuous(limits=c(-6,2),breaks=c(-6,-4,-2, 0, 2))+

theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
    axis.text.y= element_text(size=18, color="black", face= "bold"),
    axis.text.x= element_text(size=18, color="black", face= "bold"),
    axis.title.x=element_text(size=20),
    axis.title.y=element_text(size=20),
    plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p14
ggsave("FS2j.pdf", plot = p14, device = cairo_pdf, dpi = 600,width = 5.5, height = 4)
##########################################
data=read.csv("SOCNLGHG-PHBDTSOC.csv",sep=",",header=TRUE)
####11.SOC-PH###
data$Planting1<- as.factor(data$Planting1)
data$Planting1<- factor(data$Planting1,levels = c('Intercropping','Rotation'),ordered = TRUE)

p16<-ggplot(data=data,aes(x=PH1,y=lnRRPH1,color=Planting1,group=Planting1))+
labs(y=expression(paste("lnRR"," ","(","SOC"," ","stock",")")),x="pH",title="(k)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting1,fill=Planting1),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(PH1, lnRRPH1, color = Planting1), method = "spearman",size=7)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(4,8),breaks=c(4, 5,6, 7,8)) +
scale_y_continuous(limits=c(-0.5,1.5),breaks=c(-0.5, 0, 0.5, 1, 1.5))+

theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
    axis.text.y= element_text(size=18, color="black", face= "bold"),
    axis.text.x= element_text(size=18, color="black", face= "bold"),
    axis.title.x=element_text(size=20),
    axis.title.y=element_text(size=20),
    plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p16
ggsave("(k)PFS2k.pdf", plot = p16, device = cairo_pdf, dpi = 600,width = 5.5, height = 4)
#######
####12.NL-PH###
data$Planting2<- factor(data$Planting2,levels = c('Intercropping','Rotation'),ordered = TRUE)
p19<-ggplot(data=data,aes(x=PH2,y=lnRRPH2,group=Planting2,color=Planting2))+
labs(y=expression(paste("lnRR"," ","(","Nitrate"," ","leaching",")")),x="pH",title="(l)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting2,fill=Planting2),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(PH2, lnRRPH2, colour = Planting2), method = "spearman",size=7,label.x.npc = 0.25,label.y.npc = 1.0)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(4,8),breaks=c(4, 5,6, 7,8)) +
scale_y_continuous(limits=c(-2,2),breaks=c(-2,-1,0, 1,2))+

theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
    axis.text.y= element_text(size=18, color="black", face= "bold"),
    axis.text.x= element_text(size=18, color="black", face= "bold"),
    axis.title.x=element_text(size=20),
    axis.title.y=element_text(size=20),
    plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p19
ggsave("(l)PFS2l.pdf", plot = p19, device = cairo_pdf, dpi = 600,width = 5.5, height = 4#########
   ####13.CO2-PH###
data$Planting3<- factor(data$Planting3,levels = c('Intercropping','Rotation'),ordered = TRUE)
p22<-ggplot(data=data,aes(x=PH3,y=lnRRPH3,group=Planting3,color=Planting3))+
labs(y=expression(paste("lnRR"," ","(",CO[2]," ","emission",")")),x="pH",title="(m)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting3,fill=Planting3),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(PH3, lnRRPH3, colour = Planting3), method = "spearman",size=7,label.x.npc = 0.1,label.y.npc = 0.2)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(4,8),breaks=c(4, 5,6, 7,8)) +
scale_y_continuous(limits=c(-3,2),breaks=c(-3,-2,-1,0, 1,2))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p22
ggsave("(m)PFS2m.pdf", plot = p22, device = cairo_pdf, dpi = 600,width = 5.5, height = 4######
####14.N2O-PH###
data$Planting4<- factor(data$Planting4,levels = c('Intercropping','Rotation'),ordered = TRUE)
p25<-ggplot(data=data,aes(x=PH4,y=lnRRPH4,group=Planting4,color=Planting4))+
labs(y=expression(paste("lnRR"," ","(",N[2],"O"," ","emission",")")),x="pH",title="(n)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting4,fill=Planting4),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(PH4, lnRRPH4, colour = Planting4), method = "spearman",size=7,label.x.npc = 0.05,label.y.npc = 0.9)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(4,8),breaks=c(4, 5,6, 7,8)) +
scale_y_continuous(limits=c(-3,3),breaks=c(-3,-1.5,0, 1.5,3))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p25
ggsave("(n)PFS2n.pdf", plot = p25, device = cairo_pdf, dpi = 600,width = 5.5, height = 4#######
####15.CH4-PH###
data$Planting5<- factor(data$Planting5,levels = c('Intercropping','Rotation'),ordered = TRUE)
p28<-ggplot(data=data,aes(x=PH5,y=lnRRPH5,group=Planting5,color=Planting5))+
labs(y=expression(paste("lnRR"," ","(",CH[4]," ","emission",")")),x="pH",title="(o)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting5,fill=Planting5),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(PH5, lnRRPH5, colour = Planting5), method = "spearman",size=7,label.x.npc = 0.5,label.y.npc = 0.3)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(4,8),breaks=c(4, 5,6, 7,8)) +
scale_y_continuous(limits=c(-6,2),breaks=c(-6,-4,-2,0, 2))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p28
ggsave("(o)PFS2o.pdf", plot = p28, device = cairo_pdf, dpi = 600,width = 5.5, height = 4########################
######16.SOC-BD###
data=read.csv("SOCNLGHG-PHBDTSOC.csv",sep=",",header=TRUE)
data$Planting11<- factor(data$Planting11,levels = c('Intercropping','Rotation'),ordered = TRUE)
p17<-ggplot(data=data,aes(x=BD1,y=lnRRBD1,group=Planting11,color=Planting11))+
geom_point(size=5)+
geom_smooth(aes(color = Planting11,fill=Planting11),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(BD1, lnRRBD1, colour = Planting11), method = "spearman",size=7)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0.8,2),breaks=c(1,1.5,2)) +
scale_y_continuous(limits=c(-0.5,1.5),breaks=c(-0.5,0,0.5,1,1.5))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p17
ggsave("(p)BFS2p.pdf", plot = p17, device = cairo_pdf, dpi = 600,width = 5.5, height = 4#######
####17.NL-BD###
data=read.csv("SOCNLGHG-PHBDTSOC.csv",sep=",",header=TRUE)
data$Planting22<- factor(data$Planting22,levels = c('Intercropping','Rotation'),ordered = TRUE)
p20<-ggplot(data=data,aes(x=BD2,y=lnRRBD2,group=Planting22,color=Planting22))+
labs(y=expression(paste("lnRR"," ","(","Nitrate"," ","leaching",")")),x=expression(paste("Bulk density"," ","(",g," ",cm^{-3},")")),title="(q)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting22,fill=Planting22),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(BD2, lnRRBD2, colour = Planting22), method = "spearman",size=7,label.x.npc = 0.45,label.y.npc = 0.9)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0.8,2),breaks=c(1,1.5,2)) +
scale_y_continuous(limits=c(-2,2),breaks=c(-2,-1,0,1,2))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p20
ggsave("(q)BFS2q.pdf", plot = p20, device = cairo_pdf, dpi = 600,width = 5.5, height = 4#####################
####18.CO2-BD###
data=read.csv("SOCNLGHG-PHBDTSOC.csv",sep=",",header=TRUE)
data$Planting33<- factor(data$Planting33,levels = c('Intercropping','Rotation'),ordered = TRUE)
p23<-ggplot(data=data,aes(x=BD3,y=lnRRBD3,group=Planting33,color=Planting33))+
labs(y=expression(paste("lnRR"," ","(",CO[2]," ","emission",")")),x=expression(paste("Bulk density"," ","(",g," ",cm^{-3},")")),title="(r)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting33,fill=Planting33),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(BD3, lnRRBD3, colour = Planting33), method = "spearman",size=7,label.x.npc = 0.05,label.y.npc = 0.99)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0.8,2),breaks=c(1,1.5,2)) +
scale_y_continuous(limits=c(-3,2),breaks=c(-3,-2,-1,0,1,2))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p23
ggsave("(r)BFS2r.pdf", plot = p23, device = cairo_pdf, dpi = 600,width = 5.5, height = 4###########
####19.N2O-BD###
data=read.csv("SOCNLGHG-PHBDTSOC.csv",sep=",",header=TRUE)
data$Planting44<- factor(data$Planting44,levels = c('Intercropping','Rotation'),ordered = TRUE)
p26<-ggplot(data=data,aes(x=BD4,y=lnRRBD4,group=Planting44,color=Planting44))+
labs(y=expression(paste("lnRR"," ","(",N[2],"O"," ","emission",")")),x=expression(paste("Bulk density"," ","(",g," ",cm^{-3},")")),title="(s)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting44,fill=Planting44),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(BD4, lnRRBD4, colour = Planting44), method = "spearman",size=7,label.x.npc = 0.45,l35el.y.npc = 0.8)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0.8,2),breaks=c(1,1.5,2)) +
scale_y_continuous(limits=c(-3,3),breaks=c(-3,-1.5,0,1.5,3))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p26
ggsave("(s)BFS2s.pdf", plot = p26, device = cairo_pdf, dpi = 600,width = 5.5, height = 4################
####20.CH4-BD###
data=read.csv("SOCNLGHG-PHBDTSOC.csv",sep=",",header=TRUE)
data$Planting55<- factor(data$Planting55,levels = c('Intercropping','Rotation'),ordered = TRUE)
p29<-ggplot(data=data,aes(x=BD5,y=lnRRBD5,group=Planting55,color=Planting55))+
labs(y=expression(paste("lnRR"," ","(",CH[4]," ","emission",")")),x=expression(paste("Bulk density"," ","(",g," ",cm^{-3},")")),title="(t)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting55,fill=Planting55),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(BD5, lnRRBD5, colour = Planting55), method = "spearman",size=7,label.x.npc = 0.45,label.y.npc = 0.3)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0.8,2),breaks=c(1,1.5,2)) +
scale_y_continuous(limits=c(-6,2),breaks=c(-6,-4,-2,0,2))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p29
ggsave("(t)BFS2t.pdf", plot = p29, device = cairo_pdf, dpi = 600,width = 5.5, height = 4##################3
####21.SOC-TSOC###
data=read.csv("SOCNLGHG-PHBDTSOC.csv",sep=",",header=TRUE)
data$Planting111<- factor(data$Planting111,levels = c('Intercropping','Rotation'),ordered = TRUE)
p18<-ggplot(data=data,aes(x=TSOC1,y=lnRRTSOC1,group=Planting111,color=Planting111))+
labs(y=expression(paste("lnRR"," ","(","SOC"," ","stock",")")),x=expression(paste("Initial SOC"," ","(",g," ",kg^{-1},")")),title="(u)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting111,fill=Planting111),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(TSOC1, lnRRTSOC1, colour = Planting111), method = "spearman",size=7)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(3,32),breaks=c(5,10,15,20,25,30)) +
scale_y_continuous(limits=c(-0.5,1.5),breaks=c(-0.5,0,0.5,1,1.5))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p18
ggsave("(u)IFS2u.pdf", plot = p18, device = cairo_pdf, dpi = 600,width = 5.5, height = 4#########
####22.NL-TSOC###
data=read.csv("SOCNLGHG-PHBDTSOC.csv",sep=",",header=TRUE)
data$Planting222<- factor(data$Planting222,levels = c('Intercropping','Rotation'),ordered = TRUE)
p21<-ggplot(data=data,aes(x=TSOC2,y=lnRRTSOC2,group=Planting222,color=Planting222))+
labs(y=expression(paste("lnRR"," ","(","Nitrate"," ","leaching",")")),x=expression(paste("Initial SOC"," ","(",g," ",kg^{-1},")")),title="(v)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting222,fill=Planting222),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(TSOC2, lnRRTSOC2, colour = Planting222), method = "spearman",size=7)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(3,32),breaks=c(5,10,15,20,25,30)) +
scale_y_continuous(limits=c(-2,2),breaks=c(-2,-1,0,1,2))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p21
ggsave("(v)IFS2v.pdf", plot = p21, device = cairo_pdf, dpi = 600,width = 5.5, height = 4#############
####23.CO2-TSOC###
data=read.csv("SOCNLGHG-PHBDTSOC.csv",sep=",",header=TRUE)
data$Planting333<- factor(data$Planting333,levels = c('Intercropping','Rotation'),ordered = TRUE)
p24<-ggplot(data=data,aes(x=TSOC3,y=lnRRTSOC3,group=Planting333,color=Planting333))+
labs(y=expression(paste("lnRR"," ","(",CO[2]," ","emission",")")),x=expression(paste("Initial SOC"," ","(",g," ",kg^{-1},")")),title="(w)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting333,fill=Planting333),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(TSOC3, lnRRTSOC3, colour = Planting333), method = "spearman",size=7,label.x.npc = 0.4,label.y.npc = 0.99)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(3,32),breaks=c(5,10,15,20,25,30)) +
scale_y_continuous(limits=c(-3,2),breaks=c(-3,-2,-1,0,1,2))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p24
ggsave("(w)IFS2w.pdf", plot = p24, device = cairo_pdf, dpi = 600,width = 5.5, height = 4################
####24.N2O-TSOC###
data=read.csv("SOCNLGHG-PHBDTSOC.csv",sep=",",header=TRUE)
data$Planting444<- factor(data$Planting444,levels = c('Intercropping','Rotation'),ordered = TRUE)
p27<-ggplot(data=data,aes(x=TSOC4,y=lnRRTSOC4,group=Planting444,color=Planting444))+
labs(y=expression(paste("lnRR"," ","(",N[2],"O"," ","emission",")")),x=expression(paste("Initial SOC"," ","(",g," ",kg^{-1},")")),title="(x)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting444,fill=Planting444),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(TSOC4, lnRRTSOC4, colour = Planting444), method = "spearman",size=7,label.x.npc = 0.4,label.y.npc = 0.95)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(3,32),breaks=c(5,10,15,20,25,30)) +
scale_y_continuous(limits=c(-3,3),breaks=c(-3,-1.5,0,1.5,3))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p27
ggsave("(x)IFS2x.pdf", plot = p27, device = cairo_pdf, dpi = 600,width = 5.5, height = 4###########
####25.CH4-TSOC###
stat_cor(aes(TotalBiomass, lnRR1, color = Planting), method = "spearman",size=7,label.x.npc = 0.1,label.y.npc =1)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,20),breaks=c(0,5,10,15,20)) +
scale_y_continuous(limits=c(-0.5,1.5),breaks=c(-0.5,0,0.5,1,1.5))+

theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p1
###################################################################################
################################Fig.S7######################
2.SOC-RootBiomass###
p2<-ggplot(data=data,aes(x=RootBiomass,y=lnRR2,color=Planting,group=Planting))+
labs(y=expression(paste("lnRR"," ","(","SOC"," ","stock",")")),x=expression(paste("Ratio of Root Biomass")),title="(e)")+
geom_point(size=5)+

geom_smooth(aes(color = Planting,fill=Planting),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(RootBiomass, lnRR2, color = Planting), method = "spearman",size=7,label.x.npc = 0.2,label.y.npc =1)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,10),breaks=c(2,4,6,8,10)) +
scale_y_continuous(limits=c(-0.5,1.5),breaks=c(-0.5,0,0.5,1,1.5))+

theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p2
ggsave("(e)RB-SOC.pdf", p2, device = "pdf")
############
####3.SOC-FN###
data=read.csv("FS-MAP-MAT-FN-XG.csv",sep=",",header=TRUE)
data$Planting1<- factor(data$Planting1,levels = c('Intercropping','Rotation'),ordered = TRUE)
p3<-ggplot(data=data,aes(x=FN1,y=lnRRSOC,group=Planting1,color=Planting1))+
labs(y=expression(paste("lnRR"," ","(","SOC"," ","stock",")")),x=expression(paste("Nitrogen application rate"," ","(","kg"," ","N"," ",ha^{-1},")")),title="(i)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting1,fill=Planting1),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(FN1, lnRRSOC, colour = Planting1), method = "spearman",size=7)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,800),breaks=c(0, 200, 400, 600,800)) +
scale_y_continuous(limits=c(-0.5,1.5),breaks=c(-0.5, 0, 0.5, 1, 1.5))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=16, color="black", face= "bold"),
axis.text.x= element_text(size=16, color="black", face= "bold"),
axis.title.x=element_text(size=18),
axis.title.y=element_text(size=18),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p3
ggsave("(i)FN-SOC.pdf", p3, device = "pdf")
############
####3.SOC-GrowthPeriod###
data=read.csv("SOCALL2-Q3.csv",sep=",",header=TRUE)
p4<-ggplot(data=data,aes(x=GrowthPeriod,y=lnRR3,group=Planting,color=Planting))+
labs(y=expression(paste("lnRR"," ","(","SOC"," ","stock",")")),x=expression(paste("Ratio of Growth Period")),title="(m)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting,fill=Planting),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(GrowthPeriod, lnRR3, colour = Planting), method = "spearman",size=7,label.x.npc = 0.4,label.y.npc =1)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,3),breaks=c(0,1,2,3)) +
scale_y_continuous(limits=c(-0.5,1.5),breaks=c(-0.5,0,0.5,1,1.5))+

theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p4
ggsave("(m)GP-SOC.pdf", p4, device = "pdf")
#######NL-??????ͼ##################################
data=read.csv("NLALL3-Q333.csv",sep=",",header=TRUE)####?ް?????????####
####1.NL-TotalBiomass###
data$Planting<- as.factor(data$Planting)
data$Planting<- factor(data$Planting,levels = c('Intercropping','Rotation'),ordered = TRUE)
p1<-ggplot(data=data,aes(x=TotalBiomass,y=lnRR1,color=Planting,group=Planting))+
labs(y=expression(paste("lnRR"," ","(","Nitrate"," ","leaching",")")),x=expression(paste("Ratio of Total Biomass")),title="(b)")+
geom_point(size=5)+

geom_smooth(aes(color = Planting,fill=Planting),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(TotalBiomass, lnRR1, color = Planting), method = "spearman",size=7,label.x.npc = 0.5,label.y.npc =1)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,20),breaks=c(0,5,10,15,20)) +
scale_y_continuous(limits=c(-2,2),breaks=c(-2,-1,0,1,2))+

theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p1
ggsave("(b)TB-NL.pdf", p1, device = "pdf")
####2.NL-RootBiomass###
p2<-ggplot(data=data,aes(x=RootBiomass,y=lnRR2,color=Planting,group=Planting))+
labs(y=expression(paste("lnRR"," ","(","Nitrate"," ","leaching",")")),x=expression(paste("Ratio of Root Biomass")),title="(f)")+
geom_point(size=5)+

geom_smooth(aes(color = Planting,fill=Planting),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(RootBiomass, lnRR2, color = Planting), method = "spearman",size=7,label.x.npc = 0.5,label.y.npc =1)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,10),breaks=c(2,4,6,8,10)) +
scale_y_continuous(limits=c(-2,2),breaks=c(-2,-1,0,1,2))+
#scale_x_continuous(limits=c("A","B"), breaks=seq(??ʼֵ????ֵֹ??????))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p2
ggsave("(f)RB-NL.pdf", p2, device = "pdf")
#####################
data=read.csv("FS-MAP-MAT-FN-XG.csv",sep=",",header=TRUE)
data$Planting2<- as.factor(data$Planting2)
data$Planting2<- factor(data$Planting2,levels = c('Intercropping','Rotation'),ordered = TRUE)
p6<-ggplot(data=data,aes(x=FN2,y=lnRRNL,color=Planting2,group=Planting2))+
labs(y=expression(paste("lnRR"," ","(","Nitrate leaching",")")),x=expression(paste("Nitrogen application rate"," ","(","kg"," ","N"," ",ha^{-1},")")),title="(j)")+
geom_point(size=5)+

geom_smooth(aes(color = Planting2,fill=Planting2),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(FN2, lnRRNL, color = Planting2), method = "spearman",size=7,label.x.npc = 0.4,label.y.npc = 0.95)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,800),breaks=c(0,200,400,600,800)) +
scale_y_continuous(limits=c(-2,2),breaks=c(-2, -1, 0, 1, 2))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=16, color="black", face= "bold"),
axis.text.x= element_text(size=16, color="black", face= "bold"),
axis.title.x=element_text(size=18),
axis.title.y=element_text(size=18),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p6
ggsave("(j)FN-NL.pdf", p6, device = "pdf")

####3.NL-GrowthPeriod###
data=read.csv("NLALL3-Q333.csv",sep=",",header=TRUE)####?ް?????????####
####1.NL-TotalBiomass###
data$Planting<- as.factor(data$Planting)
data$Planting<- factor(data$Planting,levels = c('Intercropping','Rotation'),ordered = TRUE)
p3<-ggplot(data=data,aes(x=GrowthPeriod,y=lnRR3,group=Planting,color=Planting))+
labs(y=expression(paste("lnRR"," ","(","Nitrate"," ","leaching",")")),x=expression(paste("Ratio of Growth Period")),title="(n)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting,fill=Planting),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(GrowthPeriod, lnRR3, colour = Planting), method = "spearman",size=7,label.x.npc = 0.5,label.y.npc =1)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,3),breaks=c(0,1,2,3)) +
scale_y_continuous(limits=c(-2,2),breaks=c(-2, -1, 0, 1, 2))+

theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p3
ggsave("(n)GP-NL.pdf", p3, device = "pdf")
############CO2####################
data=read.csv("CO2ALL2-Q3.csv",sep=",",header=TRUE)
####1.CO2-TotalBiomass###
data$Planting<- as.factor(data$Planting)
data$Planting<- factor(data$Planting,levels = c('Intercropping','Rotation'),ordered = TRUE)
p1<-ggplot(data=data,aes(x=TotalBiomass,y=lnRR1,color=Planting,group=Planting))+
labs(y=expression(paste("lnRR"," ","(",CO[2]," ","emission",")")),x=expression(paste("Ratio of Total Biomass")),title="(c)")+
geom_point(size=5)+

geom_smooth(aes(color = Planting,fill=Planting),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(TotalBiomass, lnRR1, color = Planting), method = "spearman",size=7,label.x.npc = 0.2,label.y.npc =1)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,20),breaks=c(0,5,10,15,20)) +
scale_y_continuous(limits=c(-3,2),breaks=c(-3,-2, -1, 0, 1, 2))+

theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p1
ggsave("(c)TB-CO2.pdf", p1, device = "pdf")
####2.CO2-RootBiomass###
p2<-ggplot(data=data,aes(x=RootBiomass,y=lnRR2,color=Planting,group=Planting))+
labs(y=expression(paste("lnRR"," ","(",CO[2]," ","emission",")")),x=expression(paste("Ratio of Root Biomass")),title="(g)")+
geom_point(size=5)+

geom_smooth(aes(color = Planting,fill=Planting),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(RootBiomass, lnRR2, color = Planting), method = "spearman",size=7,label.x.npc = 0.05,label.y.npc =1)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,10),breaks=c(0,2,4,6,8,10)) +
scale_y_continuous(limits=c(-3,2),breaks=c(-3,-2, -1, 0, 1, 2))+
#scale_x_continuous(limits=c("A","B"), breaks=seq(??ʼֵ????ֵֹ??????))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p2
ggsave("(g)RB-CO2.pdf", p2, device = "pdf")
###########9.CO2-FN###
data=read.csv("FS-MAP-MAT-FN-XG.csv",sep=",",header=TRUE)
data$Planting3<- factor(data$Planting3,levels = c('Intercropping','Rotation'),ordered = TRUE)
p9<-ggplot(data=data,aes(x=FN3,y=lnRRCO2,group=Planting3,color=Planting3))+
labs(y=expression(paste("lnRR"," ","(",CO[2]," ","emission",")")),x=expression(paste("Nitrogen application rate"," ","(","kg"," ","N"," ",ha^{-1},")")),title="(k)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting3,fill=Planting3),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(FN3, lnRRCO2, colour = Planting3), method = "spearman",size=7,label.x.npc = 0.4,label.y.npc = 0.99)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,800),breaks=c(0,200,400,600,800)) +
scale_y_continuous(limits=c(-3,2),breaks=c(-3,-2, -1, 0, 1, 2))+

theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=16, color="black", face= "bold"),
axis.text.x= element_text(size=16, color="black", face= "bold"),
axis.title.x=element_text(size=18),
axis.title.y=element_text(size=18),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p9
ggsave("(k)FN-CO2.pdf", p9, device = "pdf")
####3.CO2-GrowthPeriod###
p3<-ggplot(data=data,aes(x=GrowthPeriod,y=lnRR3,group=Planting,color=Planting))+
labs(y=expression(paste("lnRR"," ","(",CO[2]," ","emission",")")),x=expression(paste("Ratio of Growth Period")),title="(o)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting,fill=Planting),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(GrowthPeriod, lnRR3, colour = Planting), method = "spearman",size=7,label.x.npc = 0,label.y.npc = 0.2)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,3),breaks=c(0,1,2,3)) +
scale_y_continuous(limits=c(-3,2),breaks=c(-3,-2, -1, 0, 1, 2))+

theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p3
ggsave("(o)GP-CO2.pdf", p3, device = "pdf")
############N2O###############
data=read.csv("N2OALL2-Q3.csv",sep=",",header=TRUE)
####1.N2O-TotalBiomass###
data$Planting<- as.factor(data$Planting)
data$Planting<- factor(data$Planting,levels = c('Intercropping','Rotation'),ordered = TRUE)
p1<-ggplot(data=data,aes(x=TotalBiomass,y=lnRR1,color=Planting,group=Planting))+
labs(y=expression(paste("lnRR"," ","(",N[2],"O"," ","emission",")")),x=expression(paste("Ratio of Total Biomass")),title="(d)")+
geom_point(size=5)+

geom_smooth(aes(color = Planting,fill=Planting),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(TotalBiomass, lnRR1, color = Planting), method = "spearman",size=7,label.x.npc = 0.4,label.y.npc =1)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,20),breaks=c(0,5,10,15,20)) +
scale_y_continuous(limits=c(-5,3),breaks=c(-4,-2,0,2))+

theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p1
ggsave("(d)TB-N2O.pdf", p1, device = "pdf")
####2.N2O-RootBiomass###
p2<-ggplot(data=data,aes(x=RootBiomass,y=lnRR2,color=Planting,group=Planting))+
labs(y=expression(paste("lnRR"," ","(",N[2],"O"," ","emission",")")),x=expression(paste("Ratio of Root Biomass")),title="(h)")+
geom_point(size=5)+

geom_smooth(aes(color = Planting,fill=Planting),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(RootBiomass, lnRR2, color = Planting), method = "spearman",size=7,label.x.npc = 0.2,label.y.npc = 0.2)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,10),breaks=c(0,2,4,6,8,10)) +
scale_y_continuous(limits=c(-5,3),breaks=c(-4,-2,0,2))+
#scale_x_continuous(limits=c("A","B"), breaks=seq(??ʼֵ????ֵֹ??????))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p2
ggsave("(h)RB-N2O.pdf", p2, device = "pdf")
#################
###12.N2O-FN###
data=read.csv("FS-MAP-MAT-FN-XG.csv",sep=",",header=TRUE)
data$Planting4<- factor(data$Planting4,levels = c('Intercropping','Rotation'),ordered = TRUE)
p12<-ggplot(data=data,aes(x=FN4,y=lnRRN2O,group=Planting4,color=Planting4))+
labs(y=expression(paste("lnRR"," ","(",N[2],"O"," ","emission",")")),x=expression(paste("Nitrogen application rate"," ","(","kg"," ","N"," ",ha^{-1},")")),title="(l)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting4,fill=Planting4),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(FN4, lnRRN2O, colour = Planting4), method = "spearman",size=7,label.x.npc = 0.4,label.y.npc = 0.95)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,800),breaks=c(0,200,400,600,800)) +
scale_y_continuous(limits=c(-5,3),breaks=c(-4,-2,0,2))+
theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=16, color="black", face= "bold"),
axis.text.x= element_text(size=16, color="black", face= "bold"),
axis.title.x=element_text(size=18),
axis.title.y=element_text(size=18),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p12
ggsave("(l)FN-N2O.pdf", p12, device = "pdf")
####3.N2O-GrowthPeriod###
p3<-ggplot(data=data,aes(x=GrowthPeriod,y=lnRR3,group=Planting,color=Planting))+
labs(y=expression(paste("lnRR"," ","(",N[2],"O"," ","emission",")")),x=expression(paste("Ratio of Growth Period")),title="(p)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting,fill=Planting),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(GrowthPeriod, lnRR3, colour = Planting), method = "spearman",size=7,label.x.npc = 0,label.y.npc = 0.9)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,3),breaks=c(0,1,2,3)) +
scale_y_continuous(limits=c(-5,3),breaks=c(-4,-2,0,2))+

theme(legend.position="none",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=18, color="black", face= "bold"),
axis.text.x= element_text(size=18, color="black", face= "bold"),
axis.title.x=element_text(size=20),
axis.title.y=element_text(size=20),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p3
ggsave("(p)GP-N2O.pdf", p3, device = "pdf")
####????ȡͼ??#####
p13<-ggplot(data=data,aes(x=FN4,y=lnRRN2O,group=Planting4,color=Planting4))+
labs(y=expression(paste("lnRR"," ","(",N[2],"O"," ","emission",")")),x=expression(paste("Nitrogen application rate"," ","(","kg"," ","N"," ",ha^{-1},")")),title="(l)")+
geom_point(size=5)+
geom_smooth(aes(color = Planting4,fill=Planting4),method="lm",size=2,se=TRUE,alpha=0.3,fullrange = TRUE)+
stat_cor(aes(FN4, lnRRN2O, colour = Planting4), method = "spearman",size=7,label.x.npc = 0.4,label.y.npc = 0.95)+
scale_colour_manual(values=c("#00B76D","salmon"))+
scale_fill_manual(values = c("#00B76D","salmon"))+
scale_x_continuous(limits=c(0,800),breaks=c(0,200,400,600,800)) +
scale_y_continuous(limits=c(-5,3),breaks=c(-4,-2,0,2))+
theme(legend.position="right",legend.text =element_text(size=18),panel.grid = element_blank(),panel.background = element_rect(fill='transparent',color = 'black'),
axis.text.y= element_text(size=16, color="black", face= "bold"),
axis.text.x= element_text(size=16, color="black", face= "bold"),
axis.title.x=element_text(size=18),
axis.title.y=element_text(size=18),
plot.title = element_text(size = 25))+
guides(color=guide_legend(title = " "))
p13
ggsave("??????ȡͼ??(l)FN-N2O.pdf", p13, device = "pdf")
###################################################################################
######################################Fig.S8########################
library(base)
library(graphics)
library(stringr)
library(data.table)
library(stats)
library(rJava)
library(xlsx)
library(xlsxjars)
library(readxl)
library(openxlsx)
library(stringi)
library(akima)
library(grDevices)
library(fields)
library(Hmisc)
library(sp)
library(utils)
library(pointr)
library(fBasics)
library(ggplot2)
library(RColorBrewer)
####SOC:间作Nrate+Clay#####
data1=read.csv("SOClnRR-akima-NC1.csv",sep=",",header=TRUE)###
head(data1)
dim(data1)
Xrange<-max(data1$Nrate)-min(data1$Nrate)
Yrange<-max(data1$CLAY)-min(data1$CLAY)
Zrange<-max(data1$lnRR)-min(data1$lnRR)
Xrange
Yrange
Zrange
Xmax<-ifelse(max(data1$Nrate)==0,0.1*abs(Xrange),round(max(data1$Nrate)+0.1*abs(Xrange),1))
Xmin<-ifelse(min(data1$Nrate)==0,-0.1*abs(Xrange),round(min(data1$Nrate)-0.1*abs(Xrange),1))
Xmax
Xmin
Ymax<-ifelse(max(data1$CLAY)==0,0.1*abs(Yrange),round(max(data1$CLAY)+0.1*abs(Yrange),1))
Ymin<-ifelse(min(data1$CLAY)==0,-0.1*abs(Yrange),round(min(data1$CLAY)-0.1*abs(Yrange),1))
Ymax
Ymin

interplinearS<-interp(data1$Nrate,data1$CLAY,data1$lnRR,nx=500,ny=500,linear=TRUE,duplicate = 'mean')
interplinearS$x
interplinearS$y
interplinearS$z

lgnd=1
lgndmar=5.1
lgnd
lgndmar

cairo_pdf("FS4a.pdf",width = 4.6,height = 3.6)
p1<-image.plot(interplinearS,main=expression(paste("(","a",")"," ","lnRR"," ","(","SOC stock",")")),col.main='black',xlim=c(Xmin,Xmax),ylim=c(Ymin,Ymax),xlab=expression(paste("Nitrogen application rate"," ","(","kg"," ","N"," ",ha^{-1},")")),ylab='Clay content (%)',
               legend.shrink=1,legend.width=1,legend.mar=5.1,horizontal=FALSE)
p1+scale_color_distiller(palette="Spectral")
p1+scale_color_distiller(palette="Greens")
p1
dev.off()
####SOC:轮作Nrate+Clay#####
data1=read.csv("SOClnRR-akima-NC2.csv",sep=",",header=TRUE)###
head(data1)
dim(data1)
Xrange<-max(data1$Nrate)-min(data1$Nrate)
Yrange<-max(data1$CLAY)-min(data1$CLAY)
Zrange<-max(data1$lnRR)-min(data1$lnRR)
Xrange
Yrange
Zrange
Xmax<-ifelse(max(data1$Nrate)==0,0.1*abs(Xrange),round(max(data1$Nrate)+0.1*abs(Xrange),1))
Xmin<-ifelse(min(data1$Nrate)==0,-0.1*abs(Xrange),round(min(data1$Nrate)-0.1*abs(Xrange),1))
Xmax
Xmin
Ymax<-ifelse(max(data1$CLAY)==0,0.1*abs(Yrange),round(max(data1$CLAY)+0.1*abs(Yrange),1))
Ymin<-ifelse(min(data1$CLAY)==0,-0.1*abs(Yrange),round(min(data1$CLAY)-0.1*abs(Yrange),1))
Ymax
Ymin

interplinearS<-interp(data1$Nrate,data1$CLAY,data1$lnRR,nx=500,ny=500,linear=TRUE,duplicate = 'mean')
interplinearS$x
interplinearS$y
interplinearS$z

lgnd=1
lgndmar=5.1
lgnd
lgndmar

cairo_pdf("FS4b.pdf",width = 4.6,height = 3.6)
p1<-image.plot(interplinearS,main=expression(paste("(","b",")"," ","lnRR"," ","(","SOC stock",")")),col.main='black',xlim=c(Xmin,Xmax),ylim=c(Ymin,Ymax),xlab=expression(paste("Nitrogen application rate"," ","(","kg"," ","N"," ",ha^{-1},")")),ylab='Clay content (%)',
               legend.shrink=1,legend.width=1,legend.mar=5.1,horizontal=FALSE)
p1+scale_color_distiller(palette="Spectral")
p1+scale_color_distiller(palette="Greens")
p1
dev.off()
###########
###########NL:间作：MAP-MAT###
data1=read.csv("NLlnRR-akima-MM1.csv",sep=",",header=TRUE)###
head(data1)
dim(data1)
Xrange<-max(data1$MAP)-min(data1$MAP)
Yrange<-max(data1$MAT)-min(data1$MAT)
Zrange<-max(data1$lnRR)-min(data1$lnRR)
Xrange
Yrange
Zrange
Xmax<-ifelse(max(data1$MAP)==0,0.1*abs(Xrange),round(max(data1$MAP)+0.1*abs(Xrange),1))
Xmin<-ifelse(min(data1$MAP)==0,-0.1*abs(Xrange),round(min(data1$MAP)-0.1*abs(Xrange),1))
Xmax
Xmin
Ymax<-ifelse(max(data1$MAT)==0,0.1*abs(Yrange),round(max(data1$MAT)+0.1*abs(Yrange),1))
Ymin<-ifelse(min(data1$MAT)==0,-0.1*abs(Yrange),round(min(data1$MAT)-0.1*abs(Yrange),1))
Ymax
Ymin

interplinearS<-interp(data1$MAP,data1$MAT,data1$lnRR,nx=500,ny=500,linear=TRUE,duplicate = 'mean')
interplinearS$x
interplinearS$y
interplinearS$z

lgnd=1
lgndmar=5.1
lgnd
lgndmar

cairo_pdf("FS4c.pdf",width = 4.6,height = 3.6)
p1<-image.plot(interplinearS,main=expression(paste("(","c",")"," ","lnRR"," ","(","Nitrate leaching",")")),col.main='black',xlim=c(Xmin,Xmax),ylim=c(Ymin,Ymax),xlab=expression(paste("Mean annual precipitation"," ","(mm)")),ylab='Mean annual temperature (℃)',
               legend.shrink=1,legend.width=1,legend.mar=5.1,horizontal=FALSE)
p1+scale_color_distiller(palette="Spectral")
p1+scale_color_distiller(palette="Greens")
p1
dev.off()
###########NL:轮作：MAP-MAT###
data1=read.csv("NLlnRR-akima-MM2.csv",sep=",",header=TRUE)###
head(data1)
dim(data1)
Xrange<-max(data1$MAP)-min(data1$MAP)
Yrange<-max(data1$MAT)-min(data1$MAT)
Zrange<-max(data1$lnRR)-min(data1$lnRR)
Xrange
Yrange
Zrange
Xmax<-ifelse(max(data1$MAP)==0,0.1*abs(Xrange),round(max(data1$MAP)+0.1*abs(Xrange),1))
Xmin<-ifelse(min(data1$MAP)==0,-0.1*abs(Xrange),round(min(data1$MAP)-0.1*abs(Xrange),1))
Xmax
Xmin
Ymax<-ifelse(max(data1$MAT)==0,0.1*abs(Yrange),round(max(data1$MAT)+0.1*abs(Yrange),1))
Ymin<-ifelse(min(data1$MAT)==0,-0.1*abs(Yrange),round(min(data1$MAT)-0.1*abs(Yrange),1))
Ymax
Ymin

interplinearS<-interp(data1$MAP,data1$MAT,data1$lnRR,nx=500,ny=500,linear=TRUE,duplicate = 'mean')
interplinearS$x
interplinearS$y
interplinearS$z

lgnd=1
lgndmar=5.1
lgnd
lgndmar

cairo_pdf("FS4d.pdf",width = 4.6,height = 3.6)
p1<-image.plot(interplinearS,main=expression(paste("(","d",")"," ","lnRR"," ","(","Nitrate leaching",")")),col.main='black',xlim=c(Xmin,Xmax),ylim=c(Ymin,Ymax),xlab=expression(paste("Mean annual precipitation"," ","(mm)")),ylab='Mean annual temperature (℃)',
               legend.shrink=1,legend.width=1,legend.mar=5.1,horizontal=FALSE)
p1+scale_color_distiller(palette="Spectral")
p1+scale_color_distiller(palette="Greens")
p1
dev.off
###########NL:间作：MAP-Nrate###
data1=read.csv("NLlnRR-akima-NM1.csv",sep=",",header=TRUE)###
head(data1)
dim(data1)
Xrange<-max(data1$MAP)-min(data1$MAP)
Yrange<-max(data1$Nrate)-min(data1$Nrate)
Zrange<-max(data1$lnRR)-min(data1$lnRR)
Xrange
Yrange
Zrange
Xmax<-ifelse(max(data1$MAP)==0,0.1*abs(Xrange),round(max(data1$MAP)+0.1*abs(Xrange),1))
Xmin<-ifelse(min(data1$MAP)==0,-0.1*abs(Xrange),round(min(data1$MAP)-0.1*abs(Xrange),1))
Xmax
Xmin
Ymax<-ifelse(max(data1$Nrate)==0,0.1*abs(Yrange),round(max(data1$Nrate)+0.1*abs(Yrange),1))
Ymin<-ifelse(min(data1$Nrate)==0,-0.1*abs(Yrange),round(min(data1$Nrate)-0.1*abs(Yrange),1))
Ymax
Ymin

interplinearS<-interp(data1$MAP,data1$Nrate,data1$lnRR,nx=500,ny=500,linear=TRUE,duplicate = 'mean')
interplinearS$x
interplinearS$y
interplinearS$z

lgnd=1
lgndmar=5.1
lgnd
lgndmar

cairo_pdf("FS4e.pdf",width = 4.6,height = 3.6)
p1<-image.plot(interplinearS,main=expression(paste("(","e",")"," ","lnRR"," ","(","Nitrate leaching",")")),col.main='black',xlim=c(Xmin,Xmax),ylim=c(Ymin,Ymax),xlab=expression(paste("Mean annual precipitation"," ","(mm)")),ylab=expression(paste("Nitrogen application rate"," ","(","kg"," ","N"," ",ha^{-1},")")),
               legend.shrink=1,legend.width=1,legend.mar=5.1,horizontal=FALSE)
p1+scale_color_distiller(palette="Spectral")
p1+scale_color_distiller(palette="Greens")
p1
dev.off()
###########NL:轮作：MAP-Nrate###
data1=read.csv("NLlnRR-akima-NM2.csv",sep=",",header=TRUE)###
head(data1)
dim(data1)
Xrange<-max(data1$MAP)-min(data1$MAP)
Yrange<-max(data1$Nrate)-min(data1$Nrate)
Zrange<-max(data1$lnRR)-min(data1$lnRR)
Xrange
Yrange
Zrange
Xmax<-ifelse(max(data1$MAP)==0,0.1*abs(Xrange),round(max(data1$MAP)+0.1*abs(Xrange),1))
Xmin<-ifelse(min(data1$MAP)==0,-0.1*abs(Xrange),round(min(data1$MAP)-0.1*abs(Xrange),1))
Xmax
Xmin
Ymax<-ifelse(max(data1$Nrate)==0,0.1*abs(Yrange),round(max(data1$Nrate)+0.1*abs(Yrange),1))
Ymin<-ifelse(min(data1$Nrate)==0,-0.1*abs(Yrange),round(min(data1$Nrate)-0.1*abs(Yrange),1))
Ymax
Ymin

interplinearS<-interp(data1$MAP,data1$Nrate,data1$lnRR,nx=500,ny=500,linear=TRUE,duplicate = 'mean')
interplinearS$x
interplinearS$y
interplinearS$z

lgnd=1
lgndmar=5.1
lgnd
lgndmar

cairo_pdf("FS4f.pdf",width = 4.6,height = 3.6)
p1<-image.plot(interplinearS,main=expression(paste("(","f",")"," ","lnRR"," ","(","Nitrate leaching",")")),col.main='black',xlim=c(Xmin,Xmax),ylim=c(Ymin,Ymax),xlab=expression(paste("Mean annual precipitation"," ","(mm)")),ylab=expression(paste("Nitrogen application rate"," ","(","kg"," ","N"," ",ha^{-1},")")),
               legend.shrink=1,legend.width=1,legend.mar=5.1,horizontal=FALSE)
p1+scale_color_distiller(palette="Spectral")
p1+scale_color_distiller(palette="Greens")
p1
dev.off()
###############################################################################################
###############################################################################################
###############################################################################################
################################ending##########################









