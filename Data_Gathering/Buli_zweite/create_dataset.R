setwd("~/Dokumente/Projekte/Football_Game_Prediction/Data_Gathering/Buli_zweite")


# Saison 1993-94
sai_1993_94 = read.csv("1993-94.csv")
dim(sai_1993_94)
class(sai_1993_94$Date) # "character"  26/07/93  -- format bleibt gleich
sai_1993_94 = sai_1993_94[1:380,2:7] 
# 20 Mannschaften => 20*19 = 380 Begegnungen
write.csv(sai_2019_20, file = "~/Dokumente/Projekte/Football_Game_Prediction/Data_Gathering/Buli_zweite/clean/1993-94s.csv")


# Saison 1994-95
sai_1994_95 = read.csv("1994-95.csv")
dim(sai_1994_95)
class(sai_1994_95$Date) # "character"  15/10/94  -- format bleibt gleich
sai_1994_95 = sai_1994_95[1:306,2:7] 
# wieder 18 Mannschaften => 18*17 = 306 Begegnungen
write.csv(sai_1994_95, file = "~/Dokumente/Projekte/Football_Game_Prediction/Data_Gathering/Buli_zweite/clean/1994-95s.csv")


# Saison 1995-96
sai_1995_96 = read.csv("1995-96.csv")
dim(sai_1995_96)
class(sai_1995_96$Date) # "character"  15/09/95  -- format bleibt gleich
sai_1995_96 = sai_1995_96[1:306,2:7] 
# wieder 18 Mannschaften => 18*17 = 306 Begegnungen
write.csv(sai_1995_96, file = "~/Dokumente/Projekte/Football_Game_Prediction/Data_Gathering/Buli_zweite/clean/1995-96s.csv")



# Saison 2019-20
sai_2019_20 = read.csv("2019-20.csv")
dim(sai_2019_20)
class(sai_2019_20$Date) # "character"  26/07/2019  -- format bleibt gleich
sai_2019_20 = sai_2019_20[1:306,2:8] 
sai_2019_20 = sai_2019_20[1:306,-2]
# 18 Mannschaften => 18*17 = 306 Begegnungen
write.csv(sai_2019_20, file = "~/Dokumente/Projekte/Football_Game_Prediction/Data_Gathering/Buli_zweite/clean/2019-20.csv")